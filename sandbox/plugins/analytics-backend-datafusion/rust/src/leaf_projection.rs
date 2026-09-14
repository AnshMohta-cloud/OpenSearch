/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Leaf projection for nested predicates on the plain (non-delegated) DataFusion path.
//!
//! # The problem
//!
//! A nested column is ONE top-level root holding MANY parquet leaves: `events: LIST<STRUCT<name,
//! status, code, weight>>` is four leaf column chunks. DataFusion 55 chooses leaf chunks from the
//! *expressions* (`projection_read_plan::build_projection_read_plan`), and there a whole-column
//! reference reads every leaf beneath the root. `nested_any_match_expr(events, …)` is exactly such a
//! reference, so `count(*) WHERE events.code > 100` reads all four leaves — measured at 1M rows x 1000
//! events: 4,915 MB where one leaf is 1,435 MB.
//!
//! Flat columns never had this problem: a flat column IS its own leaf, so projecting root `x` resolves
//! to precisely `x`'s chunk.
//!
//! # Why a cast and not a narrower schema
//!
//! The indexed path narrows the schema it hands to `ParquetSource` (see
//! `indexed_table::table_provider::narrow_scan_schema`), which works because it happens after the plan
//! is bound. The plain path has no such seam: narrowing the *registered table* schema is rejected before
//! execution, because the Substrait consumer compares the plan's declared type against the table's
//! including struct children:
//!
//! ```text
//! Substrait error: Field 'events' in Substrait schema has a different type
//!   (List(Struct(code, name, status, weight))) than the corresponding field in the
//!   table schema (List(Struct(code), field: 'element'))
//! ```
//!
//! So this rewrites the PREDICATE instead, wrapping the array argument in a cast to the narrow type:
//!
//! ```text
//! nested_any_match_expr(events@10, {…"field":"code"…})
//!   -> nested_any_match_expr(CAST(events@10 AS List(Struct(code))), {…"field":"code"…})
//! ```
//!
//! DataFusion 55 documents this shape as clippable: "roots consumed through a cast, and not also
//! through a `get_field` access on the same root, keep only the leaves the cast target names"
//! (`build_read_plan_with_cast_clipping`). No schema anywhere changes, so nothing rebinds.
//!
//! # Why it is safe
//!
//! - Only the predicate is touched. The scan's output schema is untouched, so every operator above it
//!   keeps the schema it was planned with.
//! - The leaf names come from the UDF's own `{"field":"…"}` nodes — the same nodes `compile_node`
//!   resolves against the struct's `Fields` **by name** at evaluation time. A struct pruned to exactly
//!   that set satisfies every lookup the UDF will perform.
//! - A root referenced any other way is left alone (`collect_nested_leaf_paths` puts it in `needs_full`).
//! - The cast is only injected when `can_cast_types` accepts it, so if clipping declines and the cast
//!   executes for real on the full struct, it is a cast Arrow can actually perform.
//! - `MUSTANG_LEAF_PROJECTION=0` disables it.
//!
//! # Two shapes, two rewrites
//!
//! Measured on the 14-shape benchmark, a nested predicate reaches a parquet scan two different ways:
//!
//! 1. **Pushed into the scan** (`ParquetSource::predicate`) — the delegated path. Casting the predicate's
//!    array argument is enough, because the scan's own predicate is one of the expression sets
//!    `build_projection_read_plan` inspects.
//!
//! 2. **Left in a `FilterExec` above the scan** — what a query with nothing for Lucene to prune gets:
//!
//!    ```text
//!    FilterExec: nested_any_match_expr(events@0, {"op":">","args":[{"field":"code"},{"lit":100}]})
//!      DataSourceExec: projection=[events], file_type=parquet        <- no predicate at all
//!    ```
//!
//!    Here the predicate is invisible to the scan, and the whole-root `projection=[events]` is what pulls
//!    all four leaves. So the PROJECTION is rewritten instead, to `CAST(events AS List(Struct(code)))`,
//!    which is the other form `build_read_plan_with_cast_clipping` clips. The scan's output type for that
//!    column then narrows, so the `FilterExec` above is rebuilt against the new child schema — safe
//!    because the UDF resolves its fields by name.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_expr::expressions::{CastExpr, Column};
use datafusion::physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use native_bridge_common::log_info;

use crate::indexed_table::table_provider::{leaf_projection_enabled, narrow_nested_root};

/// Rewrites nested predicates already pushed into a parquet scan so they read only the leaves they name.
///
/// Best-effort by construction: anything unexpected returns the plan untouched, so the worst case is
/// today's behaviour rather than a failed query.
pub(crate) fn narrow_pushed_nested_predicates(
    plan: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    if !leaf_projection_enabled() {
        return plan;
    }
    let original = Arc::clone(&plan);
    let debug = std::env::var("MUSTANG_LEAF_DEBUG").is_ok();
    if debug {
        log_info!(
            "[LEAF-PROJ] examining plan:\n{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
        );
    }
    plan.transform_up(|node| {
        // Shape 2: the predicate sits in a FilterExec above the scan, so the scan's whole-root
        // projection is the thing that has to be narrowed.
        if let Some(narrowed) = narrow_filter_over_scan(&node, debug)? {
            return Ok(Transformed::yes(narrowed));
        }
        let Some(exec) = node.downcast_ref::<DataSourceExec>() else {
            return Ok(Transformed::no(node));
        };
        let Some(config) = exec
            .data_source()
            .as_ref()
            .downcast_ref::<FileScanConfig>()
        else {
            return Ok(Transformed::no(node));
        };
        let Some(parquet) = config
            .file_source()
            .as_ref()
            .downcast_ref::<ParquetSource>()
        else {
            return Ok(Transformed::no(node));
        };
        let Some(predicate) = parquet.predicate() else {
            // No pushed predicate means the filter stayed in a FilterExec above the scan, where it is
            // invisible to `build_projection_read_plan` — the leaf set would then have to come from the
            // projection instead.
            if debug {
                log_info!("[LEAF-PROJ] parquet scan has NO pushed predicate; nothing to narrow");
            }
            return Ok(Transformed::no(node));
        };
        if debug {
            log_info!("[LEAF-PROJ] pushed predicate: {}", predicate);
        }

        // The predicate is the only place a nested root can be narrowed here: a root that is also
        // projected to the scan's output is a whole-column reference, which takes precedence over any
        // cast on the same root, so DataFusion would read every leaf regardless.
        let leaf_paths = crate::indexed_executor::collect_nested_leaf_paths(
            std::slice::from_ref(predicate),
            config.file_schema(),
        );
        if leaf_paths.is_empty() {
            if debug {
                log_info!("[LEAF-PROJ] no narrowable nested root in the pushed predicate");
            }
            return Ok(Transformed::no(node));
        }

        let Some(new_predicate) =
            cast_nested_args(Arc::clone(predicate), config.file_schema(), &leaf_paths)
        else {
            return Ok(Transformed::no(node));
        };

        // `with_predicate` takes `&self` and CLONES the source, so the projection — which in
        // DataFusion 55 lives on the file source as `ProjectionExprs`, not on `FileScanConfig` —
        // travels with it. That is why this does not need to re-apply the projection the way
        // `ProjectRowIdOptimizer` does when it builds a fresh `ParquetSource`.
        let new_source = Arc::new(parquet.with_predicate(new_predicate));
        let new_config = FileScanConfigBuilder::from(config.clone())
            .with_source(new_source)
            .build();
        Ok(Transformed::yes(
            DataSourceExec::from_data_source(new_config) as Arc<dyn ExecutionPlan>,
        ))
    })
    .data()
    .unwrap_or(original)
}

/// Wraps each `nested_any_match_expr` array argument in a cast to its narrowed `LIST<STRUCT>` type.
///
/// Returns `None` when nothing was rewritten, so the caller can leave the plan alone.
fn cast_nested_args(
    predicate: Arc<dyn PhysicalExpr>,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    leaf_paths: &std::collections::HashMap<usize, Vec<String>>,
) -> Option<Arc<dyn PhysicalExpr>> {
    let mut rewrote = false;
    let out = predicate
        .transform_down(|e| {
            // Collect what we need and drop the borrow before touching `e` again: `ScalarFunctionExpr`
            // is not `Clone`, so the rewrite goes through `with_new_children` on the original Arc.
            let info = e.downcast_ref::<ScalarFunctionExpr>().and_then(|f| {
                if f.name() != "nested_any_match_expr" || f.args().len() != 2 {
                    return None;
                }
                let col = f.args()[0].downcast_ref::<Column>()?;
                Some((
                    Arc::clone(&f.args()[0]),
                    Arc::clone(&f.args()[1]),
                    col.index(),
                ))
            });
            let Some((array_arg, json_arg, root)) = info else {
                return Ok(Transformed::no(e));
            };
            let Some(leaves) = leaf_paths.get(&root) else {
                return Ok(Transformed::no(e));
            };
            let Some(field) = schema.fields().get(root) else {
                return Ok(Transformed::no(e));
            };
            let Some(narrow) = narrow_nested_root(field, leaves) else {
                return Ok(Transformed::no(e));
            };
            // NOTE: deliberately NOT gated on `can_cast_types`. Arrow casts structs field-by-field by
            // POSITION and refuses a differing field count, so it reports `List(Struct(4)) ->
            // List(Struct(1))` as impossible — which would veto every narrowing this module exists to
            // perform. The cast is not there to be executed: DataFusion's own
            // `DefaultPhysicalExprAdapter` emits exactly this shape, and
            // `build_read_plan_with_cast_clipping` clips the read so the decoded column already has the
            // narrow type, leaving the cast an identity. `narrow_nested_root` guarantees the target is a
            // by-name subset of the source, so if clipping ever declined the failure would be a loud
            // error rather than a wrong answer.
            let cast = Arc::new(CastExpr::new(array_arg, narrow.data_type().clone(), None))
                as Arc<dyn PhysicalExpr>;
            let rebuilt = Arc::clone(&e).with_new_children(vec![cast, json_arg])?;
            rewrote = true;
            log_info!(
                "[LEAF-PROJ] cast {} to leaves {:?} for nested predicate",
                field.name(),
                leaves
            );
            Ok(Transformed::yes(rebuilt))
        })
        .data()
        .ok()?;
    if rewrote {
        Some(out)
    } else {
        None
    }
}

/// Shape 2: `FilterExec(nested_any_match_expr(events, …)) -> DataSourceExec(projection=[events])`.
///
/// Rewrites the SCAN's projection to `CAST(events AS <narrow>)` and rebuilds the filter above it. Returns
/// `Ok(None)` whenever the shape does not match or the narrowing is not provably safe, so the caller
/// leaves the plan alone.
fn narrow_filter_over_scan(
    node: &Arc<dyn ExecutionPlan>,
    debug: bool,
) -> datafusion::common::Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(filter) = node.downcast_ref::<FilterExec>() else {
        return Ok(None);
    };
    let children = filter.children();
    let [child] = children.as_slice() else {
        return Ok(None);
    };
    let Some(scan) = child.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some(config) = scan.data_source().as_ref().downcast_ref::<FileScanConfig>() else {
        return Ok(None);
    };
    let Some(parquet) = config
        .file_source()
        .as_ref()
        .downcast_ref::<ParquetSource>()
    else {
        return Ok(None);
    };
    // Only this shape: a scan that already carries its own predicate is handled by the caller's other arm.
    if parquet.predicate().is_some() {
        return Ok(None);
    }

    let file_schema = config.file_schema();
    // Index spaces differ and it matters: a `FilterExec` predicate references its CHILD's output schema
    // (`events@0` when the scan projects one column), not the file schema (`events@10`). Deriving against
    // the file schema silently resolves index 0 to an unrelated column and declines every time.
    //
    // Taken from the child plan rather than `FileScanConfig::projected_schema()`, which returns a Result:
    // a `?` there escapes the enclosing `transform_up` closure and abandons the whole rewrite silently.
    let projected = child.schema();
    if debug {
        log_info!(
            "[LEAF-PROJ] filter over scan: predicate={} child_schema={:?}",
            filter.predicate(),
            projected
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
        );
    }

    let leaf_paths = crate::indexed_executor::collect_nested_leaf_paths(
        std::slice::from_ref(filter.predicate()),
        &projected,
    );
    if leaf_paths.is_empty() {
        if debug {
            log_info!("[LEAF-PROJ] filter over scan: no narrowable nested root");
        }
        return Ok(None);
    }

    // Which index space the Column inside the cast must use is not knowable from the API contract, and
    // getting it wrong is silent: `try_pushdown_projection` MERGES this projection over the source's
    // existing one (`self.projection.try_merge(projection)`), so the Columns may have to address that
    // projection's OUTPUT (events@0) rather than the file schema (events@10). Try both and use whichever
    // the source accepts.
    let mut chosen = None;
    let mut last_err = String::from("no attempt made");
    for use_projected_index in [true, false] {
        let Some(exprs) = build_narrowed_projection(
            &projected,
            file_schema,
            &leaf_paths,
            use_projected_index,
            debug && use_projected_index,
        ) else {
            return Ok(None);
        };
        match parquet.try_pushdown_projection(&ProjectionExprs::new(exprs)) {
            Ok(Some(src)) => {
                if debug {
                    log_info!(
                        "[LEAF-PROJ] projection accepted using the {} index space",
                        if use_projected_index { "projected" } else { "file" }
                    );
                }
                chosen = Some(src);
                break;
            }
            Ok(None) => last_err = "source returned None".to_string(),
            Err(e) => last_err = format!("{e}"),
        }
    }
    let Some(new_source) = chosen else {
        // This was a bare `?` before: the error escaped the `transform_up` closure, the rewrite was
        // abandoned, and nothing was logged — the plan just silently stayed as it was.
        log_info!(
            "[LEAF-PROJ] filter over scan: parquet source rejected the projection: {}",
            last_err
        );
        return Ok(None);
    };

    if debug {
        log_info!(
            "[LEAF-PROJ] source projection after merge: {:?}",
            new_source
                .as_ref()
                .downcast_ref::<ParquetSource>()
                .and_then(|p| p.projection())
                .map(|p| format!("{p:?}"))
        );
    }

    let new_config = FileScanConfigBuilder::from(config.clone())
        .with_source(new_source)
        .build();
    let new_scan = DataSourceExec::from_data_source(new_config) as Arc<dyn ExecutionPlan>;
    if debug {
        log_info!(
            "[LEAF-PROJ] rewritten scan:\n{}",
            datafusion::physical_plan::displayable(new_scan.as_ref()).indent(false)
        );
    }

    // Rebuild the filter against the narrowed child so its input schema matches what the scan now emits.
    let mut rebuilt = FilterExec::try_new(Arc::clone(filter.predicate()), new_scan)?;
    if let Some(proj) = filter.projection() {
        rebuilt = rebuilt.with_projection(Some(proj.to_vec()))?;
    }
    Ok(Some(Arc::new(rebuilt) as Arc<dyn ExecutionPlan>))
}

/// One projection entry per projected column, with the narrowable nested roots wrapped in a cast.
///
/// `use_projected_index` picks the index space for the inner `Column` (see the caller). Returns `None`
/// when nothing would be narrowed, or when a projected column is not a plain file column.
fn build_narrowed_projection(
    projected: &datafusion::arrow::datatypes::SchemaRef,
    file_schema: &datafusion::arrow::datatypes::SchemaRef,
    leaf_paths: &std::collections::HashMap<usize, Vec<String>>,
    use_projected_index: bool,
    debug: bool,
) -> Option<Vec<ProjectionExpr>> {
    let mut exprs: Vec<ProjectionExpr> = Vec::with_capacity(projected.fields().len());
    let mut narrowed_any = false;
    for (out_idx, out_field) in projected.fields().iter().enumerate() {
        // A projected column that is not a plain file column (partition or virtual) means the index
        // mapping below would be a guess, so decline rather than risk it.
        let file_idx = file_schema.index_of(out_field.name()).ok()?;
        let col = Arc::new(Column::new(
            out_field.name(),
            if use_projected_index { out_idx } else { file_idx },
        )) as Arc<dyn PhysicalExpr>;
        match leaf_paths
            .get(&out_idx)
            .and_then(|leaves| narrow_nested_root(out_field, leaves))
        {
            // Not gated on `can_cast_types` — see the note in `cast_nested_args`.
            Some(nf) => {
                narrowed_any = true;
                if debug {
                    log_info!(
                        "[LEAF-PROJ] filter over scan: projecting {} as {:?}",
                        out_field.name(),
                        nf.data_type()
                    );
                }
                exprs.push(ProjectionExpr::new(
                    Arc::new(CastExpr::new(col, nf.data_type().clone(), None)),
                    out_field.name().clone(),
                ));
            }
            None => exprs.push(ProjectionExpr::new(col, out_field.name().clone())),
        }
    }
    if narrowed_any {
        Some(exprs)
    } else {
        None
    }
}

/// Narrows a parquet scan's projection to the named leaves, then lets `transform_up` rebuild every
/// operator above it so their schemas re-derive.
///
/// This is the unnest path. `stats count() by Events.name` and `expand` build a reshaping unnest above
/// the scan out of standard DataFusion nodes (Projection + Unnest x2 + rename, see
/// `unnest_consumer::build_reshaping_unnest`), and the scan projects the whole `Events` root, so every
/// leaf is read: measured 75.4 MB where `name` is 0.12 MB.
///
/// Narrowing the scan's SCHEMA does not work here -- it was tried and the reshaping unnest, built against
/// the full element struct, rejected the batch with
/// "column types must match schema types, expected List(Struct(cache_key, exception_message,
/// exception_type, name, retry_count))". Rewriting the PROJECTION to a cast works instead, because
/// `transform_up` rebuilds Projection/Unnest/Aggregate through `with_new_children`, and each of those
/// re-derives its schema from its input.
///
/// `leaf_paths` is keyed by column NAME (not index) so it cannot be mis-applied if the scan's schema is
/// ordered differently from the table's.
pub(crate) fn narrow_scan_projection(
    plan: Arc<dyn ExecutionPlan>,
    leaf_paths: &std::collections::HashMap<String, Vec<String>>,
) -> Arc<dyn ExecutionPlan> {
    if leaf_paths.is_empty() || !leaf_projection_enabled() {
        return plan;
    }
    let debug = std::env::var("MUSTANG_LEAF_DEBUG").is_ok();
    let original = Arc::clone(&plan);
    plan.transform_up(|node| {
        let Some(exec) = node.downcast_ref::<DataSourceExec>() else {
            return Ok(Transformed::no(node));
        };
        let Some(config) = exec.data_source().as_ref().downcast_ref::<FileScanConfig>() else {
            return Ok(Transformed::no(node));
        };
        let Some(parquet) = config
            .file_source()
            .as_ref()
            .downcast_ref::<ParquetSource>()
        else {
            return Ok(Transformed::no(node));
        };
        let file_schema = config.file_schema();
        let projected = node.schema();

        // Map the by-name leaf set onto this scan's projected output positions.
        let mut by_out_idx: std::collections::HashMap<usize, Vec<String>> =
            std::collections::HashMap::new();
        for (out_idx, f) in projected.fields().iter().enumerate() {
            if let Some(leaves) = leaf_paths.get(f.name()) {
                by_out_idx.insert(out_idx, leaves.clone());
            }
        }
        if by_out_idx.is_empty() {
            return Ok(Transformed::no(node));
        }

        let mut chosen = None;
        let mut last_err = String::from("no attempt made");
        for use_projected_index in [true, false] {
            let Some(exprs) = build_narrowed_projection(
                &projected,
                file_schema,
                &by_out_idx,
                use_projected_index,
                debug && use_projected_index,
            ) else {
                return Ok(Transformed::no(node));
            };
            match parquet.try_pushdown_projection(&ProjectionExprs::new(exprs)) {
                Ok(Some(src)) => {
                    chosen = Some(src);
                    break;
                }
                Ok(None) => last_err = "source returned None".to_string(),
                Err(e) => last_err = format!("{e}"),
            }
        }
        let Some(new_source) = chosen else {
            log_info!(
                "[LEAF-PROJ] unnest path: parquet source rejected the projection: {}",
                last_err
            );
            return Ok(Transformed::no(node));
        };
        let new_config = FileScanConfigBuilder::from(config.clone())
            .with_source(new_source)
            .build();
        if debug {
            log_info!(
                "[LEAF-PROJ] unnest path: narrowed scan projection for {:?}",
                by_out_idx
            );
        }
        Ok(Transformed::yes(
            DataSourceExec::from_data_source(new_config) as Arc<dyn ExecutionPlan>,
        ))
    })
    .data()
    .unwrap_or(original)
}

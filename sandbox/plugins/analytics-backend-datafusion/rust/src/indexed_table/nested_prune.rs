/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Row-group and page pruning for NESTED predicates.
//!
//! # The gap this closes
//!
//! A nested filter reaches the scan as one opaque UDF call:
//!
//! ```text
//! nested_any_match_expr(Events@19, {"op":"=","args":[{"field":"cache_key"},{"lit":"k5"}]})
//! ```
//!
//! `PruningPredicate::try_new` cannot derive min/max logic from an unknown UDF, so it collapses to
//! `always_true`, `build_prune_tree_config` returns `None`, and neither row-group nor page statistics are
//! ever consulted. Measured on a 1M-document index: `Events.cache_key='k5'` matches **333 rows (0.03%)**
//! and pruned **zero** pages, reading the whole 73 MB leaf, while a scalar predicate of similar
//! selectivity pruned 45 of 46 pages. Verified independent of `prune_only` (identical counters with the
//! setting on and off), so this is a long-standing gap rather than a regression.
//!
//! # Why pruning a nested existential is sound
//!
//! The predicate is `∃e ∈ Events : T(e)`. If `T = A ∧ rest`, then `∃e:T(e) ⟹ ∃e:A(e)`. A row group's
//! statistics for the leaf column `A` describe every element of every row in that group, so if no value
//! there can satisfy `A`, no element satisfies `A`, hence none satisfies `T`, hence no ROW in the group
//! matches. Dropping the group is therefore safe. This is the same superset argument
//! `PruneOnlyRelaxer` documents for the Lucene peers.
//!
//! Two consequences, both enforced below:
//! - Only leaves on an **AND-only spine** are collected. Under `OR` a leaf is not implied by the whole
//!   predicate, so it cannot prune.
//! - `NOT` and `!=` are **excluded**. Negation is non-monotone: complementing a superset yields a
//!   subset, which would drop rows that belong in the answer.
//!
//! # Naming
//!
//! Synthetic columns are named `__nested__<root>__<dotted leaf path>`. A plain `<root>.<leaf>` spelling
//! would be ambiguous: an OpenSearch `object` field produces a REAL top-level column with a dot in its
//! name (`ResourceAttributes.host_name`), so the prefix keeps the two apart.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::parquet::file::metadata::RowGroupMetaData;
use datafusion::parquet::file::page_index::column_index::ColumnIndexMetaData;
use datafusion::parquet::schema::types::SchemaDescriptor;
use datafusion::physical_expr::expressions::{lit, BinaryExpr, Column};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::pruning::PruningPredicate;
use datafusion::logical_expr::Operator;
use serde_json::Value as Json;

/// Prefix marking a synthetic pruning column. See the module note on naming.
pub const NESTED_PRUNE_PREFIX: &str = "__nested__";

/// Nested leaves report **no** null counts, on purpose.
///
/// A repeated column's `null_count` counts null *elements*, while `PruningStatistics::row_counts`
/// reports *rows*. `PruningPredicate` compares the two (a container whose `null_count` equals its
/// `row_count` holds only nulls and is prunable), so feeding element-unit nulls against row-unit
/// counts could make a page with 8 null elements across 8 two-element rows look entirely null and be
/// skipped -- dropping matching rows. Withholding them costs only the all-null-container case, which
/// min/max bounds already cover for every predicate `harvest` emits.
const NO_NULL_COUNTS: Option<ArrayRef> = None;

/// One nested leaf a synthesized predicate references.
#[derive(Debug, Clone)]
pub struct NestedLeafRef {
    /// Name used in the pruning schema and predicate, e.g. `__nested__Events__cache_key`.
    pub synthetic_name: String,
    /// Dotted path of the leaf inside the element struct, e.g. `cache_key` or `tags.name`.
    pub leaf_path: String,
    /// Top-level nested column, e.g. `Events`.
    pub root: String,
    pub data_type: DataType,
}

/// Builds the synthetic column name for a (root, leaf path) pair.
pub fn synthetic_name(root: &str, leaf_path: &str) -> String {
    format!("{NESTED_PRUNE_PREFIX}{root}__{leaf_path}")
}

/// A prunable comparison harvested from a nested predicate tree.
struct Harvested {
    leaf_path: String,
    op: Operator,
    lit_json: Json,
}

/// Collects prunable comparisons from a `nested_any_match_expr` JSON tree.
///
/// Descends only through `AND` and `{"nested":…}`; `OR`, `NOT`, `!=` and anything unrecognised are
/// skipped (not failed) so an unsupported corner simply yields less pruning, never wrong pruning.
fn harvest(node: &Json, prefix: &str, out: &mut Vec<Harvested>) {
    if let Some(op) = node.get("op").and_then(|v| v.as_str()) {
        let args = node.get("args").and_then(|v| v.as_array());
        match op {
            "AND" => {
                if let Some(args) = args {
                    for a in args {
                        harvest(a, prefix, out);
                    }
                }
                return;
            }
            // Not implied by the whole predicate / non-monotone: contribute nothing.
            "OR" | "NOT" | "!=" => return,
            _ => {}
        }
        let Some(args) = args else { return };
        if args.len() != 2 {
            return;
        }
        // One side must be a bare field, the other a literal. Arithmetic on either side is skipped:
        // min/max of `code` says nothing about min/max of `code * 2 + 1`.
        let (field, lit_json, op) = if let (Some(f), Some(l)) =
            (args[0].get("field").and_then(|v| v.as_str()), args[1].get("lit"))
        {
            (f, l, op)
        } else if let (Some(l), Some(f)) =
            (args[0].get("lit"), args[1].get("field").and_then(|v| v.as_str()))
        {
            // Literal on the left: flip the operator so the field stays on the left.
            let flipped = match op {
                ">" => "<",
                ">=" => "<=",
                "<" => ">",
                "<=" => ">=",
                other => other,
            };
            (f, l, flipped)
        } else {
            return;
        };
        let operator = match op {
            "=" => Operator::Eq,
            ">" => Operator::Gt,
            ">=" => Operator::GtEq,
            "<" => Operator::Lt,
            "<=" => Operator::LtEq,
            _ => return,
        };
        let leaf_path = if prefix.is_empty() {
            field.to_string()
        } else {
            format!("{prefix}.{field}")
        };
        out.push(Harvested { leaf_path, op: operator, lit_json: lit_json.clone() });
        return;
    }
    // Descend one nesting level: `{"nested":"tags","inner":<subtree>}`.
    if let (Some(hop), Some(inner)) = (
        node.get("nested").and_then(|v| v.as_str()),
        node.get("inner"),
    ) {
        let next = if prefix.is_empty() {
            hop.to_string()
        } else {
            format!("{prefix}.{hop}")
        };
        harvest(inner, &next, out);
    }
}

/// Resolves a dotted leaf path inside a nested root to its Arrow type, following `List<Struct>` levels.
/// Returns `None` for a MAP hop (its statistics are per column chunk, not per key, so a key predicate is
/// not prunable) or any path that does not resolve to a primitive.
fn resolve_leaf_type(root_type: &DataType, leaf_path: &str) -> Option<DataType> {
    let mut current = root_type.clone();
    let mut parts = leaf_path.split('.').peekable();
    while let Some(part) = parts.next() {
        let DataType::List(elem) = &current else {
            return None;
        };
        let DataType::Struct(children) = elem.data_type() else {
            return None;
        };
        let child = children.iter().find(|c| c.name() == part)?;
        if parts.peek().is_none() {
            return match child.data_type() {
                DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
                | DataType::Boolean => Some(child.data_type().clone()),
                // Temporal leaves are excluded until the literal's UNIT contract is pinned down.
                // Parquet has no 64-bit DATE, so a Date64 column is written as INT32 *days* while the
                // Arrow type says *milliseconds*; reading those stats as ms puts min/max off by a
                // factor of 86,400,000 and could prune a row group that does match. The JSON literal
                // side is equally unpinned (OpenSearch dates are epoch millis, Parquet Date32 is
                // days). Declining costs pruning on date leaves and nothing else.
                DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => None,
                // Map/List/Struct leaves have no scalar min/max to compare against.
                _ => None,
            };
        }
        current = child.data_type().clone();
    }
    None
}

/// Converts a JSON literal to a `ScalarValue` of the leaf's type. Returns `None` when the literal cannot
/// be represented in that type, so a type mismatch yields no pruning rather than a wrong comparison.
fn json_to_scalar(v: &Json, dt: &DataType) -> Option<ScalarValue> {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => v.as_str().map(|s| ScalarValue::Utf8(Some(s.to_string()))),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => {
            v.as_i64().and_then(|i| i32::try_from(i).ok()).map(|i| ScalarValue::Int32(Some(i)))
        }
        DataType::Int64 => v.as_i64().map(|i| ScalarValue::Int64(Some(i))),
        DataType::Float32 => v.as_f64().map(|f| ScalarValue::Float32(Some(f as f32))),
        DataType::Float64 => v.as_f64().map(|f| ScalarValue::Float64(Some(f))),
        DataType::Boolean => v.as_bool().map(|b| ScalarValue::Boolean(Some(b))),
        _ => None,
    }
}

/// Synthesizes a pruning predicate for one `nested_any_match_expr` call.
///
/// Returns the predicate plus the leaves it references, or `None` when nothing prunable was found.
pub fn synthesize(
    root: &str,
    root_type: &DataType,
    tree_json: &str,
) -> Option<(Arc<PruningPredicate>, Vec<NestedLeafRef>)> {
    let tree: Json = serde_json::from_str(tree_json).ok()?;
    let mut harvested = Vec::new();
    harvest(&tree, "", &mut harvested);
    if harvested.is_empty() {
        return None;
    }

    let mut fields: Vec<Field> = Vec::new();
    let mut refs: Vec<NestedLeafRef> = Vec::new();
    let mut conjuncts: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
    for h in &harvested {
        let Some(dt) = resolve_leaf_type(root_type, &h.leaf_path) else {
            continue;
        };
        let Some(scalar) = json_to_scalar(&h.lit_json, &dt) else {
            continue;
        };
        // Belt and braces: the literal must already BE the leaf's type. Anything else is a bug in
        // `json_to_scalar`, and rather than hand `PruningPredicate` a mismatched comparison (which it
        // may answer wrongly or panic on) this drops the conjunct, costing pruning and nothing else.
        if scalar.data_type() != dt {
            native_bridge_common::log_debug!(
                "[NESTED-PRUNE] skipping {}: literal type {:?} != leaf type {:?}",
                h.leaf_path,
                scalar.data_type(),
                dt
            );
            continue;
        }
        let name = synthetic_name(root, &h.leaf_path);
        // A leaf can appear twice (`code > 5 AND code < 90`); declare its field once.
        let idx = match fields.iter().position(|f| f.name() == &name) {
            Some(i) => i,
            None => {
                fields.push(Field::new(&name, dt.clone(), true));
                refs.push(NestedLeafRef {
                    synthetic_name: name.clone(),
                    leaf_path: h.leaf_path.clone(),
                    root: root.to_string(),
                    data_type: dt.clone(),
                });
                fields.len() - 1
            }
        };
        conjuncts.push(Arc::new(BinaryExpr::new(
            Arc::new(Column::new(&name, idx)),
            h.op,
            lit(scalar),
        )));
    }
    if conjuncts.is_empty() {
        return None;
    }
    let schema: SchemaRef = Arc::new(Schema::new(fields));
    let expr = conjuncts
        .into_iter()
        .reduce(|a, b| Arc::new(BinaryExpr::new(a, Operator::And, b)))?;
    let pp = PruningPredicate::try_new(expr, schema).ok()?;
    if pp.always_true() {
        return None;
    }
    Some((Arc::new(pp), refs))
}

/// Parquet leaf index whose column path is `<root>.list.element.<leaf path>`.
///
/// Matched on the path's own segments rather than a formatted string so the repetition wrappers the
/// writer emits (`list`/`element`) are skipped positionally instead of assumed.
pub fn parquet_leaf_index(descr: &SchemaDescriptor, root: &str, leaf_path: &str) -> Option<usize> {
    let want: Vec<&str> = std::iter::once(root).chain(leaf_path.split('.')).collect();
    (0..descr.num_columns()).find(|&i| {
        let col = descr.column(i);
        let stripped: Vec<&str> = col
            .path()
            .parts()
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s != "list" && *s != "element")
            .collect();
        stripped == want
    })
}

/// Per-row-group min/max/null-count for a nested leaf, read straight from the parquet footer.
///
/// `StatisticsConverter` cannot serve this: it resolves a TOP-LEVEL Arrow field name to a parquet leaf,
/// and `Events.cache_key` is not a top-level field. Returns `None` if any row group lacks statistics, so
/// a partially-instrumented file prunes nothing rather than pruning on incomplete bounds.
pub fn nested_leaf_rg_stats(
    rg_metas: &[&RowGroupMetaData],
    leaf_idx: usize,
    dt: &DataType,
) -> Option<(ArrayRef, ArrayRef, Option<ArrayRef>)> {
    let mut mins: Vec<Option<ScalarValue>> = Vec::with_capacity(rg_metas.len());
    let mut maxes: Vec<Option<ScalarValue>> = Vec::with_capacity(rg_metas.len());
    for rg in rg_metas {
        let col = rg.columns().get(leaf_idx)?;
        let Some(stats) = col.statistics() else {
            return None;
        };
        mins.push(stat_to_scalar(stats, dt, true));
        maxes.push(stat_to_scalar(stats, dt, false));
    }
    let min_arr = scalars_to_array(&mins, dt)?;
    let max_arr = scalars_to_array(&maxes, dt)?;
    Some((min_arr, max_arr, NO_NULL_COUNTS))
}

fn stat_to_scalar(
    stats: &datafusion::parquet::file::statistics::Statistics,
    dt: &DataType,
    want_min: bool,
) -> Option<ScalarValue> {
    use datafusion::parquet::file::statistics::Statistics as S;
    macro_rules! pick {
        ($s:expr) => {
            if want_min {
                $s.min_opt()?
            } else {
                $s.max_opt()?
            }
        };
    }
    let v = match (stats, dt) {
        (S::ByteArray(s), DataType::Utf8 | DataType::LargeUtf8) => {
            ScalarValue::Utf8(Some(pick!(s).as_utf8().ok()?.to_string()))
        }
        (S::Int32(s), _) => ScalarValue::Int32(Some(*pick!(s))),
        (S::Int64(s), _) => ScalarValue::Int64(Some(*pick!(s))),
        (S::Float(s), _) => ScalarValue::Float64(Some(*pick!(s) as f64)),
        (S::Double(s), _) => ScalarValue::Float64(Some(*pick!(s))),
        (S::Boolean(s), _) => ScalarValue::Boolean(Some(*pick!(s))),
        _ => return None,
    };
    Some(v)
}

/// Builds the stats array for a leaf, in the leaf's OWN Arrow type. The type must match the synthetic
/// field's declared type exactly: a mismatch makes `PruningPredicate` prune nothing, or panic inside
/// its simplifier when the two are not even comparable.
fn scalars_to_array(vals: &[Option<ScalarValue>], dt: &DataType) -> Option<ArrayRef> {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => Some(Arc::new(StringArray::from_iter(
            vals.iter().map(|v| match v {
                Some(ScalarValue::Utf8(s)) => s.clone(),
                _ => None,
            }),
        )) as ArrayRef),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => {
            Some(Arc::new(Int32Array::from_iter(vals.iter().map(|v| match v {
                Some(ScalarValue::Int32(i)) => *i,
                Some(ScalarValue::Int64(Some(i))) => i32::try_from(*i).ok(),
                _ => None,
            }))) as ArrayRef)
        }
        DataType::Int64 => Some(Arc::new(Int64Array::from_iter(vals.iter().map(|v| match v {
            Some(ScalarValue::Int64(i)) => *i,
            Some(ScalarValue::Int32(Some(i))) => Some(*i as i64),
            _ => None,
        }))) as ArrayRef),
        // Float32 must stay Float32: the synthetic field is declared with the leaf's own type, and a
        // width mismatch makes `PruningPredicate::prune` error out (i.e. prune nothing at all).
        DataType::Float32 => Some(Arc::new(Float32Array::from_iter(vals.iter().map(|v| match v {
            Some(ScalarValue::Float64(f)) => f.map(|f| f as f32),
            Some(ScalarValue::Float32(f)) => *f,
            _ => None,
        }))) as ArrayRef),
        DataType::Float64 => Some(Arc::new(Float64Array::from_iter(vals.iter().map(|v| match v {
            Some(ScalarValue::Float64(f)) => *f,
            Some(ScalarValue::Float32(f)) => f.map(|f| f as f64),
            _ => None,
        }))) as ArrayRef),
        DataType::Boolean => Some(Arc::new(BooleanArray::from_iter(vals.iter().map(|v| match v {
            Some(ScalarValue::Boolean(b)) => *b,
            _ => None,
        }))) as ArrayRef),
        _ => None,
    }
}

/// Stats for every synthetic column a synthesized predicate references, keyed by synthetic name so the
/// existing `PruningStatistics` impls (which look up by column name) can serve them unchanged.
pub fn nested_stats_by_name(
    refs: &[NestedLeafRef],
    descr: &SchemaDescriptor,
    rg_metas: &[&RowGroupMetaData],
) -> HashMap<String, (ArrayRef, ArrayRef, Option<ArrayRef>)> {
    let mut out = HashMap::new();
    for r in refs {
        let Some(idx) = parquet_leaf_index(descr, &r.root, &r.leaf_path) else {
            continue;
        };
        if let Some(stats) = nested_leaf_rg_stats(rg_metas, idx, &r.data_type) {
            out.insert(r.synthetic_name.clone(), stats);
        }
    }
    out
}

/// Splits a synthetic column name back into `(root, leaf path)`.
///
/// The name carries everything the stats side needs, so no side table has to be threaded from the
/// planner down into row-group evaluation.
pub fn parse_synthetic_name(name: &str) -> Option<(&str, &str)> {
    name.strip_prefix(NESTED_PRUNE_PREFIX)?.split_once("__")
}

/// Synthesizes a pruning predicate for a physical expression, if it is a nested-predicate UDF call.
///
/// Called where `build_pruning_predicate` has already returned `None` — i.e. exactly the case where
/// DataFusion could not see through the UDF.
pub fn synthesize_from_expr(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
) -> Option<(Arc<PruningPredicate>, Vec<NestedLeafRef>)> {
    use datafusion::physical_expr::expressions::Literal;
    use datafusion::physical_expr::ScalarFunctionExpr;

    let f = expr.downcast_ref::<ScalarFunctionExpr>()?;
    if f.name() != "nested_any_match_expr" || f.args().len() != 2 {
        return None;
    }
    let col = f.args()[0].downcast_ref::<Column>()?;
    let lit_expr = f.args()[1].downcast_ref::<Literal>()?;
    let json = match lit_expr.value() {
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => v.as_str(),
        _ => return None,
    };
    let field = schema.fields().get(col.index())?;
    synthesize(field.name(), field.data_type(), json)
}

/// Resolves a synthetic column's leaf type from the table schema.
pub fn leaf_type_for(schema: &SchemaRef, root: &str, leaf_path: &str) -> Option<DataType> {
    let idx = schema.index_of(root).ok()?;
    resolve_leaf_type(schema.field(idx).data_type(), leaf_path)
}

/// Per-PAGE min/max/null-count for a nested leaf, read from the ColumnIndex.
///
/// The row-group variant reads the footer's per-chunk statistics; this reads the finer per-page index,
/// which is what lets a selective nested predicate skip pages inside a row group it cannot skip whole.
///
/// Page boundaries are usable for a REPEATED column here because a record's repeated values never
/// straddle a page in files this writer produces (verified empirically at 5k/20k/50k/1M rows plus an
/// oversized-record case: zero straddles), so a page's row range is record-aligned and
/// `OffsetIndex::first_row_index` addresses rows rather than values.
///
/// Returns `None` when the column carries no min/max page index (`NONE`) or an unsupported physical
/// type, so an unindexed column prunes nothing rather than pruning on absent bounds.
pub fn nested_leaf_page_stats(
    column_index: &[Vec<ColumnIndexMetaData>],
    rg_idx: usize,
    leaf_idx: usize,
    dt: &DataType,
) -> Option<(ArrayRef, ArrayRef, Option<ArrayRef>)> {
    let idx = column_index.get(rg_idx)?.get(leaf_idx)?;

    macro_rules! prim {
        ($ci:expr, $conv:expr) => {{
            let mins: Vec<Option<ScalarValue>> = $ci.min_values_iter().map(|v| v.map($conv)).collect();
            let maxes: Vec<Option<ScalarValue>> = $ci.max_values_iter().map(|v| v.map($conv)).collect();
            (mins, maxes)
        }};
    }

    let (mins, maxes) = match idx {
        ColumnIndexMetaData::INT32(ci) => prim!(ci, |v: &i32| ScalarValue::Int32(Some(*v))),
        ColumnIndexMetaData::INT64(ci) => prim!(ci, |v: &i64| ScalarValue::Int64(Some(*v))),
        ColumnIndexMetaData::FLOAT(ci) => prim!(ci, |v: &f32| ScalarValue::Float64(Some(*v as f64))),
        ColumnIndexMetaData::DOUBLE(ci) => prim!(ci, |v: &f64| ScalarValue::Float64(Some(*v))),
        ColumnIndexMetaData::BOOLEAN(ci) => prim!(ci, |v: &bool| ScalarValue::Boolean(Some(*v))),
        ColumnIndexMetaData::BYTE_ARRAY(ci) | ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(ci) => {
            let conv = |v: &[u8]| ScalarValue::Utf8(std::str::from_utf8(v).ok().map(|s| s.to_string()));
            let mins: Vec<Option<ScalarValue>> = ci.min_values_iter().map(|v| v.map(conv)).collect();
            let maxes: Vec<Option<ScalarValue>> = ci.max_values_iter().map(|v| v.map(conv)).collect();
            (mins, maxes)
        }
        _ => return None,
    };

    let min_arr = scalars_to_array(&mins, dt)?;
    let max_arr = scalars_to_array(&maxes, dt)?;
    Some((min_arr, max_arr, NO_NULL_COUNTS))
}

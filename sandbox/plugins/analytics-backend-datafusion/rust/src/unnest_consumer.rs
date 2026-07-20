// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.

//! [NESTED-POC] Unnest-aware Substrait consumer.
//!
//! Substrait (and isthmus 0.89.1) has no first-class relational UNNEST, and DataFusion 54's
//! stock Substrait consumer (`DefaultSubstraitConsumer`) has no handler for it either — so the
//! N1 rewrite shape `Scan -> UNNEST(nested) -> Filter -> Aggregate` cannot cross the bridge as-is.
//!
//! DataFusion *does* execute unnest natively (`LogicalPlan::Unnest`), so the only missing link is
//! carrying the operator across Substrait. We do that with the spec's escape hatch: an
//! `ExtensionSingleRel` (one input + an opaque `detail`). The Java producer emits
//! `ExtensionSingleRel{ input: <scan>, detail.type_url: "unnest:<column>" }`; here we recognise
//! that type_url, convert the input, and build a real `LogicalPlan::Unnest` on the named column via
//! `LogicalPlanBuilder::unnest_column`. Every other rel delegates to the stock consumer.
//!
//! This is the general path: it works for any nested column, any depth, under any Filter/Aggregate,
//! because it produces the same `LogicalPlan::Unnest` the SQL planner would. Grep: NESTED-POC.

use std::sync::Arc;

use datafusion::common::{Column, DFSchema, DataFusionError, UnnestOptions};
use datafusion::execution::{FunctionRegistry, SessionState};
use datafusion::logical_expr::{col, Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion::sql::TableReference;
use datafusion_substrait::extensions::Extensions;
use datafusion_substrait::logical_plan::consumer::{
    from_substrait_plan_with_consumer, DefaultSubstraitConsumer, SubstraitConsumer,
};
use substrait::proto::{ExtensionLeafRel, ExtensionMultiRel, ExtensionSingleRel, Plan};

/// type_url prefix marking an ExtensionSingleRel as "unnest the named column of my input".
/// The column name follows the colon, e.g. "unnest:comments".
///
/// This is the ORIGINAL (hardcoded-path) marker: it unnests each level IN PLACE (the array column
/// is replaced by its struct fields at the same position), matching the layout `N1SubstraitBuilder`
/// hand-simulates. Used by the flag-OFF path — do not change its layout.
pub(crate) const UNNEST_TYPE_URL_PREFIX: &str = "unnest:";

/// type_url prefix marking a RESHAPING unnest, for the generic (flag-ON) path. Same comma-separated
/// path as `unnest:`, but the output is reshaped to CALCITE's Correlate+Uncollect layout:
/// `[all original columns INCLUDING the array column, positions unchanged] ++ [unnested struct fields]`.
///
/// Why: on the generic path, isthmus emits the Filter/Aggregate/Project with POSITIONAL field refs
/// against Calcite's layout. DataFusion's native `unnest_column` instead replaces the array in place,
/// a different column order — so those refs would point at the wrong columns. Reshaping here makes the
/// engine's output match what isthmus assumed, so any isthmus-emitted plan "just works" with no
/// Java-side index remapping. Mechanism: duplicate the array column to the end, then unnest the
/// duplicate — originals stay put, exploded fields append. Grep: NESTED.
pub(crate) const UNNEST_RESHAPE_TYPE_URL_PREFIX: &str = "unnest_reshape:";

/// A consumer that understands the unnest ExtensionSingleRel and otherwise behaves exactly like
/// the stock `DefaultSubstraitConsumer` (which it wraps and delegates to).
pub(crate) struct UnnestConsumer<'a> {
    inner: DefaultSubstraitConsumer<'a>,
}

impl<'a> UnnestConsumer<'a> {
    fn new(extensions: &'a Extensions, state: &'a SessionState) -> Self {
        Self { inner: DefaultSubstraitConsumer::new(extensions, state) }
    }
}

#[async_trait::async_trait]
impl SubstraitConsumer for UnnestConsumer<'_> {
    async fn resolve_table_ref(
        &self,
        table_ref: &TableReference,
    ) -> datafusion::common::Result<Option<Arc<dyn datafusion::catalog::TableProvider>>> {
        self.inner.resolve_table_ref(table_ref).await
    }

    fn get_extensions(&self) -> &Extensions {
        self.inner.get_extensions()
    }

    fn get_function_registry(&self) -> &impl FunctionRegistry {
        self.inner.get_function_registry()
    }

    // Correlated-subquery scope bookkeeping must be forwarded to the inner consumer so nested
    // expression handling stays consistent when we delegate.
    fn push_outer_schema(&self, schema: Arc<DFSchema>) {
        self.inner.push_outer_schema(schema);
    }

    fn pop_outer_schema(&self) {
        self.inner.pop_outer_schema();
    }

    fn get_outer_schema(&self, steps_out: usize) -> Option<Arc<DFSchema>> {
        self.inner.get_outer_schema(steps_out)
    }

    // Forward the OTHER extension-rel kinds to the stock consumer so this wrapper is a strict
    // superset of DefaultSubstraitConsumer: only ExtensionSingle (our unnest marker) is special-cased;
    // ExtensionLeaf/ExtensionMulti must still deserialize via the serializer registry exactly as the
    // stock consumer does (the trait DEFAULTS error, so without these overrides the wrapper would
    // regress any plan carrying a leaf/multi extension rel).
    async fn consume_extension_leaf(&self, rel: &ExtensionLeafRel) -> datafusion::common::Result<LogicalPlan> {
        self.inner.consume_extension_leaf(rel).await
    }

    async fn consume_extension_multi(&self, rel: &ExtensionMultiRel) -> datafusion::common::Result<LogicalPlan> {
        self.inner.consume_extension_multi(rel).await
    }

    /// The one override: an ExtensionSingleRel whose detail type_url is "unnest:<column>" becomes
    /// a native `LogicalPlan::Unnest` on that column of the (recursively converted) input.
    async fn consume_extension_single(
        &self,
        rel: &ExtensionSingleRel,
    ) -> datafusion::common::Result<LogicalPlan> {
        let detail = rel.detail.as_ref().ok_or_else(|| {
            DataFusionError::NotImplemented("ExtensionSingleRel without detail".to_string())
        })?;

        // Generic (flag-ON) path: reshaping unnest → Calcite Correlate+Uncollect layout.
        // Checked BEFORE the plain `unnest:` prefix because "unnest_reshape:" also starts with
        // "unnest" but must NOT be handled by the in-place branch.
        if let Some(path_spec) = detail.type_url.strip_prefix(UNNEST_RESHAPE_TYPE_URL_PREFIX) {
            let input_rel = rel.input.as_ref().ok_or_else(|| {
                DataFusionError::Execution("[NESTED] unnest_reshape ExtensionSingleRel has no input".to_string())
            })?;
            let input_plan = self.consume_rel(input_rel).await?;
            // The Java emitter stamps "<path>|w=<postUnnestWidth>" — the "|w=" suffix is a hint for the
            // Java parent-dedup post-pass only; strip it here so the reshape sees just the level path.
            let path_only = path_spec.split('|').next().unwrap_or(path_spec);
            let levels: Vec<&str> = path_only.split(',').filter(|s| !s.is_empty()).collect();
            log::info!(
                "[NESTED] unnest-consumer(reshape): expanding path {:?} to CALCITE layout \
                 (originals kept in place + struct fields appended) so isthmus positional refs align.",
                levels
            );
            return build_reshaping_unnest(input_plan, &levels);
        }

        if let Some(path_spec) = detail.type_url.strip_prefix(UNNEST_TYPE_URL_PREFIX) {
            let input_rel = rel.input.as_ref().ok_or_else(|| {
                DataFusionError::Execution("[NESTED-POC] unnest ExtensionSingleRel has no input".to_string())
            })?;
            // Recurse through THIS consumer so any nested unnest/extension inside the input is
            // also handled (consume_rel dispatches back through our overrides).
            let input_plan = self.consume_rel(input_rel).await?;

            // The tag is a comma-separated PATH of nested levels to unnest, outermost first, e.g.
            // "comments" (1-level) or "comments,comments.replies,comments.replies.reactions" (3-level).
            // Each level is a column that is a LIST<STRUCT>; unnesting it TWICE (list->struct, then
            // struct->top-level `level.field` columns) makes the next level's list column addressable
            // by its dotted name (Column::from_name does NOT split on '.', so "comments.replies" is one
            // column). Two passes per level because DataFusion's Substrait consumer only accepts
            // single-level (top-level) StructField references, not nested StructField.child access.
            let levels: Vec<&str> = path_spec.split(',').filter(|s| !s.is_empty()).collect();
            log::info!(
                "[NESTED-POC] unnest-consumer: expanding nested path {:?} -> LogicalPlan::Unnest (x2 per \
                 level: list->struct then struct->top-level fields). POC: carries the N1 UNNEST across \
                 Substrait as an ExtensionSingleRel since neither isthmus nor DF-54 model it natively.",
                levels
            );
            let mut builder = LogicalPlanBuilder::from(input_plan);
            for level in levels {
                builder = builder
                    .unnest_column(Column::from_name(level))?
                    .unnest_column(Column::from_name(level))?;
            }
            return builder.build();
        }

        // Not ours — defer to the stock behaviour (which errors with "Missing handler").
        self.inner.consume_extension_single(rel).await
    }
}

/// Builds a RESHAPING unnest producing Calcite's Correlate+Uncollect layout: original columns stay
/// in place (including the array column itself) and the exploded struct fields are appended.
///
/// Per level: we append a DUPLICATE of the array column (aliased) to the end of the row, then unnest
/// that duplicate. DataFusion's `unnest_column` replaces the duplicate in place — but since the
/// duplicate is last, the effect is "originals unchanged, struct fields appended". The two
/// `unnest_column` passes mirror the in-place branch (list→struct, then struct→top-level `alias.field`
/// columns). For a multi-level path, the next level's array column lives among the appended fields and
/// is addressed by its dotted name (`Column::from_name` does not split on '.').
///
/// Example — base `[comments: List<Struct<author,score>>, title, __row_id__]`, level "comments":
///   after project:  [comments, title, __row_id__, comments__u]         (duplicate appended)
///   after unnest×2: [comments, title, __row_id__, comments__u.author, comments__u.score]
///   after rename:   [comments, title, __row_id__, author, score]       (BARE Calcite names)
/// i.e. every original index is preserved and the child fields are appended with their BARE names —
/// exactly Calcite's Correlate row type, by name AND position. The bare names are essential for
/// multi-level `expand` (e.g. `expand products | expand variants`): the second level's
/// `unnest_reshape:variants` and isthmus's positional refs both address the column as `variants`,
/// not `products__u.variants`.
fn build_reshaping_unnest(input_plan: LogicalPlan, levels: &[&str]) -> datafusion::common::Result<LogicalPlan> {
    let mut builder = LogicalPlanBuilder::from(input_plan);
    for level in levels {
        // Alias for the duplicated array column: a name that cannot collide with a real column.
        let dup_alias = format!("{level}__u");
        // Project: keep every existing column (SELECT *), then append `array_col AS <level>__u`.
        let existing: Vec<Expr> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| col(Column::from_name(f.name())))
            .collect();
        let mut proj_exprs = existing;
        proj_exprs.push(col(Column::from_name(*level)).alias(dup_alias.clone()));
        builder = builder.project(proj_exprs)?;

        // Unnest the duplicate twice (list -> struct, struct -> top-level `<dup_alias>.field`).
        // preserve_nulls=false so an empty/absent array yields NO row (matches OpenSearch nested
        // semantics — a parent with no children contributes nothing to a nested-scoped query),
        // rather than DataFusion's default of one preserved row with null child fields.
        let opts = UnnestOptions::new().with_preserve_nulls(false);
        builder = builder
            .unnest_column_with_options(Column::from_name(&dup_alias), opts.clone())?
            .unnest_column_with_options(Column::from_name(&dup_alias), opts)?;

        // Rename the freshly-appended `<dup_alias>.field` columns to their BARE `field` names so the
        // output matches Calcite's Correlate row type by name (originals kept verbatim). Without this,
        // a following level (or isthmus's by-name refs) can't find e.g. `variants` — it'd be
        // `products__u.variants`.
        //
        // COLLISION GUARD: only bare-rename a child whose bare name does NOT already exist among the
        // ORIGINAL (pre-explosion) columns. A nested leaf that shares a parent field's name (e.g.
        // employees has parent `name` AND skills.name) would otherwise produce an ambiguous unqualified
        // `name` alongside the qualified `employees.name`, which DataFusion rejects for the WHOLE query.
        // Such a colliding child keeps its `<dup_alias>.field` name (still positionally correct — the
        // final output is relabeled by position, and only ARRAY children need a bare name to be
        // re-expanded by a following level; a colliding scalar leaf never is). Grep: NESTED name-collision.
        let prefix = format!("{dup_alias}.");
        let original_names: std::collections::HashSet<String> = builder
            .schema()
            .fields()
            .iter()
            .filter(|f| !f.name().starts_with(&prefix))
            .map(|f| f.name().clone())
            .collect();
        let renamed: Vec<Expr> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| match f.name().strip_prefix(&prefix) {
                Some(bare) if !original_names.contains(bare) => col(Column::from_name(f.name())).alias(bare.to_string()),
                _ => col(Column::from_name(f.name())),
            })
            .collect();
        builder = builder.project(renamed)?;
    }

    // [NESTED] Parent-dedup invariant: if the scan carries __row_id__ (the parent doc identity,
    // requested when the query is a parent-returning nested shape), reorder it to the LAST output
    // column. isthmus emitted every filter/project/aggregate above this unnest with POSITIONAL field
    // references computed WITHOUT __row_id__; keeping it strictly after all those columns leaves every
    // existing index undisturbed, and lets the Java dedup post-pass reference __row_id__ at a single
    // known tail index. No-op when __row_id__ is absent (the common, non-dedup case).
    //
    let fields = builder.schema().fields();
    if fields.iter().any(|f| f.name() == crate::ROW_ID_COLUMN_NAME) {
        let mut reordered: Vec<Expr> = fields
            .iter()
            .filter(|f| f.name() != crate::ROW_ID_COLUMN_NAME)
            .map(|f| col(Column::from_name(f.name())))
            .collect();
        reordered.push(col(Column::from_name(crate::ROW_ID_COLUMN_NAME)));
        builder = builder.project(reordered)?;
    }

    builder.build()
}

/// Unnest-aware replacement for `from_substrait_plan`. Builds `Extensions` from the plan exactly
/// like the stock entry point, then drives conversion through [`UnnestConsumer`].
pub(crate) async fn from_substrait_plan_unnest_aware(
    state: &SessionState,
    plan: &Plan,
) -> datafusion::common::Result<LogicalPlan> {
    let extensions = Extensions::try_from(&plan.extensions)?;
    if !extensions.type_variations.is_empty() {
        return Err(DataFusionError::NotImplemented(
            "Type variation extensions are not supported".to_string(),
        ));
    }
    let consumer = UnnestConsumer::new(&extensions, state);
    from_substrait_plan_with_consumer(&consumer, plan).await
}

// [NESTED-POC] The output-schema layout of a double unnest was verified empirically with a probe
// test (since removed to avoid a multi-GB debug build tree). For a base schema
// [comments: List<Struct<author,score,text>>, title, views, __row_id__], two `unnest_column("comments")`
// calls yield, in order:
//   [0] comments.author  [1] comments.score  [2] comments.text  [3] title  [4] views  [5] __row_id__
// i.e. the struct fields expand IN PLACE at the column's index; later columns shift right by
// (structFieldCount - 1). N1SubstraitBuilder computes filter/group-by field positions from this.

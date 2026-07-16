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

use datafusion::common::{Column, DFSchema, DataFusionError};
use datafusion::execution::{FunctionRegistry, SessionState};
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder};
use datafusion::sql::TableReference;
use datafusion_substrait::extensions::Extensions;
use datafusion_substrait::logical_plan::consumer::{
    from_substrait_plan_with_consumer, DefaultSubstraitConsumer, SubstraitConsumer,
};
use substrait::proto::{ExtensionSingleRel, Plan};

/// type_url prefix marking an ExtensionSingleRel as "unnest the named column of my input".
/// The column name follows the colon, e.g. "unnest:comments".
pub(crate) const UNNEST_TYPE_URL_PREFIX: &str = "unnest:";

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

    /// The one override: an ExtensionSingleRel whose detail type_url is "unnest:<column>" becomes
    /// a native `LogicalPlan::Unnest` on that column of the (recursively converted) input.
    async fn consume_extension_single(
        &self,
        rel: &ExtensionSingleRel,
    ) -> datafusion::common::Result<LogicalPlan> {
        let detail = rel.detail.as_ref().ok_or_else(|| {
            DataFusionError::NotImplemented("ExtensionSingleRel without detail".to_string())
        })?;

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

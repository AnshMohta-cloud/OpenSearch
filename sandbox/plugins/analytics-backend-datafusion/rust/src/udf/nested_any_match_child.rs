/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! `nested_any_match_child(array_col, field, op, value, clauseIdx)` — name-resolution stub for the
//! child-grain nested split.
//!
//! This UDF exists ONLY so DataFusion's `create_physical_expr` can resolve the function call that the
//! Java rewriter wraps in a `delegation_possible(nested_any_match_child(...), annotationId)` marker
//! (see `OpenSearchNestedFieldRewriter.tryDirectEqualityChildRewrite` and `NestedAnyMatchChildSerializer`).
//! It is NEVER evaluated: the indexed executor's classifier recognizes the child peer, pulls its
//! `clauseIdx` (5th operand) out of the resolved `PhysicalExpr`, and routes the clause's per-element
//! verdicts through the Lucene child-collect at `on_batch_mask` (see `single_collector.rs`
//! `ChildSplitState`). The body therefore fails loud — if it ever runs, classification missed the child
//! peer and would otherwise silently produce a wrong (all-true / all-false) child predicate, masking a
//! routing bug. Mirrors the `delegation_possible` / `delegated_predicate` marker-UDF pattern in
//! `substrait_to_tree.rs`.

use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::udf_identity;

pub fn register_all(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(NestedAnyMatchChildUdf::new()));
}

#[derive(Debug)]
pub struct NestedAnyMatchChildUdf {
    signature: Signature,
}

udf_identity!(NestedAnyMatchChildUdf, "nested_any_match_child");

impl NestedAnyMatchChildUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for NestedAnyMatchChildUdf {
    fn name(&self) -> &str {
        "nested_any_match_child"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Must never execute. The indexed executor recognizes the child peer during classification and
        // consumes it at child grain in on_batch_mask; DataFusion never evaluates this body on the happy
        // path. Reaching here means the child-split classifier missed the marker — fail loud rather than
        // silently emit a wrong child predicate.
        Err(datafusion::error::DataFusionError::Internal(
            "nested_any_match_child UDF body invoked — the child-grain split classifier did not recognize \
             the child peer; treat as a serious correctness bug"
                .to_string(),
        ))
    }
}

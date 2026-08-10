/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! `nested_any_match(array_col, field_name, op, value)` — parent-preserving nested predicate.
//!
//! Evaluates `any_match(array_col, element -> get_field(element, field_name) <op> value)` per row,
//! returning a BOOLEAN column. Row count is never changed — one boolean per parent row.
//!
//! This is the Rust counterpart of the Java `NESTED_ANY_MATCH_OP` emitted by
//! `OpenSearchNestedFieldRewriter` when a filter contains `ITEM($arrayCol, 'field') <op> literal`.
//! The Java side encodes the predicate as four scalar arguments; this UDF reconstructs and evaluates
//! the lambda internally using Arrow's list/struct accessors.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, BooleanBuilder, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, StringArray, StructArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{plan_err, ScalarValue};
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility};

use super::udf_identity;

pub fn register_all(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(NestedAnyMatchUdf::new()));
}

#[derive(Debug)]
pub struct NestedAnyMatchUdf {
    signature: Signature,
}

udf_identity!(NestedAnyMatchUdf, "nested_any_match");

impl NestedAnyMatchUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for NestedAnyMatchUdf {
    fn name(&self) -> &str {
        "nested_any_match"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        if args.len() != 4 {
            return plan_err!("nested_any_match expects 4 arguments, got {}", args.len());
        }

        // Extract arguments. The array column is columnar; field_name and op are string literals;
        // value may be a literal scalar or a column (for parent-field comparisons).
        let array_col = match &args[0] {
            ColumnarValue::Array(a) => Arc::clone(a),
            ColumnarValue::Scalar(s) => s.to_array_of_size(1)?,
        };

        let field_name = extract_string_scalar(&args[1], "field_name")?;
        let op = extract_string_scalar(&args[2], "op")?;

        // DEEP path (Phase B): a dotted field_name (e.g. "divisions.teams.members.mname") is the
        // descent-plus-leaf path of a DEEP keyword prune peer. NESTED_ANY_MATCH is dual-viable
        // [lucene, datafusion]; when NOT delegated to Lucene (or when DataFusion evaluates the residual),
        // the flat single-field lookup below cannot resolve a dotted name — so we build the equivalent
        // {"nested":...,"inner":...} descent tree (all segments but the last are nested arrays; the last is
        // the scalar leaf — guaranteed by the rewriter's isAllNestedScalarLeafPath) and delegate to the
        // recursive evaluator. This keeps the deep peer CORRECT on DataFusion too (Lucene stays a pure
        // pruning optimization, never a correctness dependency — the single-level invariant, at any depth).
        if field_name.contains('.') {
            // NESTED_ANY_MATCH only ever carries a string keyword value (EQUALS); read it as a string lit.
            let value = extract_string_scalar(&args[3], "value")?;
            let op_sym = match op.as_str() {
                "EQUALS" => "=",
                other => other, // NESTED_ANY_MATCH only emits EQUALS today; pass through defensively
            };
            let segs: Vec<&str> = field_name.split('.').collect();
            let leaf = segs[segs.len() - 1];
            // innermost leaf comparison, then wrap each intermediate segment as a nested descent (inner-first)
            let mut node = serde_json::json!({"op": op_sym, "args": [{"field": leaf}, {"lit": value}]});
            for i in (0..segs.len() - 1).rev() {
                node = serde_json::json!({"nested": segs[i], "inner": node});
            }
            let out = super::nested_any_match_expr::evaluate_nested_descent(&array_col, &node)?;
            return Ok(ColumnarValue::Array(Arc::new(out)));
        }

        let num_rows = array_col.len();
        let mut result = BooleanBuilder::with_capacity(num_rows);

        // The array column should be List<Struct>
        let list_array = array_col.as_list_opt::<i32>().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!(
                "nested_any_match: first argument must be List, got {:?}",
                array_col.data_type()
            ))
        })?;

        // Get the struct field index within the list's element type
        let element_type = match list_array.data_type() {
            DataType::List(f) => f.data_type().clone(),
            _ => {
                return plan_err!(
                    "nested_any_match: expected List type, got {:?}",
                    list_array.data_type()
                );
            }
        };
        let struct_fields = match &element_type {
            DataType::Struct(fields) => fields.clone(),
            _ => {
                return plan_err!(
                    "nested_any_match: expected List<Struct>, got List<{:?}>",
                    element_type
                );
            }
        };
        let field_idx = struct_fields
            .iter()
            .position(|f| f.name() == &field_name)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(format!(
                    "nested_any_match: field '{}' not found in struct. Available: {:?}",
                    field_name,
                    struct_fields.iter().map(|f| f.name()).collect::<Vec<_>>()
                ))
            })?;

        // Extract the comparison value (scalar broadcast or per-row column)
        let compare_value = match &args[3] {
            ColumnarValue::Scalar(s) => CompareValue::Scalar(s.clone()),
            ColumnarValue::Array(a) => CompareValue::Column(Arc::clone(a)),
        };

        // Iterate each row (parent document)
        let values = list_array.values();
        let struct_array = values.as_struct();
        let field_array = struct_array.column(field_idx);

        for row_idx in 0..num_rows {
            if list_array.is_null(row_idx) {
                result.append_null();
                continue;
            }

            let start = list_array.value_offsets()[row_idx] as usize;
            let end = list_array.value_offsets()[row_idx + 1] as usize;

            if start == end {
                // Empty array — no element can match
                result.append_value(false);
                continue;
            }

            let row_compare_value = match &compare_value {
                CompareValue::Scalar(s) => s.clone(),
                CompareValue::Column(col) => ScalarValue::try_from_array(col, row_idx)?,
            };

            let mut any_match = false;
            for elem_idx in start..end {
                if field_array.is_null(elem_idx) {
                    continue;
                }
                let elem_value = ScalarValue::try_from_array(field_array, elem_idx)?;
                if compare_scalar(&elem_value, &op, &row_compare_value)? {
                    any_match = true;
                    break;
                }
            }
            result.append_value(any_match);
        }

        Ok(ColumnarValue::Array(Arc::new(result.finish())))
    }
}

enum CompareValue {
    Scalar(ScalarValue),
    Column(ArrayRef),
}

fn extract_string_scalar(arg: &ColumnarValue, name: &str) -> Result<String> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s))) => Ok(s.clone()),
        ColumnarValue::Array(a) => {
            // Constant-folded literal may arrive as a 1-element array
            if a.len() == 1 && !a.is_null(0) {
                if let Some(s) = a.as_any().downcast_ref::<StringArray>() {
                    return Ok(s.value(0).to_string());
                }
            }
            plan_err!("nested_any_match: '{}' must be a string literal", name)
        }
        other => plan_err!(
            "nested_any_match: '{}' must be a string literal, got {:?}",
            name,
            other
        ),
    }
}

/// Compare two ScalarValues using the given operator string.
fn compare_scalar(left: &ScalarValue, op: &str, right: &ScalarValue) -> Result<bool> {
    // Convert both sides to f64 for numeric comparison, or string for string comparison.
    match (scalar_to_f64(left), scalar_to_f64(right)) {
        (Some(l), Some(r)) => Ok(match op {
            "GREATER_THAN" | ">" => l > r,
            "GREATER_THAN_OR_EQUAL" | ">=" => l >= r,
            "LESS_THAN" | "<" => l < r,
            "LESS_THAN_OR_EQUAL" | "<=" => l <= r,
            "EQUALS" | "=" | "==" => (l - r).abs() < f64::EPSILON,
            "NOT_EQUALS" | "!=" | "<>" => (l - r).abs() >= f64::EPSILON,
            _ => {
                return plan_err!("nested_any_match: unsupported operator '{}'", op);
            }
        }),
        _ => {
            // Fall back to string comparison
            let ls = scalar_to_string(left);
            let rs = scalar_to_string(right);
            match (ls, rs) {
                (Some(l), Some(r)) => Ok(match op {
                    "GREATER_THAN" | ">" => l > r,
                    "GREATER_THAN_OR_EQUAL" | ">=" => l >= r,
                    "LESS_THAN" | "<" => l < r,
                    "LESS_THAN_OR_EQUAL" | "<=" => l <= r,
                    "EQUALS" | "=" | "==" => l == r,
                    "NOT_EQUALS" | "!=" | "<>" => l != r,
                    _ => {
                        return plan_err!("nested_any_match: unsupported operator '{}'", op);
                    }
                }),
                _ => Ok(false), // NULL comparison — SQL semantics: NULL op X = NULL → false
            }
        }
    }
}

fn scalar_to_f64(s: &ScalarValue) -> Option<f64> {
    match s {
        ScalarValue::Int8(Some(v)) => Some(*v as f64),
        ScalarValue::Int16(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::UInt8(Some(v)) => Some(*v as f64),
        ScalarValue::UInt16(Some(v)) => Some(*v as f64),
        ScalarValue::UInt32(Some(v)) => Some(*v as f64),
        ScalarValue::UInt64(Some(v)) => Some(*v as f64),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Float64(Some(v)) => Some(*v as f64),
        _ => None,
    }
}

fn scalar_to_string(s: &ScalarValue) -> Option<String> {
    match s {
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) | ScalarValue::Utf8View(Some(v)) => {
            Some(v.clone())
        }
        _ => None,
    }
}

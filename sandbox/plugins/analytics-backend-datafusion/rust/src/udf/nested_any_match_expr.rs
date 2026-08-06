/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! `nested_any_match_expr(array_col, expr_json)` — generalized parent-preserving nested predicate.
//!
//! Generalization of `nested_any_match` (see `nested_any_match.rs`) for compound (AND/OR/NOT),
//! arithmetic (+,-,*,/,%), or otherwise-shaped per-element predicates that a flat (field, op, value)
//! triple can't express — e.g. `subs.views > 65 and subs.views % 2 = 0`. The second argument is a
//! JSON string describing the per-element expression tree; this UDF parses it once per batch and
//! evaluates it per array element, short-circuiting on the first element that satisfies the WHOLE
//! tree. Matches vanilla OpenSearch's native `nested` query + Painless script semantics: a SINGLE
//! array element must satisfy the WHOLE compound expression jointly. Row count never changes.
//!
//! Wire format (built by `OpenSearchNestedFieldRewriter.ExprTreeBuilder`, Java side):
//!   {"op":"AND"|"OR", "args":[...]}          - boolean connective, 2+ args
//!   {"op":"NOT", "args":[...]}               - negation, 1 arg
//!   {"op":">"|">="|"<"|"<="|"="|"!=", "args":[left,right]}  - comparison
//!   {"op":"+"|"-"|"*"|"/"|"%", "args":[left,right]}          - arithmetic
//!   {"field":"fieldName"}                     - read a field off the CURRENT array element
//!   {"lit":value}                             - a literal number/string/boolean

use std::sync::Arc;

use datafusion::arrow::array::{Array, AsArray, BooleanBuilder};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{plan_err, ScalarValue};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility};
use serde_json::Value as Json;

use super::udf_identity;

pub fn register_all(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(NestedAnyMatchExprUdf::new()));
}

#[derive(Debug)]
pub struct NestedAnyMatchExprUdf {
    signature: Signature,
}

udf_identity!(NestedAnyMatchExprUdf, "nested_any_match_expr");

impl NestedAnyMatchExprUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for NestedAnyMatchExprUdf {
    fn name(&self) -> &str {
        "nested_any_match_expr"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        if args.len() != 2 {
            return plan_err!("nested_any_match_expr expects 2 arguments, got {}", args.len());
        }

        let array_col = match &args[0] {
            ColumnarValue::Array(a) => Arc::clone(a),
            ColumnarValue::Scalar(s) => s.to_array_of_size(1)?,
        };
        let expr_json = extract_string_scalar(&args[1], "expr_json")?;
        let tree: Json = serde_json::from_str(&expr_json)
            .map_err(|e| DataFusionError::Execution(format!("nested_any_match_expr: invalid expr JSON: {e}")))?;

        let num_rows = array_col.len();
        let mut result = BooleanBuilder::with_capacity(num_rows);

        let list_array = array_col.as_list_opt::<i32>().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "nested_any_match_expr: first argument must be List, got {:?}",
                array_col.data_type()
            ))
        })?;

        let element_type = match list_array.data_type() {
            DataType::List(f) => f.data_type().clone(),
            _ => return plan_err!("nested_any_match_expr: expected List type, got {:?}", list_array.data_type()),
        };
        let struct_fields = match &element_type {
            DataType::Struct(fields) => fields.clone(),
            _ => {
                return plan_err!(
                    "nested_any_match_expr: expected List<Struct>, got List<{:?}>",
                    element_type
                );
            }
        };

        let values = list_array.values();
        let struct_array = values.as_struct();

        for row_idx in 0..num_rows {
            if list_array.is_null(row_idx) {
                result.append_null();
                continue;
            }
            let start = list_array.value_offsets()[row_idx] as usize;
            let end = list_array.value_offsets()[row_idx + 1] as usize;
            if start == end {
                result.append_value(false);
                continue;
            }

            let mut any_match = false;
            for elem_idx in start..end {
                // Plain path: no Lucene-delegated clauses (lucene = None). The child-grain split invokes the
                // evaluate_with_lucene entry point instead, which supplies per-element Lucene verdicts.
                match eval_bool(&tree, struct_array, &struct_fields, elem_idx, None)? {
                    Some(true) => {
                        any_match = true;
                        break;
                    }
                    _ => continue,
                }
            }
            result.append_value(any_match);
        }

        Ok(ColumnarValue::Array(Arc::new(result.finish())))
    }
}

/// Per-element evaluation context for the child-grain nested split. `clause_bits[i]` is the boolean
/// result of Lucene-delegated clause `i` at each element, indexed by the SAME global element index
/// (`elem_idx`) the UDF iterates — so a `{"lucene": i}` node is just `clause_bits[i].value(elem_idx)`.
/// The evaluator (SingleCollectorEvaluator) expands the Lucene child bitset into this element-index space
/// before invoking the UDF, so the UDF does no child-ordinal arithmetic. `None` = no split (plain path).
struct LuceneClauseBits<'a> {
    clause_bits: &'a [datafusion::arrow::array::BooleanArray],
}

impl<'a> LuceneClauseBits<'a> {
    /// The Lucene clause `idx`'s verdict for the element at global index `elem_idx`. A missing element bit
    /// (out of range / null) is treated as `false` (element did not match the keyword clause).
    fn value(&self, idx: usize, elem_idx: usize) -> bool {
        self.clause_bits
            .get(idx)
            .map(|b| elem_idx < b.len() && !b.is_null(elem_idx) && b.value(elem_idx))
            .unwrap_or(false)
    }
}

/// Evaluate a boolean-typed node of the tree for one struct element. Returns `Ok(None)` for a
/// NULL result (SQL three-valued logic — e.g. comparing against a NULL field value).
///
/// `lucene` carries per-element results for any Lucene-delegated leaves (child-grain split); `None` on the
/// plain (non-split) path. A `{"lucene": <idx>}` node consults it instead of comparing a field.
fn eval_bool(
    node: &Json,
    struct_array: &datafusion::arrow::array::StructArray,
    struct_fields: &datafusion::arrow::datatypes::Fields,
    elem_idx: usize,
    lucene: Option<&LuceneClauseBits>,
) -> Result<Option<bool>> {
    // Child-grain split leaf: a keyword clause evaluated by Lucene, its per-element verdict supplied in
    // `lucene`. `{"lucene": <clauseIdx>}`. Two-valued (matched / not) — Lucene has no NULL notion here.
    if let Some(idx) = node.get("lucene").and_then(|v| v.as_u64()) {
        let matched = lucene.map(|l| l.value(idx as usize, elem_idx)).unwrap_or(false);
        return Ok(Some(matched));
    }
    let op = node
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DataFusionError::Execution(format!("nested_any_match_expr: missing 'op' in node {node}")))?;
    let args = node
        .get("args")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DataFusionError::Execution(format!("nested_any_match_expr: missing 'args' in node {node}")))?;

    match op {
        "AND" => {
            for a in args {
                match eval_bool(a, struct_array, struct_fields, elem_idx, lucene)? {
                    Some(false) => return Ok(Some(false)),
                    None => return Ok(None), // NULL propagates: NULL AND anything-not-false = NULL
                    Some(true) => continue,
                }
            }
            Ok(Some(true))
        }
        "OR" => {
            let mut saw_null = false;
            for a in args {
                match eval_bool(a, struct_array, struct_fields, elem_idx, lucene)? {
                    Some(true) => return Ok(Some(true)),
                    None => saw_null = true,
                    Some(false) => continue,
                }
            }
            Ok(if saw_null { None } else { Some(false) })
        }
        "NOT" => {
            if args.len() != 1 {
                return plan_err!("nested_any_match_expr: NOT expects 1 arg");
            }
            Ok(eval_bool(&args[0], struct_array, struct_fields, elem_idx, lucene)?.map(|b| !b))
        }
        ">" | ">=" | "<" | "<=" | "=" | "!=" => {
            if args.len() != 2 {
                return plan_err!("nested_any_match_expr: comparison expects 2 args");
            }
            let left = eval_value(&args[0], struct_array, struct_fields, elem_idx)?;
            let right = eval_value(&args[1], struct_array, struct_fields, elem_idx)?;
            Ok(compare(&left, op, &right))
        }
        other => plan_err!("nested_any_match_expr: '{other}' is not a boolean operator"),
    }
}

/// Evaluate a value-typed node (field access, literal, or arithmetic) for one struct element.
fn eval_value(
    node: &Json,
    struct_array: &datafusion::arrow::array::StructArray,
    struct_fields: &datafusion::arrow::datatypes::Fields,
    elem_idx: usize,
) -> Result<ScalarValue> {
    if let Some(field_name) = node.get("field").and_then(|v| v.as_str()) {
        let field_idx = struct_fields
            .iter()
            .position(|f| f.name() == field_name)
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "nested_any_match_expr: field '{field_name}' not found. Available: {:?}",
                    struct_fields.iter().map(|f| f.name()).collect::<Vec<_>>()
                ))
            })?;
        let field_array = struct_array.column(field_idx);
        if field_array.is_null(elem_idx) {
            return Ok(ScalarValue::Null);
        }
        return ScalarValue::try_from_array(field_array, elem_idx);
    }

    if let Some(lit) = node.get("lit") {
        return Ok(json_to_scalar(lit));
    }

    if let Some(op) = node.get("op").and_then(|v| v.as_str()) {
        let args = node
            .get("args")
            .and_then(|v| v.as_array())
            .ok_or_else(|| DataFusionError::Execution(format!("nested_any_match_expr: missing 'args' in node {node}")))?;
        if args.len() != 2 {
            return plan_err!("nested_any_match_expr: arithmetic op '{op}' expects 2 args");
        }
        let left = eval_value(&args[0], struct_array, struct_fields, elem_idx)?;
        let right = eval_value(&args[1], struct_array, struct_fields, elem_idx)?;
        return arithmetic(&left, op, &right);
    }

    plan_err!("nested_any_match_expr: unrecognized value node {node}")
}

fn json_to_scalar(v: &Json) -> ScalarValue {
    if let Some(n) = v.as_f64() {
        return ScalarValue::Float64(Some(n));
    }
    if let Some(s) = v.as_str() {
        if s == "null" {
            return ScalarValue::Null;
        }
        return ScalarValue::Utf8(Some(s.to_string()));
    }
    if let Some(b) = v.as_bool() {
        return ScalarValue::Boolean(Some(b));
    }
    ScalarValue::Null
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
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) | ScalarValue::Utf8View(Some(v)) => Some(v.clone()),
        _ => None,
    }
}

fn is_null(s: &ScalarValue) -> bool {
    matches!(s, ScalarValue::Null) || s.is_null()
}

/// SQL three-valued comparison: NULL compared to anything is NULL (unknown), never true/false.
fn compare(left: &ScalarValue, op: &str, right: &ScalarValue) -> Option<bool> {
    if is_null(left) || is_null(right) {
        return None;
    }
    if let (Some(l), Some(r)) = (scalar_to_f64(left), scalar_to_f64(right)) {
        return Some(match op {
            ">" => l > r,
            ">=" => l >= r,
            "<" => l < r,
            "<=" => l <= r,
            "=" => (l - r).abs() < f64::EPSILON,
            "!=" => (l - r).abs() >= f64::EPSILON,
            _ => return None,
        });
    }
    if let (Some(l), Some(r)) = (scalar_to_string(left), scalar_to_string(right)) {
        return Some(match op {
            ">" => l > r,
            ">=" => l >= r,
            "<" => l < r,
            "<=" => l <= r,
            "=" => l == r,
            "!=" => l != r,
            _ => return None,
        });
    }
    None
}

fn arithmetic(left: &ScalarValue, op: &str, right: &ScalarValue) -> Result<ScalarValue> {
    if is_null(left) || is_null(right) {
        return Ok(ScalarValue::Null);
    }
    let (l, r) = match (scalar_to_f64(left), scalar_to_f64(right)) {
        (Some(l), Some(r)) => (l, r),
        _ => return plan_err!("nested_any_match_expr: arithmetic op '{op}' requires numeric operands"),
    };
    let result = match op {
        "+" => l + r,
        "-" => l - r,
        "*" => l * r,
        "/" => {
            if r == 0.0 {
                return Ok(ScalarValue::Null);
            }
            l / r
        }
        "%" => {
            if r == 0.0 {
                return Ok(ScalarValue::Null);
            }
            l % r
        }
        other => return plan_err!("nested_any_match_expr: unsupported arithmetic operator '{other}'"),
    };
    Ok(ScalarValue::Float64(Some(result)))
}

fn extract_string_scalar(arg: &ColumnarValue, name: &str) -> Result<String> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s))) => Ok(s.clone()),
        ColumnarValue::Array(a) => {
            if a.len() == 1 && !a.is_null(0) {
                if let Some(s) = a.as_any().downcast_ref::<datafusion::arrow::array::StringArray>() {
                    return Ok(s.value(0).to_string());
                }
            }
            plan_err!("nested_any_match_expr: '{}' must be a string literal", name)
        }
        other => plan_err!("nested_any_match_expr: '{}' must be a string literal, got {:?}", name, other),
    }
}

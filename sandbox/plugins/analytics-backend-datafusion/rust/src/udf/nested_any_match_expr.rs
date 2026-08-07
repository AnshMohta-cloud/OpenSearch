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

use datafusion::arrow::array::{Array, ArrayRef, AsArray, BooleanBuilder};
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

        // Plain path: no Lucene-delegated clauses. The child-grain split calls
        // evaluate_nested_with_lucene directly (below) to supply per-element Lucene verdicts.
        let result = evaluate_nested_any_match(&array_col, &tree, None)?;
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

/// Core per-element evaluation shared by the plain UDF path and the child-grain split path: for each parent
/// row of the `LIST<STRUCT>` `array_col`, returns true iff SOME single element satisfies the whole `tree`
/// (element-correlated ∃), null for a null array row, false for an empty array. `lucene`, when present,
/// supplies per-element verdicts for `{"lucene": i}` nodes (a keyword clause evaluated by Lucene in the
/// split); `None` on the plain path. This is the single place the ∃-over-elements roll-up lives — so a
/// split keyword clause and a DataFusion-evaluated range clause intersect at the SAME element index,
/// preserving vanilla nested correlation.
fn evaluate_nested_any_match(
    array_col: &ArrayRef,
    tree: &Json,
    lucene: Option<&LuceneClauseBits>,
) -> Result<datafusion::arrow::array::BooleanArray> {
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
        _ => return plan_err!("nested_any_match_expr: expected List<Struct>, got List<{:?}>", element_type),
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
            if let Some(true) = eval_bool(tree, struct_array, &struct_fields, elem_idx, lucene)? {
                any_match = true;
                break;
            }
        }
        result.append_value(any_match);
    }
    Ok(result.finish())
}

/// Child-grain split entry point (called by `SingleCollectorEvaluator`, NOT registered as a UDF). Evaluates
/// the nested predicate `tree` over `array_col` exactly like the plain path, except any `{"lucene": i}` leaf
/// takes its per-element verdict from `clause_bits[i]` (indexed by the global element index — the same order
/// this fn iterates elements). `clause_bits` is produced by expanding the Lucene child bitset into
/// element-index space upstream. Element correlation is preserved because Lucene and DataFusion clauses are
/// combined per element inside the same ∃-over-elements loop.
pub(crate) fn evaluate_nested_with_lucene(
    array_col: &ArrayRef,
    expr_json: &str,
    clause_bits: &[datafusion::arrow::array::BooleanArray],
) -> Result<datafusion::arrow::array::BooleanArray> {
    let tree: Json = serde_json::from_str(expr_json)
        .map_err(|e| DataFusionError::Execution(format!("nested_any_match_expr: invalid expr JSON: {e}")))?;
    let lucene = LuceneClauseBits { clause_bits };
    evaluate_nested_any_match(array_col, &tree, Some(&lucene))
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
    /// Whether clause `idx`'s per-element verdicts were supplied by the executor. When false, the caller
    /// must fall back to evaluating the clause's `fallback` subtree natively (do NOT treat as all-false).
    fn has_clause(&self, idx: usize) -> bool {
        idx < self.clause_bits.len()
    }

    /// The Lucene clause `idx`'s verdict for the element at global index `elem_idx`. A missing element bit
    /// (out of range / null) is treated as `false` (element did not match the keyword clause). Only call
    /// when {@link #has_clause} is true.
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
    // Child-grain split leaf: a keyword clause the rewriter chose to route to Lucene.
    // `{"lucene": <clauseIdx>, "fallback": <originalPredicateSubtree>}`.
    //
    // Lucene is a pure OPTIMIZATION here, never a correctness dependency: when the child-grain split
    // executor supplies this clause's per-element verdicts (`lucene` present AND has clause `idx`), use
    // them (two-valued: matched / not — Lucene has no NULL notion). Otherwise — the plain UDF path
    // (`lucene == None`), the Tree/OR-NOT path where the peer was demoted to native, or any path where
    // this clause wasn't delegated — evaluate the `fallback` subtree natively so the result is identical.
    // This keeps `nested_any_match_expr` self-sufficient on EVERY execution path; the split only makes it
    // faster, never changes its answer.
    if let Some(idx) = node.get("lucene").and_then(|v| v.as_u64()) {
        let idx = idx as usize;
        if let Some(l) = lucene {
            if l.has_clause(idx) {
                return Ok(Some(l.value(idx, elem_idx)));
            }
        }
        match node.get("fallback") {
            Some(fallback) => return eval_bool(fallback, struct_array, struct_fields, elem_idx, lucene),
            None => {
                return plan_err!(
                    "nested_any_match_expr: {{\"lucene\":{idx}}} node has neither delegated bits nor a \
                     \"fallback\" subtree — the rewriter must always emit a fallback so the predicate is \
                     correct when Lucene verdicts are absent"
                )
            }
        }
    }

    // Nested descent: an inner LIST<STRUCT> array level. `{"nested":"<field>","inner":<subtree>}`.
    //
    // This is what makes the evaluator arbitrary-depth. `field` names a field on the CURRENT struct
    // element that is ITSELF a nested array (Arrow List<Struct>) — e.g. `comments.replies` where both
    // `comments` and `replies` are OpenSearch `nested`. We open a fresh existential (∃) loop over the
    // inner list's elements for THIS outer element (`elem_idx`), short-circuiting on the first inner
    // element that satisfies `inner`, and return whether any did. `inner` may itself be another
    // `{"nested":...}` node, so this composes to any depth (∃-over-∃-over-∃…), one loop per array level.
    //
    // The coordinate composition is the standard Arrow nested-list descent: the inner list column's
    // `value_offsets[elem_idx]..[elem_idx+1]` delimits this outer element's inner elements in the
    // flattened inner values array — the same value_offsets indexing the outer loop used one level up.
    //
    // Semantics are TWO-VALUED (boolean), NOT 3VL — matching vanilla OpenSearch nested-query existential
    // semantics: an empty inner list, a null inner-list slot, and "inner elements present but none match"
    // ALL collapse to Some(false) (the parent has no matching nested child), never None. A null leaf on an
    // inner element is handled inside the recursion (its comparison yields None → that element doesn't
    // match → the ∃ loop continues), never poisoning the whole result. Lucene split bits do NOT cross a
    // nested boundary (the split delegates only top-level keyword clauses), so we descend with `None`.
    if let Some(field_name) = node.get("nested").and_then(|v| v.as_str()) {
        let inner_subtree = node.get("inner").ok_or_else(|| {
            DataFusionError::Execution(format!(
                "nested_any_match_expr: {{\"nested\":\"{field_name}\"}} node missing \"inner\" subtree"
            ))
        })?;
        let field_idx = struct_fields
            .iter()
            .position(|f| f.name() == field_name)
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "nested_any_match_expr: nested field '{field_name}' not found. Available: {:?}",
                    struct_fields.iter().map(|f| f.name()).collect::<Vec<_>>()
                ))
            })?;
        let inner_col = struct_array.column(field_idx);
        // A null inner-list slot for this element ⇒ no inner elements ⇒ ∃ = false.
        if inner_col.is_null(elem_idx) {
            return Ok(Some(false));
        }
        let inner_list = inner_col.as_list_opt::<i32>().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "nested_any_match_expr: nested field '{field_name}' must be List<Struct>, got {:?}",
                inner_col.data_type()
            ))
        })?;
        let inner_fields = match inner_list.data_type() {
            DataType::List(f) => match f.data_type() {
                DataType::Struct(fields) => fields.clone(),
                other => {
                    return plan_err!(
                        "nested_any_match_expr: nested field '{field_name}' must be List<Struct>, got List<{other:?}>"
                    )
                }
            },
            other => return plan_err!("nested_any_match_expr: nested field '{field_name}' must be a List, got {other:?}"),
        };
        let inner_values = inner_list.values();
        let inner_struct = inner_values.as_struct();
        let inner_start = inner_list.value_offsets()[elem_idx] as usize;
        let inner_end = inner_list.value_offsets()[elem_idx + 1] as usize;
        for inner_idx in inner_start..inner_end {
            if let Some(true) = eval_bool(inner_subtree, inner_struct, &inner_fields, inner_idx, None)? {
                return Ok(Some(true));
            }
        }
        return Ok(Some(false));
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        ArrayRef, BooleanArray, Int64Array, ListArray, StringArray, StructArray,
    };
    use datafusion::arrow::buffer::{BooleanBuffer, NullBuffer, OffsetBuffer};
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use serde_json::json;

    /// Build a `LIST<STRUCT{author: Utf8, score: Int64}>` from per-row element lists.
    /// Each inner Vec is one parent row's elements as `(author, score)` pairs.
    fn comments_array(rows: &[Vec<(&str, i64)>]) -> ArrayRef {
        let mut authors: Vec<Option<String>> = Vec::new();
        let mut scores: Vec<Option<i64>> = Vec::new();
        let mut offsets: Vec<i32> = vec![0];
        let mut acc = 0i32;
        for row in rows {
            for (a, s) in row {
                authors.push(Some((*a).to_string()));
                scores.push(Some(*s));
            }
            acc += row.len() as i32;
            offsets.push(acc);
        }
        let struct_fields: Fields = Fields::from(vec![
            Field::new("author", DataType::Utf8, true),
            Field::new("score", DataType::Int64, true),
        ]);
        let struct_array = StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(authors)) as ArrayRef,
                Arc::new(Int64Array::from(scores)) as ArrayRef,
            ],
            None,
        );
        let list_field = Arc::new(Field::new("item", DataType::Struct(struct_fields), true));
        Arc::new(ListArray::new(
            list_field,
            OffsetBuffer::new(offsets.into()),
            Arc::new(struct_array),
            None,
        )) as ArrayRef
    }

    // The JSON the rewriter emits for `comments.author='alice' AND comments.score>50` under the
    // child-grain split: the keyword clause is a {"lucene":0} node carrying its original subtree as
    // "fallback"; the range clause stays native.
    fn split_json() -> &'static str {
        r#"{"op":"AND","args":[
             {"lucene":0,"fallback":{"op":"=","args":[{"field":"author"},{"lit":"alice"}]}},
             {"op":">","args":[{"field":"score"},{"lit":50}]}
           ]}"#
    }

    // Two parents. P0: alice@40, bob@90 — NO single element satisfies author=alice AND score>50
    // (alice's element is 40, not >50). P1: alice@70 — one element satisfies both. Vanilla nested
    // semantics ⇒ [false, true]. This is the exact element-correlation the split must preserve.
    fn corpus() -> ArrayRef {
        comments_array(&[vec![("alice", 40), ("bob", 90)], vec![("alice", 70)]])
    }

    #[test]
    fn fallback_path_matches_vanilla_when_no_lucene_bits() {
        // No clause_bits supplied → the {"lucene":0} node must evaluate its "fallback" natively and
        // produce the correct element-correlated answer.
        let array = corpus();
        let out = evaluate_nested_with_lucene(&array, split_json(), &[]).unwrap();
        assert_eq!(out, BooleanArray::from(vec![false, true]));
    }

    #[test]
    fn plain_udf_path_matches_vanilla() {
        // The plain (non-split) UDF path with the same JSON (lucene=None) must also fall back and agree.
        let array = corpus();
        let tree: Json = serde_json::from_str(split_json()).unwrap();
        let out = evaluate_nested_any_match(&array, &tree, None).unwrap();
        assert_eq!(out, BooleanArray::from(vec![false, true]));
    }

    #[test]
    fn lucene_bits_are_used_and_correlate_per_element() {
        // Supply per-element Lucene verdicts for clause 0 (author='alice'). Global element order is
        // [P0.alice, P0.bob, P1.alice] → alice matches at indices 0 and 2.
        let array = corpus();
        let clause0 = BooleanArray::from(vec![true, false, true]);
        let out = evaluate_nested_with_lucene(&array, split_json(), &[clause0]).unwrap();
        // Same correlated result: P0 has no element with (alice AND score>50); P1 does.
        assert_eq!(out, BooleanArray::from(vec![false, true]));
    }

    #[test]
    fn lucene_bits_override_fallback_wrongly_would_change_result() {
        // Prove the bits are actually consulted (not the fallback): feed DELIBERATELY WRONG bits that
        // claim P0.bob (index 1) is 'alice'. Now P0 has element bob@90 matching (lucene-author AND
        // score>50) ⇒ P0 flips to true. This can only happen if the bits, not the fallback, drove it.
        let array = corpus();
        let wrong = BooleanArray::from(vec![false, true, true]);
        let out = evaluate_nested_with_lucene(&array, split_json(), &[wrong]).unwrap();
        assert_eq!(out, BooleanArray::from(vec![true, true]));
    }

    #[test]
    fn lucene_node_without_fallback_and_no_bits_errors() {
        // A {"lucene"} node with neither supplied bits nor a fallback is a rewriter bug — must fail loud,
        // never silently treat as all-false.
        let array = corpus();
        let json = r#"{"lucene":0}"#;
        let err = evaluate_nested_with_lucene(&array, json, &[]).unwrap_err();
        assert!(err.to_string().contains("fallback"), "unexpected error: {err}");
    }

    #[test]
    fn empty_and_null_arrays() {
        // Empty array → false (no element); null array → null. Independent of the split.
        let array = comments_array(&[vec![], vec![("alice", 70)]]);
        let out = evaluate_nested_with_lucene(&array, split_json(), &[]).unwrap();
        assert_eq!(out, BooleanArray::from(vec![false, true]));
    }

    // ────────────────────────────────────────────────────────────────────────────────────────────
    // Arbitrary-depth nested-of-nested (∃-over-∃-over-…) tests.
    //
    // A uniform recursive schema is used so depth is a parameter, not hardcoded: the struct at every
    // level 1..D has an Int64 leaf `v`; every level 1..D-1 ALSO has `kids: List<Struct(level+1)>`. A
    // predicate descends via {"nested":"kids","inner":…} once per array level, so testing depth D just
    // means wrapping the leaf comparison D-1 times. This exercises the real recursive value_offsets
    // descent (one independent offsets buffer per level) and the boolean (non-3VL) ∃ semantics for
    // empty inner list / null inner-list slot / null leaf — at 5, 6, and 7 levels, with correlation.
    // ────────────────────────────────────────────────────────────────────────────────────────────

    /// One nested node in the test model. `v` = this level's leaf (None = null leaf). `kids` = the inner
    /// nested array: `None` = null inner-list slot; `Some(vec![])` = empty inner list; `Some(v)` = elements.
    #[derive(Clone)]
    struct N {
        v: Option<i64>,
        kids: Option<Vec<N>>,
    }
    fn nd(v: i64, kids: Vec<N>) -> N {
        N { v: Some(v), kids: Some(kids) }
    }
    fn leaf(v: i64) -> N {
        N { v: Some(v), kids: Some(vec![]) }
    }
    fn leaf_null() -> N {
        N { v: None, kids: Some(vec![]) }
    }
    fn null_kids(v: i64) -> N {
        N { v: Some(v), kids: None }
    }
    /// A straight chain L1→L2→…→Ln carrying `vals[i]` at level i+1 (deepest value last).
    fn chain(vals: &[i64]) -> N {
        let mut node = leaf(vals[vals.len() - 1]);
        for &v in vals[..vals.len() - 1].iter().rev() {
            node = nd(v, vec![node]);
        }
        node
    }

    /// The struct fields at `depth` for a uniform tree of total `max_depth`: `v` at every level, plus
    /// `kids: List<Struct(depth+1)>` for every non-deepest level.
    fn struct_fields_for(depth: usize, max_depth: usize) -> Fields {
        if depth == max_depth {
            Fields::from(vec![Field::new("v", DataType::Int64, true)])
        } else {
            let inner = struct_fields_for(depth + 1, max_depth);
            let item = Arc::new(Field::new("item", DataType::Struct(inner), true));
            Fields::from(vec![
                Field::new("v", DataType::Int64, true),
                Field::new("kids", DataType::List(item), true),
            ])
        }
    }

    /// Build the StructArray for a flat sequence of nodes at `depth`. Recurses to build the `kids`
    /// List<Struct> for non-deepest levels, honoring null inner-list slots (validity 0, empty range) and
    /// empty inner lists (validity 1, empty range).
    fn build_struct(nodes: &[&N], depth: usize, max_depth: usize) -> StructArray {
        let fields = struct_fields_for(depth, max_depth);
        let vs: Vec<Option<i64>> = nodes.iter().map(|n| n.v).collect();
        let v_array = Arc::new(Int64Array::from(vs)) as ArrayRef;
        if depth == max_depth {
            return StructArray::new(fields, vec![v_array], None);
        }
        let mut offsets: Vec<i32> = vec![0];
        let mut acc = 0i32;
        let mut validity: Vec<bool> = Vec::with_capacity(nodes.len());
        let mut flat: Vec<&N> = Vec::new();
        for n in nodes {
            match &n.kids {
                None => {
                    validity.push(false);
                    offsets.push(acc); // null slot → empty range
                }
                Some(kids) => {
                    validity.push(true);
                    for k in kids {
                        flat.push(k);
                    }
                    acc += kids.len() as i32;
                    offsets.push(acc);
                }
            }
        }
        let child = build_struct(&flat, depth + 1, max_depth);
        let inner_fields = struct_fields_for(depth + 1, max_depth);
        let item = Arc::new(Field::new("item", DataType::Struct(inner_fields), true));
        let nulls = NullBuffer::new(BooleanBuffer::from(validity));
        let kids_list = ListArray::new(item, OffsetBuffer::new(offsets.into()), Arc::new(child), Some(nulls));
        StructArray::new(fields, vec![v_array, Arc::new(kids_list) as ArrayRef], None)
    }

    /// Build a top-level `LIST<STRUCT>` of `max_depth` levels. Each parent row is `Some(elements)` or
    /// `None` (a null top-level array row).
    fn deep_array_opt(rows: &[Option<Vec<N>>], max_depth: usize) -> ArrayRef {
        let mut offsets: Vec<i32> = vec![0];
        let mut acc = 0i32;
        let mut validity: Vec<bool> = Vec::with_capacity(rows.len());
        let mut flat: Vec<&N> = Vec::new();
        for row in rows {
            match row {
                None => {
                    validity.push(false);
                    offsets.push(acc);
                }
                Some(elems) => {
                    validity.push(true);
                    for n in elems {
                        flat.push(n);
                    }
                    acc += elems.len() as i32;
                    offsets.push(acc);
                }
            }
        }
        let struct_array = build_struct(&flat, 1, max_depth);
        let fields = struct_fields_for(1, max_depth);
        let item = Arc::new(Field::new("item", DataType::Struct(fields), true));
        let nulls = NullBuffer::new(BooleanBuffer::from(validity));
        Arc::new(ListArray::new(item, OffsetBuffer::new(offsets.into()), Arc::new(struct_array), Some(nulls))) as ArrayRef
    }
    /// Convenience for the common all-non-null-parents case.
    fn deep_array(rows: &[Vec<N>], max_depth: usize) -> ArrayRef {
        let opt: Vec<Option<Vec<N>>> = rows.iter().map(|r| Some(r.clone())).collect();
        deep_array_opt(&opt, max_depth)
    }

    /// Wrap `leaf_pred` in `descend` levels of {"nested":"kids","inner":…} — a pure ∃-through-all-levels
    /// predicate reaching a leaf `descend` array levels below the top element.
    fn deep_pred(descend: usize, leaf_pred: Json) -> Json {
        let mut node = leaf_pred;
        for _ in 0..descend {
            node = json!({"nested": "kids", "inner": node});
        }
        node
    }
    fn eq_v(target: i64) -> Json {
        json!({"op": "=", "args": [{"field": "v"}, {"lit": target}]})
    }
    fn eval_deep(arr: &ArrayRef, pred: &Json) -> BooleanArray {
        evaluate_nested_any_match(arr, pred, None).unwrap()
    }

    #[test]
    fn depth5_pure_existence() {
        // ∃ a depth-5 leaf with v==99 anywhere under the parent. P0 chain bottoms at 99, P1 at 1.
        let arr = deep_array(&[vec![chain(&[1, 2, 3, 4, 99])], vec![chain(&[1, 2, 3, 4, 1])]], 5);
        let pred = deep_pred(4, eq_v(99));
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn depth6_pure_existence() {
        let arr = deep_array(&[vec![chain(&[1, 2, 3, 4, 5, 600])], vec![chain(&[1, 2, 3, 4, 5, 6])]], 6);
        let pred = deep_pred(5, eq_v(600));
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn depth7_pure_existence() {
        // 7 levels of ∃. Target 777 sits at the very bottom of P0's chain only.
        let arr = deep_array(&[vec![chain(&[1, 2, 3, 4, 5, 6, 777])], vec![chain(&[1, 2, 3, 4, 5, 6, 7])]], 7);
        let pred = deep_pred(6, eq_v(777));
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn depth5_correlation_is_element_scoped() {
        // The Delta bug, at depth: predicate = (L1.v > 100) AND (∃ L2 with v == 7). A parent matches ONLY
        // if the SAME L1 element satisfies both. P0's single L1 element does. P1 SPREADS the two facts
        // across two different L1 elements (one has v>100, a DIFFERENT one has an L2 v==7) → must NOT match.
        let arr = deep_array(
            &[
                vec![nd(200, vec![leaf(7)])],
                vec![nd(200, vec![leaf(3)]), nd(50, vec![leaf(7)])],
            ],
            5,
        );
        let pred = json!({"op": "AND", "args": [
            {"op": ">", "args": [{"field": "v"}, {"lit": 100}]},
            {"nested": "kids", "inner": eq_v(7)}
        ]});
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn depth7_correlation_deep() {
        // Correlation at the DEEPEST boundary: reach L6, require (L6.v == 50) AND (∃ L7 with v == 900).
        // P0 has one L6 element satisfying both. P1 spreads them across two L6 elements → no match.
        let l6_match = nd(50, vec![leaf(900)]); // L6.v==50 AND has L7 v==900
        let l6_wrong_leaf = nd(50, vec![leaf(1)]); // v==50 but no L7 900
        let l6_wrong_v = nd(51, vec![leaf(900)]); // has L7 900 but v!=50
        // Build parents whose chains L1..L5 lead to these L6 elements.
        let p0 = nd(1, vec![nd(2, vec![nd(3, vec![nd(4, vec![nd(5, vec![l6_match])])])])]);
        let p1 = nd(
            1,
            vec![nd(2, vec![nd(3, vec![nd(4, vec![nd(5, vec![l6_wrong_leaf, l6_wrong_v])])])])],
        );
        let arr = deep_array(&[vec![p0], vec![p1]], 7);
        // Descend L1→L5 (5 levels), then at L6: AND[ v==50, ∃L7 v==900 ].
        let l6_pred = json!({"op": "AND", "args": [
            eq_v(50),
            {"nested": "kids", "inner": eq_v(900)}
        ]});
        let pred = deep_pred(5, l6_pred);
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn deep_all_null_cases_combined() {
        // One array mixing EVERY null/empty case at depth 4, descending to L4 v==42:
        //   - null inner-list slot (null_kids) at L2
        //   - empty inner list at L3
        //   - null leaf at L4 (must not poison; sibling still evaluated)
        //   - a real match at L4 reached past the null leaf
        // P0 contains a matching branch → true. P1 has only decoys (null list, null leaf, empty) → false.
        let p0 = nd(
            1,
            vec![
                null_kids(2),                                   // L2: null inner list
                nd(2, vec![nd(3, vec![])]),                     // L2→L3 with empty L4 list
                nd(2, vec![nd(3, vec![leaf_null(), leaf(42)])]), // L2→L3→[null L4, L4=42] ⇒ match past null
            ],
        );
        let p1 = nd(
            1,
            vec![
                null_kids(2),
                nd(2, vec![nd(3, vec![leaf_null()])]), // only a null leaf at L4
                nd(2, vec![nd(3, vec![])]),            // empty L4
            ],
        );
        let arr = deep_array(&[vec![p0], vec![p1]], 4);
        let pred = deep_pred(3, eq_v(42));
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn deep_null_parent_row_stays_null() {
        // A null TOP-LEVEL array row must yield null (not false), even with a deep predicate. Row 0 matches,
        // row 1 is a null array, row 2 is a non-matching chain.
        let arr = deep_array_opt(
            &[Some(vec![chain(&[1, 2, 3, 55])]), None, Some(vec![chain(&[1, 2, 3, 4])])],
            4,
        );
        let pred = deep_pred(3, eq_v(55));
        let out = eval_deep(&arr, &pred);
        assert_eq!(out.value(0), true);
        assert!(out.is_null(1));
        assert_eq!(out.value(2), false);
    }

    #[test]
    fn deep_empty_inner_at_every_level_is_false_not_null() {
        // A chain of empty inner lists (each level present but its kids empty) must be false, not null:
        // ∃ over empty = false, and that false must propagate up cleanly through every level.
        let empties = nd(1, vec![nd(2, vec![nd(3, vec![])])]); // L1→L2→L3 with empty L4
        let arr = deep_array(&[vec![empties]], 4);
        let pred = deep_pred(3, eq_v(42));
        assert_eq!(eval_deep(&arr, &pred), BooleanArray::from(vec![false]));
    }

    #[test]
    fn deep_nested_missing_inner_errors() {
        // A {"nested":...} node with no "inner" is a rewriter bug — must fail loud.
        let arr = deep_array(&[vec![chain(&[1, 2])]], 2);
        let bad = json!({"nested": "kids"});
        let err = evaluate_nested_any_match(&arr, &bad, None).unwrap_err();
        assert!(err.to_string().contains("inner"), "unexpected error: {err}");
    }

    #[test]
    fn deep_nested_unknown_field_errors() {
        // Descending into a field that isn't a nested list must fail loud, not silently no-match.
        let arr = deep_array(&[vec![chain(&[1, 2])]], 2);
        let bad = json!({"nested": "nope", "inner": eq_v(2)});
        let err = evaluate_nested_any_match(&arr, &bad, None).unwrap_err();
        assert!(err.to_string().contains("not found"), "unexpected error: {err}");
    }
}

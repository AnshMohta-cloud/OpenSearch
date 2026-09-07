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

    // Compile the JSON tree ONCE for the whole batch: resolve every field name to a column index,
    // every operator string to an enum, and every literal to a `Val`. The per-element loop below then
    // touches no JSON and does no string work. Previously the tree was re-walked per element, paying a
    // `serde_json` map lookup per node plus a linear name-vs-`Fields` string comparison per field read —
    // constant work repeated 20M times on a 10M-row x 2-element scan.
    //
    // A consequence worth noting: malformed predicates (unknown field, nested field that is not
    // List<Struct>) now surface once, before the loop, instead of on first evaluation. That makes them
    // detectable even when every array is empty or null, where the old code would silently never look.
    let compiled = compile_node(tree, &struct_fields)?;

    // Fast path: evaluate the element condition ONCE over the whole flattened child array with Arrow
    // kernels, then roll the per-element mask up to parents. Falls back to the per-element walk below
    // for any node whose exact semantics the vectorised path declines to reproduce (see try_vectorize).
    if let Some(mask) = try_vectorize(&compiled, struct_array, lucene, struct_array.len()) {
        for row_idx in 0..num_rows {
            if list_array.is_null(row_idx) {
                result.append_null();
                continue;
            }
            let start = list_array.value_offsets()[row_idx] as usize;
            let end = list_array.value_offsets()[row_idx + 1] as usize;
            // A null element mask entry is "unknown", which the existential treats as not-matching —
            // the same as the scalar loop, where only `Some(true)` sets `any_match`.
            result.append_value((start..end).any(|i| mask.is_valid(i) && mask.value(i)));
        }
        return Ok(result.finish());
    }

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
            if let Some(true) = eval_compiled(&compiled, struct_array, elem_idx, lucene)? {
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

/// A value produced while evaluating one element, **borrowed** from the Arrow array (for a field
/// read) or from the parsed JSON tree (for a literal).
///
/// This exists purely to keep the inner loop allocation-free. The obvious implementation reads a
/// field with `ScalarValue::try_from_array`, which for a keyword column heap-allocates a `String`
/// per element; the literal side then allocated another via `json_to_scalar`, and the old
/// `scalar_to_string` cloned *both* again to compare them — four allocations per element per
/// comparison, 80M of them on a 20M-element scan. Numeric predicates never paid this (they coerce
/// through `f64`), which is exactly why they already ran at native-DataFusion speed while keyword
/// predicates ran ~2x slower. Borrowing removes all four.
///
/// `Other` stands for a non-null value of a type this evaluator does not compare (dates,
/// timestamps, structs). It deliberately reproduces the previous behaviour: such a value is
/// *present* for `EXISTS`, but any comparison against it yields `None` (unknown).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Val<'a> {
    Null,
    Num(f64),
    Str(&'a str),
    Bool(bool),
    Other,
}

/// Read element `idx` out of a struct field column without allocating.
fn read_field(arr: &dyn Array, idx: usize) -> Val<'_> {
    use datafusion::arrow::datatypes::{
        Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    };
    if arr.is_null(idx) {
        return Val::Null;
    }
    match arr.data_type() {
        DataType::Utf8 => Val::Str(arr.as_string::<i32>().value(idx)),
        DataType::LargeUtf8 => Val::Str(arr.as_string::<i64>().value(idx)),
        DataType::Utf8View => Val::Str(arr.as_string_view().value(idx)),
        DataType::Int8 => Val::Num(arr.as_primitive::<Int8Type>().value(idx) as f64),
        DataType::Int16 => Val::Num(arr.as_primitive::<Int16Type>().value(idx) as f64),
        DataType::Int32 => Val::Num(arr.as_primitive::<Int32Type>().value(idx) as f64),
        DataType::Int64 => Val::Num(arr.as_primitive::<Int64Type>().value(idx) as f64),
        DataType::UInt8 => Val::Num(arr.as_primitive::<UInt8Type>().value(idx) as f64),
        DataType::UInt16 => Val::Num(arr.as_primitive::<UInt16Type>().value(idx) as f64),
        DataType::UInt32 => Val::Num(arr.as_primitive::<UInt32Type>().value(idx) as f64),
        DataType::UInt64 => Val::Num(arr.as_primitive::<UInt64Type>().value(idx) as f64),
        DataType::Float32 => Val::Num(arr.as_primitive::<Float32Type>().value(idx) as f64),
        DataType::Float64 => Val::Num(arr.as_primitive::<Float64Type>().value(idx)),
        DataType::Boolean => Val::Bool(arr.as_boolean().value(idx)),
        _ => Val::Other,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Compiled predicate tree
// ════════════════════════════════════════════════════════════════════════════
//
// The wire format is JSON, which is convenient to emit from Java but expensive to interpret: every
// node access is a `serde_json` map lookup and every `{"field":...}` resolves its column by scanning
// `Fields` and comparing strings. Doing that per array element meant repeating fixed work 20,000,000
// times on a 10M-row x 2-element scan.
//
// `compile_node` walks the JSON exactly once per batch and lowers it to the enums below: field names
// become column indices, operator strings become discriminants, literals become `Val`s. `eval_compiled`
// then evaluates per element with no JSON, no string comparison and no allocation.
//
// Semantics are unchanged from the interpreted version — same three-valued logic, same two-valued
// existential/`EXISTS` conventions, same error messages.

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// A value-producing node: a field read (by resolved column index), a literal, or arithmetic.
#[derive(Debug)]
enum CVal<'a> {
    Field(usize),
    Lit(Val<'a>),
    Arith(ArithOp, Box<CVal<'a>>, Box<CVal<'a>>),
}

/// A boolean-producing node.
#[derive(Debug)]
enum CNode<'a> {
    And(Vec<CNode<'a>>),
    Or(Vec<CNode<'a>>),
    Not(Box<CNode<'a>>),
    Cmp(CmpOp, CVal<'a>, CVal<'a>),
    /// `EXISTS` / `NOT_EXISTS`; the flag is set for the negated form.
    Exists(CVal<'a>, bool),
    /// Child-grain split leaf: use clause `idx`'s per-element verdict when supplied, else `fallback`.
    Lucene { idx: usize, fallback: Box<CNode<'a>> },
    /// Descent into an inner `LIST<STRUCT>` column, opening a fresh existential over its elements.
    /// `inner` is compiled against the *inner* struct's fields.
    Nested { col: usize, inner: Box<CNode<'a>> },
}

/// Resolve a field name to its column index, with the same diagnostic the interpreter produced.
fn resolve_field(fields: &datafusion::arrow::datatypes::Fields, name: &str, what: &str) -> Result<usize> {
    fields.iter().position(|f| f.name() == name).ok_or_else(|| {
        DataFusionError::Execution(format!(
            "nested_any_match_expr: {what}'{name}' not found. Available: {:?}",
            fields.iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })
}

fn compile_val<'a>(node: &'a Json, fields: &datafusion::arrow::datatypes::Fields) -> Result<CVal<'a>> {
    if let Some(field_name) = node.get("field").and_then(|v| v.as_str()) {
        return Ok(CVal::Field(resolve_field(fields, field_name, "field ")?));
    }
    if let Some(lit) = node.get("lit") {
        return Ok(CVal::Lit(json_to_val(lit)));
    }
    if let Some(op) = node.get("op").and_then(|v| v.as_str()) {
        let args = node
            .get("args")
            .and_then(|v| v.as_array())
            .ok_or_else(|| DataFusionError::Execution(format!("nested_any_match_expr: missing 'args' in node {node}")))?;
        if args.len() != 2 {
            return plan_err!("nested_any_match_expr: arithmetic op '{op}' expects 2 args");
        }
        let aop = match op {
            "+" => ArithOp::Add,
            "-" => ArithOp::Sub,
            "*" => ArithOp::Mul,
            "/" => ArithOp::Div,
            "%" => ArithOp::Rem,
            other => return plan_err!("nested_any_match_expr: unsupported arithmetic operator '{other}'"),
        };
        return Ok(CVal::Arith(
            aop,
            Box::new(compile_val(&args[0], fields)?),
            Box::new(compile_val(&args[1], fields)?),
        ));
    }
    plan_err!("nested_any_match_expr: unrecognized value node {node}")
}

fn compile_node<'a>(node: &'a Json, fields: &datafusion::arrow::datatypes::Fields) -> Result<CNode<'a>> {
    // Child-grain split leaf: `{"lucene": <clauseIdx>, "fallback": <originalPredicateSubtree>}`.
    // Lucene is a pure OPTIMIZATION, never a correctness dependency: the executor may or may not supply
    // this clause's per-element verdicts, so the `fallback` subtree is always compiled and stands ready.
    if let Some(idx) = node.get("lucene").and_then(|v| v.as_u64()) {
        let idx = idx as usize;
        let fallback = node.get("fallback").ok_or_else(|| {
            DataFusionError::Execution(format!(
                "nested_any_match_expr: {{\"lucene\":{idx}}} node has neither delegated bits nor a \
                 \"fallback\" subtree — the rewriter must always emit a fallback so the predicate is \
                 correct when Lucene verdicts are absent"
            ))
        })?;
        return Ok(CNode::Lucene {
            idx,
            fallback: Box::new(compile_node(fallback, fields)?),
        });
    }

    // Nested descent: `{"nested":"<field>","inner":<subtree>}`. `field` names a column on the CURRENT
    // struct that is itself List<Struct>, so `inner` compiles against the INNER struct's fields — which
    // is what makes the evaluator arbitrary-depth (one existential loop per array level).
    if let Some(field_name) = node.get("nested").and_then(|v| v.as_str()) {
        let inner_subtree = node.get("inner").ok_or_else(|| {
            DataFusionError::Execution(format!(
                "nested_any_match_expr: {{\"nested\":\"{field_name}\"}} node missing \"inner\" subtree"
            ))
        })?;
        let col = resolve_field(fields, field_name, "nested field ")?;
        let inner_fields = match fields[col].data_type() {
            DataType::List(f) => match f.data_type() {
                DataType::Struct(inner) => inner.clone(),
                other => {
                    return plan_err!(
                        "nested_any_match_expr: nested field '{field_name}' must be List<Struct>, got List<{other:?}>"
                    )
                }
            },
            other => return plan_err!("nested_any_match_expr: nested field '{field_name}' must be a List, got {other:?}"),
        };
        return Ok(CNode::Nested {
            col,
            inner: Box::new(compile_node(inner_subtree, &inner_fields)?),
        });
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
        "AND" | "OR" => {
            let mut kids = Vec::with_capacity(args.len());
            for a in args {
                kids.push(compile_node(a, fields)?);
            }
            Ok(if op == "AND" { CNode::And(kids) } else { CNode::Or(kids) })
        }
        "NOT" => {
            if args.len() != 1 {
                return plan_err!("nested_any_match_expr: NOT expects 1 arg");
            }
            Ok(CNode::Not(Box::new(compile_node(&args[0], fields)?)))
        }
        ">" | ">=" | "<" | "<=" | "=" | "!=" => {
            if args.len() != 2 {
                return plan_err!("nested_any_match_expr: comparison expects 2 args");
            }
            let cop = match op {
                ">" => CmpOp::Gt,
                ">=" => CmpOp::Ge,
                "<" => CmpOp::Lt,
                "<=" => CmpOp::Le,
                "=" => CmpOp::Eq,
                _ => CmpOp::Ne,
            };
            Ok(CNode::Cmp(
                cop,
                compile_val(&args[0], fields)?,
                compile_val(&args[1], fields)?,
            ))
        }
        "EXISTS" | "NOT_EXISTS" => {
            if args.len() != 1 {
                return plan_err!("nested_any_match_expr: {op} expects 1 arg");
            }
            Ok(CNode::Exists(compile_val(&args[0], fields)?, op == "NOT_EXISTS"))
        }
        other => plan_err!("nested_any_match_expr: '{other}' is not a boolean operator"),
    }
}

/// Evaluate a compiled value node for one struct element. The returned [`Val`] borrows from the array
/// or from the compiled literal — nothing is copied.
fn eval_cval<'a>(v: &'a CVal<'a>, struct_array: &'a datafusion::arrow::array::StructArray, elem_idx: usize) -> Result<Val<'a>> {
    match v {
        CVal::Field(col) => Ok(read_field(struct_array.column(*col).as_ref(), elem_idx)),
        CVal::Lit(lit) => Ok(*lit),
        CVal::Arith(op, l, r) => {
            let left = eval_cval(l, struct_array, elem_idx)?;
            let right = eval_cval(r, struct_array, elem_idx)?;
            arithmetic(&left, *op, &right)
        }
    }
}

/// Evaluate a compiled boolean node for one struct element. `Ok(None)` is a NULL result (SQL
/// three-valued logic — e.g. comparing against a NULL field value).
///
/// `lucene` carries per-element results for Lucene-delegated leaves (child-grain split); `None` on the
/// plain path. It is threaded through `Nested` descents UNCHANGED: each clause's bit array is built by
/// the executor in that clause's own coordinate space, so indexing with whatever `elem_idx` is current
/// at this recursion depth is always correct, however many levels were crossed to get here.
fn eval_compiled(
    node: &CNode<'_>,
    struct_array: &datafusion::arrow::array::StructArray,
    elem_idx: usize,
    lucene: Option<&LuceneClauseBits>,
) -> Result<Option<bool>> {
    match node {
        CNode::Lucene { idx, fallback } => {
            // Two-valued when Lucene supplied verdicts (it has no NULL notion); otherwise evaluate the
            // fallback natively so the answer is identical on every execution path.
            if let Some(l) = lucene {
                if l.has_clause(*idx) {
                    return Ok(Some(l.value(*idx, elem_idx)));
                }
            }
            eval_compiled(fallback, struct_array, elem_idx, lucene)
        }

        // Semantics here are TWO-VALUED, not 3VL — matching vanilla OpenSearch nested-query existential
        // semantics: an empty inner list, a null inner-list slot, and "inner elements present but none
        // match" ALL collapse to Some(false), never None. A null leaf on an inner element is handled
        // inside the recursion (its comparison yields None, so that element simply doesn't match).
        CNode::Nested { col, inner } => {
            let inner_col = struct_array.column(*col);
            if inner_col.is_null(elem_idx) {
                return Ok(Some(false));
            }
            let inner_list = inner_col.as_list_opt::<i32>().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "nested_any_match_expr: nested column {col} must be List<Struct>, got {:?}",
                    inner_col.data_type()
                ))
            })?;
            let inner_struct = inner_list.values().as_struct();
            let start = inner_list.value_offsets()[elem_idx] as usize;
            let end = inner_list.value_offsets()[elem_idx + 1] as usize;
            for inner_idx in start..end {
                if let Some(true) = eval_compiled(inner, inner_struct, inner_idx, lucene)? {
                    return Ok(Some(true));
                }
            }
            Ok(Some(false))
        }

        CNode::And(kids) => {
            for k in kids {
                match eval_compiled(k, struct_array, elem_idx, lucene)? {
                    Some(false) => return Ok(Some(false)),
                    None => return Ok(None), // NULL propagates: NULL AND anything-not-false = NULL
                    Some(true) => continue,
                }
            }
            Ok(Some(true))
        }

        CNode::Or(kids) => {
            let mut saw_null = false;
            for k in kids {
                match eval_compiled(k, struct_array, elem_idx, lucene)? {
                    Some(true) => return Ok(Some(true)),
                    None => saw_null = true,
                    Some(false) => continue,
                }
            }
            Ok(if saw_null { None } else { Some(false) })
        }

        CNode::Not(inner) => Ok(eval_compiled(inner, struct_array, elem_idx, lucene)?.map(|b| !b)),

        CNode::Cmp(op, l, r) => {
            let left = eval_cval(l, struct_array, elem_idx)?;
            let right = eval_cval(r, struct_array, elem_idx)?;
            Ok(compare(&left, *op, &right))
        }

        // Deliberately TWO-VALUED, never None: whether a value is present is never itself "unknown" —
        // matching vanilla OpenSearch's `is not null`/`exists` semantics and this evaluator's two-valued
        // existence convention.
        CNode::Exists(arg, negated) => {
            let present = !is_null(&eval_cval(arg, struct_array, elem_idx)?);
            Ok(Some(if *negated { !present } else { present }))
        }
    }
}

/// A literal from the wire format. The `&str` borrows straight out of the parsed tree, which is
/// parsed once per batch, so a string literal costs nothing per element.
fn json_to_val(v: &Json) -> Val<'_> {
    if let Some(n) = v.as_f64() {
        return Val::Num(n);
    }
    if let Some(s) = v.as_str() {
        if s == "null" {
            return Val::Null;
        }
        return Val::Str(s);
    }
    if let Some(b) = v.as_bool() {
        return Val::Bool(b);
    }
    Val::Null
}

fn is_null(v: &Val<'_>) -> bool {
    matches!(v, Val::Null)
}

/// SQL three-valued comparison: NULL compared to anything is NULL (unknown), never true/false.
///
/// Type pairings mirror the previous `ScalarValue` implementation exactly: numbers compare as
/// `f64`, strings compare lexicographically, and every other combination — including
/// boolean-vs-boolean and any mixed pairing — is `None`. (Boolean yielding `None` is pre-existing
/// behaviour, not something introduced here.)
fn compare(left: &Val<'_>, op: CmpOp, right: &Val<'_>) -> Option<bool> {
    match (left, right) {
        (Val::Null, _) | (_, Val::Null) => None,
        (Val::Num(l), Val::Num(r)) => Some(match op {
            CmpOp::Gt => l > r,
            CmpOp::Ge => l >= r,
            CmpOp::Lt => l < r,
            CmpOp::Le => l <= r,
            CmpOp::Eq => (l - r).abs() < f64::EPSILON,
            CmpOp::Ne => (l - r).abs() >= f64::EPSILON,
        }),
        (Val::Str(l), Val::Str(r)) => Some(match op {
            CmpOp::Gt => l > r,
            CmpOp::Ge => l >= r,
            CmpOp::Lt => l < r,
            CmpOp::Le => l <= r,
            CmpOp::Eq => l == r,
            CmpOp::Ne => l != r,
        }),
        _ => None,
    }
}

fn arithmetic<'a>(left: &Val<'a>, op: ArithOp, right: &Val<'a>) -> Result<Val<'a>> {
    if is_null(left) || is_null(right) {
        return Ok(Val::Null);
    }
    let (l, r) = match (left, right) {
        (Val::Num(l), Val::Num(r)) => (*l, *r),
        _ => return plan_err!("nested_any_match_expr: arithmetic op '{op:?}' requires numeric operands"),
    };
    let result = match op {
        ArithOp::Add => l + r,
        ArithOp::Sub => l - r,
        ArithOp::Mul => l * r,
        ArithOp::Div | ArithOp::Rem => {
            if r == 0.0 {
                return Ok(Val::Null);
            }
            if op == ArithOp::Div { l / r } else { l % r }
        }
    };
    Ok(Val::Num(result))
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

// ════════════════════════════════════════════════════════════════════════════
// Vectorised evaluation
// ════════════════════════════════════════════════════════════════════════════
//
// `eval_compiled` walks the compiled tree once per ELEMENT. With ~2 elements per row over 10M rows
// that is 20,000,000 tree walks, each a chain of virtual dispatches that re-derives the same shape.
// Measured with the Rust-side timer, `on_batch_mask` was 107 ms against 10 ms for an equivalent flat
// (non-nested) predicate on the same index — 10x, and the single largest phase of a nested query.
//
// The element-level condition is independent per element, so it can instead be evaluated ONCE over
// the whole flattened child array with Arrow kernels, yielding a per-element boolean mask, which is
// then rolled up to parents through the list offsets. Same answer, one vectorised pass.
//
// This path is deliberately PARTIAL. `try_vectorize` returns `None` for any node whose exact
// semantics it cannot reproduce, and the caller then runs the original per-element loop for the whole
// batch. That keeps the fast path honest: it never approximates, it declines. Two places where the
// scalar evaluator is subtly non-standard and declining matters:
//
//   * `AND` is NOT Kleene. It yields NULL on the first null operand without looking for a later
//     FALSE, whereas `and_kleene(NULL, FALSE)` is FALSE. The `EXISTS` roll-up hides the difference
//     (both are "not true"), but a `NOT` above it does not — `NOT (a AND b)` is a real shape (F2).
//     So AND is built from raw bitwise buffer ops that reproduce the short-circuit exactly, not from
//     `and_kleene`.
//   * numeric `=` / `!=` compare through `f64` with an `EPSILON` tolerance. That is identical to
//     Arrow's exact `eq` for integers (every i64 up to 2^53 is exactly representable) but not for
//     genuine floats, so float `=`/`!=` declines to the scalar path.

use datafusion::arrow::array::{BooleanArray, Scalar};
use datafusion::arrow::buffer::{BooleanBuffer, NullBuffer};
use datafusion::arrow::compute::kernels::{boolean, cmp};

/// Is this a type whose values are exactly representable in `f64`, so Arrow's exact `eq` agrees with
/// the scalar path's `EPSILON` comparison?
fn is_exact_in_f64(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
    )
}

fn is_numeric_col(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn is_string_col(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// `a AND b` with the scalar evaluator's short-circuit: NULL wins over a later FALSE.
///
/// Pure bitwise algebra on the value/validity buffers, so it stays vectorised:
///   values(r)   = (valid(a) & value(a)) & value(b)
///   validity(r) = valid(a) & (!value(a) | valid(b))
/// i.e. when `a` is null the result is null; when `a` is false the result is a valid false; and only
/// when `a` is true does `b` decide.
fn and_shortcircuit(a: &BooleanArray, b: &BooleanArray) -> BooleanArray {
    let av = a.values();
    let bv = b.values();
    let a_valid = a.nulls().map(|n| n.inner().clone()).unwrap_or_else(|| BooleanBuffer::new_set(a.len()));
    let b_valid = b.nulls().map(|n| n.inner().clone()).unwrap_or_else(|| BooleanBuffer::new_set(b.len()));

    let a_true = &a_valid & av;
    let values = &a_true & bv;
    let validity = &a_valid & &(&!av | &b_valid);
    BooleanArray::new(values, Some(NullBuffer::new(validity)))
}

/// Per-element mask for one compiled node over the whole child array, or `None` when this path
/// cannot reproduce the scalar semantics exactly.
fn try_vectorize(
    node: &CNode<'_>,
    struct_array: &datafusion::arrow::array::StructArray,
    lucene: Option<&LuceneClauseBits>,
    len: usize,
) -> Option<BooleanArray> {
    match node {
        CNode::And(kids) => {
            let mut acc = try_vectorize(kids.first()?, struct_array, lucene, len)?;
            for k in &kids[1..] {
                acc = and_shortcircuit(&acc, &try_vectorize(k, struct_array, lucene, len)?);
            }
            Some(acc)
        }
        // OR *is* Kleene here (TRUE wins, else NULL if any operand was null, else FALSE).
        CNode::Or(kids) => {
            let mut acc = try_vectorize(kids.first()?, struct_array, lucene, len)?;
            for k in &kids[1..] {
                acc = boolean::or_kleene(&acc, &try_vectorize(k, struct_array, lucene, len)?).ok()?;
            }
            Some(acc)
        }
        // `not` preserves nulls, matching `Option::map(|b| !b)`.
        CNode::Not(inner) => boolean::not(&try_vectorize(inner, struct_array, lucene, len)?).ok(),

        // Two-valued by design: presence is never itself unknown. `read_field` returns Val::Null
        // exactly when the slot is null, so this is precisely `is_not_null` on the column.
        CNode::Exists(CVal::Field(col), negated) => {
            let c = struct_array.column(*col);
            let m = if *negated { boolean::is_null(c).ok()? } else { boolean::is_not_null(c).ok()? };
            Some(m)
        }

        CNode::Cmp(op, CVal::Field(col), CVal::Lit(lit)) => cmp_field_lit(struct_array.column(*col), *op, lit),
        // Literal on the left: flip the operator rather than the operands.
        CNode::Cmp(op, CVal::Lit(lit), CVal::Field(col)) => {
            let flipped = match op {
                CmpOp::Gt => CmpOp::Lt,
                CmpOp::Ge => CmpOp::Le,
                CmpOp::Lt => CmpOp::Gt,
                CmpOp::Le => CmpOp::Ge,
                CmpOp::Eq => CmpOp::Eq,
                CmpOp::Ne => CmpOp::Ne,
            };
            cmp_field_lit(struct_array.column(*col), flipped, lit)
        }

        // A Lucene-delegated leaf is a straight bit lookup per element — cheap, and no tree walk.
        CNode::Lucene { idx, fallback } => match lucene {
            Some(l) if l.has_clause(*idx) => Some(BooleanArray::from_iter((0..len).map(|i| Some(l.value(*idx, i))))),
            _ => try_vectorize(fallback, struct_array, lucene, len),
        },

        // Inner LIST<STRUCT> level: vectorise the inner condition over the inner values array, then
        // roll it up to THIS level's elements. Two-valued (empty or null inner list => false),
        // matching the scalar path.
        CNode::Nested { col, inner } => {
            let inner_col = struct_array.column(*col);
            let list = inner_col.as_list_opt::<i32>()?;
            let inner_struct = list.values().as_struct();
            let inner_mask = try_vectorize(inner, inner_struct, lucene, list.values().len())?;
            let offsets = list.value_offsets();
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                if inner_col.is_null(i) {
                    out.push(Some(false));
                    continue;
                }
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                out.push(Some((s..e).any(|j| inner_mask.is_valid(j) && inner_mask.value(j))));
            }
            Some(BooleanArray::from(out))
        }

        // Arithmetic operands, field-vs-field, and EXISTS over a computed value all decline.
        _ => None,
    }
}

/// `column <op> literal` as a vectorised comparison, or `None` when the pair is not one this path
/// can reproduce exactly (mismatched families, float equality, bool/other literals).
fn cmp_field_lit(col: &ArrayRef, op: CmpOp, lit: &Val<'_>) -> Option<BooleanArray> {
    let dt = col.data_type();
    let rhs: ArrayRef = match lit {
        // Comparing against NULL is unknown for every element.
        Val::Null => {
            return Some(BooleanArray::new(
                BooleanBuffer::new_unset(col.len()),
                Some(NullBuffer::new(BooleanBuffer::new_unset(col.len()))),
            ))
        }
        Val::Num(n) if is_numeric_col(dt) => {
            if matches!(op, CmpOp::Eq | CmpOp::Ne) && !is_exact_in_f64(dt) {
                return None; // EPSILON tolerance is not Arrow's exact eq for wide ints / floats
            }
            // Compare in f64, exactly as the scalar path does.
            let casted = datafusion::arrow::compute::kernels::cast::cast(col, &DataType::Float64).ok()?;
            return apply_cmp(&casted, op, &(Arc::new(datafusion::arrow::array::Float64Array::from(vec![*n])) as ArrayRef));
        }
        Val::Str(s) if is_string_col(dt) => Arc::new(datafusion::arrow::array::StringArray::from(vec![*s])) as ArrayRef,
        // Str-vs-numeric and friends yield None per element in the scalar path; decline rather than
        // hand Arrow a type error.
        _ => return None,
    };
    let lhs: ArrayRef = if matches!(dt, DataType::Utf8) {
        Arc::clone(col)
    } else {
        datafusion::arrow::compute::kernels::cast::cast(col, &DataType::Utf8).ok()?
    };
    apply_cmp(&lhs, op, &rhs)
}

fn apply_cmp(lhs: &ArrayRef, op: CmpOp, rhs: &ArrayRef) -> Option<BooleanArray> {
    let r = Scalar::new(rhs);
    match op {
        CmpOp::Gt => cmp::gt(lhs, &r),
        CmpOp::Ge => cmp::gt_eq(lhs, &r),
        CmpOp::Lt => cmp::lt(lhs, &r),
        CmpOp::Le => cmp::lt_eq(lhs, &r),
        CmpOp::Eq => cmp::eq(lhs, &r),
        CmpOp::Ne => cmp::neq(lhs, &r),
    }
    .ok()
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

    /// Like `comments_array` but with a NULLABLE score, for EXISTS/NOT_EXISTS tests. Each row is a
    /// list of `(author, Option<score>)` elements; `None` score = null leaf.
    fn comments_array_nullable_score(rows: &[Vec<(&str, Option<i64>)>]) -> ArrayRef {
        let mut authors: Vec<Option<String>> = Vec::new();
        let mut scores: Vec<Option<i64>> = Vec::new();
        let mut offsets: Vec<i32> = vec![0];
        let mut acc = 0i32;
        for row in rows {
            for (a, s) in row {
                authors.push(Some((*a).to_string()));
                scores.push(*s);
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
            vec![Arc::new(StringArray::from(authors)) as ArrayRef, Arc::new(Int64Array::from(scores)) as ArrayRef],
            None,
        );
        let list_field = Arc::new(Field::new("item", DataType::Struct(struct_fields), true));
        Arc::new(ListArray::new(list_field, OffsetBuffer::new(offsets.into()), Arc::new(struct_array), None)) as ArrayRef
    }

    #[test]
    fn exists_matches_any_element_with_non_null_field() {
        // P0: alice has a null score, bob has 90 -> EXISTS(score) true (bob's element has one).
        // P1: alice has a null score only -> EXISTS(score) false (no element has one).
        let array = comments_array_nullable_score(&[vec![("alice", None), ("bob", Some(90))], vec![("alice", None)]]);
        let tree: Json = json!({"op": "EXISTS", "args": [{"field": "score"}]});
        let out = evaluate_nested_any_match(&array, &tree, None).unwrap();
        assert_eq!(out, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn not_exists_matches_any_element_with_null_field() {
        // P0: bob's score is null -> NOT_EXISTS(score) true (some element lacks it), even though
        // alice's score IS present -> this is an ELEMENT-EXISTENTIAL check, not "all elements".
        // P1: alice has 70 (no null) -> NOT_EXISTS(score) false.
        let array = comments_array_nullable_score(&[vec![("alice", Some(5)), ("bob", None)], vec![("alice", Some(70))]]);
        let tree: Json = json!({"op": "NOT_EXISTS", "args": [{"field": "score"}]});
        let out = evaluate_nested_any_match(&array, &tree, None).unwrap();
        assert_eq!(out, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn exists_composes_with_nested_descent() {
        // Deep EXISTS: {"nested":"kids","inner":{"op":"EXISTS","args":[{"field":"v"}]}} — reuses the
        // depth-D nested-descent model (N/nd/leaf/leaf_null/deep_array, defined further below) to
        // prove EXISTS composes through a {"nested":...} wrapper exactly like a comparison does.
        // max_depth=2: each parent row has ONE level-1 element, whose "kids" holds ONE level-2 child;
        // EXISTS(v) checks that level-2 child's own v.
        let level1_with_present_child = nd(0, vec![leaf(1)]); // child v=Some(1) -> EXISTS true
        let level1_with_absent_child = nd(0, vec![leaf_null()]); // child v=None -> EXISTS false
        let array = deep_array(&[vec![level1_with_present_child], vec![level1_with_absent_child]], 2);
        let tree: Json = json!({"nested": "kids", "inner": {"op": "EXISTS", "args": [{"field": "v"}]}});
        let out = evaluate_nested_any_match(&array, &tree, None).unwrap();
        assert_eq!(out, BooleanArray::from(vec![true, false]));
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
    /// Differential test: the vectorised path must agree with the per-element scalar path EXACTLY,
    /// element for element, including three-valued results. This is the guard that lets
    /// `try_vectorize` be trusted — if it ever diverges rather than declining, this fails.
    ///
    /// Exact comparison (not just "both not-true") is deliberate: it is what catches the non-Kleene
    /// AND short-circuit, where `NULL AND FALSE` must be NULL and not FALSE, because a `NOT` above it
    /// turns that distinction into a different answer.
    #[test]
    fn vectorized_agrees_with_scalar_element_for_element() {
        let arrays: Vec<ArrayRef> = vec![
            corpus(),
            comments_array(&[vec![], vec![("alice", 70)]]),
            comments_array_nullable_score(&[vec![("alice", None), ("bob", Some(90))], vec![("alice", None)]]),
            comments_array_nullable_score(&[vec![("alice", Some(5)), ("bob", None)], vec![("alice", Some(70))]]),
        ];
        let trees = vec![
            json!({"op": "=", "args": [{"field": "author"}, {"lit": "alice"}]}),
            json!({"op": "!=", "args": [{"field": "author"}, {"lit": "alice"}]}),
            json!({"op": ">", "args": [{"field": "score"}, {"lit": 50}]}),
            json!({"op": "<=", "args": [{"field": "score"}, {"lit": 50}]}),
            json!({"op": "AND", "args": [
                {"op": "=", "args": [{"field": "author"}, {"lit": "alice"}]},
                {"op": ">", "args": [{"field": "score"}, {"lit": 50}]}]}),
            json!({"op": "OR", "args": [
                {"op": "=", "args": [{"field": "author"}, {"lit": "zzz"}]},
                {"op": ">", "args": [{"field": "score"}, {"lit": 50}]}]}),
            // NOT over AND — the shape where the non-Kleene short-circuit becomes observable.
            json!({"op": "NOT", "args": [{"op": "AND", "args": [
                {"op": "=", "args": [{"field": "author"}, {"lit": "alice"}]},
                {"op": ">", "args": [{"field": "score"}, {"lit": 50}]}]}]}),
            json!({"op": "NOT", "args": [{"op": "OR", "args": [
                {"op": "=", "args": [{"field": "author"}, {"lit": "alice"}]},
                {"op": ">", "args": [{"field": "score"}, {"lit": 50}]}]}]}),
            json!({"op": "EXISTS", "args": [{"field": "score"}]}),
            json!({"op": "NOT_EXISTS", "args": [{"field": "score"}]}),
            // Arithmetic must DECLINE (return None), not silently differ.
            json!({"op": "=", "args": [{"op": "%", "args": [{"field": "score"}, {"lit": 2}]}, {"lit": 0}]}),
        ];
        let mut vectorized_count = 0;
        for (ai, array) in arrays.iter().enumerate() {
            let list = array.as_list_opt::<i32>().unwrap();
            let fields = match list.data_type() {
                DataType::List(f) => match f.data_type() {
                    DataType::Struct(fs) => fs.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let struct_array = list.values().as_struct();
            for (ti, tree) in trees.iter().enumerate() {
                let compiled = compile_node(tree, &fields).unwrap();
                let Some(mask) = try_vectorize(&compiled, struct_array, None, struct_array.len()) else {
                    continue; // declined — the scalar path handles it, nothing to compare
                };
                vectorized_count += 1;
                for i in 0..struct_array.len() {
                    let scalar = eval_compiled(&compiled, struct_array, i, None).unwrap();
                    let vect = if mask.is_valid(i) { Some(mask.value(i)) } else { None };
                    assert_eq!(scalar, vect, "array {ai} tree {ti} element {i}: scalar {scalar:?} != vectorized {vect:?}");
                }
                // and the roll-up must match too
                let a = evaluate_nested_any_match(array, tree, None).unwrap();
                let mut b = BooleanBuilder::with_capacity(list.len());
                for r in 0..list.len() {
                    if list.is_null(r) {
                        b.append_null();
                        continue;
                    }
                    let (s, e) = (list.value_offsets()[r] as usize, list.value_offsets()[r + 1] as usize);
                    let mut any = false;
                    for i in s..e {
                        if let Some(true) = eval_compiled(&compiled, struct_array, i, None).unwrap() {
                            any = true;
                            break;
                        }
                    }
                    b.append_value(any);
                }
                assert_eq!(a, b.finish(), "array {ai} tree {ti}: roll-up differs");
            }
        }
        assert!(vectorized_count >= 20, "expected the fast path to cover most trees, covered {vectorized_count}");
    }
}
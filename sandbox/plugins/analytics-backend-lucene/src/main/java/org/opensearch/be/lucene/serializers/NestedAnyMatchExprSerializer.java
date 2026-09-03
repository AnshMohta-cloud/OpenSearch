/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.serializers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.json.JsonMapper;

import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.opensearch.analytics.spi.FieldStorageInfo;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;

import java.util.List;

/**
 * Serializer for {@code NESTED_ANY_MATCH_EXPR(arrayCol, jsonExprTree)} — see
 * {@code OpenSearchNestedFieldRewriter}'s javadoc on {@code NESTED_ANY_MATCH_EXPR_OP} for the wire
 * format and the two-phase capability story.
 *
 * <h2>What is translated</h2>
 * The JSON tree can describe ANY per-element predicate (arithmetic, ranges, NOT, ...). This serializer
 * translates the boolean-of-equalities subset:
 * <ul>
 *   <li>{@code {"op":"=","args":[{"field":F},{"lit":V}]}} (either operand order, {@code V} a string)
 *       → {@link TermQueryBuilder} on the dotted leaf path;</li>
 *   <li>{@code {"op":"OR","args":[...]}} → {@link BoolQueryBuilder} with each arg as a {@code should}
 *       plus {@code minimumShouldMatch(1)};</li>
 *   <li>{@code {"nested":hop,"inner":...}} wrappers — descended, extending the dotted path (the shape a
 *       multi-level path such as {@code products.variants.color='red'} produces).</li>
 * </ul>
 * Anything else makes {@link #canServe} return false, so the predicate stays DataFusion-only.
 *
 * <h2>Why a flat term query is the right primitive</h2>
 * The Lucene secondary writes one document per Parquet row, with each nested leaf flattened into a
 * MULTI-VALUED field on that row doc (see {@code LuceneDocumentInput}). So a term query on the dotted
 * path already answers "does SOME element of this row have this value?" — exactly
 * {@code NESTED_ANY_MATCH_EXPR}'s existential semantics for a single leaf. No {@code NestedQueryBuilder}
 * and no block join: those would target child docs that no longer exist and match nothing.
 *
 * <h2>Only EXACT renderings are served</h2>
 * Existential quantification distributes over disjunction but not over conjunction:
 * <pre>
 *   ∃e (A(e) ∨ B(e))  ≡  (∃e A(e)) ∨ (∃e B(e))     → OR renders EXACTLY
 *   ∃e (A(e) ∧ B(e))  ⊆  (∃e A(e)) ∧ (∃e B(e))     → AND is only a SUPERSET  → NOT served
 * </pre>
 * A multi-valued field cannot express "the same element satisfied both", so an {@code AND} of two leaves
 * would also match a row whose two values live in DIFFERENT elements. A superset is fine while the
 * predicate stays <em>performance</em>-delegated ({@code strip} emits
 * {@code delegation_possible(original, id)}, retaining the original for DataFusion to re-evaluate), but
 * {@code DelegatedPredicateCombiner} reclassifies performance → correctness for any predicate under an
 * outer OR/NOT, and the correctness branch of {@code strip} calls {@code makePlaceholder}, which DROPS the
 * original. Serving only exact renderings makes this serializer safe under either delegation mode, which is
 * the invariant worth keeping — see the note on the AND case in {@link #buildQuery} for the three
 * empirically-verified reasons a superset cannot be made safe from the planner side today.
 */
public class NestedAnyMatchExprSerializer extends AbstractQuerySerializer {

    private static final org.apache.logging.log4j.Logger LOGGER = org.apache.logging.log4j.LogManager.getLogger(
        NestedAnyMatchExprSerializer.class
    );

    private static final JsonMapper MAPPER = JsonMapper.builder().build();

    /** Placeholder prefix for {@link #canServe}, which needs only the structural answer, not real paths. */
    private static final String PROBE_PREFIX = "_";

    @Override
    public boolean canServe(RexCall call, List<FieldStorageInfo> fieldStorage) {
        JsonNode root = parseJson(call);
        return root != null && buildQuery(root, PROBE_PREFIX) != null;
    }

    @Override
    public QueryBuilder buildQueryBuilder(RexCall call, List<FieldStorageInfo> fieldStorage) {
        JsonNode root = parseJson(call);
        if (root == null) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_EXPR: 2nd operand must be a JSON string literal");
        }
        List<RexNode> operands = call.getOperands();
        if (!(operands.get(0) instanceof RexInputRef arrayColRef)) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_EXPR's 1st operand must be the array column, got " + operands.get(0));
        }
        // The dotted path root. Leaf paths below are arrayCol[.hop...].field — exactly the field name the
        // Lucene secondary indexed the leaf under (MappedFieldType.name() is already fully qualified),
        // multi-valued with one value per nested element regardless of nesting depth.
        String arrayColPath = FieldStorageInfo.resolve(fieldStorage, arrayColRef.getIndex()).getFieldName();

        QueryBuilder query = buildQuery(root, arrayColPath);
        if (query == null) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_EXPR: unsupported expr tree for Lucene delegation");
        }
        LOGGER.info("[NAM-SER] arrayCol=[{}] -> flat query on multi-valued row-doc field(s): {}", arrayColPath, query);
        return query;
    }

    /** The parsed JSON expr tree from the call's 2nd operand, or {@code null} if absent/malformed. */
    private static JsonNode parseJson(RexCall call) {
        List<RexNode> operands = call.getOperands();
        if (operands.size() != 2 || !(operands.get(1) instanceof RexLiteral jsonLit)) {
            return null;
        }
        String json = jsonLit.getValueAs(String.class);
        if (json == null) {
            return null;
        }
        try {
            return MAPPER.readTree(json);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Recursively renders {@code node} as a Lucene {@link QueryBuilder} over dotted leaf paths rooted at
     * {@code pathPrefix}, or {@code null} if any part of the tree falls outside the translatable subset (a
     * range, a NOT, a non-string literal, arithmetic, a malformed node, ...). Returning {@code null} is not
     * an error — it just leaves the predicate DataFusion-only.
     */
    private static QueryBuilder buildQuery(JsonNode node, String pathPrefix) {
        if (node == null || node.isObject() == false) {
            return null;
        }
        // {"nested": hop, "inner": ...} — descend one array boundary, extending the dotted path.
        if (node.has("nested") && node.has("inner")) {
            JsonNode hop = node.get("nested");
            if (hop.isTextual() == false) {
                return null;
            }
            return buildQuery(node.get("inner"), pathPrefix + "." + hop.asText());
        }
        String op = node.path("op").asText(null);
        JsonNode args = node.get("args");
        if (op == null || args == null || args.isArray() == false || args.isEmpty()) {
            return null;
        }
        switch (op) {
            case "=" -> {
                if (args.size() != 2) {
                    return null;
                }
                FieldAndValue fv = fieldAndLiteral(args.get(0), args.get(1));
                if (fv == null) {
                    fv = fieldAndLiteral(args.get(1), args.get(0));
                }
                return fv == null ? null : new TermQueryBuilder(pathPrefix + "." + fv.field(), fv.value());
            }
            case "OR" -> {
                // EXACT: existential quantification distributes over disjunction, so a row-level
                // should-clause set answers ∃e(A ∨ B) precisely.
                BoolQueryBuilder bool = new BoolQueryBuilder();
                for (JsonNode arg : args) {
                    QueryBuilder child = buildQuery(arg, pathPrefix);
                    if (child == null) {
                        return null; // one untranslatable branch makes the whole tree untranslatable
                    }
                    bool.should(child);
                }
                return bool.minimumShouldMatch(1);
            }
            // "AND" is deliberately NOT translated, and this is a CORRECTNESS constraint, not caution.
            // A row-level must-clause set answers (∃e A) ∧ (∃e B), a strict SUPERSET of ∃e(A ∧ B): a
            // multi-valued field cannot say "the same element satisfied both". A superset is sound only
            // while the predicate stays PERFORMANCE-delegated, where strip() keeps the original for
            // DataFusion to re-check. Three things make that impossible to guarantee from here, all
            // verified empirically on a live node:
            // 1. Under a plan-level OR/NOT, DelegatedPredicateCombiner reclassifies performance ->
            // correctness, and the correctness branch of FragmentConversionDriver.strip calls
            // makePlaceholder, which DROPS the original — the superset is then trusted.
            // `where (events.name='cache_miss' and events.status='error') or trace='R2'` returned an
            // extra row whose two values lived in different elements.
            // 2. Suppressing that reclassification does not help: a PERFORMANCE-delegated leaf under an
            // OR degenerates to all-true at runtime (the same query then returned every row, with
            // collectDocs never called), which is exactly why the combiner reclassifies in the first
            // place. demote_delegation_possible() does not rescue this shape.
            // 3. Declining to delegate in the combiner does not help either, because merely reporting
            // canServe=true here makes the leaf dual-viable, and the operator-level viability
            // decided upstream in OpenSearchFilterRule is itself enough to change fragment
            // classification and break the OR shapes.
            // So an AND-bearing tree must not be Lucene-viable AT ALL until the DataFusion/Rust evaluator
            // supports a delegated leaf under OR/NOT (see the unimplemented! arms in
            // indexed_table/eval/{mod,bitmap_tree}.rs). Acceleration for the AND case still comes from
            // OpenSearchNestedFieldRewriter's per-conjunct pruning peers, which are single equality leaves
            // and therefore individually exact — safe under either delegation mode.
            default -> {
                return null;
            }
        }
    }

    private record FieldAndValue(String field, String value) {
    }

    private static FieldAndValue fieldAndLiteral(JsonNode maybeField, JsonNode maybeLiteral) {
        if (maybeField.isObject() == false
            || maybeField.has("field") == false
            || maybeLiteral.isObject() == false
            || maybeLiteral.has("lit") == false) {
            return null;
        }
        JsonNode fieldNode = maybeField.get("field");
        JsonNode litNode = maybeLiteral.get("lit");
        if (fieldNode.isTextual() == false || litNode.isTextual() == false) {
            return null;
        }
        return new FieldAndValue(fieldNode.asText(), litNode.asText());
    }
}

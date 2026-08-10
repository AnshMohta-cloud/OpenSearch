/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.serializers;

import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.opensearch.analytics.spi.FieldStorageInfo;
import org.opensearch.index.mapper.NestedPathFieldMapper;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;

import java.util.List;

/**
 * Serializer for {@code NESTED_ANY_MATCH_CHILD(arrayCol, 'field', 'EQUALS', 'value', clauseIdx)} — the
 * CHILD-GRAIN sibling of {@link NestedAnyMatchSerializer} for the opt-in child-grain nested split.
 *
 * <p>Unlike the parent-grain variant (which wraps the term in a {@link org.opensearch.index.query.NestedQueryBuilder}
 * → {@code ToParentBlockJoinQuery} returning PARENT docs), this ships a CHILD-SCOPED query whose scorer yields
 * the matching CHILD docs directly: {@code bool(must: term(deepestPath.leaf, value), filter: term(_nested_path, deepestPath))}.
 * The {@code _nested_path} filter restricts to child docs of exactly this nested level (so the term never
 * matches a parent or a sibling-path child). The executor then translates each matched child docId to its
 * {@code (root row, element offset)} via the child-ordinal map and sets the corresponding child-element bit —
 * the per-element verdict the {@code NESTED_ANY_MATCH_EXPR} residual consumes at its {@code {"lucene": clauseIdx}}
 * node, intersecting the keyword (Lucene) and range/other (DataFusion) clauses at the SAME element before the
 * ∃ roll-up to parents.
 *
 * <p>Depth (Phase C): the {@code field} operand is the descent-plus-leaf path relative to the array column —
 * a bare leaf ({@code 'author'}) at the outer element grain, or a dotted chain of intermediate nested levels
 * ending in the leaf ({@code 'divisions.teams.members.mname'}) for a deep clause. It is split exactly like the
 * parent-grain {@code NestedAnyMatchSerializer}: the leaf is the last segment, the deepest nested path is the
 * top column plus everything before the leaf, and the {@code _nested_path} scope term uses that DEEPEST path
 * so the child docs match the clause's own grain (matching the path the executor recovers from the residual's
 * {@code {"nested"}} heads above the {@code {"lucene": clauseIdx}} node).
 *
 * <p>Expected RexCall shape: {@code NESTED_ANY_MATCH_CHILD($arrayCol, 'field', 'EQUALS', 'value', <clauseIdx>)}
 * — 5 operands. {@code clauseIdx} is not needed to build the Lucene query (it pairs the peer with its JSON
 * node on the planner/executor side) and is ignored here.
 */
public class NestedAnyMatchChildSerializer extends AbstractQuerySerializer {

    @Override
    public QueryBuilder buildQueryBuilder(RexCall call, List<FieldStorageInfo> fieldStorage) {
        List<RexNode> operands = call.getOperands();
        if (operands.size() != 5) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_CHILD expects 5 operands, got " + operands.size());
        }
        if (!(operands.get(0) instanceof RexInputRef arrayColRef)) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_CHILD's 1st operand must be the array column, got " + operands.get(0));
        }
        String fieldName = literalString(operands.get(1), "field");
        String op = literalString(operands.get(2), "op");
        String value = literalString(operands.get(3), "value");
        if (!"EQUALS".equals(op)) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_CHILD only supports op=EQUALS, got " + op);
        }

        FieldStorageInfo arrayField = FieldStorageInfo.resolve(fieldStorage, arrayColRef.getIndex());
        String topPath = arrayField.getFieldName();

        // `fieldName` is the descent-plus-leaf path relative to the array column. Single-level: a bare leaf
        // ("author"). DEEP (Phase C): a dotted chain of intermediate nested levels ending in the leaf
        // ("divisions.teams.members.mname"). The leaf is the last segment; everything before it is the
        // deepest nested level's path. Split exactly as NestedAnyMatchSerializer (the parent-grain peer) does
        // so the two peers agree on paths at any depth.
        int lastDot = fieldName.lastIndexOf('.');
        String deepestPath = lastDot < 0 ? topPath : topPath + "." + fieldName.substring(0, lastDot);
        String leaf = lastDot < 0 ? fieldName : fieldName.substring(lastDot + 1);
        String leafField = deepestPath + "." + leaf;

        // Child-scoped: the term on the child leaf, restricted to children of the DEEPEST nested level via the
        // _nested_path marker. No NestedQueryBuilder/block-join wrap → the scorer yields CHILD docIds at that
        // deepest level, which the executor maps to child-element ordinals. Using the deepest path (not the
        // top col) is what makes a deep clause scope to the right level's child docs — a term on the top col's
        // path would never match a leaf that lives several nested levels down. filter (not must) on
        // _nested_path: it's a pure scope constraint, contributes no score (ScoreMode is irrelevant here —
        // this query is a filter peer).
        return new BoolQueryBuilder()
            .must(new TermQueryBuilder(leafField, value))
            .filter(new TermQueryBuilder(NestedPathFieldMapper.NAME, deepestPath));
    }

    private static String literalString(RexNode node, String operandName) {
        if (!(node instanceof RexLiteral lit)) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_CHILD's '" + operandName + "' operand must be a literal, got " + node);
        }
        String value = lit.getValueAs(String.class);
        if (value == null) {
            throw new IllegalArgumentException("NESTED_ANY_MATCH_CHILD's '" + operandName + "' operand must be a non-null string literal");
        }
        return value;
    }
}

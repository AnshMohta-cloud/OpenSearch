/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.dag;

import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.opensearch.analytics.planner.CapabilityRegistry;
import org.opensearch.analytics.planner.rel.AnnotatedPredicate;
import org.opensearch.analytics.spi.ScalarFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * Computes the <em>pruning relaxation</em> {@code R(phi)} of a filter condition: the strongest
 * predicate the peer backend (Lucene) can evaluate on its own that is still implied by the full
 * condition.
 *
 * <p>The construction is one substitution plus two simplifications:
 * <ul>
 *   <li>every leaf the peer cannot serve is replaced by {@code TRUE};</li>
 *   <li>{@code TRUE} is the <em>identity</em> under AND, so it is dropped from conjunctions —
 *       the surviving conjuncts still filter;</li>
 *   <li>{@code TRUE} is <em>absorbing</em> under OR, so a disjunction containing one collapses
 *       to {@code TRUE} entirely — a row may satisfy the branch the peer knows nothing about.</li>
 * </ul>
 *
 * <p>The resulting invariant is {@code phi => R(phi)}: {@code R} matches a superset of the rows
 * {@code phi} matches, so intersecting the peer's row set into the candidate set can never drop a
 * row that belongs in the answer. A {@code null} result means {@code R(phi) == TRUE} — nothing is
 * prunable and the peer should not be consulted at all.
 *
 * <p><b>NOT.</b> Negation is the one non-monotone connective: complementing a superset yields a
 * <em>subset</em>, which would drop rows that belong in the answer. So a {@code NOT} is relaxed
 * only when its whole operand is peer-servable (no substitution happened inside it); otherwise the
 * entire {@code NOT} node becomes {@code TRUE}. Note this is not a three-valued-logic concern —
 * the substitution and both simplifications hold in Kleene logic as well; it is plain monotonicity.
 * Nulls only make the peer's answer for a negation <em>looser</em> (Lucene's {@code must_not}
 * matches docs missing the field, which SQL's {@code NOT} excludes), never tighter, so it stays a
 * safe superset.
 *
 * @opensearch.internal
 */
final class PruneOnlyRelaxer {

    /**
     * Outcome of relaxing a condition.
     *
     * @param relaxed      the relaxation, or {@code null} when it is {@code TRUE} (nothing prunable)
     * @param backend      peer backend the relaxation targets; {@code null} when {@code relaxed} is null
     * @param annotationId annotation id to key the shipped query on — the id of the first leaf that
     *                     survived into the relaxation, so no new id space is needed
     */
    record Result(RexNode relaxed, String backend, int annotationId) {
        boolean prunable() {
            return relaxed != null;
        }
    }

    private final String operatorBackend;
    private final CapabilityRegistry registry;
    private final RexBuilder rexBuilder;

    /** First peer-servable leaf encountered, used to key the shipped query. */
    private Integer firstAnnotationId;
    private String peerBackend;

    PruneOnlyRelaxer(String operatorBackend, CapabilityRegistry registry, RexBuilder rexBuilder) {
        this.operatorBackend = operatorBackend;
        this.registry = registry;
        this.rexBuilder = rexBuilder;
    }

    Result relax(RexNode condition) {
        RexNode relaxed = relaxNode(condition);
        if (relaxed == null) {
            return new Result(null, null, -1);
        }
        return new Result(relaxed, peerBackend, firstAnnotationId);
    }

    /** Returns the relaxation of {@code node}, or {@code null} meaning {@code TRUE}. */
    private RexNode relaxNode(RexNode node) {
        if (node instanceof AnnotatedPredicate ap) {
            String peer = servingPeer(ap);
            if (peer == null) {
                return null; // peer cannot serve this leaf -> TRUE
            }
            if (firstAnnotationId == null) {
                firstAnnotationId = ap.getAnnotationId();
                peerBackend = peer;
            }
            return ap.unwrap();
        }

        if (node instanceof RexCall call) {
            switch (call.getKind()) {
                case AND: {
                    // TRUE is the identity: drop relaxed-away children, keep the rest.
                    List<RexNode> kept = new ArrayList<>(call.getOperands().size());
                    for (RexNode operand : call.getOperands()) {
                        RexNode child = relaxNode(operand);
                        if (child != null) {
                            kept.add(child);
                        }
                    }
                    if (kept.isEmpty()) {
                        return null;
                    }
                    // composeConjunction (not makeCall) so a nested AND is flattened —
                    // Calcite asserts RexUtil.isFlat on filter conditions.
                    return RexUtil.composeConjunction(rexBuilder, kept);
                }
                case OR: {
                    // TRUE is absorbing: one unservable branch kills the whole disjunction.
                    List<RexNode> kept = new ArrayList<>(call.getOperands().size());
                    for (RexNode operand : call.getOperands()) {
                        RexNode child = relaxNode(operand);
                        if (child == null) {
                            return null;
                        }
                        kept.add(child);
                    }
                    return RexUtil.composeDisjunction(rexBuilder, kept);
                }
                case NOT: {
                    // Only negate an operand the peer can evaluate exactly. Relaxing inside a NOT
                    // would substitute TRUE under a negation, and NOT(TRUE) == FALSE would prune
                    // every row.
                    RexNode operand = call.getOperands().getFirst();
                    if (!fullyServable(operand)) {
                        return null;
                    }
                    RexNode child = relaxNode(operand);
                    if (child == null) {
                        return null;
                    }
                    return rexBuilder.makeCall(SqlStdOperatorTable.NOT, child);
                }
                default:
                    // Any other call shape (comparison over expressions, unrecognised function)
                    // is not a boolean combinator the peer can decompose.
                    return null;
            }
        }
        return null;
    }

    /** True when every leaf beneath {@code node} is peer-servable, and it is built only of AND/OR/NOT. */
    private boolean fullyServable(RexNode node) {
        if (node instanceof AnnotatedPredicate ap) {
            return servingPeer(ap) != null;
        }
        if (node instanceof RexCall call) {
            SqlKind kind = call.getKind();
            if (kind != SqlKind.AND && kind != SqlKind.OR && kind != SqlKind.NOT) {
                return false;
            }
            for (RexNode operand : call.getOperands()) {
                if (!fullyServable(operand)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    /**
     * Returns a non-operator backend that can serialize this leaf, or {@code null}. Mirrors
     * {@link DelegatedPredicateCombiner}'s leaf classification: a leaf is peer-reachable either
     * because only the peer can evaluate it, or because it is dual-viable and was narrowed onto the
     * operator with the peer retained.
     */
    private String servingPeer(AnnotatedPredicate ap) {
        String first = ap.getViableBackends().getFirst();
        if (!first.equals(operatorBackend) && canSerialize(ap, first)) {
            return first;
        }
        for (String peer : ap.getPerformanceDelegationBackends()) {
            if (canSerialize(ap, peer)) {
                return peer;
            }
        }
        return null;
    }

    private boolean canSerialize(AnnotatedPredicate ap, String backend) {
        if (ap.unwrap() instanceof RexCall call) {
            ScalarFunction fn = ScalarFunction.fromSqlOperatorWithFallback(call.getOperator());
            return fn != null && registry.getBackend(backend).delegatedPredicateSerializers().containsKey(fn);
        }
        return false;
    }
}

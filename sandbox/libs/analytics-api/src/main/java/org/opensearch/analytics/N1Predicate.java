/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import java.util.List;

/**
 * [NESTED-POC] A predicate over a nested child's fields, general enough to express any single-level
 * filter the POC demonstrates: comparisons on a struct field ({@code =, !=, <, <=, >, >=}) against a
 * numeric / string / boolean literal, combined with {@code AND} / {@code OR}.
 *
 * <p>The {@code N1SubstraitBuilder} walks this tree and emits the matching Substrait expression
 * (each comparison → a {@code ScalarFunction} whose function name is one of Substrait's canonical
 * names {@code equal/not_equal/lt/lte/gt/gte}; AND/OR → the {@code and}/{@code or} scalar functions),
 * declaring each distinct function once in {@code Plan.extensions}. Field references resolve to the
 * post-unnest top-level column {@code <unnestColumn>.<field>}.
 *
 * <p>Because all comparisons in one predicate sit above a SINGLE unnest of the nested path, an
 * {@code AND} of two comparisons is a SAME-CHILD correlation (both must hold for the same array
 * element) — exactly the semantics N1 requires.
 *
 * <p>This is POC scaffolding standing in for the real customer-query -> N1 rewrite; the rewriter
 * would build an equivalent predicate tree.
 *
 * @opensearch.internal
 */
public sealed interface N1Predicate permits N1Predicate.Comparison, N1Predicate.And, N1Predicate.Or {

    /** Comparison operators, named to match Substrait's canonical scalar-function names. */
    enum Op {
        EQUAL("equal"),
        NOT_EQUAL("not_equal"),
        LT("lt"),
        LTE("lte"),
        GT("gt"),
        GTE("gte");

        private final String substraitName;

        Op(String substraitName) {
            this.substraitName = substraitName;
        }

        /** The Substrait / DataFusion canonical function name for this operator. */
        public String substraitName() {
            return substraitName;
        }
    }

    /**
     * {@code <unnestColumn>.<field> <op> <value>}. {@code field} is the struct field name (e.g.
     * {@code "score"}); {@code value} is an Integer, Long, Double, String, or Boolean literal.
     */
    record Comparison(String field, Op op, Object value) implements N1Predicate {}

    /** All children must hold (same array element — sits above one unnest). */
    record And(List<N1Predicate> children) implements N1Predicate {}

    /** Any child holds. */
    record Or(List<N1Predicate> children) implements N1Predicate {}
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

/**
 * [NESTED-POC] An aggregate applied to the result of a nested-predicate query, e.g.
 * {@code ... | where comments.score > 4 | stats count()} or {@code ... | stats avg(comments.score)}.
 *
 * <p>Two shapes, matching vanilla nested-aggregation semantics:
 * <ul>
 *   <li>{@code COUNT} with {@code argField == null} — count of matching DOCUMENTS (distinct parents),
 *       mirroring {@code reverse_nested} which collapses matching children back to distinct parents.
 *       Built as {@code count(*)} over the distinct-parent-row-id result.</li>
 *   <li>{@code AVG/SUM/MIN/MAX} with {@code argField} = a child struct field — a metric over the
 *       matched CHILD elements (the nested-agg space), built over the unnested+filtered rows.</li>
 * </ul>
 *
 * <p>{@code outputColumn} is the result column name (the PPL convention is the aggregate text, e.g.
 * {@code "count()"} / {@code "avg(comments.score)"}, or an explicit {@code as} alias).
 *
 * @opensearch.internal
 */
public record N1Aggregate(Fn fn, String argField, String outputColumn) {

    /** Aggregate functions, named to match Substrait / DataFusion canonical UDAF names. */
    public enum Fn {
        COUNT("count"),
        AVG("avg"),
        SUM("sum"),
        MIN("min"),
        MAX("max");

        private final String substraitName;

        Fn(String substraitName) {
            this.substraitName = substraitName;
        }

        public String substraitName() {
            return substraitName;
        }
    }

    /** True for {@code count()} (no argument) — counts distinct matching parent documents. */
    public boolean isCountStar() {
        return fn == Fn.COUNT && argField == null;
    }
}

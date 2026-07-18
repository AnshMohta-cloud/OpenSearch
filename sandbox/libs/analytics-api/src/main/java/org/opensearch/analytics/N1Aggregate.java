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
 * [NESTED-POC] An aggregate applied to the result of a nested-predicate query, e.g.
 * {@code ... | where comments.score > 4 | stats count()}, {@code ... | stats avg(comments.score)},
 * or grouped {@code ... | stats avg(products.price) by products.sku}.
 *
 * <p>Shapes, matching vanilla nested-aggregation semantics:
 * <ul>
 *   <li>{@code COUNT} with {@code argField == null}, no group — count of matching DOCUMENTS (distinct
 *       parents), mirroring {@code reverse_nested}. Built as {@code count(*)} over the distinct-parent
 *       result.</li>
 *   <li>{@code AVG/SUM/MIN/MAX} with {@code argField}, no group — a metric over the matched CHILD
 *       elements (the nested-agg space), built over the unnested+filtered rows (one output row).</li>
 *   <li>Any {@code fn} with {@code groupByField} set — a metric/count PER GROUP of a child dimension
 *       (the {@code nested -> terms -> metric} agg): unnest → [filter] → Aggregate(group by
 *       {@code <deepest>.<groupByField>}, measure {@code fn(argField)}). Output = one row per distinct
 *       group value, columns {@code [groupByOutputColumn, outputColumn]}. Grouped {@code count()}
 *       counts child elements per group.</li>
 * </ul>
 *
 * <p>{@code having*} (optional) filters the grouped aggregate result — the plan wraps the Aggregate in
 * a Filter comparing the measure output ({@code ... | stats count() by X}, keep groups where the
 * measure {@code havingOp havingValue}). Only meaningful with a group-by.
 *
 * <p>{@code outputColumn} is the metric result column name (PPL convention: the aggregate text, e.g.
 * {@code "avg(products.price)"}); {@code groupByOutputColumn} names the group-key column.
 *
 * @opensearch.internal
 */
public record N1Aggregate(
    Fn fn,
    String argField,
    String outputColumn,
    String groupByField,
    String groupByOutputColumn,
    N1Predicate.Op havingOp,
    Object havingValue
) {

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

    /** Convenience: global aggregate (no group-by, no having) — the original 3-arg form. */
    public N1Aggregate(Fn fn, String argField, String outputColumn) {
        this(fn, argField, outputColumn, null, null, null, null);
    }

    /** Convenience: grouped aggregate with no HAVING. */
    public N1Aggregate(Fn fn, String argField, String outputColumn, String groupByField, String groupByOutputColumn) {
        this(fn, argField, outputColumn, groupByField, groupByOutputColumn, null, null);
    }

    /** True for {@code count()} (no argument) — counts distinct matching parent documents (ungrouped). */
    public boolean isCountStar() {
        return fn == Fn.COUNT && argField == null && groupByField == null;
    }

    /** True when this aggregate groups by a child dimension. */
    public boolean hasGroupBy() {
        return groupByField != null;
    }

    /** True when a HAVING filter on the grouped measure is present. */
    public boolean hasHaving() {
        return havingOp != null;
    }

    /**
     * The ordered result column names the plan produces: {@code [groupKey, measure]} when grouped,
     * else just {@code [measure]}. Used by the front-end + executor to label/order result batches.
     */
    public List<String> outputColumns() {
        return groupByField != null ? List.of(groupByOutputColumn, outputColumn) : List.of(outputColumn);
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.ppl.action;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.N1Descriptor;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * [NESTED-POC] Hand-authored N1-rewritten plans, keyed by the EXACT PPL query string.
 *
 * <p><b>Why this exists.</b> Under N1, a predicate/aggregation on a nested field must be rewritten
 * into {@code Scan -> UNNEST(nested) -> Filter/Aggregate -> distinct(parent_row_id)} before it can
 * execute. That customer-query -> N1-rewrite conversion is owned by a different workstream and does
 * not exist yet — {@code where comments.score > 4} currently dies inside {@code UnifiedQueryPlanner.plan()}
 * with "Unsupported conversion for Relational Data type: ROW".
 *
 * <p><b>What this does.</b> For each query we want to demo, a human writes the {@link N1Descriptor}
 * it SHOULD rewrite to and registers it here under the verbatim query string. {@link UnifiedQueryService}
 * looks the incoming query up; a hit produces the descriptor (bound to the concrete index + base scan
 * row type), which the DataFusion convertor turns into a hand-assembled Substrait plan. This is a pure
 * lookup — NOT a rewriter: no query parsing/generalisation, just fixed descriptors a human authored,
 * doubling as a concrete spec ("customer query X must become this N1 plan") for the real rewrite work.
 *
 * <p>Grep the server log for {@code NESTED-POC} to see when a registered descriptor is used.
 */
final class N1PlanRegistry {

    private static final Logger logger = LogManager.getLogger(N1PlanRegistry.class);

    /**
     * Produces the {@link N1Descriptor} for a matched query, given the concrete index name and the
     * row type of the plain {@code source=<index>} scan (used to emit the ReadRel base schema and to
     * resolve column/struct-field positions).
     */
    @FunctionalInterface
    interface DescriptorFactory {
        N1Descriptor create(String indexName, RelDataType baseRowType);
    }

    /** query string (verbatim, trimmed) -> descriptor factory. LinkedHashMap for stable logging. */
    private static final Map<String, DescriptorFactory> QUERIES = new LinkedHashMap<>();

    static {
        // Parquet-only nested predicate, returning the DISTINCT PARENT ROW-IDS (the original demo).
        // `comments.score > 4` -> unnest comments, keep elements score>4, semi-join back, project row-id.
        // Expected on poc_nested: {Post X (score 5), Post Y (score 9)}; Post Z (score 1) excluded.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4",
            (indexName, rowType) ->
                new N1Descriptor(indexName, "comments", "score", 4, "__row_id__", java.util.List.of("__row_id__"), rowType)
        );
        // Same predicate, but return actual PARENT FIELDS (title, views) — proves output generality:
        // the semi-join back recovers the intact parent columns after the unnest exploded the array.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 | fields title, views",
            (indexName, rowType) ->
                new N1Descriptor(indexName, "comments", "score", 4, "__row_id__", java.util.List.of("title", "views"), rowType)
        );
        // Same predicate, SELECT * — return all intact parent columns incl. the whole comments array.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 | fields *",
            (indexName, rowType) ->
                new N1Descriptor(indexName, "comments", "score", 4, "__row_id__", java.util.List.of(), rowType)
        );
    }

    private N1PlanRegistry() {}

    /** True if we have a hand-authored N1 descriptor for this exact query. */
    static boolean has(String pplText) {
        return QUERIES.containsKey(normalize(pplText));
    }

    /**
     * Builds the hand-authored {@link N1Descriptor} for {@code pplText}. Returns {@code null} if no
     * descriptor is registered for this query.
     */
    static N1Descriptor describe(String pplText, String indexName, RelDataType baseRowType) {
        DescriptorFactory factory = QUERIES.get(normalize(pplText));
        if (factory == null) {
            return null;
        }
        logger.info("[NESTED-POC] using HAND-AUTHORED N1 descriptor for query [{}] (POC stand-in for the customer-query -> N1 rewrite)", pplText);
        return factory.create(indexName, baseRowType);
    }

    private static String normalize(String pplText) {
        return pplText == null ? "" : pplText.trim();
    }
}

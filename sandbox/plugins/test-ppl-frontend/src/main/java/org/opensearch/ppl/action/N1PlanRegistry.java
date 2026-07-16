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
import org.opensearch.analytics.N1Predicate;

import java.util.LinkedHashMap;
import java.util.List;
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

    /** Single-level unnest path over the `comments` nested column. */
    private static final List<String> COMMENTS = List.of("comments");

    static {
        // ---- Output-generality demos (predicate = comments.score > 4) ----
        // Row-ids: Expected {Post X (score 5), Post Y (score 9)}; Post Z (score 1) excluded.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4",
            (indexName, rowType) -> new N1Descriptor(indexName, COMMENTS, scoreGt(4), "__row_id__", List.of("__row_id__"), rowType)
        );
        // Actual parent fields — semi-join back recovers intact parent columns.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 | fields title, views",
            (indexName, rowType) -> new N1Descriptor(indexName, COMMENTS, scoreGt(4), "__row_id__", List.of("title", "views"), rowType)
        );
        // SELECT * — all intact parent columns incl. the whole comments array.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 | fields *",
            (indexName, rowType) -> new N1Descriptor(indexName, COMMENTS, scoreGt(4), "__row_id__", List.of(), rowType)
        );

        // ---- Predicate-generality demos ----
        // Keyword equality: comments.author = 'alice' -> only Post X has an alice comment.
        QUERIES.put(
            "source=poc_nested | where comments.author = 'alice' | fields title",
            (indexName, rowType) -> new N1Descriptor(
                indexName, COMMENTS, cmp("author", N1Predicate.Op.EQUAL, "alice"), "__row_id__", List.of("title"), rowType
            )
        );
        // Other numeric operators: comments.score <= 2 -> Post Y (bob=2) + Post Z (dave=1); Post X (5) excluded.
        QUERIES.put(
            "source=poc_nested | where comments.score <= 2 | fields title",
            (indexName, rowType) -> new N1Descriptor(
                indexName, COMMENTS, cmp("score", N1Predicate.Op.LTE, 2), "__row_id__", List.of("title"), rowType
            )
        );
        // SAME-CHILD AND (the correlation case): comments.score > 4 AND comments.author = 'carol'.
        // Post Y's carol comment has score 9 AND author carol -> MATCH. Post X (alice,5) fails author;
        // this proves both conditions bind to the SAME array element (one unnest). Expected {Post Y}.
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 and comments.author = 'carol' | fields title",
            (indexName, rowType) -> new N1Descriptor(
                indexName,
                COMMENTS,
                new N1Predicate.And(List.of(scoreGt(4), cmp("author", N1Predicate.Op.EQUAL, "carol"))),
                "__row_id__",
                List.of("title"),
                rowType
            )
        );
        // OR across children of the same nested path: comments.author='alice' OR comments.author='dave'
        // -> Post X (alice) + Post Z (dave). Expected {Post X, Post Z}.
        QUERIES.put(
            "source=poc_nested | where comments.author = 'alice' or comments.author = 'dave' | fields title",
            (indexName, rowType) -> new N1Descriptor(
                indexName,
                COMMENTS,
                new N1Predicate.Or(List.of(cmp("author", N1Predicate.Op.EQUAL, "alice"), cmp("author", N1Predicate.Op.EQUAL, "dave"))),
                "__row_id__",
                List.of("title"),
                rowType
            )
        );

        // ---- Depth-generality demo (3-level nesting, index poc_deep) ----
        // comments.replies.reactions.by = 'zoe' -> unnest the 3-level path, filter the deepest leaf,
        // semi-join back to the parent post. Path = [comments, comments.replies, comments.replies.reactions];
        // predicate field 'by' is a leaf of the deepest level. Returns the matching parent (title).
        QUERIES.put(
            "source=poc_deep | where comments.replies.reactions.by = 'zoe' | fields title",
            (indexName, rowType) -> new N1Descriptor(
                indexName,
                List.of("comments", "comments.replies", "comments.replies.reactions"),
                cmp("by", N1Predicate.Op.EQUAL, "zoe"),
                "__row_id__",
                List.of("title"),
                rowType
            )
        );

        // ---- 5-LEVEL depth demo (index poc_deep5, multiple sibling nested fields) ----
        // departments.teams.members.tasks.tags.label = 'urgent' -> unnest the 5-level path, filter the
        // deepest leaf, semi-join back to the parent org. Only Acme has an 'urgent' tag. Expected {Acme}.
        // Path has 5 nested levels; the sibling nested fields (departments.offices, reviewers) exercise
        // the post-unnest layout simulation (in-place struct expansion must not misplace siblings).
        QUERIES.put(
            "source=poc_deep5 | where departments.teams.members.tasks.tags.label = 'urgent' | fields org",
            (indexName, rowType) -> new N1Descriptor(
                indexName,
                List.of(
                    "departments",
                    "departments.teams",
                    "departments.teams.members",
                    "departments.teams.members.tasks",
                    "departments.teams.members.tasks.tags"
                ),
                cmp("label", N1Predicate.Op.EQUAL, "urgent"),
                "__row_id__",
                List.of("org"),
                rowType
            )
        );
    }

    /** Convenience: `comments.score > n`. */
    private static N1Predicate scoreGt(int n) {
        return cmp("score", N1Predicate.Op.GT, n);
    }

    /** Convenience: a single comparison {@code field <op> value}. */
    private static N1Predicate cmp(String field, N1Predicate.Op op, Object value) {
        return new N1Predicate.Comparison(field, op, value);
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

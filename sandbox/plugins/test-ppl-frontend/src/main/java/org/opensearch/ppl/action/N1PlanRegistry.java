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
import org.opensearch.analytics.N1Aggregate;
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

        // ---- Aggregate output demo (#27): count of matching DOCUMENTS over a nested predicate ----
        // `where comments.score > 4 | stats count()` -> count distinct matching parents (mirrors
        // vanilla reverse_nested). Post X + Post Y match -> count = 2. Output column "count()".
        QUERIES.put(
            "source=poc_nested | where comments.score > 4 | stats count()",
            (indexName, rowType) -> new N1Descriptor(
                indexName,
                COMMENTS,
                scoreGt(4),
                "__row_id__",
                List.of("__row_id__"),
                new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"),
                rowType
            )
        );

        // ---- 7-LEVEL depth demo (index poc_deep7): digging at the leaf (L7), L6, L5, L4. Extends
        // the verified 5-level poc_deep5 by two levels — SAME unnest-per-level + semi-join path, deeper.
        // Generated by /tmp/poc-nested/gen_deep7.py (kept in sync with cases_deep7.tsv + its computed
        // oracle). Path: regions->divisions->departments->teams->members->tasks->tags; projects `company`.
        // d7_leaf_urgent (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'urgent' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "urgent"), "__row_id__", List.of("company"), rowType));
        // d7_leaf_chore (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'chore' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "chore"), "__row_id__", List.of("company"), rowType));
        // d7_leaf_prio_gt3 (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.priority > 3 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("priority", N1Predicate.Op.GT, 3), "__row_id__", List.of("company"), rowType));
        // d7_leaf_prio_eq2 (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.priority = 2 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("priority", N1Predicate.Op.EQUAL, 2), "__row_id__", List.of("company"), rowType));
        // d7_leaf_and (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'urgent' and regions.divisions.departments.teams.members.tasks.tags.priority = 1 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), new N1Predicate.And(List.of(cmp("label", N1Predicate.Op.EQUAL, "urgent"), cmp("priority", N1Predicate.Op.EQUAL, 1))), "__row_id__", List.of("company"), rowType));
        // d7_leaf_and_empty (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'urgent' and regions.divisions.departments.teams.members.tasks.tags.priority = 2 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), new N1Predicate.And(List.of(cmp("label", N1Predicate.Op.EQUAL, "urgent"), cmp("priority", N1Predicate.Op.EQUAL, 2))), "__row_id__", List.of("company"), rowType));
        // d7_l6_migrate (level 6)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.task = 'migrate' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks"), cmp("task", N1Predicate.Op.EQUAL, "migrate"), "__row_id__", List.of("company"), rowType));
        // d7_l6_study (level 6)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.task = 'study' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks"), cmp("task", N1Predicate.Op.EQUAL, "study"), "__row_id__", List.of("company"), rowType));
        // d7_l5_alice (level 5)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.name = 'alice' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members"), cmp("name", N1Predicate.Op.EQUAL, "alice"), "__row_id__", List.of("company"), rowType));
        // d7_l5_dave (level 5)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.name = 'dave' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members"), cmp("name", N1Predicate.Op.EQUAL, "dave"), "__row_id__", List.of("company"), rowType));
        // d7_l4_platform (level 4)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.team = 'Platform' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams"), cmp("team", N1Predicate.Op.EQUAL, "Platform"), "__row_id__", List.of("company"), rowType));
        // d7_l4_west (level 4)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.team = 'West' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams"), cmp("team", N1Predicate.Op.EQUAL, "West"), "__row_id__", List.of("company"), rowType));
        // d7_count_chore (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'chore' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "chore"), "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));

        // ===== catalog differential-test cases (69, generated by /tmp/poc-nested/gen_cases.py; kept in
        // sync with cases.tsv + its computed oracle). Filters (all operators, depth 1-3, AND/OR),
        // count(), and metric aggregates avg/sum/min/max over a child field (the second nested-agg
        // shape: metric over unnested+filtered child rows, no semi-join). =====
        // p_price_eq
        QUERIES.put("source=catalog | where products.price = 100 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.EQUAL, 100), "__row_id__", List.of("name"), rowType));
        // p_price_gt
        QUERIES.put("source=catalog | where products.price > 200 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.GT, 200), "__row_id__", List.of("name"), rowType));
        // p_price_lte
        QUERIES.put("source=catalog | where products.price <= 50 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.LTE, 50), "__row_id__", List.of("name"), rowType));
        // p_price_ne
        QUERIES.put("source=catalog | where products.price != 100 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.NOT_EQUAL, 100), "__row_id__", List.of("name"), rowType));
        // p_rating_gt
        QUERIES.put("source=catalog | where products.rating > 4.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.GT, 4.0), "__row_id__", List.of("name"), rowType));
        // p_rating_gte
        QUERIES.put("source=catalog | where products.rating >= 4.8 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.GTE, 4.8), "__row_id__", List.of("name"), rowType));
        // p_instock_t
        QUERIES.put("source=catalog | where products.in_stock = true | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.EQUAL, true), "__row_id__", List.of("name"), rowType));
        // p_instock_f
        QUERIES.put("source=catalog | where products.in_stock = false | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.EQUAL, false), "__row_id__", List.of("name"), rowType));
        // p_sku_eq
        QUERIES.put("source=catalog | where products.sku = 'A1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "A1"), "__row_id__", List.of("name"), rowType));
        // v_color_red
        QUERIES.put("source=catalog | where products.variants.color = 'red' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "red"), "__row_id__", List.of("name"), rowType));
        // v_qty_gt
        QUERIES.put("source=catalog | where products.variants.qty > 10 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.GT, 10), "__row_id__", List.of("name"), rowType));
        // v_qty_zero
        QUERIES.put("source=catalog | where products.variants.qty = 0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.EQUAL, 0), "__row_id__", List.of("name"), rowType));
        // s_val_arm
        QUERIES.put("source=catalog | where products.variants.specs.value = 'arm' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.EQUAL, "arm"), "__row_id__", List.of("name"), rowType));
        // s_key_gpu
        QUERIES.put("source=catalog | where products.variants.specs.key = 'gpu' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "gpu"), "__row_id__", List.of("name"), rowType));
        // s_weight_gte
        QUERIES.put("source=catalog | where products.variants.specs.weight >= 20 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.GTE, 20), "__row_id__", List.of("name"), rowType));
        // t_label_sale
        QUERIES.put("source=catalog | where products.tags.label = 'sale' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "sale"), "__row_id__", List.of("name"), rowType));
        // r_stars_5
        QUERIES.put("source=catalog | where reviewers.stars = 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.EQUAL, 5), "__row_id__", List.of("name"), rowType));
        // r_stars_lt
        QUERIES.put("source=catalog | where reviewers.stars < 3 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.LT, 3), "__row_id__", List.of("name"), rowType));
        // p_and
        QUERIES.put("source=catalog | where products.price = 100 and products.rating = 5.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.EQUAL, 100), cmp("rating", N1Predicate.Op.EQUAL, 5.0))), "__row_id__", List.of("name"), rowType));
        // v_and
        QUERIES.put("source=catalog | where products.variants.color = 'red' and products.variants.qty > 4 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.And(List.of(cmp("color", N1Predicate.Op.EQUAL, "red"), cmp("qty", N1Predicate.Op.GT, 4))), "__row_id__", List.of("name"), rowType));
        // v_or
        QUERIES.put("source=catalog | where products.variants.color = 'gold' or products.variants.color = 'silver' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.Or(List.of(cmp("color", N1Predicate.Op.EQUAL, "gold"), cmp("color", N1Predicate.Op.EQUAL, "silver"))), "__row_id__", List.of("name"), rowType));
        // cnt_rating
        QUERIES.put("source=catalog | where products.rating > 4.0 | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.GT, 4.0), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // p_price_lt
        QUERIES.put("source=catalog | where products.price < 100 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.LT, 100), "__row_id__", List.of("name"), rowType));
        // p_price_gte
        QUERIES.put("source=catalog | where products.price >= 300 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.GTE, 300), "__row_id__", List.of("name"), rowType));
        // p_rating_lt
        QUERIES.put("source=catalog | where products.rating < 3.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.LT, 3.0), "__row_id__", List.of("name"), rowType));
        // p_rating_lte
        QUERIES.put("source=catalog | where products.rating <= 3.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.LTE, 3.0), "__row_id__", List.of("name"), rowType));
        // p_sku_ne
        QUERIES.put("source=catalog | where products.sku != 'A1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.NOT_EQUAL, "A1"), "__row_id__", List.of("name"), rowType));
        // p_sku_i1
        QUERIES.put("source=catalog | where products.sku = 'I1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "I1"), "__row_id__", List.of("name"), rowType));
        // p_and_instock
        QUERIES.put("source=catalog | where products.price = 100 and products.in_stock = true | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.EQUAL, 100), cmp("in_stock", N1Predicate.Op.EQUAL, true))), "__row_id__", List.of("name"), rowType));
        // p_or_sku
        QUERIES.put("source=catalog | where products.sku = 'G1' or products.sku = 'S1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.Or(List.of(cmp("sku", N1Predicate.Op.EQUAL, "G1"), cmp("sku", N1Predicate.Op.EQUAL, "S1"))), "__row_id__", List.of("name"), rowType));
        // p_and_empty
        QUERIES.put("source=catalog | where products.price = 800 and products.rating = 5.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.EQUAL, 800), cmp("rating", N1Predicate.Op.EQUAL, 5.0))), "__row_id__", List.of("name"), rowType));
        // v_color_blue
        QUERIES.put("source=catalog | where products.variants.color = 'blue' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "blue"), "__row_id__", List.of("name"), rowType));
        // v_qty_lt
        QUERIES.put("source=catalog | where products.variants.qty < 3 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.LT, 3), "__row_id__", List.of("name"), rowType));
        // v_qty_gte
        QUERIES.put("source=catalog | where products.variants.qty >= 7 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.GTE, 7), "__row_id__", List.of("name"), rowType));
        // v_qty_ne
        QUERIES.put("source=catalog | where products.variants.qty != 0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.NOT_EQUAL, 0), "__row_id__", List.of("name"), rowType));
        // v_color_ne_red
        QUERIES.put("source=catalog | where products.variants.color != 'red' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.NOT_EQUAL, "red"), "__row_id__", List.of("name"), rowType));
        // v_and_blue0
        QUERIES.put("source=catalog | where products.variants.color = 'blue' and products.variants.qty = 0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.And(List.of(cmp("color", N1Predicate.Op.EQUAL, "blue"), cmp("qty", N1Predicate.Op.EQUAL, 0))), "__row_id__", List.of("name"), rowType));
        // v_or_bw
        QUERIES.put("source=catalog | where products.variants.color = 'black' or products.variants.color = 'white' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.Or(List.of(cmp("color", N1Predicate.Op.EQUAL, "black"), cmp("color", N1Predicate.Op.EQUAL, "white"))), "__row_id__", List.of("name"), rowType));
        // s_key_cpu
        QUERIES.put("source=catalog | where products.variants.specs.key = 'cpu' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "cpu"), "__row_id__", List.of("name"), rowType));
        // s_key_ram
        QUERIES.put("source=catalog | where products.variants.specs.key = 'ram' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "ram"), "__row_id__", List.of("name"), rowType));
        // s_val_x86
        QUERIES.put("source=catalog | where products.variants.specs.value = 'x86' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.EQUAL, "x86"), "__row_id__", List.of("name"), rowType));
        // s_weight_lt
        QUERIES.put("source=catalog | where products.variants.specs.weight < 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.LT, 5), "__row_id__", List.of("name"), rowType));
        // s_weight_gt
        QUERIES.put("source=catalog | where products.variants.specs.weight > 10 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.GT, 10), "__row_id__", List.of("name"), rowType));
        // s_and_cpuarm
        QUERIES.put("source=catalog | where products.variants.specs.key = 'cpu' and products.variants.specs.value = 'arm' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("key", N1Predicate.Op.EQUAL, "cpu"), cmp("value", N1Predicate.Op.EQUAL, "arm"))), "__row_id__", List.of("name"), rowType));
        // s_and_ramw
        QUERIES.put("source=catalog | where products.variants.specs.key = 'ram' and products.variants.specs.weight >= 6 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("key", N1Predicate.Op.EQUAL, "ram"), cmp("weight", N1Predicate.Op.GTE, 6))), "__row_id__", List.of("name"), rowType));
        // t_label_new
        QUERIES.put("source=catalog | where products.tags.label = 'new' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "new"), "__row_id__", List.of("name"), rowType));
        // t_label_premium
        QUERIES.put("source=catalog | where products.tags.label = 'premium' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "premium"), "__row_id__", List.of("name"), rowType));
        // t_label_clearance
        QUERIES.put("source=catalog | where products.tags.label = 'clearance' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "clearance"), "__row_id__", List.of("name"), rowType));
        // t_label_ne_new
        QUERIES.put("source=catalog | where products.tags.label != 'new' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.NOT_EQUAL, "new"), "__row_id__", List.of("name"), rowType));
        // r_stars_gte4
        QUERIES.put("source=catalog | where reviewers.stars >= 4 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.GTE, 4), "__row_id__", List.of("name"), rowType));
        // r_stars_ne5
        QUERIES.put("source=catalog | where reviewers.stars != 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.NOT_EQUAL, 5), "__row_id__", List.of("name"), rowType));
        // r_user_alice
        QUERIES.put("source=catalog | where reviewers.user = 'alice' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("user", N1Predicate.Op.EQUAL, "alice"), "__row_id__", List.of("name"), rowType));
        // r_and
        QUERIES.put("source=catalog | where reviewers.user = 'grace' and reviewers.stars = 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), new N1Predicate.And(List.of(cmp("user", N1Predicate.Op.EQUAL, "grace"), cmp("stars", N1Predicate.Op.EQUAL, 5))), "__row_id__", List.of("name"), rowType));
        // r_or
        QUERIES.put("source=catalog | where reviewers.user = 'alice' or reviewers.user = 'dave' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), new N1Predicate.Or(List.of(cmp("user", N1Predicate.Op.EQUAL, "alice"), cmp("user", N1Predicate.Op.EQUAL, "dave"))), "__row_id__", List.of("name"), rowType));
        // cnt_price_ne100
        QUERIES.put("source=catalog | where products.price != 100 | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.NOT_EQUAL, 100), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // cnt_tag_new
        QUERIES.put("source=catalog | where products.tags.label = 'new' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "new"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // cnt_spec_cpu
        QUERIES.put("source=catalog | where products.variants.specs.key = 'cpu' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "cpu"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // m_avg_price
        QUERIES.put("source=catalog | stats avg(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "price", "avg(products.price)"), rowType));
        // m_sum_price
        QUERIES.put("source=catalog | stats sum(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "price", "sum(products.price)"), rowType));
        // m_min_price
        QUERIES.put("source=catalog | stats min(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "price", "min(products.price)"), rowType));
        // m_max_price
        QUERIES.put("source=catalog | stats max(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "price", "max(products.price)"), rowType));
        // m_avg_rating
        QUERIES.put("source=catalog | stats avg(products.rating)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "rating", "avg(products.rating)"), rowType));
        // m_max_rating
        QUERIES.put("source=catalog | stats max(products.rating)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "rating", "max(products.rating)"), rowType));
        // m_sum_qty
        QUERIES.put("source=catalog | stats sum(products.variants.qty)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "qty", "sum(products.variants.qty)"), rowType));
        // m_max_qty
        QUERIES.put("source=catalog | stats max(products.variants.qty)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "qty", "max(products.variants.qty)"), rowType));
        // m_avg_weight
        QUERIES.put("source=catalog | stats avg(products.variants.specs.weight)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "weight", "avg(products.variants.specs.weight)"), rowType));
        // m_min_weight
        QUERIES.put("source=catalog | stats min(products.variants.specs.weight)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "weight", "min(products.variants.specs.weight)"), rowType));
        // m_avg_stars
        QUERIES.put("source=catalog | stats avg(reviewers.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "stars", "avg(reviewers.stars)"), rowType));
        // m_avg_price_instock
        QUERIES.put("source=catalog | where products.in_stock = true | stats avg(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.EQUAL, true), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "price", "avg(products.price)"), rowType));
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

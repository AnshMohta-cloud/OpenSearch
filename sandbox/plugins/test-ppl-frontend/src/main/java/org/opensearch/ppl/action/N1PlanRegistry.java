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
        // b7_l2_division_cloud (level 2)
        QUERIES.put("source=poc_deep7 | where regions.divisions.division = 'Cloud' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions"), cmp("division", N1Predicate.Op.EQUAL, "Cloud"), "__row_id__", List.of("company"), rowType));
        // b7_l2_division_sales (level 2)
        QUERIES.put("source=poc_deep7 | where regions.divisions.division = 'Sales' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions"), cmp("division", N1Predicate.Op.EQUAL, "Sales"), "__row_id__", List.of("company"), rowType));
        // b7_l2_division_ne (level 2)
        QUERIES.put("source=poc_deep7 | where regions.divisions.division != 'Cloud' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions"), cmp("division", N1Predicate.Op.NOT_EQUAL, "Cloud"), "__row_id__", List.of("company"), rowType));
        // b7_l3_dept_eng (level 3)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.dept = 'Eng' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments"), cmp("dept", N1Predicate.Op.EQUAL, "Eng"), "__row_id__", List.of("company"), rowType));
        // b7_l3_dept_labs (level 3)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.dept = 'Labs' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments"), cmp("dept", N1Predicate.Op.EQUAL, "Labs"), "__row_id__", List.of("company"), rowType));
        // b7_l3_dept_none (level 3)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.dept = 'HR' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments"), cmp("dept", N1Predicate.Op.EQUAL, "HR"), "__row_id__", List.of("company"), rowType));
        // b7_l1_region_na (level 1)
        QUERIES.put("source=poc_deep7 | where regions.region = 'NA' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions"), cmp("region", N1Predicate.Op.EQUAL, "NA"), "__row_id__", List.of("company"), rowType));
        // b7_l1_region_ne (level 1)
        QUERIES.put("source=poc_deep7 | where regions.region != 'NA' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions"), cmp("region", N1Predicate.Op.NOT_EQUAL, "NA"), "__row_id__", List.of("company"), rowType));
        // b7_l4_team_data (level 4)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.team = 'Data' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams"), cmp("team", N1Predicate.Op.EQUAL, "Data"), "__row_id__", List.of("company"), rowType));
        // b7_l4_team_ne (level 4)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.team != 'Platform' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams"), cmp("team", N1Predicate.Op.NOT_EQUAL, "Platform"), "__row_id__", List.of("company"), rowType));
        // b7_l5_carol (level 5)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.name = 'carol' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members"), cmp("name", N1Predicate.Op.EQUAL, "carol"), "__row_id__", List.of("company"), rowType));
        // b7_l5_name_ne (level 5)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.name != 'alice' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members"), cmp("name", N1Predicate.Op.NOT_EQUAL, "alice"), "__row_id__", List.of("company"), rowType));
        // b7_l6_etl (level 6)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.task = 'etl' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks"), cmp("task", N1Predicate.Op.EQUAL, "etl"), "__row_id__", List.of("company"), rowType));
        // b7_l6_task_ne (level 6)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.task != 'study' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks"), cmp("task", N1Predicate.Op.NOT_EQUAL, "study"), "__row_id__", List.of("company"), rowType));
        // b7_leaf_backend (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'backend' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "backend"), "__row_id__", List.of("company"), rowType));
        // b7_leaf_client (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'client' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "client"), "__row_id__", List.of("company"), rowType));
        // b7_leaf_prio_lte2 (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.priority <= 2 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("priority", N1Predicate.Op.LTE, 2), "__row_id__", List.of("company"), rowType));
        // b7_leaf_prio_ne3 (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.priority != 3 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("priority", N1Predicate.Op.NOT_EQUAL, 3), "__row_id__", List.of("company"), rowType));
        // b7_leaf_or (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'urgent' or regions.divisions.departments.teams.members.tasks.tags.label = 'client' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), new N1Predicate.Or(List.of(cmp("label", N1Predicate.Op.EQUAL, "urgent"), cmp("label", N1Predicate.Op.EQUAL, "client"))), "__row_id__", List.of("company"), rowType));
        // b7_office_seattle (level 2)
        QUERIES.put("source=poc_deep7 | where regions.offices.city = 'Seattle' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.offices"), cmp("city", N1Predicate.Op.EQUAL, "Seattle"), "__row_id__", List.of("company"), rowType));
        // b7_office_floor_gt5 (level 2)
        QUERIES.put("source=poc_deep7 | where regions.offices.floor > 5 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.offices"), cmp("floor", N1Predicate.Op.GT, 5), "__row_id__", List.of("company"), rowType));
        // b7_office_floor_lte3 (level 2)
        QUERIES.put("source=poc_deep7 | where regions.offices.floor <= 3 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.offices"), cmp("floor", N1Predicate.Op.LTE, 3), "__row_id__", List.of("company"), rowType));
        // b7_aud_stars5 (level 1)
        QUERIES.put("source=poc_deep7 | where auditors.stars = 5 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("auditors"), cmp("stars", N1Predicate.Op.EQUAL, 5), "__row_id__", List.of("company"), rowType));
        // b7_aud_stars_lt3 (level 1)
        QUERIES.put("source=poc_deep7 | where auditors.stars < 3 | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("auditors"), cmp("stars", N1Predicate.Op.LT, 3), "__row_id__", List.of("company"), rowType));
        // b7_aud_who_bill (level 1)
        QUERIES.put("source=poc_deep7 | where auditors.who = 'bill' | fields company",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("auditors"), cmp("who", N1Predicate.Op.EQUAL, "bill"), "__row_id__", List.of("company"), rowType));
        // b7_cnt_platform (level 4)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.team = 'Platform' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams"), cmp("team", N1Predicate.Op.EQUAL, "Platform"), "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b7_cnt_backend (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'backend' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "backend"), "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b7_cnt_none (level 3)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.dept = 'HR' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments"), cmp("dept", N1Predicate.Op.EQUAL, "HR"), "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b7_m_avg_prio (level 7)
        QUERIES.put("source=poc_deep7 | stats avg(regions.divisions.departments.teams.members.tasks.tags.priority)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.AVG, "priority", "avg(regions.divisions.departments.teams.members.tasks.tags.priority)"), rowType));
        // b7_m_max_prio (level 7)
        QUERIES.put("source=poc_deep7 | stats max(regions.divisions.departments.teams.members.tasks.tags.priority)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.MAX, "priority", "max(regions.divisions.departments.teams.members.tasks.tags.priority)"), rowType));
        // b7_m_min_prio (level 7)
        QUERIES.put("source=poc_deep7 | stats min(regions.divisions.departments.teams.members.tasks.tags.priority)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.MIN, "priority", "min(regions.divisions.departments.teams.members.tasks.tags.priority)"), rowType));
        // b7_m_sum_prio (level 7)
        QUERIES.put("source=poc_deep7 | stats sum(regions.divisions.departments.teams.members.tasks.tags.priority)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.SUM, "priority", "sum(regions.divisions.departments.teams.members.tasks.tags.priority)"), rowType));
        // b7_m_sum_floor (level 2)
        QUERIES.put("source=poc_deep7 | stats sum(regions.offices.floor)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.offices"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.SUM, "floor", "sum(regions.offices.floor)"), rowType));
        // b7_m_avg_aud (level 1)
        QUERIES.put("source=poc_deep7 | stats avg(auditors.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("auditors"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.AVG, "stars", "avg(auditors.stars)"), rowType));
        // b7_m_avg_prio_urgent (level 7)
        QUERIES.put("source=poc_deep7 | where regions.divisions.departments.teams.members.tasks.tags.label = 'backend' | stats avg(regions.divisions.departments.teams.members.tasks.tags.priority)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), cmp("label", N1Predicate.Op.EQUAL, "backend"), "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.AVG, "priority", "avg(regions.divisions.departments.teams.members.tasks.tags.priority)"), rowType));
        // b7_m_max_label (level 7)
        QUERIES.put("source=poc_deep7 | stats max(regions.divisions.departments.teams.members.tasks.tags.label)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("regions", "regions.divisions", "regions.divisions.departments", "regions.divisions.departments.teams", "regions.divisions.departments.teams.members", "regions.divisions.departments.teams.members.tasks", "regions.divisions.departments.teams.members.tasks.tags"), null, "__row_id__", List.of("company"), new N1Aggregate(N1Aggregate.Fn.MAX, "label", "max(regions.divisions.departments.teams.members.tasks.tags.label)"), rowType));

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
        // b3_p_price_eq300
        QUERIES.put("source=catalog | where products.price = 300 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.EQUAL, 300), "__row_id__", List.of("name"), rowType));
        // b3_p_price_eq25
        QUERIES.put("source=catalog | where products.price = 25 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.EQUAL, 25), "__row_id__", List.of("name"), rowType));
        // b3_p_price_gt500
        QUERIES.put("source=catalog | where products.price > 500 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.GT, 500), "__row_id__", List.of("name"), rowType));
        // b3_p_price_gte800
        QUERIES.put("source=catalog | where products.price >= 800 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.GTE, 800), "__row_id__", List.of("name"), rowType));
        // b3_p_price_lt25
        QUERIES.put("source=catalog | where products.price < 25 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.LT, 25), "__row_id__", List.of("name"), rowType));
        // b3_p_price_lte100
        QUERIES.put("source=catalog | where products.price <= 100 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.LTE, 100), "__row_id__", List.of("name"), rowType));
        // b3_p_price_ne500
        QUERIES.put("source=catalog | where products.price != 500 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("price", N1Predicate.Op.NOT_EQUAL, 500), "__row_id__", List.of("name"), rowType));
        // b3_p_rating_eq30
        QUERIES.put("source=catalog | where products.rating = 3.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.EQUAL, 3.0), "__row_id__", List.of("name"), rowType));
        // b3_p_rating_ne50
        QUERIES.put("source=catalog | where products.rating != 5.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.NOT_EQUAL, 5.0), "__row_id__", List.of("name"), rowType));
        // b3_p_rating_gte45
        QUERIES.put("source=catalog | where products.rating >= 4.5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.GTE, 4.5), "__row_id__", List.of("name"), rowType));
        // b3_p_rating_lt45
        QUERIES.put("source=catalog | where products.rating < 4.5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.LT, 4.5), "__row_id__", List.of("name"), rowType));
        // b3_p_sku_g1
        QUERIES.put("source=catalog | where products.sku = 'G1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "G1"), "__row_id__", List.of("name"), rowType));
        // b3_p_sku_u1
        QUERIES.put("source=catalog | where products.sku = 'U1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "U1"), "__row_id__", List.of("name"), rowType));
        // b3_p_sku_w1
        QUERIES.put("source=catalog | where products.sku = 'W1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "W1"), "__row_id__", List.of("name"), rowType));
        // b3_p_sku_none
        QUERIES.put("source=catalog | where products.sku = 'ZZ' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "ZZ"), "__row_id__", List.of("name"), rowType));
        // b3_p_instock_ne_t
        QUERIES.put("source=catalog | where products.in_stock != true | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.NOT_EQUAL, true), "__row_id__", List.of("name"), rowType));
        // b3_p_and3
        QUERIES.put("source=catalog | where products.price >= 100 and products.rating > 4.0 and products.in_stock = true | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.GTE, 100), cmp("rating", N1Predicate.Op.GT, 4.0), cmp("in_stock", N1Predicate.Op.EQUAL, true))), "__row_id__", List.of("name"), rowType));
        // b3_p_or3
        QUERIES.put("source=catalog | where products.sku = 'A2' or products.sku = 'I2' or products.sku = 'U1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.Or(List.of(cmp("sku", N1Predicate.Op.EQUAL, "A2"), cmp("sku", N1Predicate.Op.EQUAL, "I2"), cmp("sku", N1Predicate.Op.EQUAL, "U1"))), "__row_id__", List.of("name"), rowType));
        // b3_p_or_mixed
        QUERIES.put("source=catalog | where products.price > 400 or products.rating < 2.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.Or(List.of(cmp("price", N1Predicate.Op.GT, 400), cmp("rating", N1Predicate.Op.LT, 2.0))), "__row_id__", List.of("name"), rowType));
        // b3_p_and_ne
        QUERIES.put("source=catalog | where products.in_stock = true and products.sku != 'A1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("in_stock", N1Predicate.Op.EQUAL, true), cmp("sku", N1Predicate.Op.NOT_EQUAL, "A1"))), "__row_id__", List.of("name"), rowType));
        // b3_p_and_empty2
        QUERIES.put("source=catalog | where products.price < 50 and products.rating > 4.0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.LT, 50), cmp("rating", N1Predicate.Op.GT, 4.0))), "__row_id__", List.of("name"), rowType));
        // b3_v_color_green
        QUERIES.put("source=catalog | where products.variants.color = 'green' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "green"), "__row_id__", List.of("name"), rowType));
        // b3_v_color_gold
        QUERIES.put("source=catalog | where products.variants.color = 'gold' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "gold"), "__row_id__", List.of("name"), rowType));
        // b3_v_color_silver
        QUERIES.put("source=catalog | where products.variants.color = 'silver' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "silver"), "__row_id__", List.of("name"), rowType));
        // b3_v_color_white
        QUERIES.put("source=catalog | where products.variants.color = 'white' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "white"), "__row_id__", List.of("name"), rowType));
        // b3_v_color_black
        QUERIES.put("source=catalog | where products.variants.color = 'black' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "black"), "__row_id__", List.of("name"), rowType));
        // b3_v_color_none
        QUERIES.put("source=catalog | where products.variants.color = 'pink' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "pink"), "__row_id__", List.of("name"), rowType));
        // b3_v_qty_eq5
        QUERIES.put("source=catalog | where products.variants.qty = 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.EQUAL, 5), "__row_id__", List.of("name"), rowType));
        // b3_v_qty_eq20
        QUERIES.put("source=catalog | where products.variants.qty = 20 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.EQUAL, 20), "__row_id__", List.of("name"), rowType));
        // b3_v_qty_lte4
        QUERIES.put("source=catalog | where products.variants.qty <= 4 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.LTE, 4), "__row_id__", List.of("name"), rowType));
        // b3_v_qty_gt2
        QUERIES.put("source=catalog | where products.variants.qty > 2 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.GT, 2), "__row_id__", List.of("name"), rowType));
        // b3_v_and_redlow
        QUERIES.put("source=catalog | where products.variants.color = 'red' and products.variants.qty < 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.And(List.of(cmp("color", N1Predicate.Op.EQUAL, "red"), cmp("qty", N1Predicate.Op.LT, 5))), "__row_id__", List.of("name"), rowType));
        // b3_v_or3
        QUERIES.put("source=catalog | where products.variants.color = 'green' or products.variants.color = 'gold' or products.variants.color = 'white' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.Or(List.of(cmp("color", N1Predicate.Op.EQUAL, "green"), cmp("color", N1Predicate.Op.EQUAL, "gold"), cmp("color", N1Predicate.Op.EQUAL, "white"))), "__row_id__", List.of("name"), rowType));
        // b3_v_and_empty
        QUERIES.put("source=catalog | where products.variants.color = 'white' and products.variants.qty > 0 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.And(List.of(cmp("color", N1Predicate.Op.EQUAL, "white"), cmp("qty", N1Predicate.Op.GT, 0))), "__row_id__", List.of("name"), rowType));
        // b3_s_val_16gb
        QUERIES.put("source=catalog | where products.variants.specs.value = '16gb' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.EQUAL, "16gb"), "__row_id__", List.of("name"), rowType));
        // b3_s_val_ultra
        QUERIES.put("source=catalog | where products.variants.specs.value = 'ultra' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.EQUAL, "ultra"), "__row_id__", List.of("name"), rowType));
        // b3_s_val_ne_x86
        QUERIES.put("source=catalog | where products.variants.specs.value != 'x86' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.NOT_EQUAL, "x86"), "__row_id__", List.of("name"), rowType));
        // b3_s_key_ne_cpu
        QUERIES.put("source=catalog | where products.variants.specs.key != 'cpu' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.NOT_EQUAL, "cpu"), "__row_id__", List.of("name"), rowType));
        // b3_s_weight_eq10
        QUERIES.put("source=catalog | where products.variants.specs.weight = 10 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.EQUAL, 10), "__row_id__", List.of("name"), rowType));
        // b3_s_weight_lte4
        QUERIES.put("source=catalog | where products.variants.specs.weight <= 4 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.LTE, 4), "__row_id__", List.of("name"), rowType));
        // b3_s_weight_ne30
        QUERIES.put("source=catalog | where products.variants.specs.weight != 30 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.NOT_EQUAL, 30), "__row_id__", List.of("name"), rowType));
        // b3_s_and_gpu30
        QUERIES.put("source=catalog | where products.variants.specs.key = 'gpu' and products.variants.specs.weight = 30 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("key", N1Predicate.Op.EQUAL, "gpu"), cmp("weight", N1Predicate.Op.EQUAL, 30))), "__row_id__", List.of("name"), rowType));
        // b3_s_and_empty
        QUERIES.put("source=catalog | where products.variants.specs.key = 'gpu' and products.variants.specs.weight < 30 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("key", N1Predicate.Op.EQUAL, "gpu"), cmp("weight", N1Predicate.Op.LT, 30))), "__row_id__", List.of("name"), rowType));
        // b3_s_or_vals
        QUERIES.put("source=catalog | where products.variants.specs.value = '128gb' or products.variants.specs.value = '64gb' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.Or(List.of(cmp("value", N1Predicate.Op.EQUAL, "128gb"), cmp("value", N1Predicate.Op.EQUAL, "64gb"))), "__row_id__", List.of("name"), rowType));
        // b3_t_or_labels
        QUERIES.put("source=catalog | where products.tags.label = 'sale' or products.tags.label = 'premium' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), new N1Predicate.Or(List.of(cmp("label", N1Predicate.Op.EQUAL, "sale"), cmp("label", N1Predicate.Op.EQUAL, "premium"))), "__row_id__", List.of("name"), rowType));
        // b3_t_label_none
        QUERIES.put("source=catalog | where products.tags.label = 'discontinued' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), cmp("label", N1Predicate.Op.EQUAL, "discontinued"), "__row_id__", List.of("name"), rowType));
        // b3_r_stars_eq4
        QUERIES.put("source=catalog | where reviewers.stars = 4 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.EQUAL, 4), "__row_id__", List.of("name"), rowType));
        // b3_r_stars_eq1
        QUERIES.put("source=catalog | where reviewers.stars = 1 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.EQUAL, 1), "__row_id__", List.of("name"), rowType));
        // b3_r_stars_gt5
        QUERIES.put("source=catalog | where reviewers.stars > 5 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.GT, 5), "__row_id__", List.of("name"), rowType));
        // b3_r_stars_lte2
        QUERIES.put("source=catalog | where reviewers.stars <= 2 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.LTE, 2), "__row_id__", List.of("name"), rowType));
        // b3_r_user_ne
        QUERIES.put("source=catalog | where reviewers.user != 'alice' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("user", N1Predicate.Op.NOT_EQUAL, "alice"), "__row_id__", List.of("name"), rowType));
        // b3_r_user_judy
        QUERIES.put("source=catalog | where reviewers.user = 'judy' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("user", N1Predicate.Op.EQUAL, "judy"), "__row_id__", List.of("name"), rowType));
        // b3_r_and_low
        QUERIES.put("source=catalog | where reviewers.user = 'bob' and reviewers.stars < 3 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), new N1Predicate.And(List.of(cmp("user", N1Predicate.Op.EQUAL, "bob"), cmp("stars", N1Predicate.Op.LT, 3))), "__row_id__", List.of("name"), rowType));
        // b3_r_or3
        QUERIES.put("source=catalog | where reviewers.user = 'frank' or reviewers.user = 'ivan' or reviewers.user = 'heidi' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), new N1Predicate.Or(List.of(cmp("user", N1Predicate.Op.EQUAL, "frank"), cmp("user", N1Predicate.Op.EQUAL, "ivan"), cmp("user", N1Predicate.Op.EQUAL, "heidi"))), "__row_id__", List.of("name"), rowType));
        // b3_cnt_instock
        QUERIES.put("source=catalog | where products.in_stock = true | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.EQUAL, true), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_cnt_v_red
        QUERIES.put("source=catalog | where products.variants.color = 'red' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "red"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_cnt_s_arm
        QUERIES.put("source=catalog | where products.variants.specs.value = 'arm' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("value", N1Predicate.Op.EQUAL, "arm"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_cnt_r_5
        QUERIES.put("source=catalog | where reviewers.stars = 5 | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.EQUAL, 5), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_cnt_and
        QUERIES.put("source=catalog | where products.price = 100 and products.rating = 5.0 | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.And(List.of(cmp("price", N1Predicate.Op.EQUAL, 100), cmp("rating", N1Predicate.Op.EQUAL, 5.0))), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_cnt_none
        QUERIES.put("source=catalog | where products.sku = 'ZZ' | stats count()",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "ZZ"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()"), rowType));
        // b3_m_min_sku
        QUERIES.put("source=catalog | stats min(products.sku)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "sku", "min(products.sku)"), rowType));
        // b3_m_max_sku
        QUERIES.put("source=catalog | stats max(products.sku)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "sku", "max(products.sku)"), rowType));
        // b3_m_min_color
        QUERIES.put("source=catalog | stats min(products.variants.color)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "color", "min(products.variants.color)"), rowType));
        // b3_m_max_color
        QUERIES.put("source=catalog | stats max(products.variants.color)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "color", "max(products.variants.color)"), rowType));
        // b3_m_max_user
        QUERIES.put("source=catalog | stats max(reviewers.user)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "user", "max(reviewers.user)"), rowType));
        // b3_m_sum_price_sale
        QUERIES.put("source=catalog | where products.rating > 4.0 | stats sum(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("rating", N1Predicate.Op.GT, 4.0), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "price", "sum(products.price)"), rowType));
        // b3_m_min_price_instock
        QUERIES.put("source=catalog | where products.in_stock = true | stats min(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("in_stock", N1Predicate.Op.EQUAL, true), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "price", "min(products.price)"), rowType));
        // b3_m_max_qty_red
        QUERIES.put("source=catalog | where products.variants.color = 'red' | stats max(products.variants.qty)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("color", N1Predicate.Op.EQUAL, "red"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "qty", "max(products.variants.qty)"), rowType));
        // b3_m_avg_qty_gt0
        QUERIES.put("source=catalog | where products.variants.qty > 0 | stats avg(products.variants.qty)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.GT, 0), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "qty", "avg(products.variants.qty)"), rowType));
        // b3_m_sum_weight_cpu
        QUERIES.put("source=catalog | where products.variants.specs.key = 'cpu' | stats sum(products.variants.specs.weight)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "cpu"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "weight", "sum(products.variants.specs.weight)"), rowType));
        // b3_m_max_weight_ram
        QUERIES.put("source=catalog | where products.variants.specs.key = 'ram' | stats max(products.variants.specs.weight)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("key", N1Predicate.Op.EQUAL, "ram"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "weight", "max(products.variants.specs.weight)"), rowType));
        // b3_m_avg_stars_high
        QUERIES.put("source=catalog | where reviewers.stars >= 4 | stats avg(reviewers.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), cmp("stars", N1Predicate.Op.GTE, 4), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "stars", "avg(reviewers.stars)"), rowType));
        // b3_m_min_stars
        QUERIES.put("source=catalog | stats min(reviewers.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "stars", "min(reviewers.stars)"), rowType));
        // b3_m_max_stars
        QUERIES.put("source=catalog | stats max(reviewers.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "stars", "max(reviewers.stars)"), rowType));
        // b3_m_sum_stars
        QUERIES.put("source=catalog | stats sum(reviewers.stars)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "stars", "sum(reviewers.stars)"), rowType));
        // b3_m_sum_weight_cpuarm
        QUERIES.put("source=catalog | where products.variants.specs.key = 'cpu' and products.variants.specs.value = 'arm' | stats sum(products.variants.specs.weight)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("key", N1Predicate.Op.EQUAL, "cpu"), cmp("value", N1Predicate.Op.EQUAL, "arm"))), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "weight", "sum(products.variants.specs.weight)"), rowType));
        // b3_m_avg_price_none
        QUERIES.put("source=catalog | where products.sku = 'ZZ' | stats avg(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.EQUAL, "ZZ"), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "price", "avg(products.price)"), rowType));
        // b3_p_sku_gt
        QUERIES.put("source=catalog | where products.sku > 'I1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.GT, "I1"), "__row_id__", List.of("name"), rowType));
        // b3_p_sku_lte
        QUERIES.put("source=catalog | where products.sku <= 'G1' | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), cmp("sku", N1Predicate.Op.LTE, "G1"), "__row_id__", List.of("name"), rowType));
        // b3_s_and_x86w
        QUERIES.put("source=catalog | where products.variants.specs.value = 'x86' and products.variants.specs.weight > 11 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), new N1Predicate.And(List.of(cmp("value", N1Predicate.Op.EQUAL, "x86"), cmp("weight", N1Predicate.Op.GT, 11))), "__row_id__", List.of("name"), rowType));
        // b3_v_or_qty
        QUERIES.put("source=catalog | where products.variants.qty = 0 or products.variants.qty = 20 | fields name",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), new N1Predicate.Or(List.of(cmp("qty", N1Predicate.Op.EQUAL, 0), cmp("qty", N1Predicate.Op.EQUAL, 20))), "__row_id__", List.of("name"), rowType));
        // b3_m_sum_price_all_or
        QUERIES.put("source=catalog | where products.in_stock = true or products.in_stock = false | stats sum(products.price)",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), new N1Predicate.Or(List.of(cmp("in_stock", N1Predicate.Op.EQUAL, true), cmp("in_stock", N1Predicate.Op.EQUAL, false))), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "price", "sum(products.price)"), rowType));
        // g_count_by_color
        QUERIES.put("source=catalog | stats count() by products.variants.color",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "color", "products.variants.color"), rowType));
        // g_count_by_key
        QUERIES.put("source=catalog | stats count() by products.variants.specs.key",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "key", "products.variants.specs.key"), rowType));
        // g_count_by_tag
        QUERIES.put("source=catalog | stats count() by products.tags.label",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "label", "products.tags.label"), rowType));
        // g_count_by_stars
        QUERIES.put("source=catalog | stats count() by reviewers.stars",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "stars", "reviewers.stars"), rowType));
        // g_count_by_instock
        QUERIES.put("source=catalog | stats count() by products.in_stock",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "in_stock", "products.in_stock"), rowType));
        // g_avg_price_by_instock
        QUERIES.put("source=catalog | stats avg(products.price) by products.in_stock",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "price", "avg(products.price)", "in_stock", "products.in_stock"), rowType));
        // g_sum_qty_by_color
        QUERIES.put("source=catalog | stats sum(products.variants.qty) by products.variants.color",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "qty", "sum(products.variants.qty)", "color", "products.variants.color"), rowType));
        // g_max_qty_by_color
        QUERIES.put("source=catalog | stats max(products.variants.qty) by products.variants.color",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "qty", "max(products.variants.qty)", "color", "products.variants.color"), rowType));
        // g_avg_weight_by_key
        QUERIES.put("source=catalog | stats avg(products.variants.specs.weight) by products.variants.specs.key",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "weight", "avg(products.variants.specs.weight)", "key", "products.variants.specs.key"), rowType));
        // g_min_weight_by_key
        QUERIES.put("source=catalog | stats min(products.variants.specs.weight) by products.variants.specs.key",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MIN, "weight", "min(products.variants.specs.weight)", "key", "products.variants.specs.key"), rowType));
        // g_sum_price_by_sku
        QUERIES.put("source=catalog | stats sum(products.price) by products.sku",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "price", "sum(products.price)", "sku", "products.sku"), rowType));
        // g_max_stars_by_user
        QUERIES.put("source=catalog | stats max(reviewers.stars) by reviewers.user",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.MAX, "stars", "max(reviewers.stars)", "user", "reviewers.user"), rowType));
        // g_count_by_color_instock
        QUERIES.put("source=catalog | where products.variants.qty > 0 | stats count() by products.variants.color",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), cmp("qty", N1Predicate.Op.GT, 0), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "color", "products.variants.color"), rowType));
        // g_avg_weight_by_key_cpu
        QUERIES.put("source=catalog | where products.variants.specs.weight >= 5 | stats avg(products.variants.specs.weight) by products.variants.specs.key",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), cmp("weight", N1Predicate.Op.GTE, 5), "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.AVG, "weight", "avg(products.variants.specs.weight)", "key", "products.variants.specs.key"), rowType));
        // g_count_by_color_h2
        QUERIES.put("source=catalog | stats count() by products.variants.color | where count() > 1",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "color", "products.variants.color", N1Predicate.Op.GT, 1), rowType));
        // g_count_by_key_h2
        QUERIES.put("source=catalog | stats count() by products.variants.specs.key | where count() >= 2",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants", "products.variants.specs"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "key", "products.variants.specs.key", N1Predicate.Op.GTE, 2), rowType));
        // g_count_by_tag_h1
        QUERIES.put("source=catalog | stats count() by products.tags.label | where count() = 1",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.tags"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "label", "products.tags.label", N1Predicate.Op.EQUAL, 1), rowType));
        // g_sum_qty_by_color_h
        QUERIES.put("source=catalog | stats sum(products.variants.qty) by products.variants.color | where sum(products.variants.qty) > 5",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("products", "products.variants"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.SUM, "qty", "sum(products.variants.qty)", "color", "products.variants.color", N1Predicate.Op.GT, 5), rowType));
        // g_count_by_stars_h
        QUERIES.put("source=catalog | stats count() by reviewers.stars | where count() > 1",
            (indexName, rowType) -> new N1Descriptor(indexName, List.of("reviewers"), null, "__row_id__", List.of("name"), new N1Aggregate(N1Aggregate.Fn.COUNT, null, "count()", "stars", "reviewers.stars", N1Predicate.Op.GT, 1), rowType));
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

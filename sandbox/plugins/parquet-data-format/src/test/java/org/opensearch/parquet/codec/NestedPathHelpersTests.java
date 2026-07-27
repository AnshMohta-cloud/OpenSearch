/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Set;

/**
 * Unit tests for the nested-path helper functions in {@link ParquetDocValuesLeafReader}:
 * {@code owningNestedPath}, {@code parquetChildLeafColumnPath}, {@code nestedDepthOf}. These are pure functions
 * of a field name plus the set of nested-object paths, and they determine how a child field maps to its physical
 * Parquet {@code LIST<STRUCT>} column at ANY depth — so exhaustive coverage here is the generalization proof that
 * the {@code .list.element.} column path is built correctly for arbitrary schemas (deep chains, sibling nested
 * fields, and non-nested object levels mixed in).
 */
public class NestedPathHelpersTests extends OpenSearchTestCase {

    // The megadeep nested paths (orgs 5-level chain + tags 3-level chain).
    private static final Set<String> MEGADEEP = Set.of(
        "orgs",
        "orgs.divisions",
        "orgs.divisions.teams",
        "orgs.divisions.teams.members",
        "orgs.divisions.teams.members.badges",
        "tags",
        "tags.variants",
        "tags.variants.specs"
    );

    // ───────────────────────── owningNestedPath ─────────────────────────

    public void testOwningNestedPathSingleLevel() {
        Set<String> paths = Set.of("comments");
        assertEquals("comments", ParquetDocValuesLeafReader.owningNestedPath("comments.author", paths));
        assertEquals("comments", ParquetDocValuesLeafReader.owningNestedPath("comments.score", paths));
        assertNull("a non-child field has no owning nested path", ParquetDocValuesLeafReader.owningNestedPath("title", paths));
        assertNull("the nested path itself is not owned", ParquetDocValuesLeafReader.owningNestedPath("comments", paths));
    }

    public void testOwningNestedPathPicksDeepest() {
        // A deep leaf is owned by the DEEPEST nested path that is a strict prefix (not a shallower ancestor).
        assertEquals(
            "orgs.divisions.teams.members.badges",
            ParquetDocValuesLeafReader.owningNestedPath("orgs.divisions.teams.members.badges.year", MEGADEEP)
        );
        assertEquals(
            "orgs.divisions.teams.members",
            ParquetDocValuesLeafReader.owningNestedPath("orgs.divisions.teams.members.age", MEGADEEP)
        );
        assertEquals("orgs", ParquetDocValuesLeafReader.owningNestedPath("orgs.oname", MEGADEEP));
        assertEquals("tags.variants.specs", ParquetDocValuesLeafReader.owningNestedPath("tags.variants.specs.v", MEGADEEP));
    }

    public void testOwningNestedPathPrefixNotBoundaryConfusion() {
        // "orgsX" must NOT be treated as owned by "orgs" (prefix match must respect the dot boundary).
        Set<String> paths = Set.of("orgs");
        assertNull(ParquetDocValuesLeafReader.owningNestedPath("orgsX.field", paths));
        assertEquals("orgs", ParquetDocValuesLeafReader.owningNestedPath("orgs.field", paths));
    }

    // ───────────────────────── nestedDepthOf ─────────────────────────

    public void testNestedDepthOf() {
        assertEquals(1, ParquetDocValuesLeafReader.nestedDepthOf("orgs", MEGADEEP));
        assertEquals(2, ParquetDocValuesLeafReader.nestedDepthOf("orgs.divisions", MEGADEEP));
        assertEquals(4, ParquetDocValuesLeafReader.nestedDepthOf("orgs.divisions.teams.members", MEGADEEP));
        assertEquals(5, ParquetDocValuesLeafReader.nestedDepthOf("orgs.divisions.teams.members.badges", MEGADEEP));
        assertEquals(1, ParquetDocValuesLeafReader.nestedDepthOf("tags", MEGADEEP));
        assertEquals(3, ParquetDocValuesLeafReader.nestedDepthOf("tags.variants.specs", MEGADEEP));
    }

    public void testNestedDepthOfIgnoresUnrelatedSiblings() {
        // Depth counts only ANCESTOR nested paths of the given path, not sibling chains.
        assertEquals(2, ParquetDocValuesLeafReader.nestedDepthOf("tags.variants", MEGADEEP));
    }

    // ───────────────────────── parquetChildLeafColumnPath ─────────────────────────

    public void testColumnPathSingleLevel() {
        Set<String> paths = Set.of("comments");
        assertEquals(
            "comments.list.element.author",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("comments.author", paths)
        );
    }

    public void testColumnPathDeepInsertsListElementAtEveryBoundary() {
        // The crux: .list.element. must be inserted between EVERY nested level, not just once.
        assertEquals(
            "orgs.list.element.oname",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("orgs.oname", MEGADEEP)
        );
        assertEquals(
            "orgs.list.element.divisions.list.element.dname",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("orgs.divisions.dname", MEGADEEP)
        );
        assertEquals(
            "orgs.list.element.divisions.list.element.teams.list.element.members.list.element.age",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("orgs.divisions.teams.members.age", MEGADEEP)
        );
        assertEquals(
            "orgs.list.element.divisions.list.element.teams.list.element.members.list.element.badges.list.element.year",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("orgs.divisions.teams.members.badges.year", MEGADEEP)
        );
        assertEquals(
            "tags.list.element.variants.list.element.specs.list.element.v",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("tags.variants.specs.v", MEGADEEP)
        );
    }

    public void testColumnPathWithNonNestedObjectLevel() {
        // A plain (non-nested) object level does NOT get a list.element. hop — only nested paths do. Here
        // `profile` is a regular object inside the nested `users`, so users.profile.name maps to
        // users.list.element.profile.name (profile stays a plain struct field, no list wrapper).
        Set<String> paths = Set.of("users"); // only `users` is nested; `users.profile` is a plain object
        assertEquals(
            "users.list.element.profile.name",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("users.profile.name", paths)
        );
    }

    public void testColumnPathTwoNestedThenPlainObject() {
        // nested a → nested a.b → plain object a.b.c → leaf a.b.c.d
        // Expect list.element after a and after b, but NOT after c (plain object).
        Set<String> paths = Set.of("a", "a.b");
        assertEquals(
            "a.list.element.b.list.element.c.d",
            ParquetDocValuesLeafReader.parquetChildLeafColumnPath("a.b.c.d", paths)
        );
    }
}

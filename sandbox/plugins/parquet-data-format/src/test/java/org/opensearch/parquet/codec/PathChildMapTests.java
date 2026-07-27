/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;
import java.util.Set;

/**
 * Unit tests for {@link ParquetDocValuesLeafReader#buildPathChildMaps} — the block-join → per-path child-doc
 * offset assignment that maps a deep child Lucene doc to its element index in the Parquet flattened column for
 * its nesting path. Getting this offset right (per path, in document/post-order, within each root block) is what
 * makes deep child-field predicates and aggregations read the correct value; a flat block offset (the old bug)
 * mis-mapped multi-level blocks. These tests drive the pure function with hand-built block layouts.
 */
public class PathChildMapTests extends OpenSearchTestCase {

    private static long[] roots(int maxDoc, Map<Integer, Long> parentToRow) {
        long[] r = new long[maxDoc];
        java.util.Arrays.fill(r, -1L);
        for (Map.Entry<Integer, Long> e : parentToRow.entrySet()) {
            r[e.getKey()] = e.getValue();
        }
        return r;
    }

    // ───────────────────────── single-level ─────────────────────────

    public void testSingleLevelBlock() {
        // Layout: [c0, c1, c2, PARENT] — 3 comment children then the root at docId 3 (row 0).
        // docPath: children are "comments", parent is null.
        String[] docPath = { "comments", "comments", "comments", null };
        long[] rootRowId = roots(4, Map.of(3, 0L));
        Map<String, ParquetDocValuesLeafReader.PathChildMap> maps = ParquetDocValuesLeafReader.buildPathChildMaps(
            4,
            docPath,
            rootRowId,
            Set.of("comments")
        );
        ParquetDocValuesLeafReader.PathChildMap m = maps.get("comments");
        // children 0,1,2 → row 0, offsets 0,1,2 ; parent doc 3 → row -1 (marks "not a child of this path").
        // The offset slot for a non-child doc is left at its default 0 and is never read (consumers gate on
        // parquetRow >= 0), so only parquetRow carries the -1 sentinel.
        assertArrayEquals(new int[] { 0, 0, 0, -1 }, m.parquetRow());
        assertEquals(0, m.offset()[0]);
        assertEquals(1, m.offset()[1]);
        assertEquals(2, m.offset()[2]);
    }

    public void testTwoRootsSingleLevel() {
        // [c0, P(row0)], [c1, c2, P(row1)]  → docIds 0..4
        String[] docPath = { "comments", null, "comments", "comments", null };
        long[] rootRowId = roots(5, Map.of(1, 0L, 4, 1L));
        ParquetDocValuesLeafReader.PathChildMap m = ParquetDocValuesLeafReader.buildPathChildMaps(
            5,
            docPath,
            rootRowId,
            Set.of("comments")
        ).get("comments");
        assertArrayEquals(new int[] { 0, -1, 1, 1, -1 }, m.parquetRow());
        // child docs 0 (block 0) and 2,3 (block 1) carry offsets 0 and 0,1 — offsets reset per block; the
        // non-child slots (parents at 1,4) keep their default 0 and are gated out by parquetRow == -1.
        assertEquals(0, m.offset()[0]);
        assertEquals(0, m.offset()[2]);
        assertEquals(1, m.offset()[3]);
    }

    public void testParentWithZeroChildren() {
        // [P(row0)] — a parent with no children (empty nested arrays). No child offsets assigned.
        String[] docPath = { null };
        long[] rootRowId = roots(1, Map.of(0, 0L));
        ParquetDocValuesLeafReader.PathChildMap m = ParquetDocValuesLeafReader.buildPathChildMaps(
            1,
            docPath,
            rootRowId,
            Set.of("comments")
        ).get("comments");
        assertArrayEquals(new int[] { -1 }, m.parquetRow());
    }

    // ───────────────────────── multi-level interleaved (the crux) ─────────────────────────

    public void testMultiLevelPerPathOffsets() {
        // A megadeep-style block where a single root contains child docs at MULTIPLE nested levels, interleaved in
        // post-order. Each path must be numbered independently (0,1,... per path), NOT by flat block position.
        // Layout (post-order) for one root at docId 6 (row 0):
        //   0: orgs.divisions.teams.members  (member A)
        //   1: orgs.divisions.teams.members  (member B)
        //   2: orgs.divisions.teams          (team 0)
        //   3: orgs.divisions                (division 0)
        //   4: orgs.divisions.teams          (team 1)   <- second team, different division? still same path counter
        //   5: orgs                          (org 0)
        //   6: PARENT (row 0)
        String[] docPath = {
            "orgs.divisions.teams.members",
            "orgs.divisions.teams.members",
            "orgs.divisions.teams",
            "orgs.divisions",
            "orgs.divisions.teams",
            "orgs",
            null };
        long[] rootRowId = roots(7, Map.of(6, 0L));
        Set<String> paths = Set.of("orgs", "orgs.divisions", "orgs.divisions.teams", "orgs.divisions.teams.members");
        Map<String, ParquetDocValuesLeafReader.PathChildMap> maps = ParquetDocValuesLeafReader.buildPathChildMaps(
            7,
            docPath,
            rootRowId,
            paths
        );

        // members: docs 0,1 → offsets 0,1
        ParquetDocValuesLeafReader.PathChildMap members = maps.get("orgs.divisions.teams.members");
        assertEquals(0, members.parquetRow()[0]);
        assertEquals(0, members.offset()[0]);
        assertEquals(0, members.parquetRow()[1]);
        assertEquals(1, members.offset()[1]);
        assertEquals(-1, members.parquetRow()[2]); // a teams doc is not a members child

        // teams: docs 2,4 → offsets 0,1 (independent counter, NOT flat block position)
        ParquetDocValuesLeafReader.PathChildMap teams = maps.get("orgs.divisions.teams");
        assertEquals(0, teams.offset()[2]);
        assertEquals(1, teams.offset()[4]);
        assertEquals(-1, teams.parquetRow()[3]); // divisions doc is not a teams child

        // divisions: doc 3 → offset 0
        assertEquals(0, maps.get("orgs.divisions").offset()[3]);
        // orgs: doc 5 → offset 0
        assertEquals(0, maps.get("orgs").offset()[5]);
    }

    public void testPathWithoutExposedFieldIsSkipped() {
        // If "orgs.divisions" is NOT in the requested paths (no exposed child field there), its child docs are
        // skipped and do not disturb the numbering of the paths that ARE requested.
        String[] docPath = { "orgs.divisions.teams", "orgs.divisions", "orgs", null };
        long[] rootRowId = roots(4, Map.of(3, 0L));
        Map<String, ParquetDocValuesLeafReader.PathChildMap> maps = ParquetDocValuesLeafReader.buildPathChildMaps(
            4,
            docPath,
            rootRowId,
            Set.of("orgs", "orgs.divisions.teams") // note: "orgs.divisions" intentionally absent
        );
        assertNull("unrequested path has no map", maps.get("orgs.divisions"));
        assertEquals(0, maps.get("orgs.divisions.teams").offset()[0]);
        assertEquals(0, maps.get("orgs").offset()[2]);
    }

    public void testChildDocOutsideRequestedPathsMarkedAbsent() {
        // A child at a path we don't track leaves parquetRow == -1 for that doc in every tracked path map.
        String[] docPath = { "other", "comments", null };
        long[] rootRowId = roots(3, Map.of(2, 5L));
        ParquetDocValuesLeafReader.PathChildMap m = ParquetDocValuesLeafReader.buildPathChildMaps(
            3,
            docPath,
            rootRowId,
            Set.of("comments")
        ).get("comments");
        assertEquals(-1, m.parquetRow()[0]); // "other" child not tracked here
        assertEquals(5, m.parquetRow()[1]);  // comments child → root row 5
        assertEquals(0, m.offset()[1]);
    }

}

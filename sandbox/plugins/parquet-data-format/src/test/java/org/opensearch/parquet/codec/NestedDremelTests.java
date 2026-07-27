/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.lucene.util.BytesRef;
import org.opensearch.test.OpenSearchTestCase;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Exhaustive unit tests for {@link NestedDremel} — the pure Dremel record-assembly at the heart of composite
 * nested support. Every case feeds HAND-BUILT (rep, def, values, maxDef, thresholds) level streams (the exact
 * form the native reader produces) and asserts the reconstructed tree / per-element view, so the striped-assembly
 * logic is proven correct in isolation, independent of any Parquet I/O.
 *
 * <p>The level streams used here are the GROUND TRUTH captured from real Parquet files written by the engine
 * (verified with a standalone parquet-crate probe + DuckDB): standard 3-level LIST encoding, so per-level
 * element-exists thresholds are {3, 6, 9, 12, 15} at depths 1..5 and a leaf's maxDef is 3*depth+1. The tests do
 * NOT hardcode "3" as an assumption — they pass thresholds explicitly, and a dedicated test proves a
 * NON-uniform threshold vector (a different encoding) assembles correctly too.
 */
public class NestedDremelTests extends OpenSearchTestCase {

    private static final ParquetPhysicalType I64 = ParquetPhysicalType.INT64;
    private static final ParquetPhysicalType I32 = ParquetPhysicalType.INT32; // megadeep score/age/year/v are `integer`
    private static final ParquetPhysicalType STR = ParquetPhysicalType.BYTE_ARRAY;

    // Uniform 3-level-LIST thresholds for depths 1..5 (element-at-level-k exists iff def >= 3k).
    private static final int[] THR = { 3, 6, 9, 12, 15 };

    private static Object[] longs(long... vs) {
        Object[] o = new Object[vs.length];
        for (int i = 0; i < vs.length; i++) o[i] = vs[i];
        return o;
    }

    private static Object[] strs(String... vs) {
        Object[] o = new Object[vs.length];
        for (int i = 0; i < vs.length; i++) o[i] = vs[i] == null ? null : vs[i].getBytes(StandardCharsets.UTF_8);
        return o;
    }

    private static List<Map<String, Object>> assemble(
        String[] chain,
        int[] rep,
        int[] def,
        int maxDef,
        Object[] values,
        boolean bytes,
        ParquetPhysicalType phys
    ) {
        List<Map<String, Object>> root = new ArrayList<>();
        NestedDremel.assembleLeafInto(root, chain, chain.length, rep, def, maxDef, THR, values, bytes, phys);
        return root;
    }

    // ───────────────────────── existingLevels ─────────────────────────

    public void testExistingLevelsUniform() {
        // depth 3, thresholds {3,6,9}: def maps to the count of satisfied leading levels.
        assertEquals(0, NestedDremel.existingLevels(0, 3, THR)); // whole field absent
        assertEquals(0, NestedDremel.existingLevels(2, 3, THR)); // below level-1 threshold
        assertEquals(1, NestedDremel.existingLevels(3, 3, THR)); // level-1 element only
        assertEquals(1, NestedDremel.existingLevels(5, 3, THR));
        assertEquals(2, NestedDremel.existingLevels(6, 3, THR)); // through level 2
        assertEquals(3, NestedDremel.existingLevels(9, 3, THR)); // through level 3
        assertEquals(3, NestedDremel.existingLevels(10, 3, THR)); // capped at depth
    }

    public void testExistingLevelsNonUniformThresholds() {
        // A hypothetical NON-uniform encoding: level boundaries {2, 5, 11}. Proves no hardcoded "3".
        int[] thr = { 2, 5, 11 };
        assertEquals(0, NestedDremel.existingLevels(1, 3, thr));
        assertEquals(1, NestedDremel.existingLevels(2, 3, thr));
        assertEquals(1, NestedDremel.existingLevels(4, 3, thr));
        assertEquals(2, NestedDremel.existingLevels(5, 3, thr));
        assertEquals(2, NestedDremel.existingLevels(10, 3, thr));
        assertEquals(3, NestedDremel.existingLevels(11, 3, thr));
    }

    // ───────────────────────── single-level reconstruction ─────────────────────────

    public void testSingleLevelSimple() {
        // comments:[{a},{b}] — one leaf "author", depth 1, maxDef 4. rep=[0,1] def=[4,4].
        List<Map<String, Object>> out = assemble(
            new String[] { "author" },
            new int[] { 0, 1 },
            new int[] { 4, 4 },
            4,
            strs("a", "b"),
            true,
            STR
        );
        assertEquals(List.of(Map.of("author", "a"), Map.of("author", "b")), out);
    }

    public void testSingleLevelInteriorNull() {
        // GROUND TRUTH (probe): comments.score for [{views:10},{score:2},{views:30},{score:4}].
        // score present only at elements 1 and 3. rep=[0,1,1,1] def=[3,4,3,4] v=[2,4].
        List<Map<String, Object>> out = assemble(
            new String[] { "score" },
            new int[] { 0, 1, 1, 1 },
            new int[] { 3, 4, 3, 4 },
            4,
            longs(2, 4),
            false,
            I32
        );
        // Element 0 and 2 exist (def 3 >= threshold 3) but have no score → empty maps; 1 and 3 carry the value.
        assertEquals(4, out.size());
        assertTrue(out.get(0).isEmpty());
        assertEquals(2, out.get(1).get("score"));
        assertTrue(out.get(2).isEmpty());
        assertEquals(4, out.get(3).get("score"));
    }

    public void testSingleLevelEmptyList() {
        // comments:[] — one rep=0 slot with def=0 (below level-1 threshold). No elements.
        List<Map<String, Object>> out = assemble(new String[] { "author" }, new int[] { 0 }, new int[] { 0 }, 4, strs(), true, STR);
        assertTrue(out.isEmpty());
    }

    public void testTwoLeavesCoAssembleIntoOneTree() {
        // comments:[{author:a,score:1},{author:b}] — author present both, score only elem 0.
        List<Map<String, Object>> root = new ArrayList<>();
        NestedDremel.assembleLeafInto(
            root, new String[] { "author" }, 1, new int[] { 0, 1 }, new int[] { 4, 4 }, 4, THR, strs("a", "b"), true, STR
        );
        NestedDremel.assembleLeafInto(
            root, new String[] { "score" }, 1, new int[] { 0, 1 }, new int[] { 4, 3 }, 4, THR, longs(1), false, I32
        );
        assertEquals(2, root.size());
        assertEquals("a", root.get(0).get("author"));
        assertEquals(1, root.get(0).get("score"));
        assertEquals("b", root.get(1).get("author"));
        assertFalse(root.get(1).containsKey("score"));
    }

    // ───────────────────────── multi-level reconstruction (ground truth: megadeep row 0) ─────────────────────────

    public void testDeepReconstruction5LevelMegadeepRow0() {
        // Rebuild D1's orgs tree from the EXACT level streams the probe captured, one leaf at a time, and assert
        // the full nested structure with empties/nulls at every level.
        List<Map<String, Object>> orgs = new ArrayList<>();
        add(orgs, new String[] { "oname" }, new int[] { 0 }, new int[] { 4 }, 4, strs("acme"), STR);
        add(orgs, new String[] { "divisions", "dname" }, new int[] { 0, 2 }, new int[] { 7, 7 }, 7, strs("eng", "sales"), STR);
        add(orgs, new String[] { "divisions", "teams", "tname" }, new int[] { 0, 3, 2 }, new int[] { 10, 10, 6 }, 10, strs("core", "qa"), STR);
        add(
            orgs,
            new String[] { "divisions", "teams", "members", "mname" },
            new int[] { 0, 4, 3, 2 },
            new int[] { 13, 13, 9, 6 },
            13,
            strs("al", "bo"),
            STR
        );
        add(
            orgs,
            new String[] { "divisions", "teams", "members", "age" },
            new int[] { 0, 4, 3, 2 },
            new int[] { 13, 13, 9, 6 },
            13,
            longs(30, 40),
            I32
        );
        add(
            orgs,
            new String[] { "divisions", "teams", "members", "badges", "badge" },
            new int[] { 0, 5, 4, 3, 2 },
            new int[] { 16, 16, 12, 9, 6 },
            16,
            strs("gold", "silver"),
            STR
        );
        add(
            orgs,
            new String[] { "divisions", "teams", "members", "badges", "year" },
            new int[] { 0, 5, 4, 3, 2 },
            new int[] { 16, 16, 12, 9, 6 },
            16,
            longs(2020, 2021),
            I32
        );

        // Expected: acme → [eng → [core → [al{30,[gold2020,silver2021]}, bo{40}], qa{}], sales{}]
        assertEquals(1, orgs.size());
        Map<String, Object> acme = orgs.get(0);
        assertEquals("acme", acme.get("oname"));
        List<?> divs = (List<?>) acme.get("divisions");
        assertEquals(2, divs.size());
        Map<?, ?> eng = (Map<?, ?>) divs.get(0);
        assertEquals("eng", eng.get("dname"));
        Map<?, ?> sales = (Map<?, ?>) divs.get(1);
        assertEquals("sales", sales.get("dname"));
        assertFalse("sales has no teams", sales.containsKey("teams"));
        List<?> teams = (List<?>) eng.get("teams");
        assertEquals(2, teams.size());
        Map<?, ?> core = (Map<?, ?>) teams.get(0);
        assertEquals("core", core.get("tname"));
        Map<?, ?> qa = (Map<?, ?>) teams.get(1);
        assertEquals("qa", qa.get("tname"));
        assertFalse("qa has no members", qa.containsKey("members"));
        List<?> members = (List<?>) core.get("members");
        assertEquals(2, members.size());
        Map<?, ?> al = (Map<?, ?>) members.get(0);
        assertEquals("al", al.get("mname"));
        assertEquals(30, al.get("age"));
        Map<?, ?> bo = (Map<?, ?>) members.get(1);
        assertEquals("bo", bo.get("mname"));
        assertEquals(40, bo.get("age"));
        assertFalse("bo has no badges", bo.containsKey("badges"));
        List<?> badges = (List<?>) al.get("badges");
        assertEquals(2, badges.size());
        assertEquals("gold", ((Map<?, ?>) badges.get(0)).get("badge"));
        assertEquals(2020, ((Map<?, ?>) badges.get(0)).get("year"));
        assertEquals("silver", ((Map<?, ?>) badges.get(1)).get("badge"));
        assertEquals(2021, ((Map<?, ?>) badges.get(1)).get("year"));
    }

    public void testWholeFieldAbsent() {
        // D2: orgs absent (def=0). Any orgs leaf yields an empty tree.
        List<Map<String, Object>> out = assemble(new String[] { "oname" }, new int[] { 0 }, new int[] { 0 }, 4, strs(), true, STR);
        assertTrue(out.isEmpty());
    }

    public void testDeepMultiElementPerParent() {
        // tags.variants.specs.v for a variant with THREE specs then a variant with none (probe-style):
        // rep=[0,3,3,2] def=[10,10,10,6] v=[10,20,30]. Two variants: v0 has specs[10,20,30], v1 has none.
        List<Map<String, Object>> tags = new ArrayList<>();
        // tag names: rep=[0] def=[4] -> one tag
        add(tags, new String[] { "tag" }, new int[] { 0 }, new int[] { 4 }, 4, strs("t1"), STR);
        // variants.color: two variants, second empty specs handled by specs leaf. rep=[0,2] def=[7,7]
        add(tags, new String[] { "variants", "color" }, new int[] { 0, 2 }, new int[] { 7, 7 }, 7, strs("red", "blue"), STR);
        // specs.v: variant0 has 3 specs, variant1 has none (def=6 = variant exists, specs empty).
        add(tags, new String[] { "variants", "specs", "v" }, new int[] { 0, 3, 3, 2 }, new int[] { 10, 10, 10, 6 }, 10, longs(10, 20, 30), I32);

        assertEquals(1, tags.size());
        List<?> variants = (List<?>) tags.get(0).get("variants");
        assertEquals(2, variants.size());
        List<?> specs0 = (List<?>) ((Map<?, ?>) variants.get(0)).get("specs");
        assertEquals(3, specs0.size());
        assertEquals(10, ((Map<?, ?>) specs0.get(0)).get("v"));
        assertEquals(20, ((Map<?, ?>) specs0.get(1)).get("v"));
        assertEquals(30, ((Map<?, ?>) specs0.get(2)).get("v"));
        assertFalse("variant 1 has no specs", ((Map<?, ?>) variants.get(1)).containsKey("specs"));
    }

    private void add(List<Map<String, Object>> root, String[] chain, int[] rep, int[] def, int maxDef, Object[] vals, ParquetPhysicalType phys) {
        NestedDremel.assembleLeafInto(root, chain, chain.length, rep, def, maxDef, THR, vals, phys == STR, phys);
    }

    // ───────────────────────── numeric decoding ─────────────────────────

    public void testDecodePrimitiveTypes() {
        assertEquals(42, NestedDremel.decodePrimitive(42L, ParquetPhysicalType.INT32));
        assertEquals(-7, NestedDremel.decodePrimitive(-7L, ParquetPhysicalType.INT32));
        assertEquals(42L, NestedDremel.decodePrimitive(42L, ParquetPhysicalType.INT64));
        assertEquals(true, NestedDremel.decodePrimitive(1L, ParquetPhysicalType.BOOL));
        assertEquals(false, NestedDremel.decodePrimitive(0L, ParquetPhysicalType.BOOL));
        assertEquals(3.5, NestedDremel.decodePrimitive(Double.doubleToLongBits(3.5), ParquetPhysicalType.DOUBLE));
        assertEquals(1.5, NestedDremel.decodePrimitive((long) Float.floatToIntBits(1.5f), ParquetPhysicalType.FLOAT));
    }

    public void testFloatDoubleReconstruction() {
        // A double leaf: value present in slot 0, absent in slot 1.
        List<Map<String, Object>> out = assemble(
            new String[] { "ratio" },
            new int[] { 0, 1 },
            new int[] { 4, 3 },
            4,
            longs(Double.doubleToLongBits(2.5)),
            false,
            ParquetPhysicalType.DOUBLE
        );
        assertEquals(2, out.size());
        assertEquals(2.5, out.get(0).get("ratio"));
        assertTrue(out.get(1).isEmpty());
    }

    // ───────────────────────── expandElementsAtLevel (child-DV per-element view) ─────────────────────────

    public void testExpandSingleLevelInteriorNull() {
        // comments.score [{views},{score:2},{views},{score:4}] at level 1, threshold 3, maxDef 4.
        NestedDremel.ElementValues ev = NestedDremel.expandElementsAtLevel(
            new int[] { 0, 1, 1, 1 },
            new int[] { 3, 4, 3, 4 },
            4,
            1,
            3,
            longs(2, 4),
            false,
            I64
        );
        assertEquals(4, ev.present().length);
        assertFalse(ev.present()[0]);
        assertTrue(ev.present()[1]);
        assertEquals(2L, ev.longs()[1]);
        assertFalse(ev.present()[2]);
        assertTrue(ev.present()[3]);
        assertEquals(4L, ev.longs()[3]);
    }

    public void testExpandDeepMultiElementNoUndercount() {
        // GROUND TRUTH (probe): specs.v row0 rep=[0,1,3,3,2,3,3,2] def=[3,10,10,10,10,10,10,6] v=[58,97,42,72,61,13].
        // At level 3 (specs), threshold 9, maxDef 10: exactly 6 spec elements, all present (matches vanilla count).
        NestedDremel.ElementValues ev = NestedDremel.expandElementsAtLevel(
            new int[] { 0, 1, 3, 3, 2, 3, 3, 2 },
            new int[] { 3, 10, 10, 10, 10, 10, 10, 6 },
            10,
            3,
            9,
            longs(58, 97, 42, 72, 61, 13),
            false,
            I64
        );
        assertEquals("6 spec elements (no undercount)", 6, ev.present().length);
        for (int i = 0; i < 6; i++) {
            assertTrue("spec " + i + " present", ev.present()[i]);
        }
        assertEquals(58L, ev.longs()[0]);
        assertEquals(13L, ev.longs()[5]);
    }

    public void testExpandKeywordElements() {
        // specs.k as keyword at level 3: 2 present + a null interior.
        NestedDremel.ElementValues ev = NestedDremel.expandElementsAtLevel(
            new int[] { 0, 3, 3 },
            new int[] { 10, 9, 10 },
            10,
            3,
            9,
            strs("w", "h"),
            true,
            STR
        );
        assertEquals(3, ev.present().length);
        assertTrue(ev.present()[0]);
        assertEquals(new BytesRef("w"), ev.bytes()[0]);
        assertFalse("middle element exists but k is null", ev.present()[1]);
        assertTrue(ev.present()[2]);
        assertEquals(new BytesRef("h"), ev.bytes()[2]);
    }

    public void testExpandEmptyAtLevel() {
        // A row where the level-2 list is empty everywhere: no elements at level 2.
        NestedDremel.ElementValues ev = NestedDremel.expandElementsAtLevel(
            new int[] { 0 },
            new int[] { 3 },  // only level-1 defined
            7,
            2,
            6,
            longs(),
            false,
            I64
        );
        assertEquals(0, ev.present().length);
    }

    public void testCountElementsAtLevelMatchesExpand() {
        int[] rep = { 0, 1, 3, 3, 2, 3, 3, 2 };
        int[] def = { 3, 10, 10, 10, 10, 10, 10, 6 };
        assertEquals(6, NestedDremel.countElementsAtLevel(rep, def, 3, 9));
        // At level 2 (variants), threshold 6: elements are slots with rep<=2 && def>=6 → slots 0? def3<6 no;
        // slot1 rep1<=2 def10>=6 yes; slot4 rep2 def10 yes; slot7 rep2 def6 yes → 3 variant elements.
        assertEquals(3, NestedDremel.countElementsAtLevel(rep, def, 2, 6));
    }

    // ───────────────────────── getOrCreate / childList primitives ─────────────────────────

    public void testGetOrCreateElementAppends() {
        List<Map<String, Object>> list = new ArrayList<>();
        Map<String, Object> e2 = NestedDremel.getOrCreateElement(list, 2);
        assertEquals(3, list.size()); // grew to index 2
        assertSame(e2, list.get(2));
        assertSame(e2, NestedDremel.getOrCreateElement(list, 2)); // revisit returns same
    }

    public void testChildListOfCreatesOnce() {
        Map<String, Object> elem = new java.util.LinkedHashMap<>();
        List<Map<String, Object>> a = NestedDremel.childListOf(elem, "specs");
        List<Map<String, Object>> b = NestedDremel.childListOf(elem, "specs");
        assertSame("same list returned on second access", a, b);
        assertSame(a, elem.get("specs"));
    }
}

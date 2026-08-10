/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.type.SqlTypeName;
import org.opensearch.test.OpenSearchTestCase;

import java.util.List;
import java.util.Map;

/**
 * Unit tests for {@link OpenSearchNestedFieldRewriter}'s deep nested-of-nested key resolution
 * ({@code ExprTreeBuilder.buildFieldOrNestedDescent}) — the Phase-2 rewriter half that turns a
 * compound-dotted nested path like {@code comments.replies.txt} into a {@code {"nested":...,"inner":...}}
 * descent tree the recursive Rust UDF consumes. Exercised via the package-private
 * {@link OpenSearchNestedFieldRewriter#resolveNestedKeyForTest} hook so we assert on the exact JSON
 * node shape without building a full Calcite plan.
 *
 * <p>The analytics schema builder types a {@code nested} field as {@code ARRAY(ROW(...))} and recurses
 * for nested-in-nested (verified in {@code OpenSearchSchemaBuilder}); these tests build the same
 * {@code ARRAY(ROW(...))} element types by hand so the resolution logic is proven at arbitrary depth
 * independent of the schema-build path.
 */
public class NestedDescentKeyResolutionTests extends OpenSearchTestCase {

    private final RelDataTypeFactory typeFactory = new JavaTypeFactoryImpl();

    private RelDataType varchar() {
        return typeFactory.createSqlType(SqlTypeName.VARCHAR);
    }

    private RelDataType intType() {
        return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }

    /** A ROW type from alternating (name, type) pairs. */
    private RelDataType row(Object... nameThenType) {
        RelDataTypeFactory.Builder b = typeFactory.builder();
        for (int i = 0; i < nameThenType.length; i += 2) {
            b.add((String) nameThenType[i], (RelDataType) nameThenType[i + 1]);
        }
        return b.build();
    }

    /** ARRAY(elementRow) — how the schema builder represents a `nested` field. */
    private RelDataType arr(RelDataType elementRow) {
        return typeFactory.createArrayType(elementRow, -1);
    }

    /**
     * A uniform recursive nested element type of {@code depth} array levels: each level has a scalar
     * leaf {@code v} plus (for non-deepest levels) a {@code kids: ARRAY(ROW(next level))}. Returns the
     * ELEMENT ROW of the top level (i.e. what {@code comments}'s ARRAY wraps).
     */
    private RelDataType deepElement(int depth) {
        if (depth == 1) {
            return row("v", intType());
        }
        return row("v", intType(), "kids", arr(deepElement(depth - 1)));
    }

    // ── leaf / single level ────────────────────────────────────────────────────────────────────

    public void testSingleLevelLeafField() {
        RelDataType element = row("author", varchar(), "score", intType());
        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("author", element);
        assertEquals(Map.of("field", "author"), node);
    }

    public void testUnknownSingleFieldFallsBack() {
        RelDataType element = row("author", varchar());
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("missing", element));
    }

    // ── two levels ─────────────────────────────────────────────────────────────────────────────

    public void testTwoLevelDescent() {
        // element = ROW(author, replies: ARRAY(ROW(txt, likes)))
        RelDataType replies = arr(row("txt", varchar(), "likes", intType()));
        RelDataType element = row("author", varchar(), "replies", replies);

        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("replies.txt", element);
        assertEquals(Map.of("nested", "replies", "inner", Map.of("field", "txt")), node);
    }

    public void testTwoLevelUnknownInnerLeafFallsBack() {
        RelDataType replies = arr(row("txt", varchar()));
        RelDataType element = row("author", varchar(), "replies", replies);
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("replies.nope", element));
    }

    // ── arbitrary depth (3..7) ───────────────────────────────────────────────────────────────────

    public void testDepth3Descent() {
        RelDataType element = deepElement(3); // v + kids(ARRAY(ROW(v + kids(ARRAY(ROW(v))))))
        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("kids.kids.v", element);
        assertEquals(
            Map.of("nested", "kids", "inner", Map.of("nested", "kids", "inner", Map.of("field", "v"))),
            node
        );
    }

    public void testDepth7Descent() {
        RelDataType element = deepElement(7);
        // 6 "kids" hops then leaf "v".
        String key = "kids.kids.kids.kids.kids.kids.v";
        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest(key, element);

        // Verify the descent chain has exactly 6 nested wrappers ending in {"field":"v"}.
        Map<?, ?> cur = node;
        for (int level = 0; level < 6; level++) {
            assertEquals("level " + level + " must be a nested descent", "kids", cur.get("nested"));
            assertTrue("level " + level + " must carry inner", cur.get("inner") instanceof Map);
            cur = (Map<?, ?>) cur.get("inner");
        }
        assertEquals("innermost must be the leaf field", Map.of("field", "v"), cur);
    }

    public void testDepth7StopsShortAtIntermediateLeaf() {
        // Reach level 3's own scalar `v` (not the deepest) — 2 hops then leaf.
        RelDataType element = deepElement(7);
        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("kids.kids.v", element);
        assertEquals(
            Map.of("nested", "kids", "inner", Map.of("nested", "kids", "inner", Map.of("field", "v"))),
            node
        );
    }

    // ── fail-closed shapes (must return null so the caller falls back, never mis-evaluate) ───────

    public void testComparingWholeNestedArrayFallsBack() {
        // `comments.replies` (path terminates ON the array, not a scalar leaf) is not a per-element
        // scalar comparison — must fall back.
        RelDataType replies = arr(row("txt", varchar()));
        RelDataType element = row("author", varchar(), "replies", replies);
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("replies", element));
    }

    public void testDottingIntoPlainStructFallsBack() {
        // An intermediate field that is a plain ROW (object), not ARRAY(ROW) (nested), is not a
        // descent boundary for this per-element ∃ evaluator — fall back rather than mis-descend.
        RelDataType meta = row("city", varchar()); // plain struct, NOT wrapped in ARRAY
        RelDataType element = row("author", varchar(), "meta", meta);
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("meta.city", element));
    }

    public void testDottingThroughLeafFallsBack() {
        // `author.x` where author is a scalar — cannot descend through a non-array leaf.
        RelDataType element = row("author", varchar());
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("author.x", element));
    }

    public void testNonStructElementFallsBack() {
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("anything", varchar()));
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("a.b", null));
    }

    public void testMidChainUnknownSegmentFallsBack() {
        // replies exists, but the NEXT hop `foo` doesn't — fall back cleanly.
        RelDataType inner = arr(row("txt", varchar()));
        RelDataType replies = arr(row("txt", varchar(), "sub", inner));
        RelDataType element = row("replies", replies);
        assertNull(OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("replies.foo.txt", element));
    }

    // ── deep COMPARISON: the OP must be pushed to the leaf, INSIDE all ∃ descents ────────────────

    public void testDeepComparisonPushesOpToLeaf() {
        // members.name = 'alice' across 4 nested levels: the "=" must live at the leaf, wrapped by the
        // descent — NOT be an operand of a value-position nested node (which the UDF rejects).
        RelDataType inner = arr(row("name", varchar()));               // members: ARRAY(ROW(name))
        RelDataType teams = arr(row("members", inner));                // teams: ARRAY(ROW(members))
        RelDataType divisions = arr(row("teams", teams));              // divisions: ARRAY(ROW(teams))
        RelDataType element = row("divisions", divisions);             // top element: ROW(divisions)

        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveDeepComparisonForTest(
            "divisions.teams.members.name", element, "=", "alice"
        );
        Map<String, Object> expectedLeaf = Map.of("op", "=", "args", List.of(Map.of("field", "name"), Map.of("lit", "alice")));
        Map<String, Object> expected = Map.of("nested", "divisions", "inner",
            Map.of("nested", "teams", "inner",
                Map.of("nested", "members", "inner", expectedLeaf)));
        assertEquals(expected, node);
    }

    public void testDeepComparisonSingleLevelStillWraps() {
        // replies.likes > 5 : one descent, op at leaf.
        RelDataType element = row("replies", arr(row("likes", intType())));
        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveDeepComparisonForTest("replies.likes", element, ">", 5);
        Map<String, Object> expected = Map.of("nested", "replies", "inner",
            Map.of("op", ">", "args", List.of(Map.of("field", "likes"), Map.of("lit", 5))));
        assertEquals(expected, node);
    }

    // ── same-prefix conjunct FUSION (innermost-AND correlation) ──────────────────────────────────

    public void testFuseSameInnerArrayCorrelatesOnSameElement() {
        // tags.label='urgent' AND tags.priority=2 → the two descents share head "tags"; must fuse so
        // the AND is INSIDE the tags ∃ (one tag satisfies both), NOT two independent tag existentials.
        Map<String, Object> labelDescent = Map.of("nested", "tags", "inner",
            Map.of("op", "=", "args", List.of(Map.of("field", "label"), Map.of("lit", "urgent"))));
        Map<String, Object> prioDescent = Map.of("nested", "tags", "inner",
            Map.of("op", "=", "args", List.of(Map.of("field", "priority"), Map.of("lit", 2))));

        Map<String, Object> fused = OpenSearchNestedFieldRewriter.fuseByNestedPrefixForTest(List.of(labelDescent, prioDescent));

        // Expect: nested(tags, AND[label=urgent, priority=2])  (AND pushed inside the shared descent)
        assertEquals("tags", fused.get("nested"));
        assertTrue(fused.get("inner") instanceof Map);
        Map<?, ?> inner = (Map<?, ?>) fused.get("inner");
        assertEquals("AND", inner.get("op"));
        assertTrue(inner.get("args") instanceof List);
        assertEquals(2, ((List<?>) inner.get("args")).size());
    }

    public void testFuseDeepSharedPrefixThenSplit() {
        // members.tasks.tags.label='urgent' AND members.tasks.tags.priority=2: shared prefix
        // members→tasks→tags, AND fused at the tags level.
        java.util.function.Function<Map<String, Object>, Map<String, Object>> wrap = leaf ->
            Map.of("nested", "members", "inner",
                Map.of("nested", "tasks", "inner",
                    Map.of("nested", "tags", "inner", leaf)));
        Map<String, Object> a = wrap.apply(Map.of("op", "=", "args", List.of(Map.of("field", "label"), Map.of("lit", "urgent"))));
        Map<String, Object> b = wrap.apply(Map.of("op", "=", "args", List.of(Map.of("field", "priority"), Map.of("lit", 2))));

        Map<String, Object> fused = OpenSearchNestedFieldRewriter.fuseByNestedPrefixForTest(List.of(a, b));
        // Walk members→tasks→tags, then expect AND at the tags inner.
        Map<?, ?> cur = fused;
        for (String h : new String[] { "members", "tasks", "tags" }) {
            assertEquals(h, cur.get("nested"));
            cur = (Map<?, ?>) cur.get("inner");
        }
        assertEquals("AND", cur.get("op"));
        assertEquals(2, ((List<?>) cur.get("args")).size());
    }

    public void testFuseKeepsTopLevelLuceneNodeFlatButFusesDeepDescents() {
        // Regression for the megadeep TRAP E false positive: a query mixing a TOP-LEVEL keyword split
        // ({"lucene":0}, no {"nested"} head) with two DEEP same-path conjuncts (members.mname='al' AND
        // members.age>35) must (a) keep the {"lucene"} node as a flat top-level sibling so the executor
        // resolves it at outer-element grain, AND (b) FUSE the two deep members-descents so both correlate
        // on the SAME member. The prior code emitted a flat AND[lucene, descent, descent] whenever a split
        // fired, un-fusing the deep pair → different members satisfy each → false positive.
        Map<String, Object> luceneNode = Map.of("lucene", 0, "fallback",
            Map.of("op", "=", "args", List.of(Map.of("field", "oname"), Map.of("lit", "acme"))));
        java.util.function.Function<Map<String, Object>, Map<String, Object>> deep = leaf ->
            Map.of("nested", "divisions", "inner",
                Map.of("nested", "teams", "inner",
                    Map.of("nested", "members", "inner", leaf)));
        Map<String, Object> mname = deep.apply(Map.of("op", "=", "args", List.of(Map.of("field", "mname"), Map.of("lit", "al"))));
        Map<String, Object> age = deep.apply(Map.of("op", ">", "args", List.of(Map.of("field", "age"), Map.of("lit", 35))));

        Map<String, Object> fused = OpenSearchNestedFieldRewriter.fuseByNestedPrefixForTest(List.of(luceneNode, mname, age));

        // Top level: AND of exactly two args — the flat {"lucene"} node and ONE fused divisions-descent.
        assertEquals("AND", fused.get("op"));
        List<?> args = (List<?>) fused.get("args");
        assertEquals("lucene node must stay flat + deep pair must collapse to ONE descent", 2, args.size());
        Map<?, ?> luceneArg = null, descentArg = null;
        for (Object a : args) {
            Map<?, ?> m = (Map<?, ?>) a;
            if (m.containsKey("lucene")) luceneArg = m;
            else if ("divisions".equals(m.get("nested"))) descentArg = m;
        }
        assertNotNull("the {\"lucene\":0} node must remain a top-level sibling", luceneArg);
        assertEquals(0, luceneArg.get("lucene"));
        assertNotNull("the two deep descents must fuse into one divisions-descent", descentArg);
        // Walk divisions→teams→members and assert the AND is pushed inside the members ∃ (same-element).
        Map<?, ?> cur = descentArg;
        for (String h : new String[] { "divisions", "teams", "members" }) {
            assertEquals(h, cur.get("nested"));
            cur = (Map<?, ?>) cur.get("inner");
        }
        assertEquals("mname AND age must correlate on the SAME member", "AND", cur.get("op"));
        assertEquals(2, ((List<?>) cur.get("args")).size());
    }

    public void testDeepPrunePeerCarriesFullDottedPath() {
        // Phase B: a DEEP keyword conjunct members.mname='al' (leaf 3 levels down) must produce a
        // parent-grain prune peer NESTED_ANY_MATCH(arr, 'divisions.teams.members.mname','EQUALS','al')
        // carrying the FULL dotted descent+leaf path, so the serializer can wrap it in a deep-path
        // NestedQueryBuilder yielding a root-doc bitset for row-group pruning at any depth.
        RelDataType members = arr(row("mname", varchar(), "age", intType()));
        RelDataType teams = arr(row("tname", varchar(), "members", members));
        RelDataType divisions = arr(row("dname", varchar(), "teams", teams));
        RelDataType element = row("oname", varchar(), "divisions", divisions);
        RelDataType arrType = arr(element);
        RelDataType inputRow = typeFactory.builder().add("orgs", arrType).build();

        RexBuilder rb = new RexBuilder(typeFactory);
        RexNode arrRef = rb.makeInputRef(arrType, 0);
        // Build ITEM(ITEM(ITEM(ITEM($0,'divisions'),'teams'),'members'),'mname') = 'al'. The code under
        // test reads ITEM chains STRUCTURALLY (operator name + operands) and validates field types against
        // inputRow itself, so we give each synthetic ITEM/EQUALS an explicit return type to bypass Calcite's
        // ITEM return-type inference (which rejects an ARRAY indexed by a string key in isolation).
        RexNode chain = itemChain(rb, arrRef, "divisions", "teams", "members", "mname");
        RexNode eq = rb.makeCall(typeFactory.createSqlType(SqlTypeName.BOOLEAN),
            org.apache.calcite.sql.fun.SqlStdOperatorTable.EQUALS, List.of(chain, rb.makeLiteral("al")));

        RexNode peer = OpenSearchNestedFieldRewriter.tryDeepEqualityPrunePeerForTest(eq, 0, inputRow, rb);
        assertNotNull("a deep all-nested keyword path must yield a prune peer", peer);
        RexCall call = (RexCall) peer;
        assertEquals("NESTED_ANY_MATCH", call.getOperator().getName());
        assertEquals(4, call.getOperands().size());
        assertEquals("divisions.teams.members.mname", ((RexLiteral) call.getOperands().get(1)).getValueAs(String.class));
        assertEquals("EQUALS", ((RexLiteral) call.getOperands().get(2)).getValueAs(String.class));
        assertEquals("al", ((RexLiteral) call.getOperands().get(3)).getValueAs(String.class));

        // A SINGLE-level keyword must NOT be claimed by the deep peer (tryDirectEqualityRewrite owns it).
        RexNode shallow = rb.makeCall(typeFactory.createSqlType(SqlTypeName.BOOLEAN),
            org.apache.calcite.sql.fun.SqlStdOperatorTable.EQUALS, List.of(itemChain(rb, arrRef, "oname"), rb.makeLiteral("acme")));
        assertNull("single-level keyword must not produce a deep prune peer",
            OpenSearchNestedFieldRewriter.tryDeepEqualityPrunePeerForTest(shallow, 0, inputRow, rb));
    }

    // ── Phase C: DEEP element-grain child peer + {"lucene"} node at the deepest inner position ──────

    public void testDeepChildPeerCarriesFullDottedPath() {
        // Phase C: a DEEP keyword conjunct members.mname='al' (leaf 3 levels down) must produce a
        // CHILD-GRAIN peer NESTED_ANY_MATCH_CHILD(arr,'divisions.teams.members.mname','EQUALS','al',clauseIdx)
        // carrying the FULL dotted descent+leaf path (5 operands, clauseIdx as a trailing STRING literal), so
        // the child serializer can scope the term to the deepest _nested_path.
        RelDataType members = arr(row("mname", varchar(), "age", intType()));
        RelDataType teams = arr(row("tname", varchar(), "members", members));
        RelDataType divisions = arr(row("dname", varchar(), "teams", teams));
        RelDataType element = row("oname", varchar(), "divisions", divisions);
        RelDataType arrType = arr(element);
        RelDataType inputRow = typeFactory.builder().add("orgs", arrType).build();

        RexBuilder rb = new RexBuilder(typeFactory);
        RexNode arrRef = rb.makeInputRef(arrType, 0);
        RexNode chain = itemChain(rb, arrRef, "divisions", "teams", "members", "mname");
        RexNode eq = rb.makeCall(typeFactory.createSqlType(SqlTypeName.BOOLEAN),
            org.apache.calcite.sql.fun.SqlStdOperatorTable.EQUALS, List.of(chain, rb.makeLiteral("al")));

        RexNode peer = OpenSearchNestedFieldRewriter.tryDirectEqualityChildRewriteForTest(eq, 0, inputRow, rb, 2);
        assertNotNull("a deep all-nested keyword path must yield a child peer", peer);
        RexCall call = (RexCall) peer;
        assertEquals("NESTED_ANY_MATCH_CHILD", call.getOperator().getName());
        assertEquals("child peer keeps arity 5 (clauseIdx in args[4])", 5, call.getOperands().size());
        assertEquals("divisions.teams.members.mname", ((RexLiteral) call.getOperands().get(1)).getValueAs(String.class));
        assertEquals("EQUALS", ((RexLiteral) call.getOperands().get(2)).getValueAs(String.class));
        assertEquals("al", ((RexLiteral) call.getOperands().get(3)).getValueAs(String.class));
        assertEquals("clauseIdx is a trailing STRING literal", "2", ((RexLiteral) call.getOperands().get(4)).getValueAs(String.class));

        // A SINGLE-level keyword must STILL be claimed at element grain — leaf-only field, arity 5.
        RexNode shallow = rb.makeCall(typeFactory.createSqlType(SqlTypeName.BOOLEAN),
            org.apache.calcite.sql.fun.SqlStdOperatorTable.EQUALS, List.of(itemChain(rb, arrRef, "oname"), rb.makeLiteral("acme")));
        RexNode shallowPeer = OpenSearchNestedFieldRewriter.tryDirectEqualityChildRewriteForTest(shallow, 0, inputRow, rb, 0);
        assertNotNull("single-level keyword must still produce a child peer (leaf-only field)", shallowPeer);
        assertEquals("oname", ((RexLiteral) ((RexCall) shallowPeer).getOperands().get(1)).getValueAs(String.class));
    }

    public void testLuceneNodeLandsAtDeepestInnerPosition() {
        // Phase C: replacing a DEEP conjunct's LEAF with {"lucene":i} must keep the {"nested"} heads ABOVE it,
        // so the Rust json_lucene_paths recovers ["divisions","teams","members"] by walking nested→nested→
        // nested→lucene. Wrapping the WHOLE descent at top level (the old single-level behavior) would strand
        // the {"lucene"} node above the heads → empty recovered path → wrong grain.
        Map<String, Object> leaf = Map.of("op", "=", "args", List.of(Map.of("field", "mname"), Map.of("lit", "al")));
        Map<String, Object> descent = Map.of("nested", "divisions", "inner",
            Map.of("nested", "teams", "inner",
                Map.of("nested", "members", "inner", leaf)));

        Map<String, Object> wrapped = OpenSearchNestedFieldRewriter.wrapLeafWithLuceneNodeForTest(descent, 3);

        // Walk divisions→teams→members; the {"lucene"} node must sit at the DEEPEST inner position.
        Map<?, ?> cur = wrapped;
        for (String h : new String[] { "divisions", "teams", "members" }) {
            assertEquals("nested head preserved above the lucene node", h, cur.get("nested"));
            assertTrue("each level carries an inner", cur.get("inner") instanceof Map);
            cur = (Map<?, ?>) cur.get("inner");
        }
        assertEquals("the lucene node sits at the deepest inner position", 3, cur.get("lucene"));
        assertEquals("its fallback is ONLY the leaf comparison", leaf, cur.get("fallback"));
        assertFalse("no {\"nested\"} head on the lucene node itself (path comes from the heads above)", cur.containsKey("nested"));
    }

    public void testLuceneNodeSingleLevelStaysFlat() {
        // A single-level conjunct (bare leaf comparison, no {"nested"} head) degenerates to a top-level
        // {"lucene":i,"fallback":<leaf>} — the original single-level bridge; recovered path is empty.
        Map<String, Object> leaf = Map.of("op", "=", "args", List.of(Map.of("field", "author"), Map.of("lit", "alice")));
        Map<String, Object> wrapped = OpenSearchNestedFieldRewriter.wrapLeafWithLuceneNodeForTest(leaf, 0);
        assertEquals(0, wrapped.get("lucene"));
        assertEquals(leaf, wrapped.get("fallback"));
        assertFalse("no nested head at top for a single-level clause", wrapped.containsKey("nested"));
    }

    /** Helper: build a left-deep ITEM chain ITEM(...ITEM($ref,seg0)...,segN) with explicit VARCHAR result
     * types so Calcite's ITEM return-type inference (which can't type an ARRAY indexed by a string) is bypassed. */
    private RexNode itemChain(RexBuilder rb, RexNode ref, String... segs) {
        RexNode cur = ref;
        for (String s : segs) {
            cur = rb.makeCall(varchar(), org.apache.calcite.sql.fun.SqlStdOperatorTable.ITEM, List.of(cur, rb.makeLiteral(s)));
        }
        return cur;
    }

    public void testFuseDistinctPrefixesStayAnded() {
        // Different inner arrays (tags vs offices) must NOT be fused — they AND at parent grain.
        Map<String, Object> tags = Map.of("nested", "tags", "inner", Map.of("field", "label"));
        Map<String, Object> offices = Map.of("nested", "offices", "inner", Map.of("field", "city"));
        Map<String, Object> fused = OpenSearchNestedFieldRewriter.fuseByNestedPrefixForTest(List.of(tags, offices));
        assertEquals("AND", fused.get("op"));
        assertEquals(2, ((List<?>) fused.get("args")).size());
    }

    public void testFuseMixedNestedAndFlatLeaf() {
        // A same-array descent (tags.label) AND a flat leaf on the outer element (a scalar field): the
        // flat leaf stays at top, the descent stays a descent, AND-ed.
        Map<String, Object> descent = Map.of("nested", "tags", "inner", Map.of("field", "label"));
        Map<String, Object> flat = Map.of("op", ">", "args", List.of(Map.of("field", "count"), Map.of("lit", 5)));
        Map<String, Object> fused = OpenSearchNestedFieldRewriter.fuseByNestedPrefixForTest(List.of(descent, flat));
        assertEquals("AND", fused.get("op"));
        assertEquals(2, ((List<?>) fused.get("args")).size());
    }

    // A realistic mixed case: 3-level with the middle level carrying multiple fields.
    public void testRealisticThreeLevelMixed() {
        RelDataType tags = arr(row("label", varchar(), "priority", intType()));
        RelDataType tasks = arr(row("task", varchar(), "tags", tags));
        RelDataType element = row("name", varchar(), "tasks", tasks);

        Map<String, Object> node = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("tasks.tags.label", element);
        assertEquals(
            Map.of("nested", "tasks", "inner", Map.of("nested", "tags", "inner", Map.of("field", "label"))),
            node
        );
        // sibling scalar at the middle level
        Map<String, Object> mid = OpenSearchNestedFieldRewriter.resolveNestedKeyForTest("tasks.task", element);
        assertEquals(Map.of("nested", "tasks", "inner", Map.of("field", "task")), mid);
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import io.substrait.proto.Expression;
import io.substrait.proto.FunctionArgument;
import org.opensearch.test.OpenSearchTestCase;

/**
 * Unit tests for {@link NestedParentDedupRewriter#shiftAllFieldRefs} — the field-reference renumbering applied
 * after {@code __row_id__} is inserted into a projection/filter. Correctness here is essential: an off-by-one in
 * ANY expression variant silently corrupts filter/projection results in a nested aggregation. The tests build
 * Substrait {@code Expression} protos directly and assert every ref {@code >= threshold} is bumped by exactly one
 * across bare refs, scalar functions, casts, CASE ({@code IfThen}), {@code SWITCH}, and {@code IN}-list forms —
 * and that opaque variants (window functions) carrying a ref {@code >= threshold} trigger a bail (null) rather
 * than a wrong shift, while ones below the threshold pass through unchanged.
 */
public class NestedParentDedupRewriterTests extends OpenSearchTestCase {

    private static Expression ref(int idx) {
        return NestedParentDedupRewriter.fieldRef(idx);
    }

    private static int idxOf(Expression e) {
        Integer i = NestedParentDedupRewriter.fieldIndexOf(e);
        assertNotNull("expected a bare field ref", i);
        return i;
    }

    private static Expression literal(long v) {
        return Expression.newBuilder().setLiteral(Expression.Literal.newBuilder().setI64(v).build()).build();
    }

    private static Expression scalar(int fnRef, Expression... args) {
        Expression.ScalarFunction.Builder sf = Expression.ScalarFunction.newBuilder().setFunctionReference(fnRef);
        for (Expression a : args) {
            sf.addArguments(FunctionArgument.newBuilder().setValue(a).build());
        }
        return Expression.newBuilder().setScalarFunction(sf.build()).build();
    }

    // ───────────────────────── bare refs ─────────────────────────

    public void testBareRefAtOrAboveThresholdShifts() {
        // threshold 5: index 7 → 8, index 5 → 6.
        assertEquals(8, idxOf(NestedParentDedupRewriter.shiftAllFieldRefs(ref(7), 5)));
        assertEquals(6, idxOf(NestedParentDedupRewriter.shiftAllFieldRefs(ref(5), 5)));
    }

    public void testBareRefBelowThresholdUnchanged() {
        assertEquals(4, idxOf(NestedParentDedupRewriter.shiftAllFieldRefs(ref(4), 5)));
        assertEquals(0, idxOf(NestedParentDedupRewriter.shiftAllFieldRefs(ref(0), 5)));
    }

    public void testLiteralUnchanged() {
        Expression lit = literal(42);
        assertEquals(lit, NestedParentDedupRewriter.shiftAllFieldRefs(lit, 0));
    }

    // ───────────────────────── scalar function (gt($7,4), and(...)) ─────────────────────────

    public void testScalarFunctionArgsShifted() {
        // gt($7, literal) with threshold 5 → gt($8, literal).
        Expression e = scalar(1, ref(7), literal(4));
        Expression shifted = NestedParentDedupRewriter.shiftAllFieldRefs(e, 5);
        Expression.ScalarFunction sf = shifted.getScalarFunction();
        assertEquals(8, idxOf(sf.getArguments(0).getValue()));
        assertTrue("literal arg preserved", sf.getArguments(1).getValue().hasLiteral());
    }

    public void testNestedScalarFunctionShifted() {
        // and(gt($7,x), lt($2,y)) threshold 5 → and(gt($8,x), lt($2,y)) — only $7 crosses.
        Expression inner1 = scalar(1, ref(7), literal(1));
        Expression inner2 = scalar(2, ref(2), literal(9));
        Expression and = scalar(3, inner1, inner2);
        Expression shifted = NestedParentDedupRewriter.shiftAllFieldRefs(and, 5);
        Expression.ScalarFunction top = shifted.getScalarFunction();
        assertEquals(8, idxOf(top.getArguments(0).getValue().getScalarFunction().getArguments(0).getValue()));
        assertEquals(2, idxOf(top.getArguments(1).getValue().getScalarFunction().getArguments(0).getValue()));
    }

    // ───────────────────────── cast ─────────────────────────

    public void testCastInputShifted() {
        Expression cast = Expression.newBuilder()
            .setCast(Expression.Cast.newBuilder().setInput(ref(6)).build())
            .build();
        Expression shifted = NestedParentDedupRewriter.shiftAllFieldRefs(cast, 5);
        assertEquals(7, idxOf(shifted.getCast().getInput()));
    }

    // ───────────────────────── IfThen (CASE) ─────────────────────────

    public void testIfThenShiftsConditionThenElse() {
        // CASE WHEN gt($7,0) THEN $8 ELSE $2 — threshold 5 → WHEN gt($8,0) THEN $9 ELSE $2.
        Expression ifThen = Expression.newBuilder()
            .setIfThen(
                Expression.IfThen.newBuilder()
                    .addIfs(
                        Expression.IfThen.IfClause.newBuilder().setIf(scalar(1, ref(7), literal(0))).setThen(ref(8)).build()
                    )
                    .setElse(ref(2))
                    .build()
            )
            .build();
        Expression shifted = NestedParentDedupRewriter.shiftAllFieldRefs(ifThen, 5);
        Expression.IfThen it = shifted.getIfThen();
        assertEquals(8, idxOf(it.getIfs(0).getIf().getScalarFunction().getArguments(0).getValue()));
        assertEquals(9, idxOf(it.getIfs(0).getThen()));
        assertEquals(2, idxOf(it.getElse()));
    }

    // ───────────────────────── SingularOrList (IN) ─────────────────────────

    public void testSingularOrListShiftsValueAndOptions() {
        // $7 IN ($8, literal) threshold 5 → $8 IN ($9, literal).
        Expression in = Expression.newBuilder()
            .setSingularOrList(
                Expression.SingularOrList.newBuilder().setValue(ref(7)).addOptions(ref(8)).addOptions(literal(3)).build()
            )
            .build();
        Expression shifted = NestedParentDedupRewriter.shiftAllFieldRefs(in, 5);
        Expression.SingularOrList sol = shifted.getSingularOrList();
        assertEquals(8, idxOf(sol.getValue()));
        assertEquals(9, idxOf(sol.getOptions(0)));
        assertTrue(sol.getOptions(1).hasLiteral());
    }

    // ───────────────────────── bail on unshiftable variant ─────────────────────────

    public void testWindowFunctionWithRefAtOrAboveThresholdBails() {
        // A window function referencing $7 (>= threshold 5) can't be safely shifted → bail (null).
        Expression window = Expression.newBuilder()
            .setWindowFunction(
                Expression.WindowFunction.newBuilder()
                    .setFunctionReference(1)
                    .addArguments(FunctionArgument.newBuilder().setValue(ref(7)).build())
                    .build()
            )
            .build();
        assertNull("must bail to original plan", NestedParentDedupRewriter.shiftAllFieldRefsOrNull(window, 5));
    }

    public void testWindowFunctionWithRefBelowThresholdPassesThrough() {
        // A window function referencing only $2 (< threshold 5) is unaffected → returned unchanged (no bail).
        Expression window = Expression.newBuilder()
            .setWindowFunction(
                Expression.WindowFunction.newBuilder()
                    .setFunctionReference(1)
                    .addArguments(FunctionArgument.newBuilder().setValue(ref(2)).build())
                    .build()
            )
            .build();
        Expression result = NestedParentDedupRewriter.shiftAllFieldRefsOrNull(window, 5);
        assertNotNull("below-threshold window is safe", result);
        assertEquals(window, result);
    }

    // ───────────────────────── containsFieldRefAtOrAbove ─────────────────────────

    public void testContainsFieldRefAtOrAbove() {
        assertTrue(NestedParentDedupRewriter.containsFieldRefAtOrAbove(ref(5), 5));
        assertFalse(NestedParentDedupRewriter.containsFieldRefAtOrAbove(ref(4), 5));
        assertTrue(NestedParentDedupRewriter.containsFieldRefAtOrAbove(scalar(1, ref(2), ref(9)), 5)); // $9 deep
        assertFalse(NestedParentDedupRewriter.containsFieldRefAtOrAbove(scalar(1, ref(2), literal(9)), 5));
        assertFalse("literal has no ref", NestedParentDedupRewriter.containsFieldRefAtOrAbove(literal(9), 0));
    }
}

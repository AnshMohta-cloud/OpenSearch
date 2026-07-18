/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Uncollect;
import org.apache.calcite.rel.logical.LogicalCorrelate;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalValues;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * [NESTED] Generic Calcite rewrite that turns references to nested fields into a real UNNEST plan,
 * for ARBITRARY queries — no per-query hardcoding. This is the production direction (behind the
 * {@code nested.generic_rewrite} flag) replacing the hand-authored {@code N1Descriptor} registry.
 *
 * <p><b>What it detects.</b> A nested field reference {@code comments.author} is represented by
 * Calcite as {@code ITEM($arrayCol, 'field')} where {@code $arrayCol} is an {@code ARRAY(ROW(...))}
 * column (see {@code OpenSearchSchemaBuilder} which exposes {@code nested} mappings that way). Such
 * {@code ITEM} calls can appear inside a {@link LogicalProject}'s expressions ({@code | fields
 * comments.author}) or inside a {@link LogicalFilter}'s condition ({@code | where comments.score>4}),
 * and — since {@code | stats avg(comments.score)} is an {@code Aggregate} over a {@code Project} that
 * contains the {@code ITEM} — handling Project + Filter also covers aggregates.
 *
 * <p><b>What it does.</b> Walking the tree, at each Project/Filter whose expressions reference an
 * array column via {@code ITEM}, it injects the textbook Calcite UNNEST shape beneath that node:
 * <pre>
 *   LogicalCorrelate(INNER, requiredColumns={arrayCol})
 *     ├─ &lt;original input&gt;                              (all original columns, indices UNCHANGED)
 *     └─ Uncollect( Project($cor0.arrayCol, LogicalValues.oneRow) )   (struct fields, APPENDED)
 * </pre>
 * and rewrites each {@code ITEM($arrayCol,'f')} to a plain {@link RexInputRef} of the appended
 * unnested column. Because the correlate keeps the left (original) columns first and appends the
 * exploded struct fields, <b>every original column index is preserved</b> — so operators above the
 * rewritten node are unaffected and the transform composes cleanly across the whole tree.
 *
 * <p>For a {@link LogicalFilter}, the appended unnested columns are projected away again above the
 * filter so the row type is restored to the parent's shape (returning parent rows). NOTE: parent
 * de-duplication (a parent with two matching children currently appears twice) and multi-array /
 * same-child correlation are the remaining runtime gaps — see the package README / task list; those
 * shapes fall back to the hardcoded path when the flag is off.
 *
 * @opensearch.internal
 */
public final class OpenSearchNestedFieldRewriter {

    private static final Logger LOGGER = LogManager.getLogger(OpenSearchNestedFieldRewriter.class);

    private OpenSearchNestedFieldRewriter() {}

    /**
     * Rewrites the tree so that every {@code ITEM}-on-array reference becomes a plain column produced
     * by an injected UNNEST. Returns the original tree unchanged if there are no nested references.
     */
    public static RelNode rewrite(RelNode root) {
        RelNode result = root.accept(new NestedShuttle());
        if (result != root) {
            LOGGER.info("[NESTED] rewrite injected UNNEST. New plan:\n{}", RelOptUtil.toString(result));
        }
        return result;
    }

    /**
     * Bottom-up shuttle: children are rewritten first (so a node always sees an already-unnested
     * input where applicable), then the node itself is rewritten if it carries {@code ITEM} refs.
     */
    private static final class NestedShuttle extends RelShuttleImpl {
        @Override
        public RelNode visit(LogicalProject project) {
            LogicalProject visited = (LogicalProject) super.visitChildren(project);
            return rewriteProject(visited);
        }

        @Override
        public RelNode visit(LogicalFilter filter) {
            LogicalFilter visited = (LogicalFilter) super.visitChildren(filter);
            return rewriteFilter(visited);
        }
    }

    // ---- Project: rewrite ITEM refs in the projected expressions -------------------------------

    private static RelNode rewriteProject(LogicalProject project) {
        RelNode input = project.getInput();
        int arrayCol = firstArrayColReferenced(project.getProjects(), input.getRowType());
        if (arrayCol < 0) {
            return project;
        }
        RelOptCluster cluster = project.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();
        UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
        if (u == null) {
            return project;
        }
        ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.correlate.getRowType());
        List<RexNode> newExprs = new ArrayList<>(project.getProjects().size());
        for (RexNode e : project.getProjects()) {
            newExprs.add(e.accept(shuttle));
        }
        return LogicalProject.create(u.correlate, List.of(), newExprs, project.getRowType().getFieldNames());
    }

    // ---- Filter: rewrite ITEM refs in the condition, then restore the parent row type ----------

    private static RelNode rewriteFilter(LogicalFilter filter) {
        RelNode input = filter.getInput();
        int arrayCol = firstArrayColReferenced(List.of(filter.getCondition()), input.getRowType());
        if (arrayCol < 0) {
            return filter;
        }
        RelOptCluster cluster = filter.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();
        int originalColCount = input.getRowType().getFieldCount();
        UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
        if (u == null) {
            return filter;
        }
        ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.correlate.getRowType());
        RexNode newCondition = filter.getCondition().accept(shuttle);
        RelNode newFilter = LogicalFilter.create(u.correlate, newCondition);

        // Restore the parent row type: project only the original columns (drop the appended unnested
        // struct fields). Original indices are unchanged, so this is a straight 0..originalColCount-1
        // projection. This yields parent rows (matching the "WHERE on nested returns the parent doc"
        // semantics). Parent de-duplication is a known follow-up (see class javadoc).
        List<RexNode> passthrough = new ArrayList<>(originalColCount);
        List<String> names = new ArrayList<>(originalColCount);
        List<RelDataTypeField> corrFields = u.correlate.getRowType().getFieldList();
        for (int i = 0; i < originalColCount; i++) {
            passthrough.add(rexBuilder.makeInputRef(corrFields.get(i).getType(), i));
            names.add(corrFields.get(i).getName());
        }
        return LogicalProject.create(newFilter, List.of(), passthrough, names);
    }

    // ---- Shared: build Correlate(input, Uncollect(array)) appending the struct fields ----------

    /** Result of injecting an unnest: the new Correlate rel + the index where unnested fields begin. */
    private record UnnestResult(LogicalCorrelate correlate, int unnestedFieldIndex, Map<String, Integer> fieldToIndex) {}

    /**
     * Injects {@code Correlate(input, Uncollect(Project($cor0.arrayCol, oneRow)))}. The correlate's
     * output is {@code [original cols..., unnested struct fields...]} — original indices preserved,
     * struct fields appended starting at {@code input.fieldCount}.
     */
    private static UnnestResult injectUnnest(RelNode input, int arrayCol, RelOptCluster cluster, RexBuilder rexBuilder) {
        RelDataType inputRowType = input.getRowType();
        RelDataTypeField arrayField = inputRowType.getFieldList().get(arrayCol);
        RelDataType elementType = arrayField.getType().getComponentType();
        if (elementType == null || !elementType.isStruct()) {
            LOGGER.warn("[NESTED] array column '{}' is not ARRAY(ROW) — skipping unnest", arrayField.getName());
            return null;
        }

        CorrelationId correlId = cluster.createCorrel();
        RexNode correlVar = rexBuilder.makeCorrel(inputRowType, correlId);
        RexNode correlArrayAccess = rexBuilder.makeFieldAccess(correlVar, arrayCol);

        RelNode oneRow = LogicalValues.createOneRow(cluster);
        RelNode rightProject = LogicalProject.create(oneRow, List.of(), List.of(correlArrayAccess), List.of(arrayField.getName()));
        RelNode uncollect = Uncollect.create(rightProject.getTraitSet(), rightProject, false, List.of());

        LogicalCorrelate correlate = LogicalCorrelate.create(
            input,
            uncollect,
            List.of(),
            correlId,
            ImmutableBitSet.of(arrayCol),
            JoinRelType.INNER
        );

        int originalColCount = inputRowType.getFieldCount();
        Map<String, Integer> fieldToIndex = new LinkedHashMap<>();
        List<RelDataTypeField> corrFields = correlate.getRowType().getFieldList();
        for (int i = originalColCount; i < corrFields.size(); i++) {
            fieldToIndex.put(corrFields.get(i).getName(), i);
        }
        LOGGER.info(
            "[NESTED] injected UNNEST on array col '{}' (idx {}); unnested fields {} at indices {}..{}",
            arrayField.getName(),
            arrayCol,
            fieldToIndex.keySet(),
            originalColCount,
            corrFields.size() - 1
        );
        return new UnnestResult(correlate, originalColCount, fieldToIndex);
    }

    // ---- ITEM detection + rewriting ------------------------------------------------------------

    /**
     * Finds the first array-column index referenced by an {@code ITEM($arrayCol,'field')} anywhere
     * within the given expressions, or -1 if none. (Single-array per rewrite step for now; multiple
     * distinct arrays in one node is a follow-up — see class javadoc.)
     */
    private static int firstArrayColReferenced(List<RexNode> exprs, RelDataType inputRowType) {
        ItemFinder finder = new ItemFinder(inputRowType);
        for (RexNode e : exprs) {
            e.accept(finder);
        }
        return finder.arrayCol;
    }

    /** Walks an expression tree recording the array-column index of the first {@code ITEM}-on-array. */
    private static final class ItemFinder extends RexShuttle {
        private final RelDataType inputRowType;
        private int arrayCol = -1;

        ItemFinder(RelDataType inputRowType) {
            this.inputRowType = inputRowType;
        }

        @Override
        public RexNode visitCall(RexCall call) {
            if (arrayCol < 0) {
                int c = itemArrayCol(call, inputRowType);
                if (c >= 0) {
                    arrayCol = c;
                }
            }
            return super.visitCall(call);
        }
    }

    /**
     * Replaces every {@code ITEM($arrayCol,'field')} (for the target array column) with a plain
     * {@link RexInputRef} to the appended unnested column of that field.
     */
    private static final class ItemRewriteShuttle extends RexShuttle {
        private final int arrayCol;
        private final Map<String, Integer> fieldToIndex;
        private final RexBuilder rexBuilder;
        private final RelDataType correlateRowType;

        ItemRewriteShuttle(int arrayCol, int unnestedStartIdx, RexBuilder rexBuilder, RelDataType correlateRowType) {
            this.arrayCol = arrayCol;
            this.rexBuilder = rexBuilder;
            this.correlateRowType = correlateRowType;
            this.fieldToIndex = new LinkedHashMap<>();
            for (int i = unnestedStartIdx; i < correlateRowType.getFieldCount(); i++) {
                fieldToIndex.put(correlateRowType.getFieldList().get(i).getName(), i);
            }
        }

        @Override
        public RexNode visitCall(RexCall call) {
            if ("ITEM".equals(call.getOperator().getName()) && call.getOperands().size() == 2) {
                RexNode arrayRef = call.getOperands().get(0);
                RexNode fieldNode = call.getOperands().get(1);
                if (arrayRef instanceof RexInputRef ref
                    && ref.getIndex() == arrayCol
                    && fieldNode instanceof RexLiteral lit
                    && lit.getTypeName() == SqlTypeName.CHAR) {
                    String field = lit.getValueAs(String.class);
                    Integer idx = fieldToIndex.get(field);
                    if (idx != null) {
                        return rexBuilder.makeInputRef(correlateRowType.getFieldList().get(idx).getType(), idx);
                    }
                }
            }
            return super.visitCall(call);
        }
    }

    /** If {@code call} is {@code ITEM($N,'field')} with {@code $N} an ARRAY column, returns N; else -1. */
    private static int itemArrayCol(RexCall call, RelDataType inputRowType) {
        if (!"ITEM".equals(call.getOperator().getName()) || call.getOperands().size() != 2) {
            return -1;
        }
        RexNode arrayRef = call.getOperands().get(0);
        RexNode fieldNode = call.getOperands().get(1);
        if (!(arrayRef instanceof RexInputRef ref)) {
            return -1;
        }
        if (!(fieldNode instanceof RexLiteral lit) || lit.getTypeName() != SqlTypeName.CHAR) {
            return -1;
        }
        int colIndex = ref.getIndex();
        if (colIndex >= inputRowType.getFieldCount()) {
            return -1;
        }
        RelDataType colType = inputRowType.getFieldList().get(colIndex).getType();
        return colType.getSqlTypeName() == SqlTypeName.ARRAY ? colIndex : -1;
    }
}

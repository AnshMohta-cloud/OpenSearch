/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rel;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Uncollect;
import org.apache.calcite.rel.logical.LogicalCorrelate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalValues;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.List;

/**
 * Backend-neutral "expand this array path into child rows, keep parent identity" operator — WHAT
 * to do, not HOW to do it. Emitted by {@code OpenSearchNestedFieldRewriter}'s {@code injectUnnest}
 * in place of a hand-built {@code Correlate(left, Uncollect(...))} pair, for the case where an
 * array-column expansion is a genuine grain change (the output IS per-child — e.g. PPL {@code stats
 * count() by comments.author} grouping on a nested field, or a filter predicate shape too irregular
 * for the {@code NESTED_ANY_MATCH_EXPR} per-element lambda) — see the rewriter's {@code
 * rewriteAggregateInputProject} Aggregate-guard for the grain-change trigger, and {@code
 * rewriteFilter}'s fallback for the filter-shape trigger.
 *
 * <p>Marked into {@link OpenSearchNestedScope} by {@code OpenSearchNestedScopeRule} during the same
 * bottom-up marking pass that marks every other operator — replacing the old hardcoded
 * {@code List.of("datafusion")} pinning ({@code OpenSearchCorrelateRule}/{@code
 * OpenSearchUncollectRule}) with a real {@link org.opensearch.analytics.spi.EngineCapability#NESTED_SCOPE}
 * capability lookup, so CBO can route this subtree to any backend that declares it — not just
 * DataFusion by construction.
 *
 * <p><b>Not used for per-element predicates.</b> A nested filter like {@code where comments.score >
 * 4} never changes row count and is represented as {@code NESTED_ANY_MATCH_EXPR} instead — this node
 * exists only for the genuine-unnest case.
 *
 * <p>Row type is {@code [original columns..., exploded struct fields...]} — struct fields appended
 * starting at {@code input.getRowType().getFieldCount()} — identical shape to the {@code
 * Correlate(left, Uncollect(...))} this node replaces, computed by actually building that shape once
 * (see {@link #buildEquivalentCorrelate}) and reading its derived row type off Calcite's own {@code
 * Correlate#deriveRowType}, so any future Calcite change to join-row-type dedup/naming is inherited
 * automatically. This means every existing {@code ItemRewriteShuttle}/index-based consumer above this
 * node needs no changes.
 *
 * @opensearch.internal
 */
public class LogicalNestedScope extends SingleRel {

    private final int arrayColumnIndex;

    public LogicalNestedScope(RelOptCluster cluster, RelTraitSet traitSet, RelNode input, int arrayColumnIndex) {
        super(cluster, traitSet, input);
        this.arrayColumnIndex = arrayColumnIndex;
        this.rowType = buildEquivalentCorrelate(input, arrayColumnIndex).getRowType();
    }

    public static LogicalNestedScope create(RelNode input, int arrayColumnIndex) {
        RelOptCluster cluster = input.getCluster();
        RelTraitSet traitSet = cluster.traitSetOf(org.apache.calcite.plan.Convention.NONE);
        return new LogicalNestedScope(cluster, traitSet, input, arrayColumnIndex);
    }

    /** Index (into this node's INPUT row type) of the array column being expanded. */
    public int getArrayColumnIndex() {
        return arrayColumnIndex;
    }

    /** Name of the array column being expanded — the nested path (e.g. {@code "comments"}). */
    public String getPath() {
        return getInput().getRowType().getFieldList().get(arrayColumnIndex).getName();
    }

    @Override
    public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
        return new LogicalNestedScope(getCluster(), traitSet, sole(inputs), arrayColumnIndex);
    }

    @Override
    public RelWriter explainTerms(RelWriter pw) {
        return super.explainTerms(pw).item("path", getPath());
    }

    /**
     * Rebuilds the equivalent {@code Correlate(input, Uncollect(Project($cor0.arrayCol, oneRow)))}
     * shape — used both to derive this node's row type (constructor) and by {@code
     * OpenSearchNestedScope#stripAnnotations} to hand backends whose {@code FragmentConvertor}
     * pattern-matches that Correlate/Uncollect structure (today: DataFusion's {@code tryEmitUnnest})
     * an unchanged input. Package-visible so {@link OpenSearchNestedScope} can reuse it directly.
     */
    static LogicalCorrelate buildEquivalentCorrelate(RelNode input, int arrayColumnIndex) {
        RelOptCluster cluster = input.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();
        RelDataType inputRowType = input.getRowType();
        RelDataTypeField arrayField = inputRowType.getFieldList().get(arrayColumnIndex);
        RelDataType elementType = arrayField.getType().getComponentType();
        if (elementType == null || !elementType.isStruct()) {
            throw new IllegalArgumentException(
                "LogicalNestedScope: array column '" + arrayField.getName() + "' is not ARRAY(ROW)"
            );
        }

        CorrelationId correlId = cluster.createCorrel();
        RexNode correlVar = rexBuilder.makeCorrel(inputRowType, correlId);
        RexNode correlArrayAccess = rexBuilder.makeFieldAccess(correlVar, arrayColumnIndex);

        RelNode oneRow = LogicalValues.createOneRow(cluster);
        RelNode rightProject = LogicalProject.create(oneRow, List.of(), List.of(correlArrayAccess), List.of(arrayField.getName()));
        RelNode uncollect = Uncollect.create(rightProject.getTraitSet(), rightProject, false, List.of());

        return LogicalCorrelate.create(input, uncollect, List.of(), correlId, ImmutableBitSet.of(arrayColumnIndex), JoinRelType.INNER);
    }
}

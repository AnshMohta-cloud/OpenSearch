/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rel;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.spi.FieldStorageInfo;

import java.util.ArrayList;
import java.util.List;

/**
 * Marked {@link LogicalNestedScope}: carries {@code viableBackends} computed by {@code
 * OpenSearchNestedScopeRule} from a real {@link org.opensearch.analytics.spi.EngineCapability#NESTED_SCOPE}
 * capability lookup (intersected with the child's viable backends) — see {@link LogicalNestedScope}'s
 * javadoc for the operator's semantics and row-type contract.
 *
 * <p>Today that capability lookup resolves to DataFusion only: it lowers this node to the same
 * {@code unnest_reshape} {@code ExtensionSingleRel} it already emits for {@code Correlate+Uncollect}
 * (see {@code DataFusionFragmentConvertor#tryEmitUnnest}, triggered off the {@code Correlate}/{@code
 * Uncollect} shape {@link #stripAnnotations} hands it — unchanged). Lucene registers no such
 * capability yet — no bucket/group-by execution path exists for nested children there today — so it
 * stays out of {@code viableBackends} until that lands; CBO can then route to either backend without
 * any change to this node, its marking rule, or the rewriter that emits {@link LogicalNestedScope}.
 *
 * @opensearch.internal
 */
public class OpenSearchNestedScope extends SingleRel implements OpenSearchRelNode {

    private final int arrayColumnIndex;
    private final List<String> viableBackends;

    public OpenSearchNestedScope(RelOptCluster cluster, RelTraitSet traitSet, RelNode input, int arrayColumnIndex, List<String> viableBackends) {
        super(cluster, traitSet, input);
        this.arrayColumnIndex = arrayColumnIndex;
        this.viableBackends = viableBackends;
        this.rowType = LogicalNestedScope.buildEquivalentCorrelate(input, arrayColumnIndex).getRowType();
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
    public List<String> getViableBackends() {
        return viableBackends;
    }

    @Override
    public List<FieldStorageInfo> getOutputFieldStorage() {
        RelNode unwrapped = RelNodeUtils.unwrapHep(getInput());
        List<FieldStorageInfo> result = new ArrayList<>();
        if (unwrapped instanceof OpenSearchRelNode os) {
            result.addAll(os.getOutputFieldStorage());
        } else {
            for (RelDataTypeField field : unwrapped.getRowType().getFieldList()) {
                result.add(FieldStorageInfo.derivedColumn(field.getName(), field.getType().getSqlTypeName()));
            }
        }
        // Exploded child struct fields are synthetic — same treatment OpenSearchUncollect gives them.
        List<RelDataTypeField> fields = getRowType().getFieldList();
        for (int i = result.size(); i < fields.size(); i++) {
            result.add(FieldStorageInfo.derivedColumn(fields.get(i).getName(), fields.get(i).getType().getSqlTypeName()));
        }
        return result;
    }

    @Override
    public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
        return new OpenSearchNestedScope(getCluster(), traitSet, sole(inputs), arrayColumnIndex, viableBackends);
    }

    @Override
    public RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        return planner.getCostFactory().makeTinyCost();
    }

    @Override
    public RelWriter explainTerms(RelWriter pw) {
        return super.explainTerms(pw).item("path", getPath()).item("viableBackends", viableBackends);
    }

    @Override
    public RelNode copyResolved(String backend, List<RelNode> children, List<OperatorAnnotation> resolvedAnnotations) {
        return new OpenSearchNestedScope(getCluster(), getTraitSet(), children.get(0), arrayColumnIndex, List.of(backend));
    }

    /**
     * Rebuilds the equivalent {@code Correlate(input, Uncollect(Project($cor0.arrayCol, oneRow)))}
     * shape for backends whose {@code FragmentConvertor} still pattern-matches that Correlate/Uncollect
     * structure (today: DataFusion's {@code tryEmitUnnest}) — see {@link LogicalNestedScope}'s javadoc.
     * Backends that add a native lowering for {@code OpenSearchNestedScope} directly can bypass this
     * by overriding their visitor for this node type instead of relying on the stripped Correlate shape.
     */
    @Override
    public RelNode stripAnnotations(List<RelNode> strippedChildren) {
        return LogicalNestedScope.buildEquivalentCorrelate(strippedChildren.get(0), arrayColumnIndex);
    }
}

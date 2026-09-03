/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rel;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.logical.LogicalCorrelate;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.spi.FieldStorageInfo;

import java.util.ArrayList;
import java.util.List;

/**
 * POC nested (N1): Marked Correlate node for UNNEST operations.
 *
 * @opensearch.internal
 */
public class OpenSearchCorrelate extends Correlate implements OpenSearchRelNode {

    private final List<String> viableBackends;

    public OpenSearchCorrelate(
        RelOptCluster cluster,
        RelTraitSet traitSet,
        RelNode left,
        RelNode right,
        CorrelationId correlationId,
        ImmutableBitSet requiredColumns,
        JoinRelType joinType,
        List<String> viableBackends
    ) {
        super(cluster, traitSet, left, right, correlationId, requiredColumns, joinType);
        this.viableBackends = viableBackends;
    }

    @Override
    public List<String> getViableBackends() {
        return viableBackends;
    }

    @Override
    public List<FieldStorageInfo> getOutputFieldStorage() {
        List<FieldStorageInfo> result = new ArrayList<>();
        appendChildStorage(getLeft(), result);
        appendChildStorage(getRight(), result);
        return result;
    }

    private void appendChildStorage(RelNode child, List<FieldStorageInfo> out) {
        RelNode unwrapped = RelNodeUtils.unwrapHep(child);
        if (unwrapped instanceof OpenSearchRelNode os) {
            out.addAll(os.getOutputFieldStorage());
        } else {
            // Synthetic entries for non-OpenSearch children
            for (var field : child.getRowType().getFieldList()) {
                out.add(FieldStorageInfo.derivedColumn(field.getName(), field.getType().getSqlTypeName()));
            }
        }
    }

    @Override
    public Correlate copy(RelTraitSet traitSet, RelNode left, RelNode right,
                          CorrelationId correlationId, ImmutableBitSet requiredColumns, JoinRelType joinType) {
        return new OpenSearchCorrelate(getCluster(), traitSet, left, right, correlationId, requiredColumns, joinType, viableBackends);
    }

    @Override
    public org.apache.calcite.plan.RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        return planner.getCostFactory().makeTinyCost();
    }

    @Override
    public RelWriter explainTerms(RelWriter pw) {
        return super.explainTerms(pw).item("viableBackends", viableBackends);
    }

    @Override
    public RelNode copyResolved(String backend, List<RelNode> children, List<OperatorAnnotation> resolvedAnnotations) {
        return new OpenSearchCorrelate(
            getCluster(), getTraitSet(), children.get(0), children.get(1),
            getCorrelationId(), getRequiredColumns(), getJoinType(), List.of(backend)
        );
    }

    @Override
    public RelNode stripAnnotations(List<RelNode> strippedChildren) {
        return LogicalCorrelate.create(
            strippedChildren.get(0),
            strippedChildren.get(1),
            List.of(),
            getCorrelationId(),
            getRequiredColumns(),
            getJoinType()
        );
    }
}

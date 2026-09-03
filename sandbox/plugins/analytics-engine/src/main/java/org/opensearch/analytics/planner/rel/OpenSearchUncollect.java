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
import org.apache.calcite.rel.core.Uncollect;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.sql.type.SqlTypeName;
import org.opensearch.analytics.spi.FieldStorageInfo;

import java.util.ArrayList;
import java.util.List;

/**
 * POC nested (N1): Marked Uncollect (UNNEST) node. Represents array explosion into rows.
 *
 * @opensearch.internal
 */
public class OpenSearchUncollect extends Uncollect implements OpenSearchRelNode {

    private final List<String> viableBackends;

    public OpenSearchUncollect(
        RelOptCluster cluster,
        RelTraitSet traitSet,
        RelNode input,
        boolean withOrdinality,
        List<String> itemAliases,
        List<String> viableBackends
    ) {
        super(cluster, traitSet, input, withOrdinality, itemAliases);
        this.viableBackends = viableBackends;
    }

    @Override
    public List<String> getViableBackends() {
        return viableBackends;
    }

    @Override
    public List<FieldStorageInfo> getOutputFieldStorage() {
        List<FieldStorageInfo> result = new ArrayList<>();
        for (var field : getRowType().getFieldList()) {
            result.add(FieldStorageInfo.derivedColumn(field.getName(), field.getType().getSqlTypeName()));
        }
        return result;
    }

    @Override
    public RelNode copy(RelTraitSet traitSet, RelNode input) {
        return new OpenSearchUncollect(getCluster(), traitSet, input, withOrdinality, getItemAliases(), viableBackends);
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
        return new OpenSearchUncollect(
            getCluster(), getTraitSet(), children.get(0), withOrdinality, getItemAliases(), List.of(backend)
        );
    }

    @Override
    public RelNode stripAnnotations(List<RelNode> strippedChildren) {
        return Uncollect.create(strippedChildren.get(0).getTraitSet(), strippedChildren.get(0), withOrdinality, getItemAliases());
    }
}

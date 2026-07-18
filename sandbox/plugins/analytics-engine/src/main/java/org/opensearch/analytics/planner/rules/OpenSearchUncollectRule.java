/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Uncollect;
import org.opensearch.analytics.planner.PlannerContext;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.planner.rel.OpenSearchConvention;
import org.opensearch.analytics.planner.rel.OpenSearchRelNode;
import org.opensearch.analytics.planner.rel.OpenSearchUncollect;

import java.util.List;

/**
 * POC nested (N1): Marking rule that converts {@link Uncollect} → {@link OpenSearchUncollect}.
 * Viable backends = DataFusion (UNNEST on LIST columns is a DataFusion-native operation).
 *
 * @opensearch.internal
 */
public class OpenSearchUncollectRule extends RelOptRule {

    private final PlannerContext context;

    public OpenSearchUncollectRule(PlannerContext context) {
        super(operand(Uncollect.class, any()), "OpenSearchUncollectRule");
        this.context = context;
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        Uncollect uncollect = call.rel(0);

        if (uncollect instanceof OpenSearchUncollect) {
            return; // Already marked
        }

        RelNode input = RelNodeUtils.unwrapHep(uncollect.getInput());

        // UNNEST is a DataFusion-native operation on LIST<STRUCT> Parquet columns
        List<String> viableBackends = List.of("datafusion");

        RelTraitSet traitSet = uncollect.getTraitSet().replace(OpenSearchConvention.INSTANCE);

        call.transformTo(new OpenSearchUncollect(
            uncollect.getCluster(),
            traitSet,
            input,
            uncollect.withOrdinality,
            uncollect.getItemAliases(),
            viableBackends
        ));
    }
}

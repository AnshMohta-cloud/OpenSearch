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
import org.opensearch.analytics.planner.PlannerContext;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.planner.rel.LogicalNestedScope;
import org.opensearch.analytics.planner.rel.OpenSearchConvention;
import org.opensearch.analytics.planner.rel.OpenSearchNestedScope;
import org.opensearch.analytics.planner.rel.OpenSearchRelNode;
import org.opensearch.analytics.spi.EngineCapability;

import java.util.List;

/**
 * Marking rule that converts {@link LogicalNestedScope} → {@link OpenSearchNestedScope}. Viable
 * backends = the child's viable backends, narrowed to those declaring {@link
 * EngineCapability#NESTED_SCOPE} — a genuine capability-registry lookup, replacing the hardcoded
 * {@code List.of("datafusion")} pinning the legacy {@code OpenSearchCorrelateRule}/{@code
 * OpenSearchUncollectRule} used for the equivalent {@code Correlate}/{@code Uncollect} shape.
 *
 * @opensearch.internal
 */
public class OpenSearchNestedScopeRule extends RelOptRule {

    private final PlannerContext context;

    public OpenSearchNestedScopeRule(PlannerContext context) {
        super(operand(LogicalNestedScope.class, any()), "OpenSearchNestedScopeRule");
        this.context = context;
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        LogicalNestedScope nestedScope = call.rel(0);

        RelNode child = RelNodeUtils.unwrapHep(nestedScope.getInput());
        if (!(child instanceof OpenSearchRelNode openSearchChild)) {
            throw new IllegalStateException("NestedScope rule encountered unmarked child [" + child.getClass().getSimpleName() + "]");
        }

        List<String> childViableBackends = openSearchChild.getViableBackends();
        List<String> nestedScopeCapable = context.getCapabilityRegistry().operatorBackends(EngineCapability.NESTED_SCOPE);
        List<String> viableBackends = childViableBackends.stream().filter(nestedScopeCapable::contains).toList();

        if (viableBackends.isEmpty()) {
            throw new IllegalStateException(
                "No backend supports NESTED_SCOPE capability among " + childViableBackends + " for path '" + nestedScope.getPath() + "'"
            );
        }

        RelTraitSet traitSet = nestedScope.getTraitSet().replace(OpenSearchConvention.INSTANCE);

        call.transformTo(new OpenSearchNestedScope(nestedScope.getCluster(), traitSet, child, nestedScope.getArrayColumnIndex(), viableBackends));
    }
}

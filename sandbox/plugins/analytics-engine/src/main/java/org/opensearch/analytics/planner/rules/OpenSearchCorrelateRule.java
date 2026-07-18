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
import org.apache.calcite.rel.logical.LogicalCorrelate;
import org.opensearch.analytics.planner.PlannerContext;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.planner.rel.OpenSearchConvention;
import org.opensearch.analytics.planner.rel.OpenSearchCorrelate;
import org.opensearch.analytics.planner.rel.OpenSearchRelNode;

import java.util.List;

/**
 * POC nested (N1): Marking rule that converts {@link LogicalCorrelate} → {@link OpenSearchCorrelate}.
 * Viable backends = intersection of left and right children's viable backends.
 *
 * @opensearch.internal
 */
public class OpenSearchCorrelateRule extends RelOptRule {

    private final PlannerContext context;

    public OpenSearchCorrelateRule(PlannerContext context) {
        super(operand(LogicalCorrelate.class, any()), "OpenSearchCorrelateRule");
        this.context = context;
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        LogicalCorrelate correlate = call.rel(0);

        RelNode left = RelNodeUtils.unwrapHep(correlate.getLeft());
        RelNode right = RelNodeUtils.unwrapHep(correlate.getRight());

        // For UNNEST correlate: the whole operation runs on DataFusion (which reads Parquet
        // LIST<STRUCT> and does the unnest). Even if the TableScan is only viable=[lucene]
        // (because the field storage resolver doesn't recognize ARRAY fields), the correlate
        // as a whole must be [datafusion]. Force it.
        //
        // POC: Also force the left child (TableScan) to include "datafusion" so PlanForker
        // can find a matching alternative. At execution time, DF reads the Parquet directly.
        if (left instanceof org.opensearch.analytics.planner.rel.OpenSearchTableScan scan) {
            if (!scan.getViableBackends().contains("datafusion")) {
                List<String> expanded = new java.util.ArrayList<>(scan.getViableBackends());
                expanded.add("datafusion");
                left = new org.opensearch.analytics.planner.rel.OpenSearchTableScan(
                    scan.getCluster(), scan.getTraitSet(), scan.getTable(), expanded, scan.getOutputFieldStorage()
                );
            }
        }

        List<String> viableBackends = List.of("datafusion");

        RelTraitSet traitSet = correlate.getTraitSet().replace(OpenSearchConvention.INSTANCE);

        call.transformTo(new OpenSearchCorrelate(
            correlate.getCluster(),
            traitSet,
            left,
            right,
            correlate.getCorrelationId(),
            correlate.getRequiredColumns(),
            correlate.getJoinType(),
            viableBackends
        ));
    }

    private static List<String> viableBackendsOf(RelNode rel) {
        // Unused after POC simplification — kept for reference
        RelNode unwrapped = RelNodeUtils.unwrapHep(rel);
        if (unwrapped instanceof OpenSearchRelNode os) {
            return os.getViableBackends();
        }
        return List.of("datafusion");
    }
}

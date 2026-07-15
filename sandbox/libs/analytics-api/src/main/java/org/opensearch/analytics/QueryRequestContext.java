/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import org.apache.calcite.schema.SchemaPlus;
import org.opensearch.cluster.ClusterState;
import org.opensearch.tasks.Task;

/**
 * Immutable per-query view of analytics-engine state, captured once at query entry.
 *
 * <p>Front-ends call {@link EngineContextProvider#getContext(ClusterState)} to obtain a
 * {@code QueryRequestContext} bound to a specific {@link ClusterState} snapshot, then thread
 * it through both schema construction <em>and</em> plan execution. This guarantees the
 * same cluster-state view is used for type resolution and runtime shard routing — without
 * it, two calls to {@code clusterService.state()} could see different snapshots between
 * planning and execution, yielding a plan that references indices the executor no longer
 * sees (or vice-versa).
 *
 * <p>{@code parentTask} is the front-end request task used to link the analytics query task for
 * cancellation propagation (see {@code DefaultPlanExecutor}). May be {@code null}.
 *
 * <p>[NESTED-POC] {@code n1Descriptor} describes a hand-authored N1-rewritten nested query (see
 * {@code N1PlanRegistry} / {@link N1Descriptor}). When non-null, the fragment convertor assembles
 * the Substrait plan by hand and skips isthmus — the POC stand-in for the customer-query -> N1
 * rewrite, needed because isthmus cannot emit relational UNNEST. May be {@code null} (normal path).
 * Carried on this context because it must cross the transport thread hop between the front-end and
 * {@code DefaultPlanExecutor.executeInternal}.
 *
 * @opensearch.internal
 */
public record QueryRequestContext(
    ClusterState clusterState,
    SchemaPlus schema,
    String querySource,
    Task parentTask,
    N1Descriptor n1Descriptor
) {

    public QueryRequestContext(ClusterState clusterState, SchemaPlus schema, String querySource, Task parentTask) {
        this(clusterState, schema, querySource, parentTask, null);
    }

    public QueryRequestContext(ClusterState clusterState, SchemaPlus schema, String querySource) {
        this(clusterState, schema, querySource, null, null);
    }

    public QueryRequestContext(ClusterState clusterState, SchemaPlus schema) {
        this(clusterState, schema, null, null, null);
    }
}

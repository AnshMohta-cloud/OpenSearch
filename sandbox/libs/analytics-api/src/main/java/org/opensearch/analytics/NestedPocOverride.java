/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

/**
 * [NESTED-POC] Thread-local bridge for a hand-authored N1 nested query.
 *
 * <p>An N1-rewritten nested query (see {@code N1PlanRegistry}) can't be converted by isthmus
 * (which cannot emit relational UNNEST), so instead of a RelNode the front-end supplies an
 * {@link N1Descriptor}. It rides on {@code QueryRequestContext#n1Descriptor()} across the
 * transport thread hop, then {@code DefaultPlanExecutor.executeInternal} stashes it here at the
 * start of the (synchronous) plan+convert chain so {@code DataFusionFragmentConvertor.convertFragment}
 * — in a different module — picks it up and assembles the Substrait plan by hand, skipping isthmus.
 *
 * <p>Set and cleared around one synchronous {@code convertAll} call on a single thread, so a plain
 * ThreadLocal is safe. Always cleared in a finally block so it never leaks onto pooled threads.
 * POC-only scaffolding; the real customer-query -> N1 rewrite emits UNNEST through the normal
 * planner path and needs none of this.
 */
public final class NestedPocOverride {

    private static final ThreadLocal<N1Descriptor> DESCRIPTOR = new ThreadLocal<>();

    private NestedPocOverride() {}

    /** Stash the N1 descriptor for the current conversion (no-op if null). */
    public static void set(N1Descriptor descriptor) {
        if (descriptor != null) {
            DESCRIPTOR.set(descriptor);
        }
    }

    /** The N1 descriptor for the current thread, or {@code null} on the normal path. */
    public static N1Descriptor get() {
        return DESCRIPTOR.get();
    }

    /** Must be called in a finally block after conversion so the value never leaks to pooled threads. */
    public static void clear() {
        DESCRIPTOR.remove();
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

/**
 * [NESTED] Feature flag enabling the generic Calcite UNNEST-rewrite for nested-field queries.
 *
 * <p>When enabled, a nested-field reference ({@code ITEM(array,'f')} on an {@code ARRAY(ROW)} column,
 * as produced by PPL {@code expand}) is rewritten in Calcite to inject {@code Correlate + Uncollect}
 * (= UNNEST) and the reference is repointed at the flattened column. isthmus then emits
 * filter/aggregate/project as usual, and only the UNNEST node is carried as an
 * {@code ExtensionSingleRel(unnest_reshape:...)}. No per-query code — works for arbitrary queries.
 *
 * <p>This is the ONLY nested-query path: the former hand-authored POC path (a per-query descriptor
 * hand-assembled into Substrait) has been removed. The flag is retained as a kill-switch: when it is
 * OFF the rewrite does not fire and nested-field queries are not supported (they fail to plan), so it
 * defaults to {@code true}. Set {@code -Dopensearch.analytics.nested.generic_rewrite=false} only to
 * disable nested support entirely (e.g. to isolate a planner issue).
 *
 * <p>Read fresh each call (not cached) so it can be toggled per-run without rebuilding.
 *
 * @opensearch.internal
 */
public final class NestedRewriteFlag {

    /** System property controlling the generic Calcite UNNEST-rewrite path. Default {@code true}. */
    public static final String PROPERTY = "opensearch.analytics.nested.generic_rewrite";

    private NestedRewriteFlag() {}

    /** True (default) when the generic Calcite UNNEST-rewrite path is enabled. */
    public static boolean genericRewriteEnabled() {
        return Boolean.parseBoolean(System.getProperty(PROPERTY, "true"));
    }
}

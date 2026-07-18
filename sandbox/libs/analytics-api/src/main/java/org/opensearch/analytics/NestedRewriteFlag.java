/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

/**
 * [NESTED] Feature flag selecting how nested-field queries are turned into a Substrait plan.
 *
 * <p>There are two implementations of nested-query support in the tree while the generic path is
 * brought to production quality:
 * <ul>
 *   <li><b>generic (flag ON)</b> — the query is rewritten in Calcite ({@code ITEM(array,'f')} on an
 *       {@code ARRAY(ROW)} column ⇒ inject {@code Correlate + Uncollect} = UNNEST, rewrite the ref to
 *       the flattened column) and then serialized: isthmus emits filter/aggregate/project as usual and
 *       only the UNNEST node is injected as an {@code ExtensionSingleRel(unnest:...)}. No per-query
 *       code — works for arbitrary queries. This is the production direction.</li>
 *   <li><b>hardcoded (flag OFF, default)</b> — the proven POC path: a hand-authored {@link N1Descriptor}
 *       (looked up per query) is hand-assembled into the whole Substrait plan by
 *       {@code N1SubstraitBuilder}. Complete for the cases we authored (filter/agg/metric, depth 1-7)
 *       but not general. Kept as the safe default and as the fallback for query shapes the generic
 *       rewrite does not handle yet.</li>
 * </ul>
 *
 * <p>Controlled by the JVM system property {@value #PROPERTY} (default {@code false}). A single
 * boolean read — the "simple if condition" seam. Set with
 * {@code -Dopensearch.analytics.nested.generic_rewrite=true} (e.g. via the run task's jvm args) to
 * exercise the generic path without blocking the default/tested path.
 *
 * <p>Read fresh each call (not cached) so it can be toggled per-run without rebuilding.
 *
 * @opensearch.internal
 */
public final class NestedRewriteFlag {

    /** System property that turns the generic Calcite-rewrite path on. */
    public static final String PROPERTY = "opensearch.analytics.nested.generic_rewrite";

    private NestedRewriteFlag() {}

    /** True when the generic Calcite UNNEST-rewrite path should be used instead of the hardcoded POC path. */
    public static boolean genericRewriteEnabled() {
        return Boolean.getBoolean(PROPERTY);
    }
}

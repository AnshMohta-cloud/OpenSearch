/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Uncollect;

/**
 * [NESTED] Home for the generic-path Substrait emission of UNNEST.
 *
 * <p>When the {@code nested.generic_rewrite} flag is ON, {@code OpenSearchNestedFieldRewriter} injects
 * a Calcite {@code Correlate + Uncollect} subtree to represent UNNEST. isthmus cannot serialize those
 * rels, so the production design is: let isthmus emit filter/aggregate/project as usual, and inject
 * only the UNNEST node as an {@code ExtensionSingleRel(detail.type_url="unnest:&lt;path&gt;")} that the
 * Rust {@code UnnestConsumer} turns back into DataFusion's native {@code LogicalPlan::Unnest}.
 *
 * <p>Today this class provides the {@link #containsUnnest} detector used to fail loudly at the emission
 * boundary until the emitter is implemented. The emitter itself (walk the tree; where a Correlate over
 * an Uncollect is found, replace it with an {@code ExtensionSingleRel} over the correlate's left input
 * carrying the unnest path, letting isthmus emit everything else) is the remaining production task.
 *
 * @opensearch.internal
 */
final class NestedUnnestEmission {

    private NestedUnnestEmission() {}

    /** True if the tree still contains a Calcite {@link Uncollect} or {@link Correlate} (an un-emitted UNNEST). */
    static boolean containsUnnest(RelNode node) {
        if (node instanceof Uncollect || node instanceof Correlate) {
            return true;
        }
        for (RelNode input : node.getInputs()) {
            if (containsUnnest(input)) {
                return true;
            }
        }
        return false;
    }
}

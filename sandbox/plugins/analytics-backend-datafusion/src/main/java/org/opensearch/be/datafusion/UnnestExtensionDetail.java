/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import com.google.protobuf.Any;
import io.substrait.relation.Extension;
import io.substrait.relation.Rel;
import io.substrait.type.Type;

/**
 * [NESTED] The {@code detail} of the {@link io.substrait.relation.ExtensionSingle} that carries an
 * UNNEST across Substrait (which has no native unnest relation — see Substrait issue #745 / closed
 * PR #917; extension relations are the sanctioned mechanism, spec docs PR #1018).
 *
 * <p>Emitted by the {@code SubstraitRelVisitor.visit(Correlate)} override in
 * {@code DataFusionFragmentConvertor} for the Calcite {@code Correlate(left, Uncollect(...))} that
 * {@code OpenSearchNestedFieldRewriter} injects for a nested-field query. The Rust
 * {@code UnnestConsumer} recognises the {@code type_url} and rebuilds a real
 * {@code LogicalPlan::Unnest}, reshaping the output to the Calcite layout (original columns in place,
 * exploded struct fields appended) so the positional field references isthmus emits for the parent
 * Filter/Aggregate/Project resolve correctly.
 *
 * <p>Contract with the consumer (documented per Substrait PR #1018 guidance):
 * <ul>
 *   <li>{@code toProto()} → an {@code Any} whose {@code type_url} is {@code "unnest_reshape:" + path}
 *       where {@code path} is the comma-separated nested levels to unnest, outermost first (e.g.
 *       {@code "products"} or {@code "products,products.variants"}). {@code value} is empty — the path
 *       is fully carried in {@code type_url}, matching the existing consumer's parse.</li>
 *   <li>{@code deriveRecordType(input)} → the output row type AFTER unnest: exactly the Calcite
 *       {@code Correlate} row type (original columns, then the appended struct fields). Supplied here
 *       so isthmus can type the extension rel and everything above it without knowing unnest.</li>
 * </ul>
 */
final class UnnestExtensionDetail implements Extension.SingleRelDetail {

    /** Marker prefix the Rust consumer matches; must equal UNNEST_RESHAPE_TYPE_URL_PREFIX in Rust. */
    static final String TYPE_URL_PREFIX = "unnest_reshape:";

    private final String pathSpec;
    private final Type.Struct outputRecordType;
    private final int postUnnestWidth;

    UnnestExtensionDetail(String pathSpec, Type.Struct outputRecordType, int postUnnestWidth) {
        this.pathSpec = pathSpec;
        this.outputRecordType = outputRecordType;
        this.postUnnestWidth = postUnnestWidth;
    }

    @Override
    public Any toProto(io.substrait.relation.RelProtoConverter converter) {
        // type_url = "unnest_reshape:<path>|w=<postUnnestWidth>". The Rust consumer strips the "|w=..."
        // suffix (it only needs <path>); the parent-dedup post-pass reads it to place __row_id__ (which
        // the reshape reorders to the tail, i.e. index == postUnnestWidth) without re-deriving the layout.
        return Any.newBuilder().setTypeUrl(TYPE_URL_PREFIX + pathSpec + "|w=" + postUnnestWidth).build();
    }

    @Override
    public Type.Struct deriveRecordType(Rel input) {
        // The post-unnest schema is fixed at build time (the Calcite Correlate row type) — it does not
        // depend on the (already-converted) input rel, which carries only the pre-unnest columns.
        return outputRecordType;
    }
}

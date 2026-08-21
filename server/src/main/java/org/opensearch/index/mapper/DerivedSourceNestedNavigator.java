/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import org.apache.lucene.search.Query;
import org.apache.lucene.util.BitSet;

import java.io.IOException;

/**
 * Read-time capability implemented by a composite (columnar-backed) leaf reader that lets
 * derived-source reconstruction walk a block-join nested document without knowing how the underlying
 * columns are physically stored.
 *
 * <p>A composite index stores a nested logical document as ONE columnar row (each nested leaf is a
 * {@code LIST}) plus a Lucene block of {@code children..., ROOT}. Rebuilding {@code _source} for such
 * a document needs just two things from the reader:
 * <ol>
 *   <li>the ROOT (parent) doc bitset — the block boundaries; and</li>
 *   <li>the per-nested-path child doc bitset — which docs are elements of a given nested path.</li>
 * </ol>
 * Both are ordinary Lucene structures ({@code _primary_term} presence and {@code _nested_path}
 * postings); this interface merely exposes the reader's already-built, per-segment-cached copies so
 * {@link ObjectMapper} can emit the nested arrays. The block-join layout (an element's whole subtree
 * precedes its own doc; siblings in array order; ROOT last) is what makes the tree recoverable from
 * doc order plus these two bitsets, at any nesting depth.
 *
 * <p>Vanilla (non-composite) leaf readers do not implement this interface, so the existing
 * derived-source behavior is left untouched for them.
 */
public interface DerivedSourceNestedNavigator {

    /** Bitset with a set bit at every ROOT (parent) doc in this segment — the nested block boundaries. */
    BitSet parentDocs() throws IOException;

    /**
     * Bitset with a set bit at every child doc matching {@code nestedTypeFilter} — a nested object's
     * {@link ObjectMapper#nestedTypeFilter()}. Implementations reuse the node-level block-join bitset
     * cache, so this is the same cached bitset the nested query and inner_hits path resolution use;
     * callers must treat it as read-only.
     */
    BitSet childDocs(Query nestedTypeFilter) throws IOException;
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.common.lucene.index;

import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Reconstructs a nested (array-of-object) field's value for derived source.
 *
 * <p>On a composite (pluggable data format) index the actual child-object values are not stored in the
 * Lucene secondary as doc values — keyword/numeric child fields are indexed as terms only, and their
 * values live in the primary format's columnar store (e.g. a Parquet {@code LIST<STRUCT>} column). The
 * standard leaf-field {@code deriveSource} path (which reads per-field doc values) therefore cannot
 * rebuild a nested array. This capability interface lets the engine's leaf reader expose the columnar
 * reconstruction so {@code ObjectMapper} (in its nested mode) can rebuild the array during
 * derived-source assembly, without the {@code server} module depending on any data-format plugin.
 *
 * <p>A {@link LeafReader} produced by a composite engine implements this interface (directly or on a
 * reader in its delegation chain); {@link #unwrap(LeafReader)} locates it. Implementations return the
 * nested elements in <b>document order</b> with duplicates preserved (the ordering/cardinality contract
 * that nested {@code inner_hits} offsets depend on).
 */
public interface NestedSourceProvider {

    /**
     * Reconstructs the elements of a nested array field for one document.
     *
     * @param path  the full dotted path of the nested field (e.g. {@code "comments"} or
     *              {@code "comments.replies"})
     * @param docId the segment-local Lucene document id of the parent (root) document
     * @return the ordered list of child objects (each a field-name → JSON-friendly value map), an empty
     *         list if the parent has no elements for this path, or {@code null} if this provider cannot
     *         resolve the path (the caller then falls back to the default per-field behavior)
     * @throws IOException if the underlying columnar store cannot be read
     */
    List<Map<String, Object>> readNestedArray(String path, int docId) throws IOException;

    /**
     * Finds a {@link NestedSourceProvider} on {@code reader} or anywhere in its delegation chain, or
     * {@code null} if none is present (a non-composite / classic Lucene reader).
     */
    static NestedSourceProvider unwrap(LeafReader reader) {
        LeafReader current = reader;
        while (current != null) {
            if (current instanceof NestedSourceProvider provider) {
                return provider;
            }
            if (current instanceof FilterLeafReader filter) {
                current = filter.getDelegate();
            } else {
                break;
            }
        }
        return null;
    }
}

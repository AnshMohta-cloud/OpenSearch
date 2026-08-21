/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.iter;

import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.parquet.bridge.BinaryPageReader;
import org.opensearch.parquet.codec.NestedElementResolver;

import java.io.IOException;

/**
 * Single-valued {@link SortedDocValues} view over a <b>nested</b> (repeated) Parquet keyword leaf.
 *
 * <p>Mirrors {@link NestedParquetNumericDocValues} for byte values: each {@code advanceExact(childDocId)}
 * resolves {@code (row, offset)} via {@link NestedElementResolver}, reads that row's list of terms in
 * document order, and exposes the element at {@code offset} as this doc's value. Depth-independent for the
 * same reason (native flattening + document-order child docIds).
 *
 * <p>Follows the streaming transient-ordinal contract of {@code ParquetSortedDocValues}: the docId is the
 * transient ordinal (unique per positioned doc, int-ranged), resolved immediately via {@link #lookupOrd};
 * segment-global operations throw rather than return wrong results (aggregations use {@code execution_hint:
 * map}). It may be upgraded to real dictionary ordinals by the same wrapper the flat keyword iterator uses.
 */
public final class NestedParquetSortedDocValues extends SortedDocValues {

    private final BinaryPageReader reader;
    private final NestedElementResolver resolver;
    private final int maxDoc;
    private final BytesRef scratch = new BytesRef();

    private int doc = -1;
    private boolean currentPresent;

    public NestedParquetSortedDocValues(BinaryPageReader reader, NestedElementResolver resolver, int maxDoc) {
        this.reader = reader;
        this.resolver = resolver;
        this.maxDoc = maxDoc;
    }

    @Override
    public boolean advanceExact(int target) throws IOException {
        if (target >= maxDoc) {
            doc = NO_MORE_DOCS;
            currentPresent = false;
            return false;
        }
        doc = target;
        long row = resolver.row(target);
        int offset = resolver.offsetInRow(target);
        // O(1), zero-copy: view the child's single element as a BytesRef into the resident page — no per-row
        // byte[][] allocation. Returns the row's element count so we can bounds-check the offset.
        int count = reader.readRepeatedBytesAt(row, offset, scratch);
        if (offset < 0 || offset >= count) {
            currentPresent = false;
            return false;
        }
        currentPresent = true;
        return true;
    }

    @Override
    public int ordValue() {
        // The document id doubles as the transient ordinal (mirrors ParquetSortedDocValues).
        return doc;
    }

    @Override
    public BytesRef lookupOrd(int ord) {
        if (ord != doc || currentPresent == false) {
            throw new UnsupportedOperationException(
                "ordinal "
                    + ord
                    + " was issued for another document (current doc "
                    + doc
                    + "): composite Parquet nested keyword fields serve per-document streaming ordinals resolvable "
                    + "only for the currently positioned document"
            );
        }
        return scratch;
    }

    @Override
    public int getValueCount() {
        throw new UnsupportedOperationException(
            "segment-global value count is unavailable for streaming nested keyword ordinals; use execution_hint:map"
        );
    }

    @Override
    public int docID() {
        return doc;
    }

    @Override
    public int nextDoc() throws IOException {
        return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
        for (int d = target; d < maxDoc; d++) {
            if (advanceExact(d)) {
                doc = d;
                return d;
            }
        }
        doc = NO_MORE_DOCS;
        return NO_MORE_DOCS;
    }

    @Override
    public long cost() {
        return maxDoc;
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.iter;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.parquet.bridge.BinaryPageReader;
import org.opensearch.parquet.codec.NestedElementResolver;

import java.io.IOException;

/**
 * {@link BinaryDocValues} view over a <b>nested</b> (repeated) Parquet {@code BYTE_ARRAY} leaf.
 *
 * <p>Binary counterpart of {@link NestedParquetNumericDocValues}/{@link NestedParquetSortedDocValues}:
 * each {@code advanceExact(childDocId)} resolves {@code (row, offset)} via {@link NestedElementResolver}
 * and returns the single element at {@code offset} as a zero-copy {@link BytesRef} into the resident page
 * (via {@link BinaryPageReader#readRepeatedBytesAt}) — O(1), no per-row allocation. Depth-independent (the
 * native reader flattens a leaf at any depth to one value per child in document order).
 *
 * <p>Unlike the keyword ({@code SORTED}) path, {@code BINARY} carries no ordinals — values are opaque byte
 * blobs returned verbatim, matching the flat {@link ParquetBinaryDocValues}.
 */
public final class NestedParquetBinaryDocValues extends BinaryDocValues {

    private final BinaryPageReader reader;
    private final NestedElementResolver resolver;
    private final int maxDoc;
    private final BytesRef scratch = new BytesRef();

    private int doc = -1;
    private boolean currentPresent;

    public NestedParquetBinaryDocValues(BinaryPageReader reader, NestedElementResolver resolver, int maxDoc) {
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
        int count = reader.readRepeatedBytesAt(row, offset, scratch); // O(1) zero-copy view into resident page
        currentPresent = offset >= 0 && offset < count;
        return currentPresent;
    }

    @Override
    public BytesRef binaryValue() {
        return scratch;
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

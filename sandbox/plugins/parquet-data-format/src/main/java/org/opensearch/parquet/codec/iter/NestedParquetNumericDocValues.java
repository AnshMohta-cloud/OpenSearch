/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.iter;

import org.apache.lucene.index.NumericDocValues;
import org.opensearch.parquet.bridge.NumericPageReader;
import org.opensearch.parquet.codec.NestedElementResolver;

import java.io.IOException;

/**
 * Single-valued {@link NumericDocValues} view over a <b>nested</b> (repeated) Parquet numeric leaf.
 *
 * <p>Lucene addresses nested child docs; the Parquet leaf is a {@code LIST} with one value per child.
 * For each child {@code advanceExact(childDocId)} resolves {@code (row, offset)} via
 * {@link NestedElementResolver}, reads that row's list in document order (no sort — the offset indexes
 * the list positionally), and exposes the single element at {@code offset} as this doc's value.
 *
 * <p>Depth-independent: the native reader flattens a leaf at any depth to one value per child in document
 * order, so the resolver's rank-based offset indexes the correct element regardless of nesting depth.
 *
 * <p>Forward-only, matching the codec iterators and the resolver's ascending-docId contract. Flat/scalar
 * numeric reads are unaffected — they keep using {@code ParquetNumericDocValues}.
 */
public final class NestedParquetNumericDocValues extends NumericDocValues {

    private final NumericPageReader reader;
    private final NestedElementResolver resolver;
    private final int maxDoc;
    private final String field; // for trace logging only — names which nested leaf this iterator serves

    /** Single-element output slot for the O(1) read; reused, no per-doc allocation. */
    private final long[] out = new long[1];

    private int doc = -1;
    private long currentValue;
    private boolean currentPresent;

    public NestedParquetNumericDocValues(NumericPageReader reader, NestedElementResolver resolver, int maxDoc, String field) {
        this.reader = reader;
        this.resolver = resolver;
        this.maxDoc = maxDoc;
        this.field = field;
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
        // O(1): read only the child's single element directly from the resident decoded page — no whole-row
        // copy. Returns the row's element count so we can bounds-check the offset (absent leaf => no value).
        int count = reader.readRepeatedLongAt(row, offset, out);
        if (offset < 0 || offset >= count) {
            org.apache.logging.log4j.LogManager.getLogger(NestedParquetNumericDocValues.class)
                .info(
                    "[DSL-TRACE] nestedNumeric field='{}' childDoc={} -> row={} offset={} rowCount={} -> ABSENT (offset out of range)",
                    field,
                    target,
                    row,
                    offset,
                    count
                );
            currentPresent = false;
            return false;
        }
        currentValue = out[0];
        currentPresent = true;
        // [DSL-TRACE] the exact element read: nested leaf field, child docId, Parquet (row, offset), row list
        // length, and the single value picked — ground truth of the (row,offset) resolution (O(1), no copy).
        org.apache.logging.log4j.LogManager.getLogger(NestedParquetNumericDocValues.class)
            .info(
                "[DSL-TRACE] nestedNumeric field='{}' childDoc={} -> row={} offset={}/{} -> value={}",
                field,
                target,
                row,
                offset,
                count,
                currentValue
            );
        return true;
    }

    @Override
    public long longValue() {
        return currentValue;
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

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.iter;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.NumericDocValues;
import org.opensearch.parquet.bridge.NumericPageReader;
import org.opensearch.parquet.codec.NestedElementResolver;
import org.opensearch.parquet.codec.cache.PageCache;

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
 * <p>Hot path mirrors {@link ParquetNumericDocValues}: a resident-page range check plus a direct indexed
 * read of the decoded {@link PageCache} — no FFM crossing and no per-doc allocation on a cache hit. Only a
 * page miss calls {@link NumericPageReader#loadPageContaining}, which rides the same forward-only native
 * cursor (transparently reopening on the rare backward request). The difference from the flat iterator is
 * purely the nested addressing: the value lives at flattened element {@code listOffsets[row] + offset}
 * rather than at {@code row} directly. Flat/scalar numeric reads keep using {@code ParquetNumericDocValues}.
 */
public final class NestedParquetNumericDocValues extends NumericDocValues {

    private static final Logger DSL_LOG = LogManager.getLogger(NestedParquetNumericDocValues.class);

    private final NumericPageReader reader;
    private final NestedElementResolver resolver;
    private final int maxDoc;
    private final String field; // for trace logging only — names which nested leaf this iterator serves
    // True when this leaf may contain nulls. When false, every path child holds exactly one value, so
    // positioning (advance/nextDoc) navigates by the path child bitset WITHOUT resolving (row, offset) or
    // decoding the value — the value is read lazily by longValue() only when actually needed (a MAYBE
    // boundary compare, or an aggregation). When true we must confirm presence, so positioning decodes
    // eagerly and skips children whose element is null.
    private final boolean columnHasNulls;

    private int doc = -1;
    private long currentValue;
    private boolean currentPresent;
    // docId whose value is currently decoded into currentValue (-1 = nothing decoded for the current doc).
    // Lets longValue() decode lazily exactly once, and lets advanceExact()/the nullable path reuse an
    // eager decode instead of decoding twice.
    private int valuedDoc = -1;

    public NestedParquetNumericDocValues(
        NumericPageReader reader,
        NestedElementResolver resolver,
        int maxDoc,
        String field,
        boolean columnHasNulls
    ) {
        this.reader = reader;
        this.resolver = resolver;
        this.maxDoc = maxDoc;
        this.field = field;
        this.columnHasNulls = columnHasNulls;
    }

    /**
     * Resolves {@code (row, offset)} for {@code docId}, reads that child's single element straight out of the
     * resident decoded {@link PageCache} (loading the page on a miss), stores it in {@code currentValue}, and
     * sets {@code currentPresent}. Returns whether the element is present. Does NOT touch {@code doc} or
     * {@code valuedDoc} — callers own iterator position and the decode marker. This is the original inline hot
     * path (mirrors the flat iterator: no reader call, no allocation on a cache hit), shared by eager
     * {@link #advanceExact} and lazy {@link #longValue}.
     */
    private boolean loadValueAt(int docId) throws IOException {
        long row = resolver.row(docId);
        int offset = resolver.offsetInRow(docId);
        PageCache cache = reader.cache();
        if (cache == null || row < cache.firstRow || row > cache.lastRow) {
            // Page miss — decode the batch containing this row (rides the forward cursor, reopens on backward).
            reader.loadPageContaining(row);
            cache = reader.cache();
            if (cache == null) { // all-nulls page
                currentPresent = false;
                currentValue = 0L;
                return false;
            }
        }
        // Cache hit: index the child's single element straight out of the resident decoded page. listOffsets are
        // ELEMENT indices into cache.values (CSR), so the child's value is at element (start + offset); elementAt
        // takes an absolute element index (NOT valueAt, which subtracts firstRow for single-valued rows).
        int relativeRow = (int) (row - cache.firstRow);
        int start = cache.listOffsets[relativeRow];
        int count = cache.listOffsets[relativeRow + 1] - start;
        if (offset < 0 || offset >= count) {
            if (DSL_LOG.isTraceEnabled()) {
                DSL_LOG.trace(
                    "[DSL-TRACE] nestedNumeric field='{}' childDoc={} -> row={} offset={} rowCount={} -> ABSENT (offset out of range)",
                    field,
                    docId,
                    row,
                    offset,
                    count
                );
            }
            currentPresent = false;
            return false;
        }
        currentValue = cache.elementAt(start + offset);
        currentPresent = true;
        // [DSL-TRACE] the exact element read: nested leaf field, child docId, Parquet (row, offset), row list
        // length, and the single value picked — ground truth of the (row,offset) resolution. Gated behind
        // isTraceEnabled() so the hot path autoboxes nothing when the trace is off (the default).
        if (DSL_LOG.isTraceEnabled()) {
            DSL_LOG.trace(
                "[DSL-TRACE] nestedNumeric field='{}' childDoc={} -> row={} offset={}/{} -> value={}",
                field,
                docId,
                row,
                offset,
                count,
                currentValue
            );
        }
        return true;
    }

    @Override
    public boolean advanceExact(int target) throws IOException {
        // Random-access presence probe (aggregations, boundary compares): decode eagerly so the value is ready.
        // Behaviour-identical to the original.
        if (target >= maxDoc) {
            doc = NO_MORE_DOCS;
            currentPresent = false;
            valuedDoc = -1;
            return false;
        }
        doc = target;
        boolean present = loadValueAt(target);
        valuedDoc = present ? target : -1; // value is cached iff present
        return present;
    }

    @Override
    public long longValue() throws IOException {
        // advance()/nextDoc() on a null-free leaf position WITHOUT decoding (the predicate/count fast path);
        // decode lazily here on the first read of the current doc, exactly once. On a nullable leaf advance()
        // already decoded eagerly (valuedDoc == doc), so this is a no-op read.
        if (valuedDoc != doc) {
            loadValueAt(doc);
            valuedDoc = doc;
        }
        return currentValue;
    }

    @Override
    public int docID() {
        return doc;
    }

    @Override
    public int nextDoc() throws IOException {
        if (doc == NO_MORE_DOCS) {
            return NO_MORE_DOCS;
        }
        return advance(doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
        // Navigate child-to-child over THIS nested path's bitset — never probing ROOT or sibling-path docs.
        // Null-free leaf: bitset membership already proves the child has a value, so position without resolving
        // (row, offset) or decoding; longValue() decodes lazily only if the value is actually read (MAYBE
        // boundary pages, aggregations). This removes the per-probe resolve+decode cost on the range/count/
        // exists predicate path — the whole point of this iterator. Nullable leaf: confirm presence by decoding,
        // skipping null elements — same semantics as the original per-doc advanceExact scan, but still jumping
        // over non-path docs via the bitset rather than incrementing one docId at a time.
        int c = Math.max(target, 0);
        while (c < maxDoc) {
            int child = resolver.nextChild(c);
            if (child == NO_MORE_DOCS || child >= maxDoc) {
                break;
            }
            if (columnHasNulls) {
                if (loadValueAt(child)) {
                    doc = child;
                    valuedDoc = child; // decoded eagerly; longValue() reuses it
                    return child;
                }
                c = child + 1; // element is null for this child — advance to the next path child
                continue;
            }
            // Null-free: present by construction. Position only; defer the value decode to longValue().
            doc = child;
            valuedDoc = -1;
            return child;
        }
        doc = NO_MORE_DOCS;
        currentPresent = false;
        valuedDoc = -1;
        return NO_MORE_DOCS;
    }

    @Override
    public long cost() {
        return maxDoc;
    }
}

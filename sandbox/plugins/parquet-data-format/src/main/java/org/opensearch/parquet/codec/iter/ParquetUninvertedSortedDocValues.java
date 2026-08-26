/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.iter;

import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.opensearch.parquet.codec.RowIdResolver;
import org.opensearch.parquet.codec.UninvertedOrdinals;

import java.io.IOException;

/**
 * Fully contract-compliant {@link SortedDocValues} for high-cardinality keyword fields, backed by
 * disk-resident {@link UninvertedOrdinals}. Access-path economics:
 *
 * <ul>
 * <li>{@code ordValue()} — one packed read from the memory-mapped ordinal file; no Parquet
 * decode at all (sorting, global-ordinal collection).</li>
 * <li>{@code lookupOrd(currentOrd)} — the per-document value pattern (map-hint terms,
 * cardinality hashing): served zero-copy from the streaming reader's resident page, never
 * through the terms index.</li>
 * <li>{@code lookupOrd(otherOrd)} — bucket-key resolution: a stateful cursor over the sidecar's
 * terms enum; ascending walks amortize to one sequential pass.</li>
 * </ul>
 */
public final class ParquetUninvertedSortedDocValues extends SortedDocValues {

    private final UninvertedOrdinals ordinals;
    private final ParquetSortedDocValues streaming;
    private final int maxDoc;
    private final RowIdResolver resolver;

    private UninvertedOrdinals.TermCursor termCursor;
    private int doc = -1;
    private int currentOrd = -1;
    private boolean streamingPositioned = false;

    public ParquetUninvertedSortedDocValues(
        UninvertedOrdinals ordinals,
        ParquetSortedDocValues streaming,
        int maxDoc,
        RowIdResolver resolver
    ) {
        this.ordinals = ordinals;
        this.streaming = streaming;
        this.maxDoc = maxDoc;
        this.resolver = resolver;
    }

    @Override
    public boolean advanceExact(int target) {
        if (target >= maxDoc) {
            doc = NO_MORE_DOCS;
            currentOrd = -1;
            return false;
        }
        doc = target;
        streamingPositioned = false; // value read is lazy; most consumers never need it
        currentOrd = ordinals.ordinal(target);
        return currentOrd >= 0;
    }

    @Override
    public int ordValue() {
        return currentOrd;
    }

    @Override
    public BytesRef lookupOrd(int ord) throws IOException {
        if (ord == currentOrd && doc >= 0 && doc != NO_MORE_DOCS) {
            // Per-document value access: the streaming reader serves the CURRENT document's
            // bytes from its resident page — O(1), not a terms-index walk.
            //
            // The ordinal array above is docId-addressed (advanceExact keeps the raw docId), but the
            // streaming VALUE reader is Parquet-row-addressed, so translate docId→row here (identity
            // for flat/un-merged segments). doc arrives non-decreasing, so the forward-only resolver
            // contract holds; this is the only resolver call on the value path.
            if (streamingPositioned == false) {
                streaming.advanceExact((int) resolver.toRowId(doc));
                streamingPositioned = true;
            }
            return streaming.lookupOrd(streaming.ordValue());
        }
        if (termCursor == null) {
            termCursor = ordinals.newTermCursor();
        }
        return termCursor.term(ord);
    }

    @Override
    public int getValueCount() {
        return ordinals.valueCount();
    }

    @Override
    public int lookupTerm(BytesRef key) {
        return ordinals.rank(key);
    }

    @Override
    public TermsEnum termsEnum() throws IOException {
        // The default implementation resolves every ordinal through lookupOrd — quadratic over
        // millions of terms when OrdinalMap walks the enum. The sidecar's own enum IS this
        // ordinal space, in order, streamed off disk.
        return ordinals.termsEnum();
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
                return d;
            }
        }
        doc = NO_MORE_DOCS;
        currentOrd = -1;
        return NO_MORE_DOCS;
    }

    @Override
    public long cost() {
        return maxDoc;
    }
}

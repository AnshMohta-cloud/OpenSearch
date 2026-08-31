/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;

import java.io.IOException;

/**
 * Maps between Lucene docId space and Parquet logical-row space for one composite segment.
 *
 * <p><b>This is now an identity map.</b> The Lucene secondary writes exactly ONE document per Parquet row
 * — a nested array element becomes a multi-valued field on the row document rather than its own child doc
 * (see {@code LuceneDocumentInput}) — and {@code LuceneWriter} asserts that every doc carries
 * {@code __row_id__ == docId}. So {@code luceneDocId == parquetRow} for every doc, and the delegated-filter
 * contract (a {@code [minRow,maxRow)} row-group window in, a logical-row bitset out) needs no translation.
 *
 * <p>Historically nested documents were indexed as a contiguous block of {@code N+1} Lucene docs
 * ({@code N} children then the root), which made {@code luceneDocId != parquetRow} and required a cached
 * per-segment {@code parentDocIds} array plus a parent bitset to roll child matches up to roots. None of
 * that exists any more: no child docs are written, no parent field is declared, and no serializer builds a
 * block join. The class is kept as a named seam (and to leave the caller untouched) but is trivial enough
 * to inline if the delegation handle is ever refactored.
 */
final class RowIdTranslator {

    /**
     * Sentinel for "this docId has no logical row". Unreachable now that every doc is a row, but retained so
     * the caller's defensive check still compiles and keeps its meaning if translation ever returns.
     */
    static final long NO_ROW = -1L;

    private final int logicalRowCount;

    private RowIdTranslator(int logicalRowCount) {
        this.logicalRowCount = logicalRowCount;
    }

    /** Returns the translator for {@code leafContext}. Always the identity map; nothing is cached or scanned. */
    static RowIdTranslator forLeaf(LeafReaderContext leafContext) throws IOException {
        return new RowIdTranslator(leafContext.reader().maxDoc());
    }

    /**
     * Always {@code false}: there are no nested child docs, so docId space and row space coincide. Retained
     * because callers use it to decide whether a {@code __row_id__} doc-values cursor is needed at all.
     */
    boolean isNested() {
        return false;
    }

    /** Number of logical rows in this leaf, which equals {@code maxDoc()}. */
    int logicalRowCount() {
        return logicalRowCount;
    }

    /**
     * The logical row for a matched docId — the docId itself.
     *
     * @param docId   the matched Lucene doc
     * @param rowIdDV unused; retained so the caller need not change (the identity map reads nothing)
     */
    long rowForDocId(int docId, SortedNumericDocValues rowIdDV) throws IOException {
        return docId;
    }

    /** The first Lucene docId to scan from to cover logical row {@code row} — the row itself. */
    int firstDocIdForRow(int row) {
        return row;
    }

    /** The exclusive docId bound covering logical rows up to {@code rowExclusive} — the bound itself. */
    int docIdScanBoundForRow(int rowExclusive) {
        return rowExclusive;
    }
}

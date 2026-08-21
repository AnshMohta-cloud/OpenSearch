/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.lucene.util.BitSet;

import java.io.IOException;

/**
 * Translates a nested-child Lucene {@code docId} into the two coordinates needed to read one element
 * of a <b>repeated</b> (LIST) Parquet leaf column: the enclosing document's Parquet <b>row</b> and the
 * child's <b>element offset</b> within that row's list.
 *
 * <p>A composite index stores a nested logical document as one Parquet row whose nested leaf columns are
 * {@code LIST}s (one value per nested child), while Lucene stores the same document as a block of
 * {@code children..., ROOT}. So a nested child's leaf value lives at {@code values[listOffsets[row] +
 * offset]} in the repeated batch — this class produces that {@code (row, offset)} pair.
 *
 * <h2>Depth independence</h2>
 * The native reader flattens a leaf at any nesting depth into one value per nested child, in document
 * order, and Lucene assigns child docIds in that same order. The offset is therefore just the child's
 * rank within its enclosing root's block for this nested path — identical logic for depth 1 or depth 40;
 * only the {@code pathChildBits} (which {@code _nested_path} identifies this level's children) differs.
 *
 * <h2>Row resolution</h2>
 * {@code __row_id__} is written on ROOT docs only. A nested child's row is therefore its enclosing root's
 * {@code __row_id__}: the root is the first parent doc at or after the child (blocks are laid out
 * {@code children..., ROOT}). The row read reuses the existing {@link RowIdResolver} over {@code __row_id__}.
 *
 * <h2>Access pattern</h2>
 * Forward-only and stateful, matching the codec's ascending-docId DV iterators: callers must pass
 * non-decreasing {@code childDocId}s. One instance is bound to one DV iterator instance. The offset uses
 * the segment-cached parent/child bitsets ({@code nextSetBit}/{@code prevSetBit}) — no per-column map is
 * materialized; every lookup is O(children-in-block) worst case and O(1) amortized for the ascending scans
 * that block-join and collection produce.
 */
public final class NestedElementResolver {

    /** Resolves a ROOT docId to its Parquet row via {@code __row_id__} (reused, merge-safe). */
    private final RowIdResolver rowResolver;

    /** Set bit at every ROOT/parent docId (the {@code _primary_term} parent bitset). */
    private final BitSet parentBits;

    /** Set bit at every child docId of THIS nested path (the {@code _nested_path == P} bitset). */
    private final BitSet pathChildBits;

    NestedElementResolver(RowIdResolver rowResolver, BitSet parentBits, BitSet pathChildBits) {
        this.rowResolver = rowResolver;
        this.parentBits = parentBits;
        this.pathChildBits = pathChildBits;
    }

    /**
     * Returns the Parquet row of the document enclosing {@code childDocId}. The enclosing root is the first
     * parent at or after the child (block layout is {@code children..., ROOT}); its {@code __row_id__} is the
     * row. {@code childDocId} must be a set bit of {@link #pathChildBits}.
     */
    public long row(int childDocId) throws IOException {
        int root = parentBits.nextSetBit(childDocId);
        assert root != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS : "nested child docId "
            + childDocId
            + " has no enclosing parent (root) doc";
        return rowResolver.toRowId(root);
    }

    /**
     * Returns the 0-based element offset of {@code childDocId} within its enclosing root's list for this
     * nested path — i.e. the number of same-path child docs between the previous root boundary and this
     * child. This equals the child's position in the row's flattened leaf values. {@code childDocId} must be
     * a set bit of {@link #pathChildBits}.
     */
    public int offsetInRow(int childDocId) {
        // The previous parent marks the end of the previous document's block; children strictly after it and
        // before this child (of this path) precede this child within the SAME document's list. (Blocks are
        // laid out children..., ROOT, so this child's own root is AFTER it; the nearest parent strictly
        // before it is the previous block's root — or none for the first block.)
        int prevRoot = childDocId == 0 ? -1 : parentBits.prevSetBit(childDocId - 1);
        int scanFrom = prevRoot + 1;
        int offset = 0;
        for (int d = pathChildBits.nextSetBit(scanFrom); d != -1 && d < childDocId; d = pathChildBits.nextSetBit(d + 1)) {
            offset++;
        }
        return offset;
    }
}

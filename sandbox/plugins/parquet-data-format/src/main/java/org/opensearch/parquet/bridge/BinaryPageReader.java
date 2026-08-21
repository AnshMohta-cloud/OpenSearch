/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.bridge;

import org.opensearch.parquet.codec.cache.PageCache;

import java.io.IOException;

/** Minimal page/batch source used by single-valued binary DocValues. */
public interface BinaryPageReader {

    /** The currently decoded row range, or {@code null} when none is loaded. */
    PageCache cache();

    /** Loads a decoded range containing {@code row}. */
    void loadPageContaining(long row) throws IOException;

    /** Reads all binary values for one repeated row. */
    byte[][] readRepeatedBytesAtRow(long row) throws IOException;

    /**
     * Views the single binary element at position {@code offset} within {@code row}'s repeated list as a
     * zero-copy {@link org.apache.lucene.util.BytesRef} into the resident decoded page — O(1), no per-row
     * {@code byte[][]} allocation. Used by the nested keyword iterator, which needs exactly one element
     * (the child's) per doc.
     *
     * @param dst a reused {@code BytesRef} to point at the element's bytes (bytes/offset/length set on hit)
     * @return the number of elements in {@code row}'s list (so callers can bounds-check {@code offset});
     *         {@code dst} is updated only when {@code 0 <= offset < returned count}
     */
    int readRepeatedBytesAt(long row, int offset, org.apache.lucene.util.BytesRef dst) throws IOException;

    /** Reads one value using the resident batch, returning {@code null} when absent. */
    default byte[] readBytesAtRow(long row) throws IOException {
        PageCache page = cache();
        if (page == null || row < page.firstRow || row > page.lastRow) {
            loadPageContaining(row);
            page = cache();
        }
        if (page == null || page.isPresent(row) == false) {
            return null;
        }
        int relativeRow = Math.toIntExact(row - page.firstRow);
        int start = page.byteOffsets[relativeRow];
        int end = page.byteOffsets[relativeRow + 1];
        return java.util.Arrays.copyOfRange(page.byteBuf, start, end);
    }
}

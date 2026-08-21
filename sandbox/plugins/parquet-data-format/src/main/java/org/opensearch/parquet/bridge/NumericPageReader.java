/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.bridge;

import org.apache.lucene.util.LongsRef;
import org.opensearch.parquet.codec.cache.PageCache;

import java.io.IOException;

/** Minimal page/batch source used by the numeric DocValues iterator. */
public interface NumericPageReader {

    /** The currently decoded row range, or {@code null} when none is loaded. */
    PageCache cache();

    /** Loads a decoded range containing {@code row}. */
    void loadPageContaining(long row) throws IOException;

    /**
     * Reads all primitive values for one repeated row into {@code dst}, growing
     * {@code dst.longs} when needed and setting {@code dst.length}. Steady-state
     * calls must not allocate: this runs once per document on the hot path.
     */
    void readRepeatedLongsAtRow(long row, LongsRef dst) throws IOException;

    /**
     * Returns the single element value at position {@code offset} within {@code row}'s repeated list,
     * indexing directly into the resident decoded page — O(1), no per-row copy. Used by the nested
     * numeric iterator, which needs exactly one element (the child's) per doc, not the whole list.
     *
     * @return the number of elements in {@code row}'s list (so callers can bounds-check {@code offset});
     *         the value is written to {@code out[0]} only when {@code 0 <= offset < returned count}.
     */
    int readRepeatedLongAt(long row, int offset, long[] out) throws IOException;
}

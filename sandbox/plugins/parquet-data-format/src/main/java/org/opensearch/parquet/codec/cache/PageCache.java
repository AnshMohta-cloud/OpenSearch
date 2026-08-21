/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec.cache;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Layer 1 (page-resident value cache) + Layer 2 (page-resident presence bitset) for a
 * single decoded Parquet page of a single-valued column.
 *
 * <p>Holds the inclusive global row range {@code [firstRow, lastRow]} and, for primitive
 * columns, <b>off-heap views</b> of the decoded values (raw {@code long} bits, one slot per
 * row) and the packed presence bitset — {@link MemorySegment} slices of the column reader's
 * rotating {@link BufferPool} slots, served in place with zero on-heap copies. For
 * {@code BYTE_ARRAY} columns the value bytes are copied to a heap {@code byte[]} + CSR
 * offsets because Lucene's {@code BytesRef} contract requires a heap array; presence stays
 * off-heap.
 *
 * <p>Iterators serve cache hits with a presence bit-test plus an indexed segment read — no
 * FFM call, no heap allocation. A single instance per column is resident at a time (sliding
 * window, ascending doc IDs, no LRU). The backing segments belong to the producer's
 * {@link BufferPool} arena: they stay valid until the producer closes (grow events replace a
 * slot's backing segment but never free the old one), and must not be read after close.
 */
public final class PageCache {

    /** {@link #values} holds one {@code long} of raw bits per row (copy mode and INT64/f64 borrows). */
    public static final int KIND_LONG = 1;
    /** {@link #values} holds one sign-extending {@code int} per row (borrowed Int32/Date32/Time32). */
    public static final int KIND_INT = 2;
    /** {@link #values} holds one zero-extending {@code int} per row (borrowed UInt32/Float32 bits). */
    public static final int KIND_UINT_BITS = 3;
    /** {@link #values} holds one sign-extending {@code short} per row (borrowed Int16). */
    public static final int KIND_SHORT = 4;
    /** {@link #values} holds one zero-extending {@code short} per row (borrowed UInt16). */
    public static final int KIND_USHORT = 5;
    /** {@link #values} holds one sign-extending {@code byte} per row (borrowed Int8). */
    public static final int KIND_BYTE = 6;
    /** {@link #values} holds one zero-extending {@code byte} per row (borrowed UInt8). */
    public static final int KIND_UBYTE = 7;

    /** Inclusive global index of the first row in the cached page. */
    public long firstRow;
    /** Inclusive global index of the last row in the cached page. */
    public long lastRow;

    /**
     * Off-heap view of the decoded values (primitive columns): either the pooled copy
     * (one {@code long} per row) or a borrowed Arrow buffer read in place according to
     * {@link #valueKind}. Null for binary columns.
     */
    public MemorySegment values;

    /** Element interpretation of {@link #values}; one of the {@code KIND_*} constants. */
    public int valueKind = KIND_LONG;

    /** Backing byte buffer for binary columns (concatenated value bytes). Null for primitives. */
    public byte[] byteBuf;
    /** CSR offsets into {@link #byteBuf}, length {@code rowsInPage + 1}. Null for primitives. */
    public int[] byteOffsets;

    /**
     * Repeated columns: CSR offsets from rows to flattened primitive/binary elements.
     * Length is {@code rowsInPage + 1}; null for single-valued columns.
     */
    public int[] listOffsets;

    /**
     * Off-heap view of the packed presence bitset: bit {@code presenceBitOffset + i} set when
     * row {@code firstRow + i} is non-null. One {@code long} word per 64 rows. {@code null}
     * means every row is present (borrowed Arrow arrays without a validity bitmap).
     */
    public MemorySegment presenceBits;

    /** First presence bit of this page within {@link #presenceBits} (borrowed Arrow bitmaps are bit-sliced). */
    public int presenceBitOffset;

    /** Number of rows in the cached page. */
    public int rowCount() {
        return (int) (lastRow - firstRow + 1);
    }

    /**
     * Constant-time presence test for a global row that lies within {@code [firstRow, lastRow]}.
     */
    public boolean isPresent(long row) {
        if (presenceBits == null) {
            return true;
        }
        long idx = row - firstRow + presenceBitOffset;
        long word = presenceBits.getAtIndex(ValueLayout.JAVA_LONG, idx >>> 6);
        return (word & (1L << (idx & 63))) != 0L;
    }

    /**
     * Returns the raw {@code long} bits for a <b>single-valued</b> primitive at the given global row.
     * {@link #values} is indexed by row-within-page here, so the page-relative index is {@code row - firstRow}.
     */
    public long valueAt(long row) {
        return elementAt(row - firstRow);
    }

    /**
     * Returns the raw {@code long} bits for the primitive at absolute element index {@code idx} within this
     * page's decoded {@link #values} buffer — no {@code firstRow} adjustment.
     *
     * <p>For <b>repeated</b> columns {@link #values} is the flattened element buffer and callers already hold
     * the batch-local element index (e.g. {@code listOffsets[relativeRow] + offset}); subtracting {@code firstRow}
     * (as {@link #valueAt(long)} does for single-valued rows) would underflow for any page with
     * {@code firstRow > 0} — i.e. every batch past the first. This is the element-indexed read matching the bulk
     * {@code DataFusionColumnReader#readRepeatedLongsAtRow} path.
     */
    public long elementAt(long idx) {
        return switch (valueKind) {
            case KIND_LONG -> values.getAtIndex(ValueLayout.JAVA_LONG, idx);
            case KIND_INT -> values.getAtIndex(ValueLayout.JAVA_INT, idx);
            case KIND_UINT_BITS -> Integer.toUnsignedLong(values.getAtIndex(ValueLayout.JAVA_INT, idx));
            case KIND_SHORT -> values.getAtIndex(ValueLayout.JAVA_SHORT, idx);
            case KIND_USHORT -> Short.toUnsignedLong(values.getAtIndex(ValueLayout.JAVA_SHORT, idx));
            case KIND_BYTE -> values.get(ValueLayout.JAVA_BYTE, idx);
            default -> Byte.toUnsignedLong(values.get(ValueLayout.JAVA_BYTE, idx));
        };
    }

    /** True when the given global row falls within this cached page's range. */
    public boolean contains(long row) {
        return row >= firstRow && row <= lastRow;
    }
}

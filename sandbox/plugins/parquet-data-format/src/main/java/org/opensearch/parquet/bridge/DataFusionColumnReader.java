/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.bridge;

import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.LongsRef;
import org.opensearch.parquet.codec.ParquetPhysicalType;
import org.opensearch.parquet.codec.cache.BufferPool;
import org.opensearch.parquet.codec.cache.ColumnPageIndex;
import org.opensearch.parquet.codec.cache.PageCache;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Column reader backed by DataFusion I/O and Arrow's retained Parquet reader.
 *
 * <p>The native cursor is forward-only and adaptive: it starts at the configured window, grows
 * while access stays page-dense, and decays on large jumps. This wrapper makes the reader
 * logically random-access — a backward request transparently reopens the native cursor (cheap:
 * file metadata and the scoped page index are cached node-wide) so multiple Lucene iterators
 * over the same column can share one reader.
 *
 * <p>Variable-width batches follow a grow-and-retry overflow protocol; the native side stages
 * the decoded batch across the retry so nothing is decoded twice.
 */
public final class DataFusionColumnReader implements Closeable, NumericPageReader, BinaryPageReader {

    private static final org.apache.logging.log4j.Logger DSL_LOG = org.apache.logging.log4j.LogManager.getLogger(
        DataFusionColumnReader.class
    );

    private static final long CLOSED_HANDLE = -1L;

    /**
     * Mirrors {@code MAX_BATCH_SIZE} in the native {@code doc_values_cursor.rs} and the upper
     * bound of {@code parquet.docvalues.initial_batch_size}; keep the three in sync. The native
     * cursor never returns more rows than this in one batch.
     */
    private static final int MAX_BATCH_ROWS = 8192;

    private static final byte[] EMPTY_BYTES = new byte[0];

    /** Distinguishes pool slots across reader instances: several dedicated readers may serve the
     * same column concurrently (one per search slice), and slots must never be shared. */
    private static final java.util.concurrent.atomic.AtomicLong INSTANCE_IDS = new java.util.concurrent.atomic.AtomicLong();

    /**
     * GC backstop for native cursors. Readers opened by segment-lifetime shared producers are
     * handed to cache-retained iterators with no close hook; when such an iterator becomes
     * unreachable, the cleaner releases its cursor instead of waiting for segment close.
     * Explicit {@link #close()} remains the primary path and unregisters the action.
     */
    private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();

    /** Cursor handle shared with the cleaner action; cleared on explicit close. */
    private static final class CursorState implements Runnable {
        private final java.util.concurrent.atomic.AtomicLong handle;

        CursorState(long handle) {
            this.handle = new java.util.concurrent.atomic.AtomicLong(handle);
        }

        @Override
        public void run() {
            long stale = handle.getAndSet(CLOSED_HANDLE);
            if (stale != CLOSED_HANDLE) {
                try {
                    RustBridge.dfCloseIter(stale);
                } catch (java.io.IOException e) {
                    // Nothing actionable during GC-driven cleanup.
                }
            }
        }
    }

    private final BufferPool bufferPool;
    private final Path file;
    private final String column;
    private final ParquetPhysicalType type;
    private final boolean repeated;
    private final int initialBatchSize;

    // Pre-built pool slot names: one reader owns a stable slot family, so no per-batch concat.
    private final String firstRowSlot;
    private final String lastRowSlot;
    private final String valueLenSlot;
    private final String valueCountSlot;
    private final String valuesSlot;
    private final String presenceSlot;
    private final String bytesSlot;
    private final String byteOffsetsSlot;
    private final String rowOffsetsSlot;
    private final String valuesAddrSlot;
    private final String validityAddrSlot;
    private final String validityBitOffsetSlot;
    private final String valueKindSlot;
    private final String slotPrefix;

    private long handle;
    private final CursorState cursorState;
    private ColumnPageIndex pageIndex;
    private PageCache cache;
    private int outputRowsCapacity;
    private int repeatedValueCapacity;
    private long binaryValueCapacity = 64 * 1024L;

    private DataFusionColumnReader(
        long handle,
        Path file,
        String column,
        ParquetPhysicalType type,
        boolean repeated,
        BufferPool bufferPool,
        int initialBatchSize
    ) {
        this.handle = handle;
        this.cursorState = new CursorState(handle);
        CLEANER.register(this, cursorState);
        this.file = file;
        this.column = column;
        this.type = type;
        this.repeated = repeated;
        this.bufferPool = bufferPool;
        this.initialBatchSize = initialBatchSize;
        this.slotPrefix = "df:" + INSTANCE_IDS.incrementAndGet() + ":" + column + ":";
        this.firstRowSlot = slotPrefix + "firstRow";
        this.lastRowSlot = slotPrefix + "lastRow";
        this.valueLenSlot = slotPrefix + "valueLen";
        this.valueCountSlot = slotPrefix + "valueCount";
        this.valuesSlot = slotPrefix + "values";
        this.presenceSlot = slotPrefix + "presence";
        this.bytesSlot = slotPrefix + "bytes";
        this.byteOffsetsSlot = slotPrefix + "byteOffsets";
        this.rowOffsetsSlot = slotPrefix + "rowOffsets";
        this.valuesAddrSlot = slotPrefix + "valuesAddr";
        this.validityAddrSlot = slotPrefix + "validityAddr";
        this.validityBitOffsetSlot = slotPrefix + "validityBitOffset";
        this.valueKindSlot = slotPrefix + "valueKind";
        // Capacity is independent of the adaptive decode window. Reserving the
        // bounded maximum avoids an FFM overflow probe on every window growth.
        this.outputRowsCapacity = Math.max(initialBatchSize, MAX_BATCH_ROWS);
        this.repeatedValueCapacity = this.outputRowsCapacity;
    }

    /** Opens a retained INT64 cursor with the default starting window (tests / simple callers). */
    public static DataFusionColumnReader open(Path file, String column, BufferPool pool) throws IOException {
        return open(file, column, ParquetPhysicalType.INT64, false, pool, 32);
    }

    /** Opens a retained cursor with an explicit Java-facing physical representation. */
    public static DataFusionColumnReader open(
        Path file,
        String column,
        ParquetPhysicalType type,
        boolean repeated,
        BufferPool pool,
        int initialBatchSize
    ) throws IOException {
        long handle = RustBridge.dfOpenIter(file.toString(), column, initialBatchSize);
        return new DataFusionColumnReader(handle, file, column, type, repeated, pool, initialBatchSize);
    }

    /** Page statistics loaded through DataFusion's scoped page-index cache. */
    public ColumnPageIndex pageIndex() throws IOException {
        if (pageIndex == null) {
            pageIndex = loadPageIndex();
        }
        return pageIndex;
    }

    @Override
    public PageCache cache() {
        return cache;
    }

    /**
     * Loads an adaptive Arrow batch beginning at {@code row}. Forward requests ride the retained
     * native cursor; a backward request (another iterator over this column fell behind the shared
     * decode position) transparently reopens the cursor first.
     */
    @Override
    public void loadPageContaining(long row) throws IOException {
        ensureOpen();
        // [DSL-TRACE] PAGE MISS on column '{}' at row {} -> crossing FFM into Rust/DataFusion to DECODE a
        // Parquet page batch. This is the actual Parquet byte read (the "direct parquet read"). Note it is
        // a single-column, forward cursor decode — NOT a DataFusion query plan (no predicate/agg pushdown).
        DSL_LOG.info(
            "[DSL-TRACE] loadPageContaining column='{}' row={} repeated={} type={} -> decode Parquet page (FFM->Rust)",
            column,
            row,
            repeated,
            type
        );
        PageCache current = cache;
        if (current != null && row < current.firstRow) {
            reopen();
        }
        if (repeated && type.isPrimitive()) {
            loadRepeatedNumericBatch(row);
        } else if (repeated) {
            loadRepeatedBinaryBatch(row);
        } else if (type.isPrimitive()) {
            loadNumericBatch(row);
        } else {
            loadBinaryBatch(row);
        }
    }

    /**
     * Replaces the forward-only native cursor with a fresh one positioned at row zero. Only
     * reached on a backward request, which the common single-iterator forward path never makes.
     */
    private void reopen() throws IOException {
        cache = null;
        RustBridge.dfResetIter(handle);
    }

    /** One native batch call. Fetches pooled buffers itself so a retry after growth sees the new capacity. */
    @FunctionalInterface
    private interface NativeBatchCall {
        long invoke() throws IOException;
    }

    /**
     * Runs {@code call}; on {@link RustBridge#RC_OVERFLOW} grows the capacities reported through
     * the out-params via {@code grow} and retries exactly once, then validates the final status.
     */
    private void invokeGrowingOnOverflow(long row, String kind, NativeBatchCall call, Runnable grow) throws IOException {
        long rc = call.invoke();
        if (rc == RustBridge.RC_OVERFLOW) {
            grow.run();
            rc = call.invoke();
            if (rc == RustBridge.RC_OVERFLOW) {
                throw new IOException("DataFusion " + kind + " batch overflow persisted after retry at row " + row);
            }
        }
        checkStatus(rc, row, kind);
    }

    private void loadNumericBatch(long row) throws IOException {
        MemorySegment firstRowOut = bufferPool.longOut(firstRowSlot);
        MemorySegment lastRowOut = bufferPool.longOut(lastRowSlot);
        MemorySegment valueLenOut = bufferPool.longOut(valueLenSlot);
        MemorySegment valuesAddrOut = bufferPool.longOut(valuesAddrSlot);
        MemorySegment validityAddrOut = bufferPool.longOut(validityAddrSlot);
        MemorySegment validityBitOffsetOut = bufferPool.longOut(validityBitOffsetSlot);
        MemorySegment valueKindOut = bufferPool.longOut(valueKindSlot);

        // outputRowsCapacity always covers MAX_BATCH_ROWS (see constructor), so this call cannot
        // overflow in practice; the retry protocol is kept for uniformity with the other shapes.
        invokeGrowingOnOverflow(row, "numeric", () -> {
            long valueCap = (long) outputRowsCapacity * Long.BYTES;
            int presenceWords = presenceWords(outputRowsCapacity);
            return RustBridge.dfNextBatch(
                handle,
                row,
                firstRowOut,
                lastRowOut,
                bufferPool.bytes(valuesSlot, valueCap),
                valueCap,
                valueLenOut,
                bufferPool.longs(presenceSlot, presenceWords),
                presenceWords,
                valuesAddrOut,
                validityAddrOut,
                validityBitOffsetOut,
                valueKindOut
            );
        }, () -> growRowCapacity(firstRowOut, lastRowOut));

        long firstRow = firstRowOut.get(ValueLayout.JAVA_LONG, 0);
        long lastRow = lastRowOut.get(ValueLayout.JAVA_LONG, 0);
        int batchRows = Math.toIntExact(lastRow - firstRow + 1);

        PageCache next = new PageCache();
        next.firstRow = firstRow;
        next.lastRow = lastRow;
        int kind = (int) valueKindOut.get(ValueLayout.JAVA_LONG, 0);
        if (kind != 0) {
            // Borrowed Arrow buffers, read in place: O(rows accessed), no copy.
            // Valid until the next native call on this cursor; the resident
            // PageCache is always replaced before that call.
            long valuesAddr = valuesAddrOut.get(ValueLayout.JAVA_LONG, 0);
            long validityAddr = validityAddrOut.get(ValueLayout.JAVA_LONG, 0);
            int bitOffset = (int) validityBitOffsetOut.get(ValueLayout.JAVA_LONG, 0);
            int width = switch (kind) {
                case PageCache.KIND_LONG -> Long.BYTES;
                case PageCache.KIND_INT, PageCache.KIND_UINT_BITS -> Integer.BYTES;
                case PageCache.KIND_SHORT, PageCache.KIND_USHORT -> Short.BYTES;
                default -> Byte.BYTES;
            };
            next.valueKind = kind;
            next.values = MemorySegment.ofAddress(valuesAddr).reinterpret((long) batchRows * width);
            if (validityAddr == 0) {
                next.presenceBits = null;
            } else {
                long presenceWords = ((long) bitOffset + batchRows + 63) >>> 6;
                next.presenceBits = MemorySegment.ofAddress(validityAddr).reinterpret(presenceWords * Long.BYTES);
                next.presenceBitOffset = bitOffset;
            }
        } else {
            next.values = longsSlice(valuesSlot, batchRows);
            next.presenceBits = longsSlice(presenceSlot, presenceWords(batchRows));
        }
        cache = next;
    }

    private void loadBinaryBatch(long row) throws IOException {
        MemorySegment firstRowOut = bufferPool.longOut(firstRowSlot);
        MemorySegment lastRowOut = bufferPool.longOut(lastRowSlot);
        MemorySegment valueLenOut = bufferPool.longOut(valueLenSlot);

        invokeGrowingOnOverflow(row, "binary", () -> {
            long rows = outputRowsCapacity;
            int presenceWords = presenceWords(outputRowsCapacity);
            return RustBridge.dfNextBinaryBatch(
                handle,
                row,
                firstRowOut,
                lastRowOut,
                bufferPool.bytes(bytesSlot, binaryValueCapacity),
                binaryValueCapacity,
                valueLenOut,
                bufferPool.ints(byteOffsetsSlot, rows + 1),
                rows + 1,
                bufferPool.longs(presenceSlot, presenceWords),
                presenceWords
            );
        }, () -> {
            growRowCapacity(firstRowOut, lastRowOut);
            growBinaryValueCapacity(valueLenOut);
        });

        long firstRow = firstRowOut.get(ValueLayout.JAVA_LONG, 0);
        long lastRow = lastRowOut.get(ValueLayout.JAVA_LONG, 0);
        int batchRows = Math.toIntExact(lastRow - firstRow + 1);

        PageCache next = new PageCache();
        next.firstRow = firstRow;
        next.lastRow = lastRow;
        next.byteBuf = bytesArray(valueLenOut.get(ValueLayout.JAVA_LONG, 0));
        next.byteOffsets = intsArray(byteOffsetsSlot, batchRows + 1);
        next.presenceBits = longsSlice(presenceSlot, presenceWords(batchRows));
        cache = next;
    }

    private void loadRepeatedNumericBatch(long row) throws IOException {
        MemorySegment firstRowOut = bufferPool.longOut(firstRowSlot);
        MemorySegment lastRowOut = bufferPool.longOut(lastRowSlot);
        MemorySegment valueCountOut = bufferPool.longOut(valueCountSlot);

        invokeGrowingOnOverflow(row, "repeated numeric", () -> {
            long rows = outputRowsCapacity;
            return RustBridge.dfNextRepeatedBatch(
                handle,
                row,
                firstRowOut,
                lastRowOut,
                bufferPool.longs(valuesSlot, repeatedValueCapacity),
                repeatedValueCapacity,
                valueCountOut,
                bufferPool.ints(rowOffsetsSlot, rows + 1),
                rows + 1
            );
        }, () -> {
            growRowCapacity(firstRowOut, lastRowOut);
            growRepeatedValueCapacity(valueCountOut);
        });

        long firstRow = firstRowOut.get(ValueLayout.JAVA_LONG, 0);
        long lastRow = lastRowOut.get(ValueLayout.JAVA_LONG, 0);
        int batchRows = Math.toIntExact(lastRow - firstRow + 1);
        int valueCount = Math.toIntExact(valueCountOut.get(ValueLayout.JAVA_LONG, 0));

        PageCache next = new PageCache();
        next.firstRow = firstRow;
        next.lastRow = lastRow;
        next.values = longsSlice(valuesSlot, valueCount);
        next.listOffsets = intsArray(rowOffsetsSlot, batchRows + 1);
        cache = next;
    }

    private void loadRepeatedBinaryBatch(long row) throws IOException {
        MemorySegment firstRowOut = bufferPool.longOut(firstRowSlot);
        MemorySegment lastRowOut = bufferPool.longOut(lastRowSlot);
        MemorySegment valueLenOut = bufferPool.longOut(valueLenSlot);
        MemorySegment valueCountOut = bufferPool.longOut(valueCountSlot);

        invokeGrowingOnOverflow(row, "repeated binary", () -> {
            long rows = outputRowsCapacity;
            return RustBridge.dfNextRepeatedBinaryBatch(
                handle,
                row,
                firstRowOut,
                lastRowOut,
                bufferPool.bytes(bytesSlot, binaryValueCapacity),
                binaryValueCapacity,
                valueLenOut,
                bufferPool.ints(byteOffsetsSlot, repeatedValueCapacity + 1L),
                repeatedValueCapacity + 1L,
                valueCountOut,
                bufferPool.ints(rowOffsetsSlot, rows + 1),
                rows + 1
            );
        }, () -> {
            growRowCapacity(firstRowOut, lastRowOut);
            growBinaryValueCapacity(valueLenOut);
            growRepeatedValueCapacity(valueCountOut);
        });

        long firstRow = firstRowOut.get(ValueLayout.JAVA_LONG, 0);
        long lastRow = lastRowOut.get(ValueLayout.JAVA_LONG, 0);
        int batchRows = Math.toIntExact(lastRow - firstRow + 1);
        int valueCount = Math.toIntExact(valueCountOut.get(ValueLayout.JAVA_LONG, 0));

        PageCache next = new PageCache();
        next.firstRow = firstRow;
        next.lastRow = lastRow;
        next.byteBuf = bytesArray(valueLenOut.get(ValueLayout.JAVA_LONG, 0));
        next.byteOffsets = intsArray(byteOffsetsSlot, valueCount + 1);
        next.listOffsets = intsArray(rowOffsetsSlot, batchRows + 1);
        cache = next;
    }

    @Override
    public void readRepeatedLongsAtRow(long row, LongsRef dst) throws IOException {
        if (repeated == false || type.isPrimitive() == false) {
            throw new IOException("Column " + column + " is not a repeated primitive");
        }
        PageCache page = pageFor(row);
        int relativeRow = Math.toIntExact(row - page.firstRow);
        int start = page.listOffsets[relativeRow];
        int count = page.listOffsets[relativeRow + 1] - start;
        dst.longs = ArrayUtil.grow(dst.longs, count);
        dst.offset = 0;
        dst.length = count;
        MemorySegment.copy(page.values, ValueLayout.JAVA_LONG, (long) start * Long.BYTES, dst.longs, 0, count);
    }

    @Override
    public int readRepeatedLongAt(long row, int offset, long[] out) throws IOException {
        if (repeated == false || type.isPrimitive() == false) {
            throw new IOException("Column " + column + " is not a repeated primitive");
        }
        PageCache page = pageFor(row);
        int relativeRow = Math.toIntExact(row - page.firstRow);
        int start = page.listOffsets[relativeRow];
        int count = page.listOffsets[relativeRow + 1] - start;
        // O(1): index the single element directly in the resident decoded page. listOffsets are ELEMENT
        // indices into page.values, so the child's value is at element (start + offset). Use elementAt() —
        // an absolute element index — NOT valueAt(), which subtracts firstRow for single-valued row lookups
        // and would underflow to a negative index on any batch past the first (firstRow > 0). Matches the
        // element-indexed bulk readRepeatedLongsAtRow path. No per-row copy.
        if (offset >= 0 && offset < count) {
            out[0] = page.elementAt(start + offset);
        }
        return count;
    }

    @Override
    public byte[][] readRepeatedBytesAtRow(long row) throws IOException {
        if (repeated == false || type.isPrimitive()) {
            throw new IOException("Column " + column + " is not repeated binary");
        }
        PageCache page = pageFor(row);
        int relativeRow = Math.toIntExact(row - page.firstRow);
        int firstElement = page.listOffsets[relativeRow];
        int lastElement = page.listOffsets[relativeRow + 1];
        byte[][] values = new byte[lastElement - firstElement][];
        for (int element = firstElement; element < lastElement; element++) {
            int start = page.byteOffsets[element];
            int end = page.byteOffsets[element + 1];
            values[element - firstElement] = Arrays.copyOfRange(page.byteBuf, start, end);
        }
        return values;
    }

    @Override
    public int readRepeatedBytesAt(long row, int offset, org.apache.lucene.util.BytesRef dst) throws IOException {
        if (repeated == false || type.isPrimitive()) {
            throw new IOException("Column " + column + " is not repeated binary");
        }
        PageCache page = pageFor(row);
        int relativeRow = Math.toIntExact(row - page.firstRow);
        int firstElement = page.listOffsets[relativeRow];
        int count = page.listOffsets[relativeRow + 1] - firstElement;
        // O(1), zero-copy: point BytesRef straight at the element's slice of the resident byteBuf. byteOffsets
        // are per-ELEMENT (CSR) into byteBuf, so element (firstElement + offset) is the child's value. No
        // byte[][] allocation — the whole point vs readRepeatedBytesAtRow.
        if (offset >= 0 && offset < count) {
            int element = firstElement + offset;
            int start = page.byteOffsets[element];
            dst.bytes = page.byteBuf;
            dst.offset = start;
            dst.length = page.byteOffsets[element + 1] - start;
        }
        return count;
    }

    /** The resident batch containing {@code row}, loading (and reopening if backward) on a miss. */
    private PageCache pageFor(long row) throws IOException {
        PageCache page = cache;
        if (page == null || page.contains(row) == false) {
            loadPageContaining(row);
            page = cache;
        }
        return page;
    }

    private void growRowCapacity(MemorySegment firstRowOut, MemorySegment lastRowOut) {
        long firstRow = firstRowOut.get(ValueLayout.JAVA_LONG, 0);
        long lastRow = lastRowOut.get(ValueLayout.JAVA_LONG, 0);
        outputRowsCapacity = Math.max(outputRowsCapacity, Math.toIntExact(lastRow - firstRow + 1));
    }

    private void growBinaryValueCapacity(MemorySegment valueLenOut) {
        binaryValueCapacity = Math.max(binaryValueCapacity, Math.max(valueLenOut.get(ValueLayout.JAVA_LONG, 0), 1L));
    }

    private void growRepeatedValueCapacity(MemorySegment valueCountOut) {
        repeatedValueCapacity = Math.max(repeatedValueCapacity, Math.toIntExact(valueCountOut.get(ValueLayout.JAVA_LONG, 0)));
    }

    private static int presenceWords(int rows) {
        return (rows + 63) >>> 6;
    }

    /** Off-heap slice of the slot's current backing segment, {@code count} longs wide. */
    private MemorySegment longsSlice(String slot, int count) {
        return bufferPool.longs(slot, Math.max(count, 1)).asSlice(0, (long) count * Long.BYTES);
    }

    /** Heap copy of the slot's first {@code count} ints (amortized: one copy per batch). */
    private int[] intsArray(String slot, int count) {
        return bufferPool.ints(slot, count).asSlice(0, (long) count * Integer.BYTES).toArray(ValueLayout.JAVA_INT);
    }

    /** Heap copy of the binary slot's first {@code length} bytes (amortized: one copy per batch). */
    private byte[] bytesArray(long length) {
        return length == 0 ? EMPTY_BYTES : bufferPool.bytes(bytesSlot, length).asSlice(0, length).toArray(ValueLayout.JAVA_BYTE);
    }

    private void checkStatus(long rc, long row, String kind) throws IOException {
        if (rc == RustBridge.RC_EOF) {
            throw new IOException("DataFusion " + kind + " cursor exhausted before row " + row + " (" + file + "/" + column + ")");
        }
        if (rc != RustBridge.RC_OK) {
            throw new IOException(
                "Unexpected DataFusion " + kind + " cursor status " + rc + " at row " + row + " (" + file + "/" + column + ")"
            );
        }
    }

    private void ensureOpen() {
        if (handle == CLOSED_HANDLE) {
            throw new IllegalStateException("DataFusionColumnReader is closed");
        }
    }

    @Override
    public void close() throws IOException {
        if (handle == CLOSED_HANDLE) {
            return;
        }
        long current = handle;
        handle = CLOSED_HANDLE;
        cache = null;
        cursorState.handle.set(CLOSED_HANDLE);
        RustBridge.dfCloseIter(current);
    }

    private ColumnPageIndex loadPageIndex() throws IOException {
        int pageCount = Math.toIntExact(RustBridge.dfPageCount(handle));
        int capacity = Math.max(pageCount, 1);
        MemorySegment firstRow = bufferPool.longs(slotPrefix + "idxFirstRow", capacity);
        MemorySegment fileOffset = bufferPool.longs(slotPrefix + "idxFileOffset", capacity);
        MemorySegment compressed = bufferPool.ints(slotPrefix + "idxCompressed", capacity);
        MemorySegment nullCount = bufferPool.longs(slotPrefix + "idxNullCount", capacity);
        MemorySegment minLong = bufferPool.longs(slotPrefix + "idxMin", capacity);
        MemorySegment maxLong = bufferPool.longs(slotPrefix + "idxMax", capacity);
        MemorySegment actualPages = bufferPool.longOut(slotPrefix + "idxActualPages");

        long rc = RustBridge.dfPageIndex(handle, firstRow, fileOffset, compressed, nullCount, minLong, maxLong, pageCount, actualPages);
        if (rc == RustBridge.RC_OVERFLOW) {
            throw new IOException(
                "DataFusion page count changed while opening cursor: expected "
                    + pageCount
                    + " but found "
                    + actualPages.get(ValueLayout.JAVA_LONG, 0)
            );
        }

        return new ColumnPageIndex(
            toLongArray(firstRow, pageCount),
            toLongArray(fileOffset, pageCount),
            toIntArray(compressed, pageCount),
            toLongArray(nullCount, pageCount),
            toLongArray(minLong, pageCount),
            toLongArray(maxLong, pageCount),
            RustBridge.dfRowCount(handle)
        );
    }

    private static long[] toLongArray(MemorySegment segment, int length) {
        return length == 0
            ? new long[0]
            : segment.asSlice(0, (long) length * ValueLayout.JAVA_LONG.byteSize()).toArray(ValueLayout.JAVA_LONG);
    }

    private static int[] toIntArray(MemorySegment segment, int length) {
        return length == 0 ? new int[0] : segment.asSlice(0, (long) length * ValueLayout.JAVA_INT.byteSize()).toArray(ValueLayout.JAVA_INT);
    }
}

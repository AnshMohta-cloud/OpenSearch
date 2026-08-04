/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.exec.ArrowSourceSerializer;
import org.opensearch.analytics.exec.ArrowValues;
import org.opensearch.analytics.spi.DocumentLookupService;
import org.opensearch.analytics.spi.DocumentRowReader;
import org.opensearch.be.datafusion.nativelib.NativeBridge;
import org.opensearch.be.datafusion.nativelib.ReaderHandle;
import org.opensearch.be.datafusion.nativelib.StreamHandle;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.engine.exec.DocumentMetadataResolver;
import org.opensearch.index.engine.exec.MonoFileWriterSet;
import org.opensearch.index.engine.exec.WriterFileSet;
import org.opensearch.index.engine.exec.coord.CatalogSnapshot;
import org.opensearch.index.mapper.IdFieldMapper;
import org.opensearch.index.mapper.Uid;

import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * DataFusion-backed get-by-id executor. The resolver maps an {@code _id} to a
 * {@code (writerGeneration, rowId)}; this locates the parquet file in the {@link CatalogSnapshot}
 * and reads that row via a Substrait scan on the native runtime, flattening the Arrow batch into a
 * source map.
 */
@ExperimentalApi
public class GetService implements Closeable {

    private static final Logger logger = LogManager.getLogger(GetService.class);

    private final DocumentRowReader executor;

    /** Production constructor. */
    public GetService(DataFusionPlugin dfPlugin) {
        this(new NativeBridgeExecutor(dfPlugin));
    }

    GetService(DocumentRowReader executor) {
        this.executor = executor;
    }

    /** Returns a core-layer DocumentLookupService wired with this backend's executor and the supplied document resolver. */
    public DocumentLookupService documentLookupService(DocumentMetadataResolver resolver) {
        return new DocumentLookupService(resolver, executor);
    }

    /**
     * Reconstructs the nested field {@code path}'s ordered element array for the row {@code rowId} in
     * {@code parquetFilePath}, by reading that one file through the native runtime and rebuilding its
     * {@code LIST<STRUCT>} column via {@link org.opensearch.analytics.exec.ArrowSourceSerializer} with mapping-faithful
     * rendering. Returns {@code null} if the executor cannot serve row-id reconstruction (caller falls back).
     */
    public List<Map<String, Object>> reconstructNested(
        String parquetFilePath,
        long writerGeneration,
        String path,
        long rowId,
        org.opensearch.index.mapper.MapperService mapperService
    ) throws IOException {
        if (executor instanceof NativeBridgeExecutor nbe) {
            return nbe.reconstructNested(parquetFilePath, writerGeneration, path, rowId, mapperService);
        }
        return null;
    }

    /** Per-element raw values of one nested-child leaf for a row (see the executor method). {@code null} if unservable. */
    public List<Object> reconstructNestedLeaf(
        String parquetFilePath,
        String owningPath,
        String leafName,
        long rowId,
        org.opensearch.index.mapper.MapperService mapperService
    ) throws IOException {
        if (executor instanceof NativeBridgeExecutor nbe) {
            return nbe.reconstructNestedLeaf(parquetFilePath, owningPath, leafName, rowId, mapperService);
        }
        return null;
    }

    /**
     * Materializes a flat numeric column across the whole file as per-row {@code long} bits, keyed by
     * {@code __row_id__}. Serves the codec's single-valued numeric doc-values from DataFusion. Returns
     * {@code null} if the executor cannot serve column scans.
     */
    public org.opensearch.index.mapper.FlatColumnValueSource.LongColumn scanLongColumn(
        String parquetFilePath,
        String field,
        int rowCount
    ) throws IOException {
        if (executor instanceof NativeBridgeExecutor nbe) {
            return nbe.scanLongColumn(parquetFilePath, field, rowCount);
        }
        return null;
    }

    /**
     * Materializes a flat byte-array column (keyword/text/ip/binary) across the whole file as per-row raw bytes,
     * keyed by {@code __row_id__}. Returns {@code null} if the executor cannot serve column scans.
     */
    public org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn scanBytesColumn(
        String parquetFilePath,
        String field,
        int rowCount
    ) throws IOException {
        if (executor instanceof NativeBridgeExecutor nbe) {
            return nbe.scanBytesColumn(parquetFilePath, field, rowCount);
        }
        return null;
    }

    @Override
    public void close() throws IOException {
        if (executor instanceof Closeable c) {
            c.close();
        }
    }

    /**
     * Production executor driving {@link NativeBridge}. Spins up a short-lived
     * reader scoped to one parquet file, executes the plan, imports the single resulting batch
     * via the Arrow C Data Interface, and flattens the first row into a Java map.
     */
    static final class NativeBridgeExecutor implements DocumentRowReader, Closeable {

        private static final String GET_BY_ID_TABLE_ALIAS = "_t";
        private static final String PARQUET_FORMAT = "parquet";
        /** Empty Substrait plan — the internal-search path builds its plan natively and ignores it. */
        private static final byte[] EMPTY_PLAN = new byte[0];

        private static final long GET_BY_ID_TIMEOUT_MILLIS = 30_000L;

        private final DataFusionPlugin dfPlugin;
        private final BufferAllocator sharedAllocator = new RootAllocator(64 * 1024 * 1024);

        NativeBridgeExecutor(DataFusionPlugin dfPlugin) {
            this.dfPlugin = dfPlugin;
        }

        @Override
        public void close() {
            sharedAllocator.close();
        }

        @Override
        public String formatName() {
            return PARQUET_FORMAT;
        }

        @Override
        public Map<String, Object> executeSingleRow(long rowId, WriterFileSet parquetSet) throws IOException {
            if (rowId < 0) {
                throw new IllegalArgumentException("rowId must be non-negative, got: " + rowId);
            }
            String parquetDir = parquetSet.directory();
            String parquetFile = parquetSet.files().iterator().next();
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            // ReaderHandle registers the native pointer with NativeHandle so downstream
            // validatePointer() calls in executeQueryAsync() find it in the live set.
            MonoFileWriterSet segment = MonoFileWriterSet.of(parquetDir, parquetSet.writerGeneration(), parquetFile, 0L);
            try (ReaderHandle readerHandle = new ReaderHandle(parquetDir, List.of(segment), null, List.of(), List.of())) {
                long readerPtr = readerHandle.getPointer();
                // Internal-search get-by-row-id: the native side ignores Substrait and builds a
                // DataFrame plan filtering `__row_id__ = rowId` with pushdown enabled. __row_id__ is
                // the physical row position the parquet writer stamps at flush (sequential 0..N after
                // any index sort) and remaps into the Lucene secondary index, so the resolver's rowId
                // equals the row's __row_id__. An equality predicate returns exactly that row
                // independent of scan order and lets DataFusion prune row-groups/pages via the
                // column's min/max statistics.
                long streamPtr = executeInternalSearch(
                    readerPtr,
                    runtimePtr,
                    NativeBridge.INTERNAL_SEARCH_BY_ROW_ID,
                    rowId,
                    "DataFusion get-by-id query failed"
                );
                return readSingleRow(streamPtr);
            }
        }

        /**
         * Fetches one row from {@code parquetFilePath} by {@code __row_id__} and reconstructs the nested field
         * {@code path}'s element array from that row's {@code LIST<STRUCT>} column, rendering each leaf faithfully
         * by its OpenSearch mapping type via {@link org.opensearch.analytics.exec.ArrowSourceSerializer}. The
         * per-leaf formatter comes from the shared {@link org.opensearch.index.mapper.SourceValueFormatters}, so the
         * result is byte-identical to the codec's own FFI reconstruction. Returns {@code null} if the row/column is
         * absent, so the caller can fall back.
         */
        List<Map<String, Object>> reconstructNested(
            String parquetFilePath,
            long writerGeneration,
            String path,
            long rowId,
            org.opensearch.index.mapper.MapperService mapperService
        ) throws IOException {
            if (rowId < 0) {
                throw new IllegalArgumentException("rowId must be non-negative, got: " + rowId);
            }
            java.nio.file.Path filePath = java.nio.file.Paths.get(parquetFilePath);
            String parquetDir = filePath.getParent().toString();
            String parquetFile = filePath.getFileName().toString();
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            MonoFileWriterSet segment = MonoFileWriterSet.of(parquetDir, writerGeneration, parquetFile, 0L);
            try (ReaderHandle readerHandle = new ReaderHandle(parquetDir, List.of(segment), null, List.of(), List.of())) {
                long streamPtr = executeInternalSearch(
                    readerHandle.getPointer(),
                    runtimePtr,
                    NativeBridge.INTERNAL_SEARCH_BY_ROW_ID,
                    rowId,
                    "DataFusion nested-reconstruction get-by-row-id failed"
                );
                // The nested field is stored as one top-level LIST<STRUCT> column named `path`. A mapping-driven
                // TypeResolver renders each leaf (ip→dotted, binary→base64, date→configured format, float→single).
                ArrowSourceSerializer.TypeResolver resolver = dottedPath -> {
                    org.opensearch.index.mapper.MappedFieldType mft = mapperService.fieldType(dottedPath);
                    return mft == null ? null : org.opensearch.index.mapper.SourceValueFormatters.forField(mft);
                };
                try (
                    StreamHandle streamHandle = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                    DatafusionResultStream stream = new DatafusionResultStream(streamHandle, sharedAllocator)
                ) {
                    var iter = stream.iterator();
                    if (!iter.hasNext()) {
                        return null;
                    }
                    var batch = iter.next();
                    try (VectorSchemaRoot root = batch.getArrowRoot()) {
                        if (root.getRowCount() == 0) {
                            return null;
                        }
                        FieldVector col = root.getVector(path);
                        if (col instanceof org.apache.arrow.vector.complex.ListVector == false) {
                            return null; // not a nested LIST<STRUCT> column here — let the caller fall back
                        }
                        return ArrowSourceSerializer.toNestedSource(
                            (org.apache.arrow.vector.complex.ListVector) col,
                            0,
                            path,
                            resolver
                        );
                    }
                }
            }
        }

        /**
         * Reconstructs one nested-child leaf's ordered per-element RAW values for a row: reconstructs the owning
         * path's element list (via the same LIST&lt;STRUCT&gt; read as {@link #reconstructNested}, but with an
         * identity resolver so values stay raw — the codec's child doc-values want raw {@code long} bits / bytes,
         * not the formatted {@code _source} form), then pulls each element's {@code leafName} in flatten order.
         * Element {@code k} is the k-th child of {@code owningPath} in the block — the offset the codec scatters by.
         */
        List<Object> reconstructNestedLeaf(
            String parquetFilePath,
            String owningPath,
            String leafName,
            long rowId,
            org.opensearch.index.mapper.MapperService mapperService
        ) throws IOException {
            if (rowId < 0) {
                throw new IllegalArgumentException("rowId must be non-negative, got: " + rowId);
            }
            // DataFusion only projects the TOP-LEVEL nested column (e.g. "products"), reconstructing the whole
            // tree. For a deep owning path (products.variants.specs) we reconstruct the top-level array (raw
            // values, no formatting) then descend the intermediate container segments, flattening in document
            // order — which equals the block/post-order the codec's child-doc offsets use (verified live at
            // depth 3 and 4). Element k of the returned list is the k-th child of owningPath in the block.
            String[] segs = owningPath.split("\\.");
            String topPath = segs[0];
            java.nio.file.Path filePath = java.nio.file.Paths.get(parquetFilePath);
            String parquetDir = filePath.getParent().toString();
            String parquetFile = filePath.getFileName().toString();
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            MonoFileWriterSet segment = MonoFileWriterSet.of(parquetDir, 0L, parquetFile, 0L);
            try (ReaderHandle readerHandle = new ReaderHandle(parquetDir, List.of(segment), null, List.of(), List.of())) {
                long streamPtr = executeInternalSearch(
                    readerHandle.getPointer(),
                    runtimePtr,
                    NativeBridge.INTERNAL_SEARCH_BY_ROW_ID,
                    rowId,
                    "DataFusion nested-leaf get-by-row-id failed"
                );
                // Identity resolver keeps values RAW (byte[]/boxed Number/Boolean) — child doc-values want raw
                // long bits / bytes, not the formatted _source form.
                ArrowSourceSerializer.TypeResolver rawResolver = dottedPath -> null;
                try (
                    StreamHandle streamHandle = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                    DatafusionResultStream stream = new DatafusionResultStream(streamHandle, sharedAllocator)
                ) {
                    var iter = stream.iterator();
                    if (!iter.hasNext()) {
                        return null;
                    }
                    var batch = iter.next();
                    try (VectorSchemaRoot root = batch.getArrowRoot()) {
                        if (root.getRowCount() == 0) {
                            return null;
                        }
                        FieldVector col = root.getVector(topPath);
                        if (col instanceof org.apache.arrow.vector.complex.ListVector == false) {
                            return null;
                        }
                        List<Map<String, Object>> topElems = ArrowSourceSerializer.toNestedSource(
                            (org.apache.arrow.vector.complex.ListVector) col,
                            0,
                            topPath,
                            rawResolver
                        );
                        // Descend the intermediate container segments (segs[1..n-1]) flattening in order down to
                        // owningPath's element level, then collect each element's leaf value.
                        List<Map<String, Object>> level = topElems;
                        for (int s = 1; s < segs.length; s++) {
                            List<Map<String, Object>> next = new ArrayList<>();
                            for (Map<String, Object> e : level) {
                                Object child = e.get(segs[s]);
                                if (child instanceof List<?> childList) {
                                    for (Object ce : childList) {
                                        if (ce instanceof Map<?, ?> cm) {
                                            @SuppressWarnings("unchecked")
                                            Map<String, Object> m = (Map<String, Object>) cm;
                                            next.add(m);
                                        }
                                    }
                                }
                            }
                            level = next;
                        }
                        List<Object> out = new ArrayList<>(level.size());
                        for (Map<String, Object> e : level) {
                            out.add(e.get(leafName)); // null preserved when the leaf is absent for this element
                        }
                        return out;
                    }
                }
            }
        }

        /**
         * Runs a full-column scan {@code SELECT field, __row_id__ FROM t} over one file via SQL→Substrait and the
         * normal query path ({@link NativeBridge#INTERNAL_SEARCH_OFF}), returning the result stream pointer. The
         * caller iterates and scatters each row's value by its {@code __row_id__}.
         */
        private long scanColumnStream(long readerPtr, long runtimePtr, String field) throws IOException {
            String sql = "SELECT \"" + field + "\", \"__row_id__\" FROM " + GET_BY_ID_TABLE_ALIAS;
            byte[] plan = NativeBridge.sqlToSubstrait(readerPtr, GET_BY_ID_TABLE_ALIAS, sql, runtimePtr);
            CompletableFuture<Long> future = new CompletableFuture<>();
            WireConfigSnapshot configSnapshot = WireConfigSnapshot.builder(dfPlugin.getDatafusionSettings().getSnapshot()).build();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment configSegment = arena.allocate(WireConfigSnapshot.BYTE_SIZE);
                configSnapshot.writeTo(configSegment);
                NativeBridge.executeQueryAsync(
                    readerPtr,
                    GET_BY_ID_TABLE_ALIAS,
                    plan,
                    runtimePtr,
                    0L,
                    configSegment.address(),
                    new ActionListener<>() {
                        @Override
                        public void onResponse(Long v) {
                            future.complete(v);
                        }

                        @Override
                        public void onFailure(Exception e) {
                            future.completeExceptionally(e);
                        }
                    }
                );
                try {
                    return future.get(GET_BY_ID_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS);
                } catch (Exception e) {
                    throw new IOException("DataFusion full-column scan failed for field " + field, e);
                }
            }
        }

        org.opensearch.index.mapper.FlatColumnValueSource.LongColumn scanLongColumn(
            String parquetFilePath,
            String field,
            int rowCount
        ) throws IOException {
            long[] values = new long[rowCount];
            boolean[] present = new boolean[rowCount];
            java.nio.file.Path fp = java.nio.file.Paths.get(parquetFilePath);
            MonoFileWriterSet seg = MonoFileWriterSet.of(fp.getParent().toString(), 0L, fp.getFileName().toString(), 0L);
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            try (ReaderHandle rh = new ReaderHandle(fp.getParent().toString(), List.of(seg), null, List.of(), List.of())) {
                long streamPtr = scanColumnStream(rh.getPointer(), runtimePtr, field);
                try (
                    StreamHandle sh = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                    DatafusionResultStream stream = new DatafusionResultStream(sh, sharedAllocator)
                ) {
                    var iter = stream.iterator();
                    while (iter.hasNext()) {
                        try (VectorSchemaRoot root = iter.next().getArrowRoot()) {
                            FieldVector valVec = root.getVector(field);
                            FieldVector idVec = root.getVector("__row_id__");
                            for (int i = 0; i < root.getRowCount(); i++) {
                                if (idVec.isNull(i) || valVec.isNull(i)) {
                                    continue;
                                }
                                int row = (int) ((Number) idVec.getObject(i)).longValue();
                                if (row < 0 || row >= rowCount) {
                                    continue;
                                }
                                values[row] = numericBits(valVec, i);
                                present[row] = true;
                            }
                        }
                    }
                }
            }
            return new org.opensearch.index.mapper.FlatColumnValueSource.LongColumn(values, present);
        }

        org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn scanBytesColumn(
            String parquetFilePath,
            String field,
            int rowCount
        ) throws IOException {
            byte[][] values = new byte[rowCount][];
            boolean[] present = new boolean[rowCount];
            java.nio.file.Path fp = java.nio.file.Paths.get(parquetFilePath);
            MonoFileWriterSet seg = MonoFileWriterSet.of(fp.getParent().toString(), 0L, fp.getFileName().toString(), 0L);
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            try (ReaderHandle rh = new ReaderHandle(fp.getParent().toString(), List.of(seg), null, List.of(), List.of())) {
                long streamPtr = scanColumnStream(rh.getPointer(), runtimePtr, field);
                try (
                    StreamHandle sh = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                    DatafusionResultStream stream = new DatafusionResultStream(sh, sharedAllocator)
                ) {
                    var iter = stream.iterator();
                    while (iter.hasNext()) {
                        try (VectorSchemaRoot root = iter.next().getArrowRoot()) {
                            FieldVector valVec = root.getVector(field);
                            FieldVector idVec = root.getVector("__row_id__");
                            for (int i = 0; i < root.getRowCount(); i++) {
                                if (idVec.isNull(i) || valVec.isNull(i)) {
                                    continue;
                                }
                                int row = (int) ((Number) idVec.getObject(i)).longValue();
                                if (row < 0 || row >= rowCount) {
                                    continue;
                                }
                                Object o = valVec.getObject(i);
                                if (o instanceof byte[] b) {
                                    values[row] = b;
                                } else if (o instanceof org.apache.arrow.vector.util.Text t) {
                                    values[row] = t.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
                                } else {
                                    values[row] = String.valueOf(o).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                                }
                                present[row] = true;
                            }
                        }
                    }
                }
            }
            return new org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn(values, present);
        }

        /** Encodes an Arrow numeric cell to the raw {@code long} bits the codec's numeric doc-values serve. */
        private static long numericBits(FieldVector vec, int idx) {
            Object o = vec.getObject(idx);
            if (o instanceof Float f) {
                return Float.floatToRawIntBits(f) & 0xFFFF_FFFFL;
            }
            if (o instanceof Double d) {
                return Double.doubleToRawLongBits(d);
            }
            if (o instanceof Boolean b) {
                return b ? 1L : 0L;
            }
            return ((Number) o).longValue();
        }

        @Override
        public List<Map<String, Object>> executeRowsAboveSeqNo(List<WriterFileSet> fileSets, long seqNoFloor) throws IOException {
            long runtimePtr = dfPlugin.getDataFusionService().getNativeRuntime().get();
            List<Map<String, Object>> all = new ArrayList<>();
            for (WriterFileSet parquetSet : fileSets) {
                String parquetDir = parquetSet.directory();
                String parquetFile = parquetSet.files().iterator().next();
                MonoFileWriterSet writerSet = MonoFileWriterSet.of(parquetDir, parquetSet.writerGeneration(), parquetFile, 0L);
                try (ReaderHandle readerHandle = new ReaderHandle(parquetDir, List.of(writerSet), null, List.of(), List.of())) {
                    long readerPtr = readerHandle.getPointer();
                    // Internal-search seq-no scan: the native side ignores Substrait and builds a
                    // DataFrame plan filtering `_seq_no > seqNoFloor`, projecting only the version
                    // metadata columns, with pushdown enabled.
                    long streamPtr = executeInternalSearch(
                        readerPtr,
                        runtimePtr,
                        NativeBridge.INTERNAL_SEARCH_SEQ_NO_ABOVE,
                        seqNoFloor,
                        "DataFusion range query failed"
                    );
                    all.addAll(readAllRows(streamPtr));
                }
            }
            return all;
        }

        private List<Map<String, Object>> readAllRows(long streamPtr) {
            List<Map<String, Object>> results = new ArrayList<>();
            try (
                StreamHandle streamHandle = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                DatafusionResultStream stream = new DatafusionResultStream(streamHandle, sharedAllocator)
            ) {
                var iter = stream.iterator();
                while (iter.hasNext()) {
                    var batch = iter.next();
                    try (VectorSchemaRoot root = batch.getArrowRoot()) {
                        FieldVector idVec = root.getVector(IdFieldMapper.NAME);
                        for (int i = 0; i < root.getRowCount(); i++) {
                            Map<String, Object> row = ArrowValues.toSourceMap(root, i);
                            if (idVec != null && !idVec.isNull(i)) {
                                row.put(IdFieldMapper.NAME, Uid.decodeId((byte[]) idVec.getObject(i)));
                            }
                            results.add(row);
                        }
                    }
                }
            }
            return results;
        }

        private Map<String, Object> readSingleRow(long streamPtr) {
            try (
                StreamHandle streamHandle = new StreamHandle(streamPtr, dfPlugin.getDataFusionService().getNativeRuntime());
                DatafusionResultStream stream = new DatafusionResultStream(streamHandle, sharedAllocator)
            ) {
                var iter = stream.iterator();
                if (!iter.hasNext()) return null;
                var batch = iter.next();
                try (VectorSchemaRoot root = batch.getArrowRoot()) {
                    if (root.getRowCount() == 0) return null;
                    return ArrowValues.toSourceMap(root, 0);
                }
            }
        }

        /**
         * Runs an engine-internal point lookup through {@link NativeBridge#executeQueryAsync} and
         * returns the result stream pointer. No Substrait is generated: the native side builds the
         * filter plan from {@code mode} + {@code bound} via the DataFrame API. The plan is empty,
         * but a valid {@link WireConfigSnapshot} is still required.
         */
        private long executeInternalSearch(long readerPtr, long runtimePtr, long mode, long bound, String errorMessage) throws IOException {
            CompletableFuture<Long> future = new CompletableFuture<>();
            WireConfigSnapshot configSnapshot = WireConfigSnapshot.builder(dfPlugin.getDatafusionSettings().getSnapshot()).build();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment configSegment = arena.allocate(WireConfigSnapshot.BYTE_SIZE);
                configSnapshot.writeTo(configSegment);
                NativeBridge.executeQueryAsync(
                    readerPtr,
                    GET_BY_ID_TABLE_ALIAS,
                    EMPTY_PLAN,
                    runtimePtr,
                    0L,
                    configSegment.address(),
                    mode,
                    bound,
                    new ActionListener<>() {
                        @Override
                        public void onResponse(Long v) {
                            future.complete(v);
                        }

                        @Override
                        public void onFailure(Exception e) {
                            future.completeExceptionally(e);
                        }
                    }
                );
                try {
                    return future.get(GET_BY_ID_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS);
                } catch (Exception e) {
                    throw new IOException(errorMessage, e);
                }
            }
        }
    }

}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.parquet.bridge.ParquetColumnReader;
import org.opensearch.parquet.bridge.ParquetFileMetadata;
import org.opensearch.parquet.bridge.RustBridge;
import org.opensearch.parquet.codec.cache.BufferPool;
import org.opensearch.parquet.codec.cache.QueryParquetStats;
import org.opensearch.parquet.codec.iter.ParquetBinaryDocValues;
import org.opensearch.parquet.codec.iter.ParquetNumericDocValues;
import org.opensearch.parquet.codec.iter.ParquetSortedDocValues;
import org.opensearch.parquet.codec.iter.ParquetSortedNumericDocValues;
import org.opensearch.parquet.codec.iter.ParquetSortedSetDocValues;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Read-only {@link DocValuesProducer} that materializes per-document values from a Parquet
 * file through Lucene's DocValues iterator API, for <b>flat indices only</b>.
 *
 * <h2>Row ID = Doc ID invariant (precondition)</h2>
 * This producer relies on the composite indexing engine's guarantee that Lucene document
 * {@code N} corresponds to Parquet row position {@code N} within the same segment's Parquet
 * file, with one Lucene document per Parquet row and no translation table. The invariant is
 * verified at construction by asserting the Parquet file's {@code numRows} equals the segment's
 * {@code maxDoc}; a mismatch throws {@link IllegalStateException}. Nested documents are out of
 * scope.
 *
 * <h2>Laziness and lifecycle</h2>
 * The constructor resolves the Parquet file and checks the invariant but opens no native
 * column-reader handle. Each {@code getX(field)} lazily opens (and caches) a
 * {@link ParquetColumnReader}; {@code getSorted}/{@code getSortedSet} additionally build (and
 * cache) an {@link OrdinalTable} on first access. {@link #close()} releases every reader,
 * ordinal table, and the shared {@link BufferPool}, and is idempotent.
 *
 * <p>Not thread-safe: one producer serves one segment on one query thread.
 */
public final class ParquetDocValuesProducer extends DocValuesProducer {

    // [NESTED-TRACE] see ParquetDocValuesLeafReader — enable logger.org.opensearch.parquet.codec=TRACE.
    private static final Logger TRACE = LogManager.getLogger(ParquetDocValuesProducer.class);

    /**
     * Node-level kill switch for routing nested {@code _source} reconstruction through a registered
     * {@link org.opensearch.index.mapper.NestedSourceReconstructor} (DataFusion) instead of the FFI + Dremel path.
     * Default off; enable with {@code -Dopensearch.parquet.nested_source.datafusion_fetch=true}. Read once at class
     * load — a static toggle is sufficient because selection is per-call and the fallback is safe.
     */
    private static final boolean DATAFUSION_FETCH_ENABLED = Boolean.getBoolean(
        "opensearch.parquet.nested_source.datafusion_fetch"
    );

    private final Path parquetFile;
    private final MapperService mapperService;
    private final int maxDoc;
    private final long parquetRowCount;

    private final BufferPool bufferPool = new BufferPool();
    private final Map<String, ParquetColumnReader> columnReaders = new HashMap<>();
    private final Map<String, OrdinalTable> ordinalTables = new HashMap<>();

    /** Optional per-query accumulator; propagated to each column reader so its stats roll up at close. */
    private QueryParquetStats queryStats;

    private boolean closed;

    /**
     * Constructs the producer for {@code state}'s segment.
     *
     * @param mapperService resolves OpenSearch mapping types for DV-type validation (may be
     *                      {@code null} only in low-level tests that bypass type validation)
     * @throws IOException if the Parquet file for the segment cannot be resolved (Req 9.3)
     * @throws IllegalStateException if the Row ID = Doc ID invariant is violated (Req 12.3)
     */
    public ParquetDocValuesProducer(SegmentReadState state, MapperService mapperService) throws IOException {
        this.mapperService = mapperService;
        this.maxDoc = state.segmentInfo.maxDoc();

        Path resolved = ParquetSegmentLayout.resolve(state);
        if (resolved == null) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "no Parquet file found for segment '%s' (maxDoc=%d); cannot serve Parquet doc values",
                    state.segmentInfo.name,
                    maxDoc
                )
            );
        }
        this.parquetFile = resolved;

        ParquetFileMetadata metadata = RustBridge.getFileMetadata(parquetFile.toString());
        this.parquetRowCount = metadata.numRows();
        // Row-count invariant. On a FLAT segment every Lucene doc is one Parquet row, so the counts must be
        // equal. On a NESTED segment (block-join), a logical document is N+1 Lucene docs (N hidden children
        // followed by the parent); the Parquet primary stores exactly ONE row per parent (the nested array
        // is a LIST<STRUCT> column on that row). So Parquet numRows == parent (logical) count, which is
        // strictly LESS than Lucene maxDoc. A segment is nested iff the Lucene writer set a parent field
        // (FieldInfos.getParentField() == "__nested_parent"). docId→row translation for parents is handled
        // by ParquetDocValuesLeafReader's __row_id__-backed RowIdResolver, so here we only sanity-check the
        // counts: exact equality when flat, and 0 < numRows <= maxDoc when nested (rows == parents < docs).
        boolean nestedSegment = state.fieldInfos.getParentField() != null;
        boolean valid = nestedSegment ? (parquetRowCount > 0 && parquetRowCount <= maxDoc) : (parquetRowCount == maxDoc);
        if (valid == false) {
            throw new IllegalStateException(
                String.format(
                    Locale.ROOT,
                    "Parquet/Lucene row-count mismatch for segment '%s' (nested=%b): Lucene maxDoc=%d but Parquet numRows=%d (file=%s). "
                        + "Flat segments require numRows==maxDoc; nested segments require 0<numRows<=maxDoc (one Parquet row per "
                        + "parent doc). docId→row translation is handled separately via __row_id__.",
                    state.segmentInfo.name,
                    nestedSegment,
                    maxDoc,
                    parquetRowCount,
                    parquetFile
                )
            );
        }
    }

    /**
     * Attaches the per-query accumulator. The accumulator is propagated to every column reader
     * (existing and future) so each reader's stats roll up into the query total when it closes.
     */
    public void setQueryStats(QueryParquetStats queryStats) {
        this.queryStats = queryStats;
        if (queryStats != null) {
            for (ParquetColumnReader reader : columnReaders.values()) {
                reader.setQueryStats(queryStats);
            }
        }
    }

    /** Absolute path of the Parquet file backing this segment (bound to the search's pinned snapshot). */
    public String parquetFilePath() {
        return parquetFile.toString();
    }

    /** Number of Parquet rows (parents) in this segment's file. */
    public int parquetRowCount() {
        return Math.toIntExact(parquetRowCount);
    }

    // ── DocValuesProducer API ──

    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        ensureOpen();
        validate(field, DocValuesType.NUMERIC);
        ParquetColumnReader reader = readerFor(field, false);
        return new ParquetNumericDocValues(reader, maxDoc);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        ensureOpen();
        validate(field, DocValuesType.SORTED_NUMERIC);
        ParquetColumnReader reader = readerFor(field, true);
        return new ParquetSortedNumericDocValues(reader, maxDoc);
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        ensureOpen();
        validate(field, DocValuesType.BINARY);
        ParquetColumnReader reader = readerFor(field, false);
        return new ParquetBinaryDocValues(reader, maxDoc);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        ensureOpen();
        validate(field, DocValuesType.SORTED);
        OrdinalTable table = ordinalTableFor(field, false);
        return new ParquetSortedDocValues(table, maxDoc);
    }

    @Override
    public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        ensureOpen();
        validate(field, DocValuesType.SORTED_SET);
        OrdinalTable table = ordinalTableFor(field, true);
        return new ParquetSortedSetDocValues(table, maxDoc);
    }

    /**
     * Serves a {@link DocValuesSkipper} backed by the column's Parquet ColumnIndex (per-page
     * min/max/null-count), letting Lucene's range machinery skip whole pages whose stats
     * exclude the query range — no decode, no FFM crossing for skipped pages.
     *
     * <p>Integer-shaped columns only (INT32/INT64/BOOL physical): their raw-bits order is
     * numeric order. Float/double doc values are IEEE-754 raw bits whose order diverges from
     * numeric order for negative values, so page min/max computed on bits would be wrong for
     * them; they get no skipper. BYTE_ARRAY min/max is not exchanged as i64 at all.
     */
    @Override
    public DocValuesSkipper getSkipper(FieldInfo field) throws IOException {
        ensureOpen();
        ParquetPhysicalType phys = physicalType(field);
        if (phys != ParquetPhysicalType.INT32 && phys != ParquetPhysicalType.INT64 && phys != ParquetPhysicalType.BOOL) {
            return null;
        }
        // Match the repeated flag the field's DV accessor will use — readerFor caches by field
        // name, so opening here with a mismatched flag would poison the cache for the accessor.
        boolean repeated = field.getDocValuesType() == DocValuesType.SORTED_NUMERIC;
        ParquetColumnReader reader = readerFor(field, repeated);
        return new ParquetDocValuesSkipper(reader.pageIndex(), maxDoc);
    }

    /**
     * Verifies the underlying Parquet file is accessible and its metadata is consistent: the
     * file opens, {@code numRows} matches the value cached at construction, and the metadata
     * round-trip (which includes the writer-side CRC) succeeds.
     */
    @Override
    public void checkIntegrity() throws IOException {
        ParquetFileMetadata metadata = RustBridge.getFileMetadata(parquetFile.toString());
        if (metadata.numRows() != parquetRowCount) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "checkIntegrity: Parquet numRows changed for %s: expected %d, found %d",
                    parquetFile,
                    parquetRowCount,
                    metadata.numRows()
                )
            );
        }
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        // Cache-effectiveness is summarized once per query on the dedicated stats channel
        // ([PARQUET_DV_QUERY_STATS]); no per-segment detail line here.
        closed = true;
        IOException first = null;
        for (ParquetColumnReader reader : columnReaders.values()) {
            try {
                reader.close();
            } catch (IOException | RuntimeException e) {
                if (first == null && e instanceof IOException io) {
                    first = io;
                }
                // Suppress per-reader errors so every reader gets a chance to close.
            }
        }
        columnReaders.clear();
        ordinalTables.clear();
        bufferPool.close();
        if (first != null) {
            throw first;
        }
    }

    // ── internals ──

    /** Validates the field's mapping type supports the requested DV type, when a mapper is present. */
    private void validate(FieldInfo field, DocValuesType requested) {
        if (mapperService == null) {
            return; // low-level tests may bypass mapping validation
        }
        FieldTypeMapping.validate(field.getName(), mappingType(field), requested);
    }

    private String mappingType(FieldInfo field) {
        MappedFieldType mft = mapperService.fieldType(field.getName());
        if (mft == null) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "field '%s' has no mapping; cannot resolve Parquet column type", field.getName())
            );
        }
        return mft.typeName();
    }

    /** Resolves the Parquet physical type for a field from its mapping (or infers for tests). */
    private ParquetPhysicalType physicalType(FieldInfo field) {
        if (mapperService != null) {
            return FieldTypeMapping.forType(mappingType(field)).physical();
        }
        // Without a mapper, infer from the Lucene DV type recorded on the field.
        return switch (field.getDocValuesType()) {
            case BINARY, SORTED, SORTED_SET -> ParquetPhysicalType.BYTE_ARRAY;
            default -> ParquetPhysicalType.INT64;
        };
    }

    private ParquetColumnReader readerFor(FieldInfo field, boolean repeated) throws IOException {
        ParquetColumnReader reader = columnReaders.get(field.getName());
        if (reader == null) {
            reader = ParquetColumnReader.open(parquetFile, field.getName(), physicalType(field), repeated, bufferPool);
            reader.setQueryStats(queryStats);
            columnReaders.put(field.getName(), reader);
        }
        return reader;
    }

    /**
     * Opens (and caches) a repeated Parquet column reader for an arbitrary leaf column path + physical type.
     * Package-private so the leaf reader can materialize nested-child columns (addressed by their parquet
     * element leaf path, e.g. {@code comments.list.element.score}) for child-field doc values.
     */
    ParquetColumnReader repeatedReaderFor(String columnPath, ParquetPhysicalType physical) throws IOException {
        ParquetColumnReader reader = columnReaders.get(columnPath);
        if (reader == null) {
            reader = ParquetColumnReader.open(parquetFile, columnPath, physical, true, bufferPool);
            reader.setQueryStats(queryStats);
            columnReaders.put(columnPath, reader);
        }
        return reader;
    }

    /**
     * Reconstructs the full (arbitrarily-deep) element tree of a nested {@code LIST<STRUCT>} field at Parquet
     * row {@code parquetRow}, as the ordered {@code List<Map<String,Object>>} that the derived {@code _source}
     * serializes for that field — matching vanilla OpenSearch nested {@code _source} exactly (element order and
     * duplicates preserved; absent/null leaves omitted; empty sub-arrays produce no key).
     *
     * <p><b>How.</b> A nested field is stored in Parquet as the standard 3-level {@code LIST<STRUCT>} encoding, one primitive
     * leaf column per scalar field at any depth (e.g. {@code orgs.list.element.divisions.list.element.dname}).
     * Grouping, cardinality, nulls, and empty lists at every level are encoded by each value's Dremel
     * <b>repetition</b> (r) and <b>definition</b> (d) levels. We:
     * <ol>
     *   <li>enumerate EVERY scalar leaf under {@code path} recursively (any depth) from the mapping;</li>
     *   <li>read each leaf's {@code (rep[], def[], values)} at the row via the native reader (levels include
     *       null/empty slots, values only the present ones); and</li>
     *   <li>replay the standard Dremel striped-assembly per leaf: an index path {@code idx[1..L]} is advanced by
     *       each slot's {@code r} (reset all deeper indices), the deepest existing element level is {@code d/3},
     *       and a value is placed only when {@code d == max_def}. Elements are get-or-created by index at each
     *       level, so leaves assemble into a single shared tree regardless of read order.</li>
     * </ol>
     * This is grain-, depth-, null-, and empty-list-correct for any schema; it supersedes the earlier
     * single-level position-zip (which mis-handled interior nulls and dropped depth ≥ 2).
     *
     * @param path       the nested field's full dotted path (e.g. {@code "orgs"})
     * @param parquetRow the parent's Parquet row position (already resolved from docId via {@code __row_id__})
     * @return the ordered element objects, or an empty list if the parent's array is empty/absent
     */
    public List<Map<String, Object>> readNestedArray(String path, long parquetRow) throws IOException {
        List<NestedLeaf> leaves = nestedLeavesUnder(path);
        if (leaves.isEmpty()) {
            return List.of();
        }
        // Optional DataFusion-backed reconstruction: when enabled and a backend has registered a reconstructor,
        // rebuild this nested array by reading the SAME segment file (this.parquetFile) through DataFusion +
        // ArrowSourceSerializer instead of the FFI + Dremel assembly below. Reads the identical file the query
        // phase saw, so there is no snapshot-alignment concern. Any null return / failure falls back to the FFI
        // path — the two produce byte-identical output (both render leaves via SourceValueFormatters).
        if (DATAFUSION_FETCH_ENABLED) {
            org.opensearch.index.mapper.NestedSourceReconstructor reconstructor =
                org.opensearch.index.mapper.NestedSourceReconstructor.get();
            if (reconstructor != null) {
                try {
                    List<Map<String, Object>> df = reconstructor.reconstructNested(
                        parquetFile.toString(),
                        0L, // single-file read by path; writer generation is not needed to open the file
                        path,
                        parquetRow,
                        mapperService
                    );
                    if (df != null) {
                        return df;
                    }
                } catch (Exception e) {
                    // Fall back to FFI reconstruction below; log once at debug to avoid per-hit spam.
                    TRACE.debug("[NESTED-TRACE] DataFusion nested reconstruction failed for path='{}' row={}; "
                        + "falling back to FFI. cause: {}", path, parquetRow, e.toString());
                }
            }
        }
        if (TRACE.isTraceEnabled()) {
            List<String> cols = new ArrayList<>();
            for (NestedLeaf l : leaves) {
                cols.add(l.columnPath() + "(" + l.typeName() + ")");
            }
            TRACE.trace("[NESTED-TRACE] producer.readNestedArray(path='{}', row={}): {} leaf column(s) to assemble: {}",
                path, parquetRow, leaves.size(), cols);
        }
        // The reconstructed array of this top-level nested field. Each leaf assembles INTO this same tree by
        // navigating its index path; a shared mutable root keeps cross-leaf assembly O(total slots).
        List<Map<String, Object>> root = new ArrayList<>();
        for (NestedLeaf leaf : leaves) {
            assembleLeaf(root, leaf, parquetRow);
        }
        return root;
    }

    /**
     * One scalar leaf under a nested path: its parquet column path, physical type, field chain, depth, and the
     * OpenSearch mapping type ({@code typeName}, e.g. {@code date}/{@code ip}/{@code keyword}) — needed to format
     * the reconstructed {@code _source} value faithfully (a {@code date} leaf stores epoch millis physically but
     * must render as its date string; an {@code ip} leaf stores encoded bytes but must render dotted).
     */
    private record NestedLeaf(String columnPath, ParquetPhysicalType physical, String[] fieldChain, int depth, String typeName) {}

    /**
     * Enumerates every scalar leaf reachable under the nested {@code path}, at any depth, from the mapping.
     * A leaf name like {@code orgs.divisions.teams.tname} yields column path
     * {@code orgs.list.element.divisions.list.element.teams.list.element.tname}, field chain
     * {@code [divisions, teams, tname]} (relative to {@code path}), and depth = number of {@code list.element}
     * hops = (chain length). Order follows the mapping's field iteration (stable).
     */
    private List<NestedLeaf> nestedLeavesUnder(String path) {
        String prefix = path + ".";
        List<NestedLeaf> leaves = new ArrayList<>();
        for (MappedFieldType mft : mapperService.fieldTypes()) {
            String name = mft.name();
            if (name.startsWith(prefix) == false) {
                continue;
            }
            if (FieldTypeMapping.isSupported(mft.typeName()) == false) {
                continue; // non-scalar (object/nested container) or unsupported type — not a value leaf
            }
            // Field chain relative to the nested root, e.g. "divisions.teams.tname" -> [divisions, teams, tname].
            String relative = name.substring(prefix.length());
            String[] chain = relative.split("\\.");
            // Parquet leaf column path inserts ".list.element." between every chain segment and before the leaf.
            StringBuilder col = new StringBuilder(path);
            for (String seg : chain) {
                col.append(".list.element.").append(seg);
            }
            ParquetPhysicalType physical = FieldTypeMapping.forType(mft.typeName()).physical();
            leaves.add(new NestedLeaf(col.toString(), physical, chain, chain.length, mft.typeName()));
        }
        return leaves;
    }

    /**
     * Replays one leaf's Dremel {@code (rep, def, values)} stream at {@code parquetRow} and writes each present
     * value into the shared {@code root} tree at its reconstructed index path. The leaf's field chain has length
     * {@code L == depth}: chain[0..L-2] are nested-list container names, chain[L-1] is the scalar field name; the
     * value lives at list-level {@code L}.
     */
    private void assembleLeaf(List<Map<String, Object>> root, NestedLeaf leaf, long parquetRow) throws IOException {
        ParquetColumnReader reader = repeatedReaderFor(leaf.columnPath(), leaf.physical());
        boolean byteArray = leaf.physical() == ParquetPhysicalType.BYTE_ARRAY;
        ParquetColumnReader.LeafLevels levels = reader.readLeafLevelsAtRow(parquetRow, byteArray);
        // Per-level "element exists" definition thresholds, read from the actual Parquet schema (thresholds[k-1]
        // = min def at which a level-k element is present). Using these instead of a hardcoded per-level
        // increment makes the depth mapping correct for any LIST encoding.
        int[] thresholds = reader.levelDefThresholds();
        // The present values in slot order, boxed for the pure assembler: byte[] (or null) for BYTE_ARRAY leaves,
        // Long bits for numeric leaves.
        int valueCount = byteArray ? (levels.bytes() == null ? 0 : levels.bytes().length) : levels.longs().length;
        Object[] values = new Object[valueCount];
        for (int i = 0; i < valueCount; i++) {
            values[i] = byteArray ? levels.bytes()[i] : Long.valueOf(levels.longs()[i]);
        }
        TRACE.trace("[NESTED-TRACE]   assembleLeaf chain={} type={} at row {}: rep={} def={} maxDef={} thresholds={} presentValues={}",
            java.util.Arrays.toString(leaf.fieldChain()), leaf.typeName(), parquetRow,
            java.util.Arrays.toString(levels.rep()), java.util.Arrays.toString(levels.def()), levels.maxDef(),
            java.util.Arrays.toString(thresholds), byteArray ? valueCount + " bytes" : java.util.Arrays.toString(levels.longs()));
        // Delegate the (pure, I/O-free) Dremel record assembly. See NestedDremel for the algorithm + its tests.
        // The value formatter renders each leaf's raw physical value into its faithful _source form per the
        // OpenSearch mapping type (keyword/text bytes → String, ip → dotted, binary → base64, date → date string).
        NestedDremel.assembleLeafInto(
            root,
            leaf.fieldChain(),
            leaf.depth(),
            levels.rep(),
            levels.def(),
            levels.maxDef(),
            thresholds,
            values,
            byteArray,
            leaf.physical(),
            sourceFormatterFor(leaf.typeName())
        );
    }

    /**
     * Returns the {@code _source} value formatter for a nested-child leaf of the given OpenSearch mapping type.
     * Delegates to the shared {@link org.opensearch.index.mapper.SourceValueFormatters} so this Parquet-direct
     * reconstruction and the Arrow-batch reconstruction render every type identically (no byte-for-byte drift).
     */
    private static java.util.function.UnaryOperator<Object> sourceFormatterFor(String typeName) {
        return org.opensearch.index.mapper.SourceValueFormatters.forType(typeName);
    }

    private OrdinalTable ordinalTableFor(FieldInfo field, boolean multiValued) throws IOException {
        OrdinalTable table = ordinalTables.get(field.getName());
        if (table == null) {
            ParquetColumnReader reader = readerFor(field, multiValued);
            // The ordinal table is indexed by PARQUET ROW, so it must be built over the Parquet row count
            // (parquetRowCount), NOT Lucene maxDoc. On a flat segment these are equal; on a nested segment
            // maxDoc counts child docs too (parquetRowCount == parents < maxDoc), and iterating up to maxDoc
            // would read past the file ("row N out of range"). docId→row translation is applied separately
            // by ParquetDocValuesLeafReader's RowIdResolver before indexing into this table.
            int numRows = Math.toIntExact(parquetRowCount);
            table = multiValued ? OrdinalTable.buildMultiValued(reader, numRows) : OrdinalTable.buildSingleValued(reader, numRows);
            ordinalTables.put(field.getName(), table);
        }
        return table;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ParquetDocValuesProducer is closed");
        }
    }
}

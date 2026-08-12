/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.index;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LogByteSizeMergePolicy;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.be.lucene.LuceneDataFormat;
import org.opensearch.be.lucene.stats.LuceneShardStatsTracker;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.index.engine.dataformat.DeleteInput;
import org.opensearch.index.engine.dataformat.DeleteResult;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.dataformat.FileInfos;
import org.opensearch.index.engine.dataformat.FlushInput;
import org.opensearch.index.engine.dataformat.RowIdMapping;
import org.opensearch.index.engine.dataformat.WriteResult;
import org.opensearch.index.engine.dataformat.Writer;
import org.opensearch.index.engine.dataformat.WriterState;
import org.opensearch.index.engine.exec.WriterFileSet;
import org.opensearch.index.mapper.Uid;
import org.opensearch.plugin.stats.StatsRecorder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Executor;

/**
 * Per-generation Lucene writer that creates segments in an isolated temporary directory.
 *
 * Each instance owns its own {@link IndexWriter} and {@link Directory}. Documents are
 * added via {@link #addDoc(LuceneDocumentInput)}, and on {@link #flush(FlushInput)}, the writer
 * performs a force merge to exactly 1 segment to maintain a 1:1 mapping between the
 * Lucene segment and the corresponding Parquet file for the same writer generation.
 *
 * The writer uses a large RAM buffer (256 MB by default) to minimize the chance of
 * intermediate flushes, since all writes to a single writer come from a single thread
 * (the writer pool in {@code DataFormatAwareEngine} locks the writer during use).
 *
 * After flush, the returned {@link FileInfos} contains the temp directory path so that
 * {@link LuceneIndexingExecutionEngine#refresh} can incorporate the segment into the
 * shared {@link LuceneCommitter} writer via {@code addIndexes}.
 *
 * Row ID invariant: Each document must have a {@code __row_id__} field set via
 * {@link LuceneDocumentInput#setRowId}. The row ID must equal the Lucene doc ID
 * (0-based sequential within this writer). This is asserted during flush to ensure
 * the 1:1 offset correspondence with Parquet.
 *
 * @opensearch.experimental
 */
@ExperimentalApi
public class LuceneWriter implements Writer<LuceneDocumentInput> {

    private static final Logger logger = LogManager.getLogger(LuceneWriter.class);

    /** Segment info attribute key storing the writer generation for post-addIndexes correlation. */
    public static final String WRITER_GENERATION_ATTRIBUTE = "writer_generation";

    /**
     * Lucene parent field for nested document-block support. Declaring it via
     * {@code IndexWriterConfig.setParentField} lets {@code addDocuments} blocks (children + root) coexist
     * with index sorting: the block-aware sorter keeps each parent+children block contiguous and in
     * root-{@code __row_id__} order through sorted merges, preserving the Lucene-docId ↔ Parquet-row
     * correspondence the reader relies on. Name must not collide with any mapped field.
     */
    public static final String NESTED_PARENT_FIELD = "__nested_parent";

    /** Large RAM buffer to avoid intermediate segment flushes within a single writer in production. */
    private static final double RAM_BUFFER_SIZE_MB = 256.0;

    private final long writerGeneration;
    private final LuceneDataFormat dataFormat;
    private final LuceneShardStatsTracker stats;
    private final Path tempDirectory;
    private final Directory directory;
    private final IndexWriter indexWriter;
    private final Set<LuceneWriter> registry;
    private long mappingVersion;
    /**
     * Physical Lucene doc count — advances by the whole block (N children + 1 root) per nested doc, by 1 per
     * flat doc. This is the docId cursor and what the flushed segment's {@code maxDoc()} must equal.
     */
    private volatile long docCount;
    /**
     * LOGICAL doc count (== Parquet rows) — advances by exactly 1 per {@link #addDoc} call, flat or nested.
     * For a flat-only segment {@code logicalRowCount == docCount}; with nested blocks it is the ROOT count
     * ({@code < docCount}). This is what {@code WriterFileSet.numRows()} reports so it reconciles with the
     * Parquet primary's logical row count in the cross-format parity checks.
     */
    private volatile long logicalRowCount;
    /** Physical size of the most recently written block (1 for a flat doc), for block-aware {@link #rollbackTo}. */
    private volatile int lastBlockSize;
    private volatile boolean flushed;
    private volatile WriterState state = WriterState.ACTIVE;

    /**
     * Creates a new LuceneWriter for the given generation.
     *
     * @param writerGeneration the writer generation number
     * @param mappingVersion   the initial mapping version
     * @param dataFormat       the Lucene data format descriptor
     * @param baseDirectory    the base directory under which to create the temp directory
     * @param analyzer         the analyzer to use for tokenized fields, or null for default
     * @param codec            the codec to use, or null for default
     * @param indexSort        the index sort to apply (null when Lucene is secondary format)
     * @param stats            the shard-level stats collector
     * @throws IOException if directory creation or IndexWriter opening fails
     */
    public LuceneWriter(
        long writerGeneration,
        long mappingVersion,
        LuceneDataFormat dataFormat,
        Path baseDirectory,
        Analyzer analyzer,
        Codec codec,
        Sort indexSort,
        Set<LuceneWriter> registry,
        LuceneShardStatsTracker stats
    ) throws IOException {
        this.writerGeneration = writerGeneration;
        this.mappingVersion = mappingVersion;
        this.dataFormat = dataFormat;
        this.stats = stats;
        this.docCount = 0;
        this.logicalRowCount = 0;
        this.lastBlockSize = 0;
        this.registry = registry;

        // Create an isolated temp directory for this writer's segment
        this.tempDirectory = baseDirectory.resolve("lucene_gen_" + writerGeneration);
        logger.info("Creating directory for temp lucene writer: " + tempDirectory);
        Files.createDirectories(tempDirectory);
        this.directory = createDirectory(tempDirectory);

        IndexWriterConfig iwc = analyzer != null ? new IndexWriterConfig(analyzer) : new IndexWriterConfig();
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        // Two flush triggers wired in via overridable hooks:
        // - ramBufferSizeMB(): RAM-buffer based flushing. Default 256 MB so production
        // accumulates docs into a single in-memory segment before flush.
        // - maxBufferedDocs(): doc-count based flushing. Default DISABLE_AUTO_FLUSH so
        // production never flushes by doc count. Tests that need deterministic
        // multi-segment paths override ramBufferSizeMB() to Double.MAX_VALUE and
        // maxBufferedDocs() to a small value (e.g. 20) to force exact spill points
        // independent of JVM/Lucene version.
        iwc.setRAMBufferSizeMB(ramBufferSizeMB());
        iwc.setMaxBufferedDocs(maxBufferedDocs());
        if (indexSort != null) {
            iwc.setMergePolicy(new LogByteSizeMergePolicy());
            iwc.setIndexSort(indexSort);
            // Nested support: declaring the parent field is what allows document blocks (addDocuments)
            // to be used together with index sorting — Lucene keeps each parent+children block contiguous
            // through the sorted merge instead of scattering children away from their root.
            iwc.setParentField(NESTED_PARENT_FIELD);
        } else {
            // We are taking control here hence not allowing any merge to happen automatically.
            iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        }
        iwc.setCodec(new LuceneWriterCodec(codec, writerGeneration));
        this.indexWriter = new IndexWriter(directory, iwc);
        registry.add(this);
    }

    /**
     * Hook for tests to wrap the underlying {@link Directory} (e.g., to inject I/O
     * faults). Default is a plain {@link MMapDirectory}. Invoked once from the
     * constructor with the writer's per-generation temp directory.
     */
    protected Directory createDirectory(Path tempDirectory) throws IOException {
        return new MMapDirectory(tempDirectory);
    }

    /**
     * Hook for tests to override the IndexWriter RAM buffer size. Default is
     * {@link #RAM_BUFFER_SIZE_MB}, large enough that production accumulates all
     * docs into a single in-memory segment before flush.
     *
     * <p>For deterministic multi-segment tests, override this to a very large
     * value (e.g. {@code 1024.0}) together with a small
     * {@link #maxBufferedDocs()} value, so doc count alone drives spilling
     * independent of JVM/Lucene version and per-doc encoding size.
     *
     * <p>This method is invoked from the constructor, so overrides must be
     * stateless (return a constant) — they cannot depend on any instance fields.
     */
    protected double ramBufferSizeMB() {
        return RAM_BUFFER_SIZE_MB;
    }

    /**
     * Hook for tests to override the IndexWriter max-buffered-docs threshold.
     * Default is {@link IndexWriterConfig#DISABLE_AUTO_FLUSH}, so production
     * never flushes based on doc count and relies on {@link #ramBufferSizeMB()}.
     *
     * <p>Tests exercising the multi-segment merge path should override
     * {@link #ramBufferSizeMB()} to a very large value (e.g. {@code 1024.0})
     * and return a small value (e.g. 20) here to force exact, version-independent
     * spill points.
     *
     * <p>This method is invoked from the constructor, so overrides must be
     * stateless (return a constant) — they cannot depend on any instance fields.
     */
    protected int maxBufferedDocs() {
        return IndexWriterConfig.DISABLE_AUTO_FLUSH;
    }

    /**
     * Adds a document to the underlying {@link IndexWriter}. {@link IOException} (low-level
     * I/O fault) and {@link IllegalArgumentException} (per-doc rejection from
     * {@code IndexingChain}, e.g. oversized term, missing binary value, duplicate term) are
     * translated to {@link WriteResult.Failure}, with {@code docCount} advanced to capture
     * the partial slot and the writer moved to {@link WriterState#PENDING_ROLLBACK}; the caller must
     * then invoke {@link #rollbackTo(long)}. Anything else (e.g.
     * {@link org.apache.lucene.store.AlreadyClosedException} after a tragic event) propagates.
     */
    @Override
    public WriteResult addDoc(LuceneDocumentInput input) throws IOException {
        return StatsRecorder.recordTimeMillis(() -> {
            if (state != WriterState.ACTIVE) {
                throw new IllegalStateException("addDoc requires ACTIVE state but was " + state);
            }
            // A nested doc is written as a BLOCK of (children..., root) via addDocuments; a flat doc is a
            // single addDocument. `docCount` counts LUCENE docs (it must advance by the whole block so the
            // next block starts at the right docId), while `rowId` counts LOGICAL docs (Parquet rows) and
            // advances by 1 per logical doc. For a FLAT doc these coincide, so we keep the strict
            // rowId == docCount defense-in-depth check; for a nested block they legitimately diverge
            // (root docId = docCount + blockSize - 1 != rowId), so that invariant does not apply.
            boolean nested = input.hasNestedChildren();
            // Defense-in-depth: the caller's rowId must equal our LOGICAL row count (Parquet rows), NOT the
            // physical docCount. They coincide for a flat-only stream, but once a nested block writes N+1
            // physical docs the two diverge — comparing against logicalRowCount keeps the check correct for
            // flat docs AND flat/nested docs that follow a nested block. Applies uniformly to flat and nested.
            if (input.getRowId() != logicalRowCount) {
                throw new IllegalStateException(
                    "rowId [" + input.getRowId() + "] does not match logical row count [" + logicalRowCount + "]"
                );
            }
            int blockSize = nested ? input.getDocumentBlock().size() : 1;
            try {
                if (nested) {
                    // Contiguous Lucene block: children first (post-order), root last. The root carries
                    // __row_id__ / _primary_term; children carry _nested_path. This is what block-join needs.
                    indexWriter.addDocuments(input.getDocumentBlock());
                } else {
                    indexWriter.addDocument(input.getFinalInput());
                }
            } catch (IOException | IllegalArgumentException e) {
                // Lucene's IndexWriter may have consumed docId(s) before throwing; advance the physical
                // counter by the whole block so rollbackTo can tombstone the partial slot(s) and the docId
                // invariant for subsequent writes is preserved. The logical row was NOT admitted, so
                // logicalRowCount is not advanced (the failed rowId can be retried/rolled back cleanly).
                docCount += blockSize;
                lastBlockSize = blockSize;
                state = WriterState.PENDING_ROLLBACK;
                stats.incDocsIndexedFailures();
                return new WriteResult.Failure(e, -1L, -1L, -1L);
            }
            // The root doc is the LAST doc in the block; report its docId as the write's docId.
            long currentDocId = docCount + blockSize - 1;
            docCount += blockSize;
            logicalRowCount++; // exactly one LOGICAL doc admitted (a whole nested block counts as one row)
            lastBlockSize = blockSize;
            stats.addDocsIndexed(1);
            return new WriteResult.Success(1L, 1L, currentDocId);
        }, stats::addIndexTimeMillis);
    }

    @Override
    public void rollbackTo(long rowCount) throws IOException {
        if (state != WriterState.PENDING_ROLLBACK && state != WriterState.ACTIVE) {
            throw new IllegalStateException("rollbackTo requires ACTIVE or PENDING_ROLLBACK state but was " + state);
        }
        // `rowCount` is a LOGICAL row count (Parquet rows), matching CompositeWriter's acceptedRows — NOT the
        // physical Lucene docCount (which is larger when nested blocks were written). Compare against
        // logicalRowCount throughout.
        if (rowCount > logicalRowCount) {
            throw new IllegalStateException(
                "Cannot rollback to " + rowCount + ": only " + logicalRowCount + " logical rows admitted"
            );
        }
        if (rowCount == logicalRowCount) {
            state = WriterState.RETIRED_FLUSHABLE;
            return;
        }
        // Contract (see Writer#rollbackTo): exactly one in-flight logical doc is rolled back. For a flat doc
        // that is one physical Lucene doc; for a nested doc it is the whole trailing block (children + root =
        // lastBlockSize physical docs). Deleting only the root by its __row_id__ would orphan the child docs
        // (they carry no __row_id__), breaking block-join, so we delete the whole trailing docId range.
        if (logicalRowCount - rowCount != 1) {
            throw new IllegalStateException(
                "rollbackTo supports rolling back exactly 1 logical row, but asked to roll back "
                    + (logicalRowCount - rowCount)
            );
        }
        int blockSize = lastBlockSize <= 0 ? 1 : lastBlockSize;
        if (blockSize == 1) {
            // Flat (or single-doc) row — delete by its __row_id__ exactly as before. The tombstone is
            // expunged by the reordering/forceMerge in flush(), so the segment ends at maxDoc == docCount.
            indexWriter.deleteDocuments(NumericDocValuesField.newSlowExactQuery(DocumentInput.ROW_ID_FIELD, rowCount));
        } else {
            // Nested block — the rolled-back logical row is the whole trailing block occupying the contiguous
            // docId range [docCount - blockSize, docCount). Deleting only the root by its __row_id__ would
            // orphan the child docs (they carry no __row_id__), breaking block-join, so delete every docId in
            // the block via tryDeleteDocument against a single point-in-time reader. Tombstones are expunged by
            // the forceMerge in flush(), leaving maxDoc == docCount (post-decrement).
            long firstDoc = docCount - blockSize;
            try (DirectoryReader reader = DirectoryReader.open(indexWriter)) {
                for (long d = firstDoc; d < docCount; d++) {
                    indexWriter.tryDeleteDocument(reader, (int) d);
                }
            }
        }
        docCount -= blockSize;
        logicalRowCount = rowCount;
        lastBlockSize = 0;
        state = WriterState.RETIRED_FLUSHABLE;
    }

    @Override
    public WriterState state() {
        return state;
    }

    /**
     * Force-merges all buffered documents into exactly one segment, commits the IndexWriter,
     * and returns a {@link FileInfos} describing the resulting segment files in the temp directory.
     * <p>
     * After flush, the IndexWriter and Directory are closed. The temp directory files remain
     * on disk for {@link LuceneIndexingExecutionEngine#refresh} to incorporate via
     * {@code addIndexes}.
     *
     * <p>If the {@link FlushInput} carries a sort permutation from the primary data format
     * (e.g., Parquet sort-on-close), the Lucene segment is reordered using Lucene's IndexSort
     * mechanism with a custom SortField that remaps {@code ___row_id} values through the
     * permutation. This ensures the Lucene doc order matches the sorted Parquet row order.
     *
     * @param flushInput optional context; if it carries a sort permutation, the segment is sorted
     * @return file infos containing the temp directory path and segment file names,
     *         or {@link FileInfos#empty()} if no documents were added
     * @throws IOException if force merge, commit, or file listing fails
     */
    @Override
    public FileInfos flush(FlushInput flushInput) throws IOException {
        if (docCount == 0) {
            return FileInfos.empty();
        }

        long flushStart = System.nanoTime();
        long totalFlushDurationMs = 0;
        try {
            logger.debug(
                "flush: START generation={}, docCount={}, hasRowIdMapping={}",
                writerGeneration,
                docCount,
                flushInput.hasRowIdMapping()
            );
            indexWriter.flush();

            // If sort permutation is provided, configure the reorder merge policy
            if (flushInput.hasRowIdMapping()) {
                // RowIdMapping shouldn't be available if index has sort configurations.
                Sort configuredIndexSort = indexWriter.getConfig().getIndexSort();
                if (configuredIndexSort != null) {
                    throw new IllegalStateException(
                        "RowIdMapping should not be available when child IndexWriter is configured with IndexSort ["
                            + configuredIndexSort
                            + "] for writer generation ["
                            + writerGeneration
                            + "]"
                    );
                }
                RowIdMapping mapping = flushInput.rowIdMapping();
                // The mapping remaps __row_id__ values, which exist only on root/logical docs — so it is
                // sized in LOGICAL rows, matching logicalRowCount (== docCount for a flat-only segment).
                if (mapping.size() != logicalRowCount) {
                    throw new IllegalStateException(
                        "RowIdMapping size ["
                            + mapping.size()
                            + "] does not match logical row count ["
                            + logicalRowCount
                            + "] for writer generation ["
                            + writerGeneration
                            + "]"
                    );
                }
                configureSortedMerge(mapping, state == WriterState.RETIRED_FLUSHABLE);
            } else if (indexWriter.getConfig().getIndexSort() != null) {
                // Lucene is primary with IndexSort: Lucene natively reorders docs during
                // forceMerge, but __row_id__ values were assigned at insertion time and will
                // be scrambled after sort. Force a merge so the codec rewrites row IDs to
                // sequential 0..N-1 in the final doc order.
                if (flushInput.hasRowIdMapping()) {
                    throw new IllegalStateException(
                        "RowIdMapping must not be provided when IndexSort is configured for writer generation [" + writerGeneration + "]"
                    );
                }
            }

            // Common path: forceMerge to 1 segment, commit, build FileInfos
            long forceMergeStartNanos = System.nanoTime();
            StatsRecorder.recordTimeMillis(() -> indexWriter.forceMerge(1, true), stats::addFlushForceMergeTimeMillis);
            long forceMergeDurationMs = TimeValue.nsecToMSec(System.nanoTime() - forceMergeStartNanos);
            logger.debug(
                "flush: forceMerge complete: generation={}, docCount={}, duration={}ms",
                writerGeneration,
                docCount,
                forceMergeDurationMs
            );

            long commitStartNanos = System.nanoTime();
            indexWriter.commit();
            long commitDurationMs = TimeValue.nsecToMSec(System.nanoTime() - commitStartNanos);
            logger.debug("flush: commit complete: generation={}, duration={}ms", writerGeneration, commitDurationMs);

            // Close the IndexWriter before rewriting segment metadata.
            // This prevents IndexFileDeleter from removing our rewritten segments_N
            // file (which it wouldn't recognize as its own commit).
            indexWriter.close();

            // Verify the invariant: exactly 1 segment with docCount documents
            SegmentInfos segmentInfos = SegmentInfos.readLatestCommit(directory);
            assert segmentInfos.size() == 1 : "Expected exactly 1 segment after force merge, got " + segmentInfos.size();

            SegmentCommitInfo segmentInfo = segmentInfos.info(0);
            // After flush the segment must contain exactly docCount live docs. Any tombstones
            // from rollbackTo are expunged by the reordering forceMerge (secondary path)
            // or by LogByteSizeMergePolicy's forceMerge (primary-with-indexSort path).
            // TODO: this assertion will trip if Lucene is configured as the primary format
            // without an IndexSort and a rollbackTo has run — the no-sort/no-mapping
            // branch uses NoMergePolicy, so forceMerge is a no-op and the tombstone remains.
            // Production never wires Lucene as primary without IndexSort, but tests that do
            // need to either configure index.sort.field or avoid the rollback path.
            assert segmentInfo.info.maxDoc() == docCount : "Expected " + docCount + " docs in segment, got " + segmentInfo.info.maxDoc();

            // Invariant: ___row_id__ doc values must be sequential 0..maxDoc-1 after forceMerge.
            // This holds in all cases:
            // - Lucene secondary: docs reordered via OneMerge.reorder() + row ID rewrite
            // - Lucene primary with IndexSort: Lucene sorts natively + row ID rewrite
            // - No sort: docs added sequentially, row IDs naturally sequential
            // Wrapped in `assert` so the I/O cost is paid only when assertions are enabled.
            assert assertRowIdsSequential(directory) : "___row_id__ doc values not sequential after forceMerge for writer generation ["
                + writerGeneration
                + "]";

            // Stamp the IndexSort on the segment metadata post-commit so that
            // addIndexes(Directory...) on the shared writer sees matching sort.
            // The segment is always sorted by __row_id__ — either naturally (docs
            // written sequentially) or via OneMerge.reorder() + row ID rewrite.
            if (flushInput.hasRowIdMapping()) {
                logger.debug("Overriding segment info manually");
                rewriteSegmentInfoWithSort(segmentInfos, segmentInfo);
            }

            // Build the WriterFileSet pointing to the temp directory. numRows is the LOGICAL row count
            // (== Parquet rows), NOT the physical Lucene doc count — for a nested segment they differ
            // (docCount counts children + roots), and the cross-format parity checks compare this against
            // the Parquet primary's logical row count. For a flat-only segment logicalRowCount == docCount.
            WriterFileSet.Builder wfsBuilder = WriterFileSet.builder()
                .directory(tempDirectory)
                .writerGeneration(writerGeneration)
                .addNumRows(logicalRowCount);

            // Add all files in the segment
            for (String file : directory.listAll()) {
                if (file.startsWith("segments") == false && file.equals("write.lock") == false) {
                    wfsBuilder.addFile(file);
                }
            }

            directory.close();
            flushed = true;

            totalFlushDurationMs = TimeValue.nsecToMSec(System.nanoTime() - flushStart);
            logger.debug(
                "flush: DONE generation={}, totalRows={}, forceMerge={}ms, commit={}ms, total={}ms",
                writerGeneration,
                docCount,
                forceMergeDurationMs,
                commitDurationMs,
                totalFlushDurationMs
            );

            return FileInfos.builder().putWriterFileSet(dataFormat, wfsBuilder.build()).build();
        } finally {
            stats.incFlushTotal();
            stats.addFlushTimeMillis(totalFlushDurationMs);
        }
    }

    /**
     * Configures the child writer for sorted flush: sets a ReorderingMergePolicy
     * that forces a merge of all segments (with optional doc reordering via the mapping).
     * Row ID rewrite is already enabled on the codec at construction time.
     */
    private void configureSortedMerge(RowIdMapping mapping, boolean hasRolledBack) {
        indexWriter.getConfig().setMergePolicy(new ReorderingMergePolicy(mapping, hasRolledBack ? mapping.size() : -1));
        Codec currentCodec = indexWriter.getConfig().getCodec();
        if (currentCodec instanceof LuceneWriterCodec lwc) {
            lwc.enableRowIdRewrite();
        }
    }

    /**
     * Asserts that the {@code ___row_id} doc values in the freshly committed segment
     * are sequential 0..maxDoc-1 in doc order. This invariant must hold whether the
     * segment was produced naturally (docs added sequentially) or via reorder + row ID
     * rewrite during a sorted flush. It guards the 1:1 offset correspondence with the
     * Parquet file for the same writer generation.
     *
     * <p>This method is intended to be invoked from inside an {@code assert} statement
     * so the I/O cost is paid only when assertions are enabled (tests, dev, CI).
     *
     * @return {@code true} if all row IDs are sequential, {@code false} otherwise
     * @throws AssertionError with a descriptive message if I/O fails or the field is missing
     */
    private boolean assertRowIdsSequential(Directory directory) {
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            assert reader.leaves().size() == 1 : "Expected exactly 1 leaf reader, got " + reader.leaves().size();
            LeafReader leaf = reader.leaves().get(0).reader();
            SortedNumericDocValues rowIdDV = leaf.getSortedNumericDocValues(LuceneDocumentInput.ROW_ID_FIELD);
            if (rowIdDV == null) {
                throw new AssertionError(
                    "Field [" + LuceneDocumentInput.ROW_ID_FIELD + "] missing from segment for writer generation [" + writerGeneration + "]"
                );
            }
            int maxDoc = leaf.maxDoc();
            // Block-aware invariant. Nested documents are written as blocks (children first, root last), so
            // the old "every docId has __row_id__ == docId" no longer holds: child docs carry only
            // _nested_path (no __row_id__), and a root sits at docId = blockStart + blockSize - 1. The design
            // invariant we validate instead: child docs (no __row_id__) are skipped, and ROOT docs (those
            // WITH __row_id__) must carry sequential row-ids 0,1,2,... in ascending doc order. For a flat
            // segment every doc is a root, so this reduces exactly to the original per-docId check.
            long expectedRootRowId = 0;
            for (int docId = 0; docId < maxDoc; docId++) {
                boolean hasRowId = rowIdDV.advanceExact(docId);
                if (hasRowId == false) {
                    continue; // nested child doc — legitimately has no __row_id__ (only roots carry it)
                }
                long rowId = rowIdDV.nextValue();
                if (rowId != expectedRootRowId) {
                    throw new AssertionError(
                        "Non-sequential "
                            + LuceneDocumentInput.ROW_ID_FIELD
                            + " at docId="
                            + docId
                            + " (got "
                            + rowId
                            + ", expected "
                            + expectedRootRowId
                            + ") for writer generation ["
                            + writerGeneration
                            + "]"
                    );
                }
                expectedRootRowId++;
            }
            return true;
        } catch (IOException e) {
            throw new AssertionError("Failed to verify ___row_id__ invariant for writer generation [" + writerGeneration + "]", e);
        }
    }

    /**
     * Rewrites the segment's .si file and segments_N commit to declare the IndexSort.
     * <p>
     * After the child writer commits, the segment on disk has no IndexSort metadata
     * (because the writer operates without IndexSort to allow OneMerge.reorder()).
     * However, the segment is logically sorted by __row_id__ (either naturally sequential
     * or via reorder + row ID rewrite). This method reconstructs the SegmentInfo with
     * the expected sort, rewrites the .si file, and re-commits the SegmentInfos so that
     * addIndexes(Directory...) on the shared writer sees matching sort metadata.
     *
     * @param segmentInfos the current committed SegmentInfos
     * @param segmentCommitInfo the single segment's commit info
     * @throws IOException if rewriting fails
     */
    private void rewriteSegmentInfoWithSort(SegmentInfos segmentInfos, SegmentCommitInfo segmentCommitInfo) throws IOException {
        SegmentInfo originalInfo = segmentCommitInfo.info;
        Sort sort = new Sort(new SortedNumericSortField(LuceneDocumentInput.ROW_ID_FIELD, SortField.Type.LONG));

        // Reconstruct SegmentInfo with the IndexSort declared
        SegmentInfo sortedInfo = new SegmentInfo(
            originalInfo.dir,
            originalInfo.getVersion(),
            originalInfo.getMinVersion(),
            originalInfo.name,
            originalInfo.maxDoc(),
            originalInfo.getUseCompoundFile(),
            originalInfo.getHasBlocks(),
            originalInfo.getCodec(),
            originalInfo.getDiagnostics(),
            originalInfo.getId(),
            originalInfo.getAttributes(),
            sort
        );
        sortedInfo.setFiles(originalInfo.files());

        // Delete the existing .si file before rewriting — Lucene's createOutput
        // does not overwrite existing files.
        String siFileName = originalInfo.name + ".si";
        directory.deleteFile(siFileName);

        // Rewrite the .si file with sort metadata
        originalInfo.getCodec().segmentInfoFormat().write(directory, sortedInfo, IOContext.DEFAULT);

        // Replace the segment in SegmentInfos and re-commit so segments_N is consistent.
        // The flush() caller has already asserted segmentInfos.size() == 1, so removing
        // the original SegmentCommitInfo expresses intent better than clear() + add().
        SegmentCommitInfo newCommitInfo = new SegmentCommitInfo(
            sortedInfo,
            segmentCommitInfo.getDelCount(),
            segmentCommitInfo.getSoftDelCount(),
            segmentCommitInfo.getDelGen(),
            segmentCommitInfo.getFieldInfosGen(),
            segmentCommitInfo.getDocValuesGen(),
            segmentCommitInfo.getId()
        );
        segmentInfos.remove(segmentCommitInfo);
        segmentInfos.add(newCommitInfo);
        segmentInfos.commit(directory);
    }

    /**
     * MergePolicy that unconditionally forces a merge of all segments on {@code forceMerge},
     * even if there is only one segment. When a {@link RowIdMapping} is provided, the merge
     * also reorders docs via {@link ReorderingOneMerge}. When mapping is null, a plain
     * {@code OneMerge} is used (the merge still rewrites the segment, enabling the codec's
     * row ID rewrite to fire).
     *
     * <p>This policy is used in two scenarios:
     * <ul>
     *   <li>Lucene secondary with sort permutation from Parquet: mapping is non-null,
     *       docs are reordered to match Parquet's sort order.</li>
     *   <li>Lucene primary with IndexSort: mapping is null, Lucene's native sort already
     *       reordered docs, but the merge is needed to trigger the codec's row ID rewrite.</li>
     * </ul>
     */
    static class ReorderingMergePolicy extends MergePolicy {
        private final RowIdMapping mapping; // nullable
        private volatile boolean mergeDone = false;
        private final int rowIdToPurge;

        /**
         * @param mapping the sort permutation to apply during merge, or null for a plain
         *                rewrite merge (used when Lucene's IndexSort already sorted the docs)
         */
        ReorderingMergePolicy(RowIdMapping mapping, int rowIdToPurge) {
            if (mapping != null && mapping.isNewToOldSupported() == false) {
                throw new IllegalArgumentException("RowIdMapping must support reverse lookup (newToOld) for sorted flush reordering");
            }
            this.mapping = mapping;
            this.rowIdToPurge = rowIdToPurge;
        }

        @Override
        public MergeSpecification findMerges(MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext) {
            return null; // no automatic merges
        }

        @Override
        public MergeSpecification findForcedMerges(
            SegmentInfos segmentInfos,
            int maxSegmentCount,
            Map<SegmentCommitInfo, Boolean> segmentsToMerge,
            MergeContext mergeContext
        ) {
            if (mergeDone) {
                return null; // already done, stop the loop
            }
            mergeDone = true;

            List<SegmentCommitInfo> segments = new ArrayList<>();
            for (int i = 0; i < segmentInfos.size(); i++) {
                segments.add(segmentInfos.info(i));
            }
            if (segments.isEmpty()) {
                return null;
            }
            MergeSpecification spec = new MergeSpecification();
            if (mapping != null) {
                spec.add(new ReorderingOneMerge(segments, mapping, rowIdToPurge));
            } else {
                spec.add(new MergePolicy.OneMerge(segments));
            }
            return spec;
        }

        @Override
        public MergeSpecification findForcedDeletesMerges(SegmentInfos segmentInfos, MergeContext mergeContext) {
            return null;
        }
    }

    /**
     * Custom OneMerge that overrides {@code reorder()} to provide the sort permutation
     * as a {@link Sorter.DocMap}. This causes Lucene to physically reorder docs during
     * the merge according to the Parquet sort order.
     */
    static class ReorderingOneMerge extends MergePolicy.OneMerge {
        private final RowIdMapping mapping;
        private final int rowIdToPurge;

        ReorderingOneMerge(List<SegmentCommitInfo> segments, RowIdMapping mapping, int rowIdToPurge) {
            super(segments);
            this.mapping = mapping;
            this.rowIdToPurge = rowIdToPurge;
        }

        @Override
        public Sorter.DocMap reorder(CodecReader reader, Directory dir, Executor executor) throws IOException {
            return new Sorter.DocMap() {
                @Override
                public int oldToNew(int docID) {
                    if (docID == rowIdToPurge) {
                        return rowIdToPurge;
                    }
                    return (int) mapping.getNewRowId(docID, RowIdMapping.SINGLE_GEN);
                }

                @Override
                public int newToOld(int docID) {
                    if (docID == rowIdToPurge) {
                        return rowIdToPurge;
                    }
                    return (int) mapping.getOldRowId(docID);
                }

                @Override
                public int size() {
                    return mapping.size() + (rowIdToPurge != -1 ? 1 : 0);
                }
            };
        }

        @Override
        public void setMergeInfo(SegmentCommitInfo info) {
            super.setMergeInfo(info);
            if (info != null) {
                info.info.putAttribute(WRITER_GENERATION_ATTRIBUTE, String.valueOf(0));
            }
        }
    }

    /** {@inheritDoc} Returns the writer generation number assigned at construction. */
    @Override
    public long generation() {
        return writerGeneration;
    }

    @Override
    public boolean isSchemaMutable() {
        return true;
    }

    @Override
    public long mappingVersion() {
        return mappingVersion;
    }

    @Override
    public void updateMappingVersion(long newVersion) {
        if (newVersion > this.mappingVersion) {
            this.mappingVersion = newVersion;
        }
    }

    /** Returns heap bytes used by this writer's IndexWriter RAM buffer. Returns 0 after flush. */
    public long getHeapBytesUsed() {
        if (indexWriter.isOpen()) {
            try {
                return indexWriter.ramBytesUsed();
            } catch (AlreadyClosedException e) {
                return 0;
            }
        }
        return 0;
    }

    /**
     * Closes this writer, rolling back the IndexWriter if still open, closing the directory,
     * and deleting the temp directory. Safe to call multiple times.
     *
     * @throws IOException if cleanup fails
     */
    @Override
    public void close() throws IOException {
        // Close the IndexWriter and Directory if they haven't been closed by flush()
        try {
            if (indexWriter.isOpen()) {
                indexWriter.rollback();
            }
        } catch (Exception e) {
            logger.warn("Failed to rollback IndexWriter for generation[{}]: {}", writerGeneration, e);
        }
        try {
            directory.close();
        } catch (Exception e) {
            logger.warn("Failed to close directory for generation[{}]: {}", writerGeneration, e);
        } finally {
            registry.remove(this);
            if (flushed == false) IOUtils.rm(tempDirectory);
            state = WriterState.CLOSED;
        }
    }

    /**
     * Deletes the temporary directory created by this writer. Called externally by the
     * engine after {@code addIndexes} has hardlinked the segment files into the shared
     * writer's directory, making the originals safe to remove.
     */
    public void cleanupTempDirectory() {
        try {
            IOUtils.rm(tempDirectory);
        } catch (IOException e) {
            logger.warn("Failed to delete lucene temp directory [{}]: {}", tempDirectory, e.getMessage());
        }
    }

    /**
     * Deletes all documents containing the given term from this writer's {@link IndexWriter}.
     *
     * @param deleteInput the {@code _id} term identifying the document(s) to delete
     * @return the result of the delete operation
     * @throws IOException if a low-level I/O error occurs
     */
    @Override
    public DeleteResult deleteDocument(DeleteInput deleteInput) throws IOException {
        Term uid = new Term(deleteInput.fieldName(), Uid.encodeId(deleteInput.id()));
        indexWriter.deleteDocuments(uid);
        return new DeleteResult.Success(1L, 1L, 1L);
    }

    @Override
    public Optional<Writer<?>> getWriterForFormat(String formatName) {
        if (LuceneDataFormat.LUCENE_FORMAT_NAME.equals(formatName)) {
            return Optional.of(this);
        }
        return Optional.empty();
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.analytics.spi.DelegatedExpression;
import org.opensearch.analytics.spi.FilterDelegationHandle;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.exec.coord.CatalogSnapshot;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

/**
 * Lucene implementation of {@link FilterDelegationHandle}. Compiles delegated expressions
 * into Lucene Queries, creates Weights on demand, and produces bitsets via Scorers.
 *
 * <p>Segments are resolved by <b>writer generation</b>. The mapping
 * {@code generation → Lucene leaf index} is provided by {@link LuceneReader}, which is
 * built once at refresh time in {@link LuceneReaderManager}.
 *
 * @opensearch.internal
 */
final class LuceneFilterDelegationHandle implements FilterDelegationHandle {

    private static final Logger LOGGER = LogManager.getLogger(LuceneFilterDelegationHandle.class);

    /** The {@code __row_id__} doc-values field written on root docs; read per-collector for docId->row mapping. */
    private static final String ROW_ID_FIELD = DocumentInput.ROW_ID_FIELD;

    // TODO: lazy query compilation for performance-delegated predicates. Today
    // every delegated expression is compiled (QueryBuilder → Lucene Query) at
    // ctor time. For correctness-delegated predicates (always called) this is
    // fine. For performance-delegated predicates that DF page-pruning may never
    // consult, the compile cost is wasted. Deferring needs a way to distinguish
    // the two kinds (e.g. add a kind field on DelegatedExpression) and clear
    // semantics for compile-failure timing (eager = fail at ctor, lazy = fail
    // at first use). Revisit if this surfaces as a real cost — needs revisiting.
    private final Map<Integer, Query> queriesByAnnotationId;
    private final DirectoryReader directoryReader;
    private final IndexSearcher searcher;
    private final List<LeafReaderContext> leaves;
    private final BooleanSupplier isCancelledSupplier;
    private final Map<Long, String> generationToSegmentName;

    private final ConcurrentHashMap<Integer, Weight> weightsByProviderKey = new ConcurrentHashMap<>();
    /** providerKey -> annotationId, so every emitted bitset can be attributed to the predicate that asked for it. */
    private final ConcurrentHashMap<Integer, Integer> annotationIdByProviderKey = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, ScorerHandle> scorersByCollectorKey = new ConcurrentHashMap<>();
    private final AtomicInteger nextProviderKey = new AtomicInteger(1);
    private final AtomicInteger nextCollectorKey = new AtomicInteger(1);

    LuceneFilterDelegationHandle(
        List<DelegatedExpression> expressions,
        QueryShardContext queryShardContext,
        LuceneReader luceneReader,
        CatalogSnapshot catalogSnapshot,
        NamedWriteableRegistry namedWriteableRegistry,
        BooleanSupplier isCancelledSupplier
    ) {
        assert luceneReader != null : "luceneReader must not be null";
        assert catalogSnapshot != null : "catalogSnapshot must not be null";
        this.directoryReader = luceneReader.directoryReader();
        this.searcher = queryShardContext.searcher();
        this.leaves = directoryReader.leaves();
        this.generationToSegmentName = luceneReader.generationToSegmentName();
        this.queriesByAnnotationId = compileQueries(expressions, queryShardContext, namedWriteableRegistry);
        this.isCancelledSupplier = isCancelledSupplier;
    }

    private static Map<Integer, Query> compileQueries(
        List<DelegatedExpression> expressions,
        QueryShardContext context,
        NamedWriteableRegistry registry
    ) {
        Map<Integer, Query> queries = new HashMap<>();
        for (DelegatedExpression expr : expressions) {
            try {
                StreamInput rawInput = StreamInput.wrap(expr.getExpressionBytes());
                StreamInput input = new NamedWriteableAwareStreamInput(rawInput, registry);
                QueryBuilder queryBuilder = input.readNamedWriteable(QueryBuilder.class);
                // Rewrite FieldExistsQuery → a postings-only equivalent: the lucene-secondary segment
                // has no doc_values/norms (they live in the parquet primary), so a FieldExistsQuery
                // built from an _exists_ clause (PPL `search field!=value`) would throw at rewrite().
                Query query = LuceneQueryConversionUtils.rewriteFieldExistsForSecondary(queryBuilder.toQuery(context));
                LOGGER.info("[NAM-QDBG] annotationId={} queryBuilder=[{}] compiledQuery=[{}]", expr.getAnnotationId(), queryBuilder, query);
                queries.put(expr.getAnnotationId(), query);
            } catch (IOException exception) {
                throw new IllegalStateException(
                    "Failed to deserialize delegated expression for annotationId=" + expr.getAnnotationId(),
                    exception
                );
            }
        }
        return queries;
    }

    @Override
    public int createProvider(int annotationId) {
        Query query = queriesByAnnotationId.get(annotationId);
        if (query == null) {
            return -1;
        }
        try {
            Query rewritten = searcher.rewrite(query);
            Weight weight = searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
            int providerKey = nextProviderKey.getAndIncrement();
            weightsByProviderKey.put(providerKey, weight);
            annotationIdByProviderKey.put(providerKey, annotationId);
            LOGGER.info("[scf] createProvider annotationId={} → providerKey={} rewrittenQuery=[{}]", annotationId, providerKey, rewritten);
            // ONE-SHOT DIAGNOSTIC: scan every leaf with a fresh scorer, doc 0..end, logging where the
            // block-join's matches land in docId space and their translated rows — independent of any window.
            for (LeafReaderContext lrc : leaves) {
                try {
                    Scorer probe = weight.scorer(lrc);
                    if (probe == null) {
                        LOGGER.info(
                            "[NAM-PROBE] annotationId={} leaf(ord={},docBase={},maxDoc={}) scorer=NULL",
                            annotationId,
                            lrc.ord,
                            lrc.docBase,
                            lrc.reader().maxDoc()
                        );
                        continue;
                    }
                    SortedNumericDocValues rid = lrc.reader().getSortedNumericDocValues(ROW_ID_FIELD);
                    DocIdSetIterator it = probe.iterator();
                    StringBuilder sb = new StringBuilder();
                    int d = it.nextDoc();
                    int c = 0;
                    while (d != DocIdSetIterator.NO_MORE_DOCS && c < 40) {
                        String rowStr = "?";
                        if (rid != null && rid.advanceExact(d)) {
                            rowStr = Long.toString(rid.nextValue());
                        } else if (rid != null) {
                            rowStr = "CHILD(noRowId)";
                        }
                        sb.append("d").append(d).append("/row=").append(rowStr).append(" ");
                        d = it.nextDoc();
                        c++;
                    }
                    LOGGER.info(
                        "[NAM-PROBE] annotationId={} leaf(ord={},docBase={},maxDoc={}) blockJoinMatches(first{})=[{}]",
                        annotationId,
                        lrc.ord,
                        lrc.docBase,
                        lrc.reader().maxDoc(),
                        c,
                        sb
                    );
                } catch (Exception pe) {
                    LOGGER.warn("[NAM-PROBE] leaf probe failed", pe);
                }
            }
            return providerKey;
        } catch (IOException exception) {
            LOGGER.error("createProvider failed for annotationId=" + annotationId, exception);
            return -1;
        }
    }

    @Override
    public int createCollector(int providerKey, long writerGeneration, int minDoc, int maxDoc) {
        Weight weight = weightsByProviderKey.get(providerKey);
        if (weight == null) {
            return -1;
        }
        String segName = generationToSegmentName.get(writerGeneration);
        if (segName == null) {
            LOGGER.error(
                "createCollector: no Lucene segment for writer_generation={} (providerKey={}). Known generations: {}",
                writerGeneration,
                providerKey,
                generationToSegmentName.keySet()
            );
            return -1;
        }
        LeafReaderContext leaf = null;
        for (LeafReaderContext lrc : leaves) {
            if (unwrapSegmentReader(lrc.reader()).getSegmentInfo().info.name.equals(segName)) {
                leaf = lrc;
                break;
            }
        }
        if (leaf == null) {
            LOGGER.error(
                "createCollector: segment name [{}] not found in leaves (writerGeneration={}, providerKey={})",
                segName,
                writerGeneration,
                providerKey
            );
            return -1;
        }

        // Build (or fetch from the per-segment cache) the docId<->logical-row translator for this leaf. On a
        // NESTED segment a logical document is a block of N+1 Lucene docs (children first, root last), so
        // Lucene docIds do NOT equal Parquet logical rows; the translator maps between the two spaces via the
        // root __row_id__ doc-values. On a non-nested segment it is a pass-through (docId == row). The
        // delegated-filter window [minDoc,maxDoc) is a LOGICAL-ROW window (a Parquet row-group slice), so it
        // is validated against the logical-row count (== parent count on a nested leaf, == maxDoc on a flat).
        RowIdTranslator translator;
        try {
            translator = RowIdTranslator.forLeaf(leaf);
        } catch (IOException exception) {
            LOGGER.error(
                "createCollector: failed building row-id translator for segment="
                    + segName
                    + " (writerGeneration="
                    + writerGeneration
                    + ")",
                exception
            );
            return -1;
        }

        int logicalRowCount = translator.logicalRowCount();
        assert minDoc >= 0 && minDoc <= maxDoc && maxDoc <= logicalRowCount : "createCollector(providerKey="
            + providerKey
            + ", writerGeneration="
            + writerGeneration
            + " -> segment="
            + segName
            + "): logical-row window ["
            + minDoc
            + ","
            + maxDoc
            + ") exceeds logical-row count="
            + logicalRowCount;

        try {
            Scorer scorer = weight.scorer(leaf);
            // Per-collector cursor over __row_id__ for the docId->row lookup in collectDocs. On a nested leaf
            // this is the live doc-values the translator reads with advanceExact; null on a flat leaf (unused).
            SortedNumericDocValues rowIdDV = translator.isNested() ? leaf.reader().getSortedNumericDocValues(ROW_ID_FIELD) : null;
            int collectorKey = nextCollectorKey.getAndIncrement();
            scorersByCollectorKey.put(
                collectorKey,
                new ScorerHandle(
                    scorer,
                    annotationIdByProviderKey.getOrDefault(providerKey, -1),
                    String.valueOf(weight.getQuery()),
                    minDoc,
                    maxDoc,
                    translator,
                    rowIdDV
                )
            );
            LOGGER.debug(
                "[scf] createCollector providerKey={} writerGeneration={} rowWindow=[{},{}) nested={} → collectorKey={}",
                providerKey,
                writerGeneration,
                minDoc,
                maxDoc,
                translator.isNested(),
                collectorKey
            );
            return collectorKey;
        } catch (IOException exception) {
            LOGGER.error(
                "createCollector failed for providerKey=" + providerKey + ", writerGeneration=" + writerGeneration + ", segment=" + segName,
                exception
            );
            return -1;
        }
    }

    @Override
    public boolean isCancelled() {
        return isCancelledSupplier != null && isCancelledSupplier.getAsBoolean();
    }

    @Override
    public int collectDocs(int collectorKey, int minDoc, int maxDoc, MemorySegment out) {
        // [PERF-COLL] time the whole collectDocs call (block-join scorer scan + docId->row translate +
        // bitset build) so the RowSelection-build cost of the nested-predicate->DataFusion bridge is
        // visible per invocation in the log. See project_mustang_latency.
        long collStartNanos = System.nanoTime();
        ScorerHandle handle = scorersByCollectorKey.get(collectorKey);
        if (handle == null) {
            return -1;
        }
        if (maxDoc <= minDoc) {
            return 0;
        }
        // [minDoc,maxDoc) is a LOGICAL-ROW window (a Parquet row-group slice). The returned bitset is indexed
        // by logical-row offset (row - minDoc) — the coordinate space the Rust indexed scan turns into a
        // Parquet RowSelection. `span` is therefore a count of logical rows, correct for flat and nested.
        int span = maxDoc - minDoc;
        FixedBitSet bits = new FixedBitSet(span);

        if (handle.scorer != null) {
            RowIdTranslator translator = handle.translator;
            // Clamp the requested row window to this collector's partition (also logical-row space).
            int scanRowFrom = Math.max(minDoc, handle.partitionRowMin);
            int scanRowTo = Math.min(maxDoc, handle.partitionRowMax);

            if (scanRowFrom < scanRowTo) {
                // Translate the row window to the Lucene docId range to scan. Flat: docId == row (identity).
                // Nested: the parent docs for rows [scanRowFrom,scanRowTo) live in a contiguous docId
                // sub-range (children precede their root), resolved by direct index into the cached
                // parentDocIds array. The block-join query returns ROOT docs only, so we scan that range and
                // translate each matched root back to its logical row.
                int docFrom = translator.firstDocIdForRow(scanRowFrom);
                int docTo = translator.docIdScanBoundForRow(scanRowTo);
                int matched = 0;
                int skippedNoRow = 0;
                int skippedOutOfWindow = 0;
                StringBuilder matchTrace = new StringBuilder();
                try {
                    DocIdSetIterator iterator = handle.scorer.iterator();
                    int docId = handle.currentDoc;
                    LOGGER.info(
                        "[NAM-COLL] collectorKey={} rowWindow=[{},{}) scanRow=[{},{}) docRange=[{},{}) startCursor={} scorerCost={}",
                        collectorKey,
                        minDoc,
                        maxDoc,
                        scanRowFrom,
                        scanRowTo,
                        docFrom,
                        docTo,
                        docId,
                        iterator.cost()
                    );
                    if (docId != DocIdSetIterator.NO_MORE_DOCS) {
                        if (docId < docFrom) {
                            docId = iterator.advance(docFrom);
                        }
                        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < docTo) {
                            // Map the matched Lucene docId to its logical row. Flat: identity. Nested: the
                            // block-join match is a ROOT doc carrying __row_id__; NO_ROW means the docId had
                            // no row (e.g. a stray child on a mis-shaped query) and is skipped rather than
                            // corrupting the bitset.
                            long row = translator.rowForDocId(docId, handle.rowIdDV);
                            matched++;
                            if (matchTrace.length() < 200) {
                                matchTrace.append("doc=").append(docId).append("->row=").append(row).append(" ");
                            }
                            if (row == RowIdTranslator.NO_ROW) {
                                skippedNoRow++;
                            } else if (row >= minDoc && row < maxDoc) {
                                bits.set((int) (row - minDoc));
                            } else {
                                skippedOutOfWindow++;
                            }
                            docId = iterator.nextDoc();
                        }
                        handle.currentDoc = docId;
                    }
                    // [NAM-BITSET] the exact row bitset handed to DataFusion for ONE delegated predicate.
                    // `selectedRows` are ABSOLUTE logical row ids (bit i of the returned bitset means row
                    // minDoc+i), which is the coordinate space DataFusion turns into a Parquet RowSelection —
                    // so this line is literally "these rows survive Lucene; DF may skip the rest". Pair it with
                    // the Rust-side [SCF-DF] summary (rows_matched / rows_pruned / output_rows) to see DF act on
                    // it. Logged per (predicate x row-group window), so a query with N delegated predicates
                    // emits N of these per window.
                    LOGGER.info(
                        "[NAM-BITSET] annotationId={} collectorKey={} rowWindow=[{},{}) cardinality={}/{} selectedRows={} "
                            + "matchedDocs={} skippedNoRow={} skippedOutOfWindow={} collect_ms={} query=[{}]",
                        handle.annotationId,
                        collectorKey,
                        minDoc,
                        maxDoc,
                        bits.cardinality(),
                        span,
                        selectedRows(bits, minDoc),
                        matched,
                        skippedNoRow,
                        skippedOutOfWindow,
                        (System.nanoTime() - collStartNanos) / 1_000_000L,
                        handle.queryDesc
                    );
                } catch (IOException exception) {
                    LOGGER.warn("IOException during collectDocs, returning partial bitset", exception);
                }
            } else {
                LOGGER.info(
                    "[NAM-COLL] collectorKey={} rowWindow=[{},{}) EMPTY scan window (scanRowFrom={} >= scanRowTo={}, partition=[{},{}))",
                    collectorKey,
                    minDoc,
                    maxDoc,
                    scanRowFrom,
                    scanRowTo,
                    handle.partitionRowMin,
                    handle.partitionRowMax
                );
            }
        } else {
            LOGGER.info("[NAM-COLL] collectorKey={} scorer is NULL (query matched nothing on this leaf)", collectorKey);
        }

        long[] words = bits.getBits();
        int wordCount = (span + 63) >>> 6;
        MemorySegment.copy(words, 0, out, ValueLayout.JAVA_LONG, 0, wordCount);
        if (LOGGER.isDebugEnabled()) {
            LOGGER.debug(
                "[scf] collectDocs collectorKey={} rowWindow=[{},{}) nested={} → cardinality={} words={}",
                collectorKey,
                minDoc,
                maxDoc,
                handle.translator.isNested(),
                bits.cardinality(),
                wordCount
            );
        }
        return wordCount;
    }

    @Override
    public void releaseCollector(int collectorKey) {
        scorersByCollectorKey.remove(collectorKey);
    }

    @Override
    public void releaseProvider(int providerKey) {
        weightsByProviderKey.remove(providerKey);
    }

    @Override
    public void close() {
        weightsByProviderKey.clear();
        scorersByCollectorKey.clear();
    }

    private SegmentReader unwrapSegmentReader(LeafReader reader) {
        LeafReader current = reader;
        while (current instanceof FilterLeafReader flr) {
            current = flr.getDelegate();
        }
        return (SegmentReader) current;
    }

    /**
     * Renders the set bits of {@code bits} as ABSOLUTE logical row ids ({@code minDoc + bitIndex}) — the row
     * ids DataFusion will scan. Capped at {@value #ROW_TRACE_CAP} ids so a wide row-group window cannot flood
     * the log; the full count is always available from {@code cardinality} on the same line.
     */
    private static String selectedRows(FixedBitSet bits, int minDoc) {
        StringBuilder sb = new StringBuilder("[");
        int shown = 0;
        for (int i = bits.nextSetBit(0); i != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;) {
            if (shown == ROW_TRACE_CAP) {
                sb.append(", ...");
                break;
            }
            if (shown++ > 0) {
                sb.append(", ");
            }
            sb.append(minDoc + i);
            i = i + 1 >= bits.length() ? org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS : bits.nextSetBit(i + 1);
        }
        return sb.append(']').toString();
    }

    /** Max row ids printed by {@link #selectedRows}. */
    private static final int ROW_TRACE_CAP = 64;

    private static final class ScorerHandle {
        final Scorer scorer;
        /** The delegated predicate this collector serves — lets each emitted bitset be attributed in the log. */
        final int annotationId;
        /** The compiled Lucene query, logged alongside the bitset so the two can be read together. */
        final String queryDesc;
        /** Partition bounds in LOGICAL-ROW space (Parquet row-group slice), inclusive-exclusive. */
        final int partitionRowMin;
        final int partitionRowMax;
        /** docId-to-row translator for the collector's leaf (pass-through on a non-nested leaf). */
        final RowIdTranslator translator;
        /**
         * Live {@code __row_id__} doc-values cursor for the docId->row lookup in collectDocs; null on a flat
         * leaf (the translator is identity there and never reads it). Advanced forward across matches.
         */
        final SortedNumericDocValues rowIdDV;
        /** Forward cursor in Lucene docId space, monotonic across successive collectDocs calls. */
        int currentDoc = -1;

        ScorerHandle(
            Scorer scorer,
            int annotationId,
            String queryDesc,
            int partitionRowMin,
            int partitionRowMax,
            RowIdTranslator translator,
            SortedNumericDocValues rowIdDV
        ) {
            this.scorer = scorer;
            this.annotationId = annotationId;
            this.queryDesc = queryDesc;
            this.partitionRowMin = partitionRowMin;
            this.partitionRowMax = partitionRowMax;
            this.translator = translator;
            this.rowIdDV = rowIdDV;
        }
    }
}

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
import org.opensearch.be.lucene.index.LuceneDocumentInput;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.index.engine.exec.coord.CatalogSnapshot;
import org.opensearch.index.mapper.NestedPathFieldMapper;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;

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

    // TODO: lazy query compilation for performance-delegated predicates. Today
    // every delegated expression is compiled (QueryBuilder → Lucene Query) at
    // ctor time. For correctness-delegated predicates (always called) this is
    // fine. For performance-delegated predicates that DF page-pruning may never
    // consult, the compile cost is wasted. Deferring needs a way to distinguish
    // the two kinds (e.g. add a kind field on DelegatedExpression) and clear
    // semantics for compile-failure timing (eager = fail at ctor, lazy = fail
    // at first use). Revisit if this surfaces as a real cost — needs revisiting.
    private final Map<Integer, Query> queriesByAnnotationId;
    /**
     * Child-grain nested split only: for a delegated expression that is a CHILD-scoped query (a
     * {@code NESTED_ANY_MATCH_CHILD} peer — see {@code NestedAnyMatchChildSerializer}), the nested path
     * its {@code _nested_path} filter restricts to. Absent for every non-child delegated expression. Used
     * to select the correct per-level ordinal in {@link #collectChildDocs} without marshaling the path
     * across the FFM boundary.
     */
    private final Map<Integer, String> childPathByAnnotationId;
    /** {@code providerKey → child nested path}, populated in {@link #createProvider} from {@link #childPathByAnnotationId}. */
    private final ConcurrentHashMap<Integer, String> childPathByProviderKey = new ConcurrentHashMap<>();
    private final DirectoryReader directoryReader;
    private final IndexSearcher searcher;
    private final List<LeafReaderContext> leaves;
    private final BooleanSupplier isCancelledSupplier;
    private final Map<Long, String> generationToSegmentName;

    private final ConcurrentHashMap<Integer, Weight> weightsByProviderKey = new ConcurrentHashMap<>();
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
        this.childPathByAnnotationId = new HashMap<>();
        this.queriesByAnnotationId = compileQueries(expressions, queryShardContext, namedWriteableRegistry, this.childPathByAnnotationId);
        this.isCancelledSupplier = isCancelledSupplier;
    }

    private static Map<Integer, Query> compileQueries(
        List<DelegatedExpression> expressions,
        QueryShardContext context,
        NamedWriteableRegistry registry,
        Map<Integer, String> childPathOut
    ) {
        Map<Integer, Query> queries = new HashMap<>();
        for (DelegatedExpression expr : expressions) {
            try {
                StreamInput rawInput = StreamInput.wrap(expr.getExpressionBytes());
                StreamInput input = new NamedWriteableAwareStreamInput(rawInput, registry);
                QueryBuilder queryBuilder = input.readNamedWriteable(QueryBuilder.class);
                // Child-grain split: if this is a CHILD-scoped query (NestedAnyMatchChildSerializer emits
                // bool(must: term(path.field), filter: term(_nested_path, path))), record the nested path
                // so collectChildDocs can pick the right per-level ordinal without an FFM path arg.
                String childPath = extractChildNestedPath(queryBuilder);
                if (childPath != null) {
                    childPathOut.put(expr.getAnnotationId(), childPath);
                }
                // Rewrite FieldExistsQuery → a postings-only equivalent: the lucene-secondary segment
                // has no doc_values/norms (they live in the parquet primary), so a FieldExistsQuery
                // built from an _exists_ clause (PPL `search field!=value`) would throw at rewrite().
                Query query = LuceneQueryConversionUtils.rewriteFieldExistsForSecondary(queryBuilder.toQuery(context));
                // [NESTED] Trace the compiled delegated query. queryBuilder type tells us whether a nested
                // predicate was wrapped as a NestedQueryBuilder (→ ToParentBlockJoinQuery, returns parents)
                // or shipped as a flat term/range (returns child docs — the nested-delegation bug).
                // Grep: NESTED lucene-delegation.
                LOGGER.info(
                    "[NESTED] compileQueries annotationId={} builder={} → luceneQuery={} ({})",
                    expr.getAnnotationId(),
                    queryBuilder.getClass().getSimpleName(),
                    query.getClass().getSimpleName(),
                    query
                );
                LOGGER.info(
                    "[TRACE-STEP] Lucene RECEIVED delegated expression annotationId={} (deserialized from bytes shipped by the coordinator) -> compiled to runnable Lucene Query [{}], ready and waiting for DataFusion's FFM call",
                    expr.getAnnotationId(),
                    query
                );
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

    /**
     * If {@code queryBuilder} is the CHILD-scoped shape {@code NestedAnyMatchChildSerializer} emits —
     * {@code bool(must: term(path.field, value), filter: term(_nested_path, path))} — return the nested
     * {@code path} carried by the {@code _nested_path} filter clause; otherwise {@code null}. This is the
     * only builder shape that carries a {@code _nested_path} term, so the match is unambiguous.
     */
    private static String extractChildNestedPath(QueryBuilder queryBuilder) {
        if (queryBuilder instanceof BoolQueryBuilder bool) {
            for (QueryBuilder filter : bool.filter()) {
                if (filter instanceof TermQueryBuilder term && NestedPathFieldMapper.NAME.equals(term.fieldName())) {
                    Object value = term.value();
                    return value == null ? null : value.toString();
                }
            }
        }
        return null;
    }

    @Override
    public int createProvider(int annotationId) {
        Query query = queriesByAnnotationId.get(annotationId);
        if (query == null) {
            return -1;
        }
        try {
            Weight weight = searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1.0f);
            int providerKey = nextProviderKey.getAndIncrement();
            weightsByProviderKey.put(providerKey, weight);
            // Carry the child nested path (if this is a child-scoped peer) to the provider key so the
            // per-collector ScorerHandle can pick the right ordinal in collectChildDocs.
            String childPath = childPathByAnnotationId.get(annotationId);
            if (childPath != null) {
                childPathByProviderKey.put(providerKey, childPath);
            }
            LOGGER.debug("[scf] createProvider annotationId={} → providerKey={}", annotationId, providerKey);
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

        // Build the docId↔logical-row translator for this leaf. On a NESTED segment a logical document
        // is a block of N+1 Lucene docs (children first, parent last), so Lucene docIds do NOT equal
        // Parquet logical rows; the translator maps between the two spaces via the parent __row_id__
        // doc-values. On a non-nested segment it is a pass-through (docId == row). See RowIdTranslator.
        RowIdTranslator translator;
        try {
            translator = RowIdTranslator.forLeaf(leaf);
        } catch (IOException exception) {
            LOGGER.error(
                "createCollector: failed building row-id translator for segment=" + segName + " (writerGeneration=" + writerGeneration + ")",
                exception
            );
            return -1;
        }

        // [minDoc,maxDoc) is a LOGICAL-ROW window (a Parquet row-group slice), NOT a Lucene docId window
        // (see indexed_executor stream.rs: min_doc = rg.first_row). Validate it against the logical-row
        // count (== parent count on a nested leaf, == leaf.maxDoc() on a non-nested leaf).
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
            int collectorKey = nextCollectorKey.getAndIncrement();
            scorersByCollectorKey.put(
                collectorKey,
                new ScorerHandle(scorer, minDoc, maxDoc, translator, childPathByProviderKey.get(providerKey))
            );
            // [NESTED] createCollector trace: rowWindow is a Parquet row-group slice (logical-row space);
            // nested=true means the docId→row translation is active for this leaf. Grep: NESTED lucene-delegation.
            LOGGER.info(
                "[NESTED] createCollector providerKey={} writerGeneration={} rowWindow=[{},{}) nested={} logicalRows={} scorerNull={} → collectorKey={}",
                providerKey,
                writerGeneration,
                minDoc,
                maxDoc,
                translator.isNested(),
                logicalRowCount,
                scorer == null,
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
        ScorerHandle handle = scorersByCollectorKey.get(collectorKey);
        if (handle == null) {
            return -1;
        }
        if (maxDoc <= minDoc) {
            return 0;
        }
        // [minDoc,maxDoc) is a LOGICAL-ROW window (a Parquet row-group slice). The returned bitset is
        // indexed by logical-row offset (row - minDoc), which is the coordinate space the Rust indexed
        // scan turns into a Parquet RowSelection. `span` is therefore a count of logical rows, correct
        // for both nested and non-nested leaves.
        int rowMin = minDoc;
        int rowMax = maxDoc;
        int span = rowMax - rowMin;
        FixedBitSet bits = new FixedBitSet(span);

        if (handle.scorer != null) {
            // Clamp the requested row window to the collector's partition, then translate it to the
            // Lucene docId range to scan. On a non-nested leaf docId == row so this is identity; on a
            // nested leaf the parent docs for rows [scanRowFrom,scanRowTo) live in a contiguous docId
            // sub-range resolved by binary search over the (row-id sorted) parent docs.
            int scanRowFrom = Math.max(rowMin, handle.partitionRowMin);
            int scanRowTo = Math.min(rowMax, handle.partitionRowMax);

            if (scanRowFrom < scanRowTo) {
                RowIdTranslator translator = handle.translator;
                int docFrom = translator.firstDocIdForRow(scanRowFrom);
                int docTo = translator.docIdScanBoundForRow(scanRowTo);
                // [NESTED] Trace the row-window → docId-range translation. On a flat leaf docFrom/docTo
                // equal the row bounds; on a nested leaf they span the block docId range. Any matched
                // docId with no logical row (skippedNoRow) points at a bug in the delegated query shape.
                int matchedDocs = 0;
                int skippedNoRow = 0;
                int skippedOutOfWindow = 0;
                try {
                    DocIdSetIterator iterator = handle.scorer.iterator();
                    int docId = handle.currentDoc;
                    if (docId != DocIdSetIterator.NO_MORE_DOCS) {
                        if (docId < docFrom) {
                            docId = iterator.advance(docFrom);
                        }
                        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < docTo) {
                            // Translate the matched Lucene docId to its logical row. A nested leaf's
                            // block-join query returns PARENT docs, each of which carries __row_id__;
                            // NO_ROW (-1) means the match had no logical row (e.g. a stray child doc on
                            // a mis-shaped query) and is skipped rather than corrupting the bitset.
                            matchedDocs++;
                            long row = translator.rowForDocId(docId);
                            if (row == RowIdTranslator.NO_ROW) {
                                skippedNoRow++;
                            } else if (row >= rowMin && row < rowMax) {
                                bits.set((int) (row - rowMin));
                            } else {
                                skippedOutOfWindow++;
                            }
                            docId = iterator.nextDoc();
                        }
                        handle.currentDoc = docId;
                    }
                    LOGGER.info(
                        "[NESTED] collectDocs collectorKey={} rowWindow=[{},{}) nested={} docScan=[{},{}) matchedDocs={} "
                            + "skippedNoRow={} skippedOutOfWindow={} → cardinality={}",
                        collectorKey,
                        rowMin,
                        rowMax,
                        translator.isNested(),
                        docFrom,
                        docTo,
                        matchedDocs,
                        skippedNoRow,
                        skippedOutOfWindow,
                        bits.cardinality()
                    );
                } catch (IOException exception) {
                    LOGGER.warn("[NESTED] IOException during collectDocs, returning partial bitset", exception);
                }
            }
        }

        long[] words = bits.getBits();
        int wordCount = (span + 63) >>> 6;
        MemorySegment.copy(words, 0, out, ValueLayout.JAVA_LONG, 0, wordCount);
        LOGGER.info(
            "[TRACE-STEP] Lucene RETURNED bitset for collectorKey={}: cardinality={} matching rows out of rowWindow=[{},{}) -> copied into the MemorySegment `out`, this is the FFM return path back into Rust, where DataFusion will AND/intersect this with its own page-pruning and other predicates",
            collectorKey,
            bits.cardinality(),
            minDoc,
            maxDoc
        );
        return wordCount;
    }

    /**
     * CHILD-GRAIN collect lane for the nested predicate split (mirror of {@link #collectDocs}, but the
     * returned bitset is indexed by CHILD ELEMENT ORDINAL, not parent row).
     *
     * <p>For a nested predicate split across engines (keyword child clause → Lucene, numeric/range child
     * clause → Parquet/DataFusion), the keyword clause is shipped as a CHILD-scoped query (a plain
     * {@code TermQuery} on {@code path.field} restricted to {@code _nested_path:path} children — NOT wrapped
     * in a NestedQueryBuilder/block-join), so its scorer yields CHILD docIds. Each matched child docId is
     * translated to its element ordinal within the query's per-RG dense child-id space via
     * {@link RowIdTranslator#childOffset} + the caller-supplied {@code childBase} prefix-sums, and that bit is
     * set. The driving backend (DataFusion) then intersects this per-element bitset with its own per-element
     * evaluation of the range clause, at CHILD grain, before the ∃ roll-up to parents — preserving element
     * correlation (the same child must satisfy both), matching vanilla nested semantics.
     *
     * <p>{@code childBase[r - rowMin]} = the CALLER's element index at which parent row r's elements begin
     * (built from the decoded Arrow LIST {@code value_offsets} of the current batch); element k of row r has
     * child-id {@code childBase[r-rowMin] + k}. A {@code childBase} entry of {@code -1} marks a parent row
     * NOT present in the caller's current batch — matched children of such a row are skipped. The bitset
     * spans {@code totalChildren} bits. The nested path is taken from the collector's own child-scoped query
     * ({@link ScorerHandle#childPath}); no path is marshaled across FFM.
     *
     * @return number of u64 words written, or -1 on error.
     */
    @Override
    public int collectChildDocs(int collectorKey, int minDoc, int maxDoc, int[] childBase, int totalChildren, MemorySegment out) {
        ScorerHandle handle = scorersByCollectorKey.get(collectorKey);
        if (handle == null) {
            return -1;
        }
        String path = handle.childPath;
        if (path == null) {
            LOGGER.error("[NESTED-SPLIT] collectChildDocs collectorKey={} has no child path (not a child-scoped collector)", collectorKey);
            return -1;
        }
        int span = maxDoc - minDoc;
        if (span <= 0 || totalChildren <= 0) {
            return 0;
        }
        if (childBase == null || childBase.length != span) {
            LOGGER.error(
                "[NESTED-SPLIT] collectChildDocs collectorKey={} childBase length={} != row span={}",
                collectorKey,
                childBase == null ? -1 : childBase.length,
                span
            );
            return -1;
        }
        FixedBitSet bits = new FixedBitSet(totalChildren);
        if (handle.scorer != null) {
            int scanRowFrom = Math.max(minDoc, handle.partitionRowMin);
            int scanRowTo = Math.min(maxDoc, handle.partitionRowMax);
            if (scanRowFrom < scanRowTo) {
                RowIdTranslator translator = handle.translator;
                int docFrom = translator.firstDocIdForRow(scanRowFrom);
                int docTo = translator.docIdScanBoundForRow(scanRowTo);
                int matchedChildren = 0;
                int skippedNoChild = 0;
                try {
                    DocIdSetIterator iterator = handle.scorer.iterator();
                    int docId = iterator.docID() == -1 ? iterator.nextDoc() : iterator.docID();
                    if (docId < docFrom) {
                        docId = iterator.advance(docFrom);
                    }
                    while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < docTo) {
                        // Matched a CHILD doc → (root row, element offset). row must be in [minDoc,maxDoc);
                        // child-id = childBase[row-minDoc] + offset. A matched doc with no child ordinal for
                        // this path (offset < 0), or whose row is not in the caller's current batch
                        // (childBase[row-minDoc] == -1), is skipped, not corrupting the set — the -1 sentinel
                        // must be checked BEFORE adding `off`, else -1 + off could land on a valid index.
                        int row = translator.childRow(path, docId);
                        int off = translator.childOffset(path, docId);
                        if (row < 0 || off < 0 || row < minDoc || row >= maxDoc) {
                            skippedNoChild++;
                        } else {
                            int base = childBase[row - minDoc];
                            if (base < 0) {
                                skippedNoChild++; // parent row not delivered in this batch
                            } else {
                                int childId = base + off;
                                if (childId >= 0 && childId < totalChildren) {
                                    bits.set(childId);
                                    matchedChildren++;
                                } else {
                                    skippedNoChild++;
                                }
                            }
                        }
                        docId = iterator.nextDoc();
                    }
                    LOGGER.info(
                        "[NESTED-SPLIT] collectChildDocs collectorKey={} path={} rowWindow=[{},{}) totalChildren={} "
                            + "matchedChildren={} skippedNoChild={} → cardinality={}",
                        collectorKey,
                        path,
                        minDoc,
                        maxDoc,
                        totalChildren,
                        matchedChildren,
                        skippedNoChild,
                        bits.cardinality()
                    );
                } catch (IOException exception) {
                    LOGGER.warn("[NESTED-SPLIT] IOException during collectChildDocs, returning partial bitset", exception);
                }
            }
        }
        long[] words = bits.getBits();
        int wordCount = (totalChildren + 63) >>> 6;
        MemorySegment.copy(words, 0, out, ValueLayout.JAVA_LONG, 0, wordCount);
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

    private static final class ScorerHandle {
        final Scorer scorer;
        /** Partition bounds in LOGICAL-ROW space (Parquet row-group slice), inclusive-exclusive. */
        final int partitionRowMin;
        final int partitionRowMax;
        /** docId↔row translator for the collector's leaf (pass-through on a non-nested leaf). */
        final RowIdTranslator translator;
        /**
         * Child-grain split only: the nested path this collector's child-scoped query targets, so
         * {@link #collectChildDocs} can select the right per-level ordinal. Null for every non-child
         * (flat / whole-node block-join) collector.
         */
        final String childPath;
        /** Forward cursor in Lucene docId space, monotonic across successive collectDocs calls. */
        int currentDoc = -1;

        ScorerHandle(Scorer scorer, int partitionRowMin, int partitionRowMax, RowIdTranslator translator, String childPath) {
            this.scorer = scorer;
            this.partitionRowMin = partitionRowMin;
            this.partitionRowMax = partitionRowMax;
            this.translator = translator;
            this.childPath = childPath;
        }
    }

    /**
     * Translates between Lucene docId space and Parquet logical-row space for one leaf.
     *
     * <p><b>Why this exists.</b> Under OpenSearch {@code nested} mapping a single logical document is
     * indexed as a contiguous block of N+1 Lucene docs (N children, then the parent — see
     * {@link LuceneDocumentInput}), while the Parquet primary stores exactly one row per logical
     * document. So on a nested segment {@code luceneDocId != parquetRow}. The delegated-filter contract
     * is expressed in logical-row space (the {@code [minDoc,maxDoc)} window is a Parquet row-group
     * slice, and the returned bitset indexes logical rows), so a match found in Lucene docId space must
     * be translated back to its logical row before it can restrict the Parquet scan.
     *
     * <p><b>The key.</b> Only parent docs carry the {@code __row_id__} {@link org.apache.lucene.index.NumericDocValues}
     * (set in {@link LuceneDocumentInput#setRowId}); it equals the Parquet logical row. Children have no
     * {@code __row_id__}. The segment is index-sorted by {@code __row_id__}, so the parent docs occur in
     * ascending {@code __row_id__} (== ascending docId) order — which makes the row↔docId maps a simple
     * ordered scan plus binary search.
     *
     * <p><b>Non-nested leaves.</b> When the leaf has no parent field, every doc is its own logical row
     * and {@code docId == row}; the translator is a zero-overhead pass-through, so the non-nested
     * delegation path is byte-for-byte unchanged.
     */
    static final class RowIdTranslator {
        static final long NO_ROW = -1L;

        /** True when the leaf is a nested segment (has a Lucene parent field). */
        private final boolean nested;
        /** Number of logical rows = parent count (nested) or leaf.maxDoc() (non-nested). */
        private final int logicalRowCount;
        /**
         * Nested only: parent docIds in ascending order (index i is the parent whose ordinal is i), and
         * the parallel {@code __row_id__} value for each. {@code parentDocIds[i]} ↔ {@code rowIds[i]}.
         * Null on a non-nested (pass-through) leaf.
         */
        private final int[] parentDocIds;
        private final long[] rowIds;
        /**
         * Nested child-ordinal lane (child docId → (root row, element offset)), used ONLY by the child-grain
         * split delegation path. Lazily built + cached the first time {@link #childOffset} / {@link #childRow}
         * is called for a path, since most delegations (flat / whole-node block-join) never need it. Holds the
         * leaf so the lazy build can read {@code _nested_path} postings on demand. Null on a non-nested leaf.
         */
        private final LeafReader leafForChildOrdinal;
        private NestedChildOrdinalMap childOrdinalMap;

        private RowIdTranslator(boolean nested, int logicalRowCount, int[] parentDocIds, long[] rowIds, LeafReader leafForChildOrdinal) {
            this.nested = nested;
            this.logicalRowCount = logicalRowCount;
            this.parentDocIds = parentDocIds;
            this.rowIds = rowIds;
            this.leafForChildOrdinal = leafForChildOrdinal;
        }

        /**
         * The element offset of a matched CHILD docId within its root's list for {@code path} (child-grain
         * split lane), or {@code -1} if {@code docId} is not a child at {@code path}. Lazily builds the
         * per-segment {@link NestedChildOrdinalMap} covering {@code path} on first use. Only valid on a
         * nested leaf.
         */
        int childOffset(String path, int docId) throws IOException {
            if (nested == false || leafForChildOrdinal == null) {
                return -1;
            }
            ensureChildOrdinal(path);
            return childOrdinalMap.offsetForDocId(path, docId);
        }

        /** The root Parquet row of a matched CHILD docId for {@code path}, or {@code -1}. See {@link #childOffset}. */
        int childRow(String path, int docId) throws IOException {
            if (nested == false || leafForChildOrdinal == null) {
                return -1;
            }
            ensureChildOrdinal(path);
            return childOrdinalMap.rowForDocId(path, docId);
        }

        private void ensureChildOrdinal(String path) throws IOException {
            if (childOrdinalMap == null || childOrdinalMap.paths().contains(path) == false) {
                // Build (or rebuild widening) the map to cover `path`. Rebuild is rare (one delegated nested
                // path per query in the common case) and cached thereafter.
                java.util.Set<String> want = new java.util.HashSet<>();
                if (childOrdinalMap != null) {
                    want.addAll(childOrdinalMap.paths());
                }
                want.add(path);
                childOrdinalMap = NestedChildOrdinalMap.build(leafForChildOrdinal, want);
            }
        }

        /**
         * Builds the translator for {@code leaf}. Detects nested via {@link org.apache.lucene.index.FieldInfos#getParentField()}
         * (set to {@code __nested_parent} by the writer for nested indices). On a nested leaf, materializes
         * the parent docId → {@code __row_id__} correspondence once by scanning the {@code __row_id__}
         * doc-values (present only on parents).
         */
        static RowIdTranslator forLeaf(LeafReaderContext leafContext) throws IOException {
            LeafReader reader = leafContext.reader();
            int maxDoc = reader.maxDoc();
            String parentField = reader.getFieldInfos().getParentField();
            // __row_id__ is written as a SortedNumericDocValuesField (see LuceneDocumentInput.setRowId),
            // so it MUST be read via getSortedNumericDocValues — getNumericDocValues returns null for it.
            SortedNumericDocValues rowIdDV = reader.getSortedNumericDocValues(LuceneDocumentInput.ROW_ID_FIELD);

            // Non-nested segment (no parent field, or no __row_id__ doc-values): pass-through, docId==row.
            if (parentField == null || rowIdDV == null) {
                // [NESTED] Trace which leaves are treated as flat (docId==row). On a nested index a null
                // parentField here is the smoking gun that the block-join parent field isn't visible to
                // the delegation reader. Grep: NESTED lucene-delegation.
                LOGGER.info(
                    "[NESTED] RowIdTranslator.forLeaf: FLAT leaf (docId==row) maxDoc={} parentField={} hasRowIdDV={}",
                    maxDoc,
                    parentField,
                    rowIdDV != null
                );
                return new RowIdTranslator(false, maxDoc, null, null, null);
            }

            // Nested segment: collect (parentDocId, rowId) pairs in docId order. __row_id__ exists only on
            // parents, so iterating the doc-values naturally visits parents in ascending docId order —
            // and, because the segment is index-sorted by __row_id__, in ascending rowId order too.
            int[] docIds = new int[maxDoc];
            long[] rows = new long[maxDoc];
            int n = 0;
            for (int docId = rowIdDV.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = rowIdDV.nextDoc()) {
                docIds[n] = docId;
                // Each parent carries exactly one __row_id__ value (docCount()==1); nextValue() reads it.
                rows[n] = rowIdDV.nextValue();
                n++;
            }
            // [NESTED] Trace the block layout the delegation reader sees: maxDoc (parents+children),
            // parent count (== logical rows == Parquet rows), and the parent docId→row_id mapping. This is
            // the correspondence that makes a Lucene block-join match restrict the right Parquet row.
            LOGGER.info(
                "[NESTED] RowIdTranslator.forLeaf: NESTED leaf parentField={} maxDoc={} parentCount={} firstParents={} firstRowIds={}",
                parentField,
                maxDoc,
                n,
                java.util.Arrays.toString(java.util.Arrays.copyOf(docIds, Math.min(n, 8))),
                java.util.Arrays.toString(java.util.Arrays.copyOf(rows, Math.min(n, 8)))
            );
            if (n == docIds.length) {
                return new RowIdTranslator(true, n, docIds, rows, reader);
            }
            int[] trimmedDocIds = new int[n];
            long[] trimmedRows = new long[n];
            System.arraycopy(docIds, 0, trimmedDocIds, 0, n);
            System.arraycopy(rows, 0, trimmedRows, 0, n);
            return new RowIdTranslator(true, n, trimmedDocIds, trimmedRows, reader);
        }

        boolean isNested() {
            return nested;
        }

        int logicalRowCount() {
            return logicalRowCount;
        }

        /**
         * The logical row for a matched docId. Non-nested: identity. Nested: the docId must be a parent
         * (block-join queries return parents); returns its {@code __row_id__}, or {@link #NO_ROW} if the
         * docId is not a parent (defensive — a correctly block-joined query never yields a child).
         */
        long rowForDocId(int docId) {
            if (nested == false) {
                return docId;
            }
            int idx = java.util.Arrays.binarySearch(parentDocIds, docId);
            return idx >= 0 ? rowIds[idx] : NO_ROW;
        }

        /**
         * The first Lucene docId to start scanning from to cover logical row {@code row} (inclusive).
         * Non-nested: {@code row}. Nested: the docId of the block that owns {@code row} starts just after
         * the previous parent (children precede their parent), so scanning from the previous parent's
         * docId+1 (or 0) covers this row's children and parent.
         */
        int firstDocIdForRow(int row) {
            if (nested == false) {
                return row;
            }
            int idx = rowOrdinalIndex(row);
            if (idx >= 0) {
                // `row` is a parent at ordinal idx; its block starts just after the previous parent.
                return blockStart(idx);
            }
            // `row` isn't an exact parent rowId (defensive — rows are dense in practice): floor to the
            // nearest parent ≤ row and start just after it; if none, start at docId 0.
            int floor = rowOrdinalFloorIndex(row);
            return floor < 0 ? 0 : parentDocIds[floor] + 1;
        }

        /**
         * The exclusive Lucene docId bound to stop scanning at when covering logical rows up to
         * {@code rowExclusive}. Non-nested: {@code rowExclusive}. Nested: one past the parent docId of
         * the last row in range (a block ends at its parent, the highest docId in the block).
         */
        int docIdScanBoundForRow(int rowExclusive) {
            if (nested == false) {
                return rowExclusive;
            }
            if (rowExclusive <= 0) {
                return 0;
            }
            // Last logical row in range is rowExclusive-1; its parent docId is the block's max docId.
            int idx = rowOrdinalIndex(rowExclusive - 1);
            if (idx < 0) {
                // rowExclusive-1 beyond the last parent → scan to end of leaf.
                return parentDocIds.length == 0 ? 0 : parentDocIds[parentDocIds.length - 1] + 1;
            }
            return parentDocIds[idx] + 1;
        }

        /** docId where the block of the parent at ordinal {@code idx} begins (just after the previous parent). */
        private int blockStart(int idx) {
            return idx == 0 ? 0 : parentDocIds[idx - 1] + 1;
        }

        /** The parent-array index whose rowId equals {@code row}, or -1. rowIds are ascending. */
        private int rowOrdinalIndex(int row) {
            return java.util.Arrays.binarySearch(rowIds, row);
        }

        /** The greatest parent-array index whose rowId is ≤ {@code row}, or -1 if row precedes all. */
        private int rowOrdinalFloorIndex(int row) {
            int idx = java.util.Arrays.binarySearch(rowIds, row);
            if (idx >= 0) {
                return idx;
            }
            // insertion point (-idx-1) is the first rowId > row; floor is one before it.
            return (-idx - 1) - 1;
        }
    }
}

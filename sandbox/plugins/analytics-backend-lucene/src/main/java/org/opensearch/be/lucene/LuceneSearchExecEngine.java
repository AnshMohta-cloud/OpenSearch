/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.CDataDictionaryProvider;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.opensearch.analytics.backend.EngineResultStream;
import org.opensearch.analytics.backend.SearchExecEngine;
import org.opensearch.analytics.backend.ShardScanExecutionContext;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Lucene-side {@link SearchExecEngine}. Mirrors {@code DatafusionSearchExecEngine}'s role
 * for the Lucene backend: takes the {@link LuceneSearcherState} produced upstream by the
 * instruction handler, executes the operation, and returns an {@link EngineResultStream}
 * the framework drains into the Flight transport.
 *
 * <p>Today's only operation is the count fast path —
 * {@link org.apache.lucene.search.IndexSearcher#count(org.apache.lucene.search.Query)} —
 * exported through the Arrow C-Data interface so the result VSR has the same
 * foreign-allocation-managed buffer layout DataFusion's result stream produces. Pure-Java
 * {@code setSafe}-built VSRs don't survive Flight's {@code VectorTransfer.transferRoot};
 * see {@link LuceneResultStream} for the detailed comparison.
 *
 * <p>No deletes gate. {@code IndexSearcher.count} is self-healing: per-leaf
 * {@code Weight.count(leaf)} returns -1 on dirty leaves and falls back to full iteration —
 * correct under deletes, just slower. Even the slow case is substantially cheaper than
 * DataFusion decoding rows.
 *
 * @opensearch.internal
 */
final class LuceneSearchExecEngine implements SearchExecEngine<ShardScanExecutionContext, EngineResultStream> {

    private static final Logger LOGGER = LogManager.getLogger(LuceneSearchExecEngine.class);

    private final LuceneSearcherState state;

    LuceneSearchExecEngine(LuceneSearcherState state) {
        this.state = state;
    }

    @Override
    public void prepare(ShardScanExecutionContext context) {
        // No preparation needed — the LuceneSearcherState was fully built by the instruction
        // handler. {@code prepare} is part of the SearchExecEngine contract for backends that
        // need to assemble plans from the context (e.g. DataFusion); Lucene has nothing to do.
    }

    @Override
    public EngineResultStream execute(ShardScanExecutionContext context) throws IOException {
        // [NESTED] count(*) must return LOGICAL-document count (== Parquet rows), not raw Lucene doc
        // count. On a nested index a logical doc is a block of N+1 Lucene docs, so a plain
        // count(MatchAll) over-counts by the children. Restrict the count to PARENT docs by AND-ing the
        // filter with a parents filter (FieldExistsQuery on the block-join parent field). On a non-nested
        // index there is no parent field, so the query is used unchanged and behaviour is identical.
        // Grep: NESTED count-fastpath.
        // Count via IndexSearcher.count(), identical to how vanilla OpenSearch counts a nested query
        // (size:0 search / _count over the block-join): TotalHitCount over the block-join scorer, which
        // honors liveDocs and needs no bespoke rollup. parentScopedCountQuery scopes the count to logical
        // parent docs — and for a pure ToParentBlockJoinQuery it returns the bare block-join unwrapped
        // (fix #2), so the count runs on the single-clause block-join path exactly like vanilla's.
        Query countQuery = parentScopedCountQuery(state.searcher(), state.filterQuery());
        long countStartNanos = System.nanoTime();
        long count = state.searcher().count(countQuery);
        long countMs = (System.nanoTime() - countStartNanos) / 1_000_000L;
        LOGGER.info(
            "[NESTED] lucene-count shardId={} count_ms={} count={} originalQuery={} countQuery={} columns={}",
            context.getShardId(),
            countMs,
            count,
            state.filterQuery(),
            countQuery,
            state.outputColumnNames()
        );
        BufferAllocator allocator = context.getAllocator();
        Schema schema = buildSchema(state.outputColumnNames());
        ArrowArray array = ArrowArray.allocateNew(allocator);
        ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        boolean transferred = false;
        try {
            populateBatchToCData(allocator, schema, state.outputColumnNames(), count, array, arrowSchema);
            LuceneResultStream stream = new LuceneResultStream(array, arrowSchema, allocator);
            transferred = true;
            return stream;
        } finally {
            if (transferred == false) {
                try {
                    array.close();
                } finally {
                    arrowSchema.close();
                }
            }
        }
    }

    /**
     * Returns a count query scoped to PARENT (logical) documents when the index uses nested block-join
     * storage, else the original query unchanged.
     *
     * <p>A nested index stores each logical document as a block of N+1 Lucene docs (N children + parent),
     * so raw {@code count(query)} over-counts by the children. Only parents carry the {@code __row_id__}
     * doc-value, so {@code query AND FieldExists(__row_id__)} counts exactly the logical documents — which
     * equals the Parquet primary's row count. We detect "nested index" via the Lucene parent field
     * ({@code FieldInfos.getParentField()}, set to {@code __nested_parent} by the writer); when absent the
     * index is flat and the original query is returned untouched (zero behaviour change for non-nested).
     *
     * <p><b>Pure block-join fast path.</b> When the whole filter is a single {@link
     * OpenSearchToParentBlockJoinQuery} (the shape a lone nested-equality predicate serializes to, e.g.
     * {@code where comments.replies.author = "alice" | stats count()}), the extra {@code #FieldExists(__row_id__)}
     * conjunct is <em>redundant</em>: a {@code ToParentBlockJoinQuery} already resolves each matching child
     * block up to its single parent doc, so it emits <em>only</em> parent docs. AND-ing it with the parent
     * marker changes nothing about the result, but it turns a 1-clause count into a 2-required-clause
     * {@code BooleanQuery}. That forces {@code IndexSearcher.count()} down the {@code ConjunctionScorer}
     * path — full candidate-by-candidate iteration with a redundant {@code __row_id__} doc-values advance
     * per parent — because {@code ToParentBlockJoinQuery} has no fast {@code Weight.count()} and the
     * conjunction disables the single-clause bulk path. Returning the bare block-join lets {@code count()}
     * use the block-join's own scorer directly (the same path vanilla's nested count takes), which is the
     * dominant cost of nested-eq {@code count()} at scale (see project_mustang_latency). The count is
     * identical — logical parents with >=1 matching child. We keep the wrapper for every other filter
     * shape (MatchAll, flat/keyword terms, or any query that can match child docs), where the
     * {@code __row_id__} conjunct is what prevents over-counting.
     */
    private static Query parentScopedCountQuery(IndexSearcher searcher, Query filterQuery) {
        boolean nested = searcher.getIndexReader()
            .leaves()
            .stream()
            .anyMatch(leaf -> leaf.reader().getFieldInfos().getParentField() != null);
        if (nested == false) {
            return filterQuery;
        }
        // Pure block-join: already parent-scoped, so the __row_id__ conjunct is redundant — return it
        // unwrapped to keep count() on the fast single-clause block-join path.
        if (filterQuery instanceof OpenSearchToParentBlockJoinQuery) {
            return filterQuery;
        }
        Query parents = new FieldExistsQuery(DocumentInput.ROW_ID_FIELD);
        if (filterQuery instanceof MatchAllDocsQuery) {
            return parents;
        }
        return new BooleanQuery.Builder().add(filterQuery, BooleanClause.Occur.MUST)
            .add(parents, BooleanClause.Occur.FILTER)
            .build();
    }

    private static Schema buildSchema(List<String> columnNames) {
        FieldType int64Nullable = new FieldType(true, new ArrowType.Int(64, true), null);
        List<Field> fields = new ArrayList<>(columnNames.size());
        for (String name : columnNames) {
            fields.add(new Field(name, int64Nullable, null));
        }
        return new Schema(fields);
    }

    /**
     * Builds a one-row scratch VSR carrying {@code count} for every column, exports it to
     * the supplied {@code array}/{@code arrowSchema} via the Arrow C-Data interface, then
     * closes the scratch VSR. Mirrors the export side of {@code DatafusionResultStream}'s
     * contract: the populated {@link ArrowArray} is what {@link LuceneResultStream}
     * re-imports into its result VSR — same call shape DataFusion uses for native batches.
     */
    private static void populateBatchToCData(
        BufferAllocator allocator,
        Schema schema,
        List<String> columnNames,
        long count,
        ArrowArray array,
        ArrowSchema arrowSchema
    ) {
        VectorSchemaRoot scratch = VectorSchemaRoot.create(schema, allocator);
        try {
            scratch.allocateNew();
            for (int i = 0; i < columnNames.size(); i++) {
                BigIntVector v = (BigIntVector) scratch.getVector(i);
                v.setSafe(0, count);
            }
            scratch.setRowCount(1);
            try (CDataDictionaryProvider dictProvider = new CDataDictionaryProvider()) {
                Data.exportVectorSchemaRoot(allocator, scratch, dictProvider, array, arrowSchema);
            }
        } finally {
            scratch.close();
        }
    }
}

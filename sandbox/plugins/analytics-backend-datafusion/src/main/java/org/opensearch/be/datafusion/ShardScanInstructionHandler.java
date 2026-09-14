/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import org.opensearch.analytics.backend.ShardScanExecutionContext;
import org.opensearch.analytics.spi.BackendExecutionContext;
import org.opensearch.analytics.spi.CommonExecutionContext;
import org.opensearch.analytics.spi.FilterTreeShape;
import org.opensearch.analytics.spi.FragmentInstructionHandler;
import org.opensearch.analytics.spi.ShardScanInstructionNode;
import org.opensearch.be.datafusion.nativelib.NativeBridge;
import org.opensearch.be.datafusion.nativelib.SessionContextHandle;
import org.opensearch.index.engine.dataformat.DataFormatRegistry;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Handles ShardScan instruction: creates a SessionContext via FFM and registers
 * the default ListingTable provider for parquet scans.
 */
public class ShardScanInstructionHandler implements FragmentInstructionHandler<ShardScanInstructionNode> {

    private final DataFusionPlugin plugin;

    ShardScanInstructionHandler(DataFusionPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public BackendExecutionContext apply(
        ShardScanInstructionNode node,
        CommonExecutionContext commonContext,
        BackendExecutionContext backendContext
    ) {
        ShardScanExecutionContext context = (ShardScanExecutionContext) commonContext;
        DataFusionService dataFusionService = plugin.getDataFusionService();
        DataFormatRegistry registry = plugin.getDataFormatRegistry();

        DatafusionReader dfReader = null;
        for (String formatName : plugin.getSupportedFormats()) {
            dfReader = context.getReader().getReader(registry.format(formatName), DatafusionReader.class);
            if (dfReader != null) break;
        }
        if (dfReader == null) {
            throw new IllegalStateException("No DatafusionReader available in the acquired reader");
        }

        long readerPtr = dfReader.getReaderHandle().getPointer();
        long runtimePtr = dataFusionService.getNativeRuntime().get();
        long contextId = context.getTask() != null ? context.getTask().getId() : 0L;
        // The coordinator captured the logical table name (alias / index pattern / index the query
        // referenced) from the plan's table-scan leaf. Register the shard's table under it so the
        // Substrait plan's NamedTable binds. Fall back to the concrete shard index name when absent.
        String tableName = node.getLogicalTableName() != null ? node.getLogicalTableName() : context.getTableName();

        WireConfigSnapshot snapshot = plugin.getDatafusionSettings().getSnapshot();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(WireConfigSnapshot.BYTE_SIZE);
            snapshot.writeTo(segment);
            SessionContextHandle sessionCtxHandle;
            // A nested predicate needs the indexed context even with nothing delegated to Lucene. On
            // the plain ListingTable path a nested leaf cannot be pruned by anyone: DataFusion's own
            // PruningPredicate resolves top-level fields only, and the predicate reaches it as one
            // opaque nested_any_match_expr UDF, so it reads every page. The indexed path prunes the
            // same predicate from the leaf's own row-group and page statistics. Nested KEYWORD leaves
            // already arrive here delegated (via ShardScanWithDelegationHandler); nested NUMERIC ones
            // report treeShape=NO_DELEGATION and would otherwise land on the ListingTable and scan in
            // full. Detected by scanning the fragment for the UDF name rather than by a new field on
            // ShardScanInstructionNode, which would be a wire-format change needing BWC handling.
            boolean hasNestedPredicate = containsNestedPredicate(context.getFragmentBytes());
            if (node.requestsRowIds() || hasNestedPredicate) {
                // QTF query phase — narrowed scan emits __row_id__. Use the indexed session
                // context so the IndexedTableProvider injects shard-global row ids during scan.
                // No delegated predicates here (delegation goes through ShardScanWithDelegationHandler),
                // so treeShape=NO_DELEGATION and delegatedPredicateCount=0.
                sessionCtxHandle = NativeBridge.createSessionContextForIndexedExecution(
                    readerPtr,
                    runtimePtr,
                    tableName,
                    contextId,
                    FilterTreeShape.NO_DELEGATION.ordinal(),
                    0,
                    // Only QTF wants row ids. The nested case takes this context purely for pruning,
                    // and emitting __row_id__ there would add a column its projection never asked for.
                    node.requestsRowIds(),
                    context.hasPartialAggregate(),
                    segment.address(),
                    context.getFragmentBytes()
                );
            } else {
                // Plan bytes let Rust widen the schema for multi-index queries (null-fill missing columns).
                sessionCtxHandle = NativeBridge.createSessionContext(
                    readerPtr,
                    runtimePtr,
                    tableName,
                    contextId,
                    context.hasPartialAggregate(),
                    segment.address(),
                    context.getFragmentBytes()
                );
            }
            return new DataFusionSessionState(sessionCtxHandle);
        }
    }

    /**
     * The Substrait function name the nested-field rewriter emits, as written by
     * DataFusionFragmentConvertor's function mapping.
     */
    private static final byte[] NESTED_ANY_MATCH_EXPR = "nested_any_match_expr".getBytes(java.nio.charset.StandardCharsets.US_ASCII);

    /**
     * Does this fragment carry a nested predicate? A substring scan of the Substrait bytes, not a
     * parse — the function name appears in the plan's extension declarations whenever the rewriter
     * emitted a {@code NESTED_ANY_MATCH_EXPR} call.
     *
     * <p>A false positive costs nothing: the indexed path with nothing delegated is a predicate-only
     * scan that reads every row and applies the whole predicate as a residual, which is what the
     * ListingTable path does anyway.
     */
    private static boolean containsNestedPredicate(byte[] fragmentBytes) {
        if (fragmentBytes == null || fragmentBytes.length < NESTED_ANY_MATCH_EXPR.length) {
            return false;
        }
        outer: for (int i = 0; i <= fragmentBytes.length - NESTED_ANY_MATCH_EXPR.length; i++) {
            for (int j = 0; j < NESTED_ANY_MATCH_EXPR.length; j++) {
                if (fragmentBytes[i + j] != NESTED_ANY_MATCH_EXPR[j]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }
}

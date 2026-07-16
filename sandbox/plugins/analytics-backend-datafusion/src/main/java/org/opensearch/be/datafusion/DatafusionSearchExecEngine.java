/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.backend.EngineResultStream;
import org.opensearch.analytics.backend.SearchExecEngine;
import org.opensearch.analytics.backend.ShardScanExecutionContext;
import org.opensearch.analytics.exec.task.AnalyticsShardTask;
import org.opensearch.be.datafusion.nativelib.NativeBridge;
import org.opensearch.be.datafusion.nativelib.StreamHandle;
import org.opensearch.common.annotation.ExperimentalApi;

import java.io.IOException;

/**
 * DataFusion-backed search execution engine.
 * <p>
 * Delegates execution to the native DataFusion runtime via {@link DatafusionSearcher}.
 *
 * @opensearch.experimental
 */
@ExperimentalApi
public class DatafusionSearchExecEngine implements SearchExecEngine<ShardScanExecutionContext, EngineResultStream> {

    private static final Logger LOGGER = LogManager.getLogger(DatafusionSearchExecEngine.class);

    private final DatafusionContext datafusionContext;

    public DatafusionSearchExecEngine(DatafusionContext datafusionContext) {
        this.datafusionContext = datafusionContext;
    }

    @Override
    public void prepare(ShardScanExecutionContext requestContext) {
        byte[] substraitBytes = requestContext.getFragmentBytes();
        long contextId = datafusionContext.task() != null ? datafusionContext.task().getId() : 0L;
        // [NESTED-POC] Data-node receipt: the DataFusion backend has received the fragment's Substrait
        // bytes for this table and is about to hand them to the native runtime (from_substrait_plan).
        // This is the coordinator→data-node handoff point. Grep: NESTED-POC.
        LOGGER.info(
            "[NESTED-POC] DataFusion backend received fragment for table [{}]: {} substrait bytes, contextId={}",
            requestContext.getTableName(),
            substraitBytes == null ? 0 : substraitBytes.length,
            contextId
        );
        datafusionContext.setDatafusionQuery(new DatafusionQuery(requestContext.getTableName(), substraitBytes, contextId));
    }

    @Override
    public EngineResultStream execute(ShardScanExecutionContext requestContext) throws IOException {
        BufferAllocator allocator = requestContext.getAllocator();
        if (allocator == null) {
            throw new IllegalStateException("ExecutionContext.allocator must be set by the caller before execute()");
        }

        // Register cancellation hook so HTTP disconnect / _tasks/_cancel / timeout
        // immediately fires the Rust CancellationToken.
        long contextId = datafusionContext.getContextId();
        AnalyticsShardTask shardTask = datafusionContext.task() instanceof AnalyticsShardTask t ? t : null;
        if (shardTask != null) {
            shardTask.setCancellationListener(() -> NativeBridge.cancelQuery(contextId));
        }

        DatafusionSearcher searcher = datafusionContext.getSearcher();
        searcher.search(datafusionContext);
        StreamHandle handle = datafusionContext.takeStreamHandle();
        return new DatafusionResultStream(handle, allocator);
    }

    @Override
    public void close() throws IOException {
        datafusionContext.close();
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.indices;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.message.ParameterizedMessage;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.opensearch.ExceptionsHelper;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.common.cache.Cache;
import org.opensearch.common.cache.CacheBuilder;
import org.opensearch.common.cache.RemovalListener;
import org.opensearch.common.cache.RemovalNotification;
import org.opensearch.common.lease.Releasable;
import org.opensearch.common.lucene.index.OpenSearchDirectoryReader;
import org.opensearch.common.lucene.search.Queries;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Setting.Property;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ConcurrentCollections;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.IndexWarmer;
import org.opensearch.index.IndexWarmer.TerminationHandle;
import org.opensearch.index.cache.bitset.BitsetFilterCache;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.index.shard.ShardUtils;
import org.opensearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.ToLongBiFunction;

/**
 * Node-level cache for {@link BitSet} based filters. Manages a single flat cache shared across
 * all indices on the node, with a configurable size limit and async stale entry cleanup.
 * Stale entries from closed readers are purged periodically by a background cleanup task.
 *
 * @opensearch.api
 */
@ExperimentalApi
public class IndicesBitsetFilterCache
    implements
        IndexReader.ClosedListener,
        RemovalListener<IndicesBitsetFilterCache.BitsetCacheKey, IndicesBitsetFilterCache.Value>,
        Closeable {

    private static final Logger logger = LogManager.getLogger(IndicesBitsetFilterCache.class);

    public static final Setting<Boolean> INDEX_LOAD_RANDOM_ACCESS_FILTERS_EAGERLY_SETTING = Setting.boolSetting(
        "index.load_fixed_bitset_filters_eagerly",
        true,
        Property.IndexScope
    );

    public static final Setting<ByteSizeValue> INDICES_BITSET_FILTER_CACHE_SIZE_SETTING = Setting.memorySizeSetting(
        "indices.cache.bitset.size",
        "5%",
        Property.NodeScope
    );

    public static final Setting<TimeValue> INDICES_BITSET_FILTER_CACHE_CLEAN_INTERVAL_SETTING = Setting.positiveTimeSetting(
        "indices.cache.bitset.cleanup_interval",
        TimeValue.timeValueSeconds(60),
        Property.NodeScope
    );

    private final Cache<BitsetCacheKey, Value> cache;
    private final Set<IndexReader.CacheKey> staleCacheKeys = ConcurrentCollections.newConcurrentSet();
    private final Set<IndexReader.CacheKey> registeredKeys = ConcurrentCollections.newConcurrentSet();
    private final BitsetCacheCleaner cacheCleaner;

    public IndicesBitsetFilterCache(Settings settings, ThreadPool threadPool) {
        long sizeInBytes = INDICES_BITSET_FILTER_CACHE_SIZE_SETTING.get(settings).getBytes();
        CacheBuilder<BitsetCacheKey, Value> cacheBuilder = CacheBuilder.<BitsetCacheKey, Value>builder().removalListener(this);
        if (sizeInBytes > 0) {
            cacheBuilder.setMaximumWeight(sizeInBytes).weigher(new BitsetWeigher());
        }
        this.cache = cacheBuilder.build();

        TimeValue cleanInterval = INDICES_BITSET_FILTER_CACHE_CLEAN_INTERVAL_SETTING.get(settings);
        this.cacheCleaner = new BitsetCacheCleaner(this, threadPool, cleanInterval);
        threadPool.schedule(cacheCleaner, cleanInterval, ThreadPool.Names.SAME);
    }

    public BitSetProducer getBitSetProducer(Query query, BitsetFilterCache.Listener listener) {
        return new QueryWrapperBitSetProducer(query, listener);
    }

    public IndexWarmer.Listener createListener(ThreadPool threadPool) {
        return new BitSetProducerWarmer(threadPool);
    }

    public static BitSet bitsetFromQuery(Query query, LeafReaderContext context) throws IOException {
        final IndexReaderContext topLevelContext = ReaderUtil.getTopLevelContext(context);
        final IndexSearcher searcher = new IndexSearcher(topLevelContext);
        searcher.setQueryCache(null);
        final Weight weight = searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);
        Scorer s = weight.scorer(context);
        if (s == null) {
            // [DSL-TRACE] parent-bitset build returned NO scorer -> empty bitset for this query on this leaf.
            // For newNonNestedFilter()==FieldExistsQuery(_primary_term) this means the field is ABSENT or the
            // wrong doc-values type (e.g. SORTED_NUMERIC when NUMERIC is expected) -> parent bitset broken.
            org.apache.logging.log4j.LogManager.getLogger(IndicesBitsetFilterCache.class)
                .info("[DSL-TRACE] parentBitset: query={} maxDoc={} -> scorer=NULL (bitset EMPTY)", query, context.reader().maxDoc());
            return null;
        } else {
            BitSet bs = BitSet.of(s.iterator(), context.reader().maxDoc());
            // [DSL-TRACE] Report the built bitset: cardinality = number of docs matched (for the parent filter
            // this must equal the number of PARENT/root docs), and the first set doc ids so we can eyeball
            // that they are the roots (root is LAST in each block: child,child,ROOT).
            int card = bs.cardinality();
            StringBuilder firstIds = new StringBuilder();
            int shown = 0;
            for (int d = bs.nextSetBit(0); d != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS && shown < 40; d = (d + 1 >= bs
                .length() ? org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS : bs.nextSetBit(d + 1)), shown++) {
                firstIds.append(d).append(',');
            }
            org.apache.logging.log4j.LogManager.getLogger(IndicesBitsetFilterCache.class)
                .info(
                    "[DSL-TRACE] parentBitset: query={} maxDoc={} cardinality={} bitsetId={} setDocIds=[{}{}]",
                    query,
                    context.reader().maxDoc(),
                    card,
                    System.identityHashCode(bs),
                    firstIds,
                    card > 40 ? "..." : ""
                );
            return bs;
        }
    }

    /**
     * The ascending set-bit positions of {@code rootBits} — i.e. the Lucene doc IDs of the block-join ROOT
     * (parent) documents, indexed by their 0-based rank. Because children precede their root in each block,
     * {@code rank == __row_id__}, so this is the {@code row -> root-docId} map the composite/Parquet nested
     * numeric doc-values skipper needs to translate a page's first row into a child-docId tiling boundary.
     * A pure function of the ROOT bitset, so it is co-located with (and rebuilt alongside) that bitset in
     * {@link Value}. Returns an empty array for a {@code null} bitset (field absent on the segment).
     */
    public static int[] buildParentDocIdByRow(BitSet rootBits) {
        if (rootBits == null) {
            return new int[0];
        }
        final int cardinality = rootBits.cardinality();
        final int[] parentDocIdByRow = new int[cardinality];
        final BitSetIterator it = new BitSetIterator(rootBits, cardinality);
        int row = 0;
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
            parentDocIdByRow[row++] = doc;
        }
        return parentDocIdByRow;
    }

    BitSet getAndLoadIfNotPresent(final Query query, final LeafReaderContext context, final BitsetFilterCache.Listener listener)
        throws ExecutionException {
        return getOrLoadValue(query, context, listener, false).bitset;
    }

    /**
     * The composite/Parquet nested {@code row -> root-docId} map for {@code context}'s segment, co-loaded into
     * the SAME cache {@link Value} as the block-join ROOT (parent) bitset it is derived from — so it inherits
     * the bitset cache's size cap, eviction, and segment-scoped lifetime for free (evicted with, and rebuilt
     * alongside, that bitset). Keyed on {@link Queries#newNonNestedFilter()}, the exact ROOT filter. Built once
     * per segment inside the loading function (so the size weigher counts it); every later call is a cache hit.
     */
    public int[] getAndLoadParentDocIdByRow(final LeafReaderContext context, final BitsetFilterCache.Listener listener)
        throws ExecutionException {
        return getOrLoadValue(Queries.newNonNestedFilter(), context, listener, true).parentDocIdByRow();
    }

    /**
     * Loads (or returns the cached) {@link Value} for {@code (segment, query)}. When {@code buildParentDocIdByRow}
     * is set and the value is being created, the {@code row -> root-docId} array is materialized <b>inside</b> the
     * loading function — before the value is inserted — so {@link BitsetWeigher} accounts for it. On a cache hit
     * the flag is ignored (the value already exists); callers that need the array co-loaded with weight must be
     * the value's first accessor, which the ROOT-filter warm and the codec's read path both are for composite.
     */
    private Value getOrLoadValue(
        final Query query,
        final LeafReaderContext context,
        final BitsetFilterCache.Listener listener,
        final boolean buildParentDocIdByRow
    ) throws ExecutionException {
        final IndexReader.CacheHelper cacheHelper = FilterLeafReader.unwrap(context.reader()).getCoreCacheHelper();
        if (cacheHelper == null) {
            throw new IllegalArgumentException("Reader " + context.reader() + " does not support caching");
        }
        final IndexReader.CacheKey coreCacheReader = cacheHelper.getKey();
        final ShardId shardId = ShardUtils.extractShardId(context.reader());

        if (registeredKeys.add(coreCacheReader)) {
            cacheHelper.addClosedListener(this);
        }

        final BitsetCacheKey cacheKey = new BitsetCacheKey(coreCacheReader, query);
        return cache.computeIfAbsent(cacheKey, key -> {
            final BitSet bitSet = bitsetFromQuery(query, context);
            Value value = new Value(bitSet, shardId, listener);
            if (buildParentDocIdByRow) {
                value.parentDocIdByRow(); // materialize before insertion so the weigher counts the array
            }
            listener.onCache(shardId, value.bitset);
            return value;
        });
    }

    @Override
    public void onClose(IndexReader.CacheKey ownerCoreCacheKey) {
        staleCacheKeys.add(ownerCoreCacheKey);
    }

    @Override
    public void close() {
        cacheCleaner.close();
        clear();
    }

    public void clear() {
        cache.invalidateAll();
        staleCacheKeys.clear();
        registeredKeys.clear();
    }

    @Override
    public void onRemoval(RemovalNotification<BitsetCacheKey, Value> notification) {
        Value value = notification.getValue();
        if (value == null || value.listener == null) {
            return;
        }
        value.listener.onRemoval(value.shardId, value.bitset);
    }

    public void purgeStaleEntries() {
        if (staleCacheKeys.isEmpty()) {
            return;
        }
        Set<IndexReader.CacheKey> staleSnapshot = new HashSet<>(staleCacheKeys);

        for (BitsetCacheKey key : cache.keys()) {
            if (staleSnapshot.contains(key.readerCacheKey)) {
                cache.invalidate(key);
            }
        }

        staleCacheKeys.removeAll(staleSnapshot);
        registeredKeys.removeAll(staleSnapshot);
    }

    public Cache<BitsetCacheKey, Value> getCache() {
        return cache;
    }

    /**
     * Composite key combining a reader segment key with a query.
     *
     * @opensearch.internal
     */
    @ExperimentalApi
    public static final class BitsetCacheKey {
        final IndexReader.CacheKey readerCacheKey;
        final Query query;

        public BitsetCacheKey(IndexReader.CacheKey readerCacheKey, Query query) {
            this.readerCacheKey = Objects.requireNonNull(readerCacheKey);
            this.query = Objects.requireNonNull(query);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof BitsetCacheKey other)) return false;
            return readerCacheKey == other.readerCacheKey && query.equals(other.query);
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(readerCacheKey) + query.hashCode();
        }
    }

    /**
     * Cached value holding the bitset, shard identity, and the per-index listener for stats.
     *
     * @opensearch.internal
     */
    @ExperimentalApi
    public static final class Value {
        final BitSet bitset;
        final ShardId shardId;
        final BitsetFilterCache.Listener listener;

        /**
         * Composite/Parquet nested only: the {@code row -> root-docId} map derived from {@link #bitset} (the ROOT
         * block-join bitset), memoized so it is built at most once per segment and shares this value's cache
         * lifetime. {@code null} until first requested via {@link #parentDocIdByRow()} — for the composite path
         * that is inside the loading function (so {@link BitsetWeigher} counts it); never populated for vanilla
         * values, which keep the field {@code null} and unweighed. See {@link #buildParentDocIdByRow(BitSet)}.
         */
        private volatile int[] parentDocIdByRow;

        Value(BitSet bitset, ShardId shardId, BitsetFilterCache.Listener listener) {
            this.bitset = bitset;
            this.shardId = shardId;
            this.listener = listener;
        }

        /**
         * The memoized {@code row -> root-docId} map for this value's ROOT bitset, built on first call under a
         * lock (double-checked) and reused thereafter. Pure function of {@link #bitset}, so it is safe to build
         * lazily and share.
         */
        int[] parentDocIdByRow() {
            int[] local = parentDocIdByRow;
            if (local == null) {
                synchronized (this) {
                    local = parentDocIdByRow;
                    if (local == null) {
                        local = buildParentDocIdByRow(bitset);
                        parentDocIdByRow = local;
                    }
                }
            }
            return local;
        }
    }

    static class BitsetWeigher implements ToLongBiFunction<BitsetCacheKey, Value> {
        @Override
        public long applyAsLong(BitsetCacheKey key, Value value) {
            long weight = (value.bitset != null) ? value.bitset.ramBytesUsed() : 0;
            // Composite/Parquet nested: the co-loaded row->root-docId array shares this value's lifetime, so
            // charge it to the same size cap. Materialized before insertion (in getOrLoadValue) for the
            // composite path, so it is visible here; null (unweighed) for vanilla values.
            final int[] parentDocIdByRow = value.parentDocIdByRow;
            if (parentDocIdByRow != null) {
                weight += (long) parentDocIdByRow.length * Integer.BYTES + 16;
            }
            return weight == 0 ? 1 : weight;
        }
    }

    final class QueryWrapperBitSetProducer implements BitSetProducer {
        final Query query;
        final BitsetFilterCache.Listener listener;

        QueryWrapperBitSetProducer(Query query, BitsetFilterCache.Listener listener) {
            this.query = Objects.requireNonNull(query);
            this.listener = Objects.requireNonNull(listener);
        }

        @Override
        public BitSet getBitSet(LeafReaderContext context) throws IOException {
            try {
                return getAndLoadIfNotPresent(query, context, listener);
            } catch (ExecutionException e) {
                throw ExceptionsHelper.convertToOpenSearchException(e);
            }
        }

        @Override
        public String toString() {
            return "random_access(" + query + ")";
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof QueryWrapperBitSetProducer other)) return false;
            return this.query.equals(other.query);
        }

        @Override
        public int hashCode() {
            return 31 * getClass().hashCode() + query.hashCode();
        }
    }

    final class BitSetProducerWarmer implements IndexWarmer.Listener {
        private final Executor executor;

        BitSetProducerWarmer(ThreadPool threadPool) {
            this.executor = threadPool.executor(ThreadPool.Names.WARMER);
        }

        @Override
        public IndexWarmer.TerminationHandle warmReader(final IndexShard indexShard, final OpenSearchDirectoryReader reader) {
            if (!indexShard.indexSettings().getValue(INDEX_LOAD_RANDOM_ACCESS_FILTERS_EAGERLY_SETTING)) {
                return TerminationHandle.NO_WAIT;
            }

            boolean hasNested = false;
            final Set<Query> warmUp = new HashSet<>();
            final MapperService mapperService = indexShard.mapperService();
            DocumentMapper docMapper = mapperService.documentMapper();
            if (docMapper != null) {
                if (docMapper.hasNestedObjects()) {
                    hasNested = true;
                    for (ObjectMapper objectMapper : docMapper.objectMappers().values()) {
                        if (objectMapper.nested().isNested()) {
                            ObjectMapper parentObjectMapper = objectMapper.getParentObjectMapper(mapperService);
                            if (parentObjectMapper != null && parentObjectMapper.nested().isNested()) {
                                warmUp.add(parentObjectMapper.nestedTypeFilter());
                            }
                        }
                    }
                }
            }

            // Whether this index uses the composite/Parquet data format. For composite we ALWAYS warm the ROOT
            // (non-nested) bitset and co-load the row->root-docId map, regardless of whether the mapping declares
            // nested objects: that map is what the nested numeric doc-values skipper reads, and gating it on
            // mapping introspection is fragile, so we build it for the whole composite path (in the pure-flat
            // case the map is the identity row==docId — cheap, and weighed under the same cache size cap).
            final boolean composite = indexShard.indexSettings().isPluggableDataFormatEnabled();

            // The ROOT (non-nested) filter. Held as a reference so the warm task can recognize it and, for the
            // composite/Parquet path, additionally co-load the row->root-docId map into the SAME cache value.
            // Warmed when the index has nested objects (vanilla behavior) OR is composite (always).
            final Query nonNestedFilter;
            if (hasNested || composite) {
                nonNestedFilter = Queries.newNonNestedFilter();
                warmUp.add(nonNestedFilter);
            } else {
                nonNestedFilter = null;
            }

            // Build a listener that routes stats to the correct shard.
            final BitsetFilterCache.Listener listener = new BitsetFilterCache.Listener() {
                @Override
                public void onCache(ShardId shardId, Accountable accountable) {
                    if (shardId != null && accountable != null) {
                        indexShard.shardBitsetFilterCache().onCached(accountable.ramBytesUsed());
                    }
                }

                @Override
                public void onRemoval(ShardId shardId, Accountable accountable) {
                    if (shardId != null && accountable != null) {
                        indexShard.shardBitsetFilterCache().onRemoval(accountable.ramBytesUsed());
                    }
                }
            };

            final CountDownLatch latch = new CountDownLatch(reader.leaves().size() * warmUp.size());
            for (final LeafReaderContext ctx : reader.leaves()) {
                for (final Query filterToWarm : warmUp) {
                    executor.execute(() -> {
                        try {
                            final long start = System.nanoTime();
                            if (composite && filterToWarm.equals(nonNestedFilter)) {
                                // Loads the ROOT bitset AND the co-located row->root-docId map in one shot.
                                getAndLoadParentDocIdByRow(ctx, listener);
                            } else {
                                getAndLoadIfNotPresent(filterToWarm, ctx, listener);
                            }
                            if (indexShard.warmerService().logger().isTraceEnabled()) {
                                indexShard.warmerService()
                                    .logger()
                                    .trace(
                                        "warmed bitset for [{}], took [{}]",
                                        filterToWarm,
                                        TimeValue.timeValueNanos(System.nanoTime() - start)
                                    );
                            }
                        } catch (Exception e) {
                            indexShard.warmerService()
                                .logger()
                                .warn(() -> new ParameterizedMessage("failed to load bitset for [{}]", filterToWarm), e);
                        } finally {
                            latch.countDown();
                        }
                    });
                }
            }
            return () -> latch.await();
        }
    }

    private static final class BitsetCacheCleaner implements Runnable, Releasable {
        private final IndicesBitsetFilterCache cache;
        private final ThreadPool threadPool;
        private final TimeValue interval;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        BitsetCacheCleaner(IndicesBitsetFilterCache cache, ThreadPool threadPool, TimeValue interval) {
            this.cache = cache;
            this.threadPool = threadPool;
            this.interval = interval;
        }

        @Override
        public void run() {
            try {
                cache.purgeStaleEntries();
            } catch (Exception e) {
                logger.warn("Exception during periodic bitset filter cache cleanup:", e);
            }
            if (closed.get() == false) {
                threadPool.scheduleUnlessShuttingDown(interval, ThreadPool.Names.SAME, this);
            }
        }

        @Override
        public void close() {
            closed.compareAndSet(false, true);
        }
    }
}

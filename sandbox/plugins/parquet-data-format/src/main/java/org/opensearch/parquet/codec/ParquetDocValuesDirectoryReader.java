/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.opensearch.index.cache.bitset.BitsetFilterCache;
import org.opensearch.index.mapper.MapperService;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * A {@link FilterDirectoryReader} that wraps each leaf in a {@link ParquetDocValuesLeafReader} so
 * that Parquet-resident doc values (fields that live only in Parquet and have no Lucene segment
 * {@link org.apache.lucene.index.FieldInfo}) are served to the standard OpenSearch search and
 * aggregation path at read time.
 *
 * <p>Registered as the index reader wrapper via
 * {@code IndexModule.setReaderWrapper(...)}; OpenSearch applies it inside
 * {@code IndexShard#wrapSearcher}, which runs on the composite engine's
 * {@code DataFormatAwareEngine} searcher path. Per-leaf wrapping self-gates: a leaf is only wrapped
 * when a Parquet file resolves for its segment and the mapping declares at least one Parquet-codec
 * field missing doc values in the Lucene segment (see {@link ParquetDocValuesLeafReader#wrapIfApplicable}).
 */
public final class ParquetDocValuesDirectoryReader extends FilterDirectoryReader {

    // Dedicated stats channel, NOT the class-named logger, so the per-query summary can be toggled
    // in isolation (logger.org.opensearch.parquet.stats.query=TRACE) without turning on any other
    private final MapperService mapperService;
    private final BitsetFilterCache bitsetFilterCache;

    private ParquetDocValuesDirectoryReader(DirectoryReader in, MapperService mapperService, BitsetFilterCache bitsetFilterCache)
        throws IOException {
        super(in, new ParquetSubReaderWrapper(mapperService, bitsetFilterCache));
        this.mapperService = mapperService;
        this.bitsetFilterCache = bitsetFilterCache;
    }

    /**
     * Wraps {@code in} so Parquet-resident doc values become visible to Lucene query/aggregation
     * code paths.
     */
    public static DirectoryReader wrap(DirectoryReader in, MapperService mapperService, BitsetFilterCache bitsetFilterCache)
        throws IOException {
        return new ParquetDocValuesDirectoryReader(in, mapperService, bitsetFilterCache);
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
        return new ParquetDocValuesDirectoryReader(in, mapperService, bitsetFilterCache);
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        // Delegate to the wrapped reader's cache helper: this reader does not change the set of
        // live docs, so it is cache-coherent with the underlying OpenSearchDirectoryReader.
        return in.getReaderCacheHelper();
    }

    @Override
    protected void doClose() throws IOException {
        IOException first = null;
        try {
            for (LeafReader leaf : getSequentialSubReaders()) {
                if (leaf instanceof ParquetDocValuesLeafReader parquetLeaf) {
                    try {
                        parquetLeaf.closeParquetResources();
                    } catch (IOException e) {
                        if (first == null) {
                            first = e;
                        }
                    }
                }
            }
        } finally {
            try {
                super.doClose();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    /** Per-leaf wrapper that swaps in {@link ParquetDocValuesLeafReader} when applicable. */
    private static final class ParquetSubReaderWrapper extends SubReaderWrapper {
        private final MapperService mapperService;
        private final BitsetFilterCache bitsetFilterCache;

        private ParquetSubReaderWrapper(MapperService mapperService, BitsetFilterCache bitsetFilterCache) {
            this.mapperService = mapperService;
            this.bitsetFilterCache = bitsetFilterCache;
        }

        @Override
        public LeafReader wrap(LeafReader reader) {
            try {
                return ParquetDocValuesLeafReader.wrapIfApplicable(reader, mapperService, bitsetFilterCache);
            } catch (IOException e) {
                // SubReaderWrapper.wrap cannot throw checked exceptions; surface as unchecked so
                // the search fails loudly rather than silently dropping Parquet doc values.
                throw new UncheckedIOException("failed to wrap leaf reader for Parquet doc values", e);
            }
        }
    }
}

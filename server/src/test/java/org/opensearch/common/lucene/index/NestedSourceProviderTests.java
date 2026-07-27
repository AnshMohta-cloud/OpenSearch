/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.common.lucene.index;

import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.document.Document;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for {@link NestedSourceProvider#unwrap(LeafReader)} — the chain-walk that lets {@code ObjectMapper}
 * find the composite engine's nested-array reconstructor on a leaf reader (directly or anywhere in its
 * {@link FilterLeafReader} delegation stack), without the {@code server} module depending on any data-format
 * plugin. Correctness of the unwrap is what makes nested derived-source dispatch on composite indices while
 * cleanly returning {@code null} (→ default behavior) on classic indices.
 */
public class NestedSourceProviderTests extends OpenSearchTestCase {

    /** A test provider that records the paths/docs it was asked to reconstruct. */
    private static final class RecordingProvider implements NestedSourceProvider {
        @Override
        public List<Map<String, Object>> readNestedArray(String path, int docId) {
            return List.of(Map.of("path", path, "doc", docId));
        }
    }

    /** A FilterLeafReader that also implements NestedSourceProvider (models the composite leaf reader). */
    private static final class ProviderFilterReader extends FilterLeafReader implements NestedSourceProvider {
        ProviderFilterReader(LeafReader in) {
            super(in);
        }

        @Override
        public List<Map<String, Object>> readNestedArray(String path, int docId) {
            return List.of(Map.of("wrapped", path));
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    /** A plain FilterLeafReader that does NOT implement the provider (models an intermediate wrapper). */
    private static final class PlainFilterReader extends FilterLeafReader {
        PlainFilterReader(LeafReader in) {
            super(in);
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    private LeafReader baseLeaf(Directory dir) throws IOException {
        RandomIndexWriter w = new RandomIndexWriter(random(), dir);
        w.addDocument(new Document());
        DirectoryReader r = w.getReader();
        w.close();
        return r.leaves().get(0).reader();
    }

    public void testUnwrapFindsDirectProvider() throws IOException {
        try (Directory dir = newDirectory()) {
            LeafReader base = baseLeaf(dir);
            ProviderFilterReader provider = new ProviderFilterReader(base);
            NestedSourceProvider found = NestedSourceProvider.unwrap(provider);
            assertSame("the reader itself implements the provider", provider, found);
            base.close();
        }
    }

    public void testUnwrapFindsProviderThroughFilterChain() throws IOException {
        try (Directory dir = newDirectory()) {
            LeafReader base = baseLeaf(dir);
            // plain wrapper OVER a provider wrapper — unwrap must descend to find it.
            ProviderFilterReader provider = new ProviderFilterReader(base);
            PlainFilterReader outer = new PlainFilterReader(provider);
            NestedSourceProvider found = NestedSourceProvider.unwrap(outer);
            assertSame("unwrap should descend the FilterLeafReader chain", provider, found);
            base.close();
        }
    }

    public void testUnwrapReturnsNullWhenAbsent() throws IOException {
        try (Directory dir = newDirectory()) {
            LeafReader base = baseLeaf(dir);
            // No provider anywhere in the chain (classic index) → null → caller uses default per-field behavior.
            assertNull(NestedSourceProvider.unwrap(base));
            assertNull(NestedSourceProvider.unwrap(new PlainFilterReader(base)));
            base.close();
        }
    }

    public void testProviderContractReturnsElements() {
        RecordingProvider p = new RecordingProvider();
        List<Map<String, Object>> out = p.readNestedArray("comments", 3);
        assertEquals(1, out.size());
        assertEquals("comments", out.get(0).get("path"));
        assertEquals(3, out.get(0).get("doc"));
    }
}

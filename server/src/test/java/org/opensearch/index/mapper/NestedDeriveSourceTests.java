/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.store.Directory;
import org.opensearch.common.compress.CompressedXContent;
import org.opensearch.common.lucene.index.NestedSourceProvider;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Focused tests for {@link ObjectMapper#deriveSource} on a NESTED object mapper — the branch that reconstructs a
 * nested field's {@code _source} as a JSON ARRAY via a {@link NestedSourceProvider} (composite indices), and that
 * must OMIT the field (never emit a bare object) when no provider can reconstruct it (classic indices). Emitting
 * an object for a nested field would make the fetch layer throw ("extracted source isn't an object or an array"),
 * so these cases pin the exact shape.
 */
public class NestedDeriveSourceTests extends OpenSearchSingleNodeTestCase {

    private ObjectMapper nestedMapper() throws IOException {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("comments")
            .field("type", "nested")
            .startObject("properties")
            .startObject("author")
            .field("type", "keyword")
            .endObject()
            .startObject("score")
            .field("type", "integer")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        DocumentMapper docMapper = createIndex("test").mapperService()
            .documentMapperParser()
            .parse("_doc", new CompressedXContent(mapping));
        ObjectMapper comments = docMapper.objectMappers().get("comments");
        assertTrue("comments must be nested", comments.nested().isNested());
        return comments;
    }

    /** A leaf reader that carries a canned nested reconstruction (models the composite engine's leaf reader). */
    private static final class ProviderLeafReader extends FilterLeafReader implements NestedSourceProvider {
        private final List<Map<String, Object>> elements;

        ProviderLeafReader(LeafReader in, List<Map<String, Object>> elements) {
            super(in);
            this.elements = elements;
        }

        @Override
        public List<Map<String, Object>> readNestedArray(String path, int docId) {
            return elements;
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

    private LeafReader emptyLeaf(Directory dir) throws IOException {
        try (IndexWriter iw = new IndexWriter(dir, new IndexWriterConfig())) {
            iw.addDocument(new Document());
        }
        DirectoryReader r = DirectoryReader.open(dir);
        return r.leaves().get(0).reader();
    }

    private String derive(ObjectMapper mapper, LeafReader leaf) throws IOException {
        XContentBuilder b = XContentFactory.jsonBuilder().startObject();
        mapper.deriveSource(b, leaf, 0);
        b.endObject();
        return b.toString();
    }

    public void testNestedDeriveSourceEmitsArray() throws IOException {
        ObjectMapper comments = nestedMapper();
        try (Directory dir = newDirectory()) {
            LeafReader base = emptyLeaf(dir);
            // Use insertion-ordered maps so the serialized field order is deterministic (Map.of iteration order
            // is unspecified). The reconstruction preserves the element map's order faithfully.
            Map<String, Object> alice = new java.util.LinkedHashMap<>();
            alice.put("author", "alice");
            alice.put("score", 5);
            Map<String, Object> bob = new java.util.LinkedHashMap<>();
            bob.put("author", "bob");
            LeafReader provider = new ProviderLeafReader(base, List.of(alice, bob));
            String src = derive(comments, provider);
            // Must be an ARRAY of the two elements, in order, with the interior-null (bob has no score) preserved.
            assertEquals("{\"comments\":[{\"author\":\"alice\",\"score\":5},{\"author\":\"bob\"}]}", src);
            base.close();
        }
    }

    public void testNestedDeriveSourceEmptyArrayOmitsField() throws IOException {
        ObjectMapper comments = nestedMapper();
        try (Directory dir = newDirectory()) {
            LeafReader base = emptyLeaf(dir);
            LeafReader provider = new ProviderLeafReader(base, List.of());
            String src = derive(comments, provider);
            // An empty nested array is OMITTED (no key) — matches how vanilla drops an absent nested field.
            assertEquals("{}", src);
            base.close();
        }
    }

    public void testNestedDeriveSourceNoProviderOmitsField() throws IOException {
        ObjectMapper comments = nestedMapper();
        try (Directory dir = newDirectory()) {
            LeafReader base = emptyLeaf(dir); // classic reader — NOT a NestedSourceProvider
            String src = derive(comments, base);
            // No provider → the field is OMITTED, NOT emitted as an object. Emitting `comments:{...}` would make
            // the fetch layer reject the source ("isn't an object or an array").
            assertEquals("{}", src);
            assertFalse("must not contain an object-shaped nested field", src.contains("comments\":{"));
            base.close();
        }
    }
}

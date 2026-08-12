/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.composite;

import org.opensearch.index.engine.dataformat.DataFormat;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.dataformat.FieldTypeCapabilities;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Verifies that {@link CompositeDocumentInput} broadcasts the nested-block signals
 * ({@code startNestedChild}/{@code endNestedChild}) to the primary AND every secondary
 * {@link DocumentInput}, in order — so all formats build their nested representation from the same
 * parse-order signal stream.
 */
public class CompositeNestedDocumentInputTests extends OpenSearchTestCase {

    /** A single nested open/close is broadcast to the primary and every secondary, preserving order. */
    public void testNestedSignalsBroadcastToPrimaryAndAllSecondaries() {
        RecordingDocumentInput primary = new RecordingDocumentInput();
        RecordingDocumentInput secondary1 = new RecordingDocumentInput();
        RecordingDocumentInput secondary2 = new RecordingDocumentInput();

        Map<DataFormat, DocumentInput<?>> secondaries = new HashMap<>();
        secondaries.put(mockFormat("parquet", 2), secondary1);
        secondaries.put(mockFormat("arrow", 3), secondary2);

        CompositeDocumentInput composite = new CompositeDocumentInput(mockFormat("lucene", 1), primary, secondaries);

        composite.startNestedChild("comments");
        composite.endNestedChild();

        List<String> expected = List.of("start:comments", "end");
        assertEquals("primary must receive the nested signals in order", expected, primary.events);
        assertEquals("secondary1 must receive the nested signals in order", expected, secondary1.events);
        assertEquals("secondary2 must receive the nested signals in order", expected, secondary2.events);
    }

    /**
     * A multi-level nested sequence (comments -> replies) with interleaved field additions is broadcast
     * verbatim and in order to every format, so each reconstructs the identical tree.
     */
    public void testMultiLevelNestedSequenceBroadcastInOrder() {
        RecordingDocumentInput primary = new RecordingDocumentInput();
        RecordingDocumentInput secondary = new RecordingDocumentInput();

        CompositeDocumentInput composite = new CompositeDocumentInput(
            mockFormat("lucene", 1),
            primary,
            Map.of(mockFormat("parquet", 2), secondary)
        );

        MappedFieldType author = mockFieldType("comments.author");
        MappedFieldType replyText = mockFieldType("comments.replies.text");

        composite.startNestedChild("comments");
        composite.addField(author, "alice");
        composite.startNestedChild("comments.replies");
        composite.addField(replyText, "nice");
        composite.endNestedChild();
        composite.endNestedChild();

        List<String> expected = List.of(
            "start:comments",
            "field:comments.author=alice",
            "start:comments.replies",
            "field:comments.replies.text=nice",
            "end",
            "end"
        );
        assertEquals(expected, primary.events);
        assertEquals("secondary reconstructs the identical signal stream", expected, secondary.events);
    }

    /** With no secondaries, the primary still receives every nested signal. */
    public void testNestedSignalsWithNoSecondaries() {
        RecordingDocumentInput primary = new RecordingDocumentInput();
        CompositeDocumentInput composite = new CompositeDocumentInput(mockFormat("lucene", 1), primary, Map.of());

        composite.startNestedChild("comments");
        composite.endNestedChild();

        assertEquals(List.of("start:comments", "end"), primary.events);
    }

    /**
     * The composite does not swallow errors from endNestedChild — if the primary's underlying input
     * throws (e.g. an unbalanced end), it propagates so the parse fails fast.
     */
    public void testEndNestedChildPropagatesPrimaryError() {
        DocumentInput<?> throwingPrimary = new RecordingDocumentInput() {
            @Override
            public void endNestedChild() {
                throw new IllegalStateException("endNestedChild called with no open nested child");
            }
        };
        CompositeDocumentInput composite = new CompositeDocumentInput(mockFormat("lucene", 1), throwingPrimary, Map.of());
        expectThrows(IllegalStateException.class, composite::endNestedChild);
    }

    // --- helpers ---

    private DataFormat mockFormat(String name, long priority) {
        return new DataFormat() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public long priority() {
                return priority;
            }

            @Override
            public Set<FieldTypeCapabilities> supportedFields() {
                return Set.of();
            }
        };
    }

    private MappedFieldType mockFieldType(String name) {
        MappedFieldType ft = org.mockito.Mockito.mock(MappedFieldType.class);
        org.mockito.Mockito.when(ft.name()).thenReturn(name);
        return ft;
    }

    /** Records the ordered stream of signals a DocumentInput receives. */
    static class RecordingDocumentInput implements DocumentInput<Object> {
        final List<String> events = new ArrayList<>();

        @Override
        public void addField(MappedFieldType fieldType, Object value) {
            events.add("field:" + fieldType.name() + "=" + value);
        }

        @Override
        public void setRowId(String rowIdFieldName, long rowId) {
            events.add("rowId:" + rowId);
        }

        @Override
        public void startNestedChild(String nestedPath) {
            events.add("start:" + nestedPath);
        }

        @Override
        public void endNestedChild() {
            events.add("end");
        }

        @Override
        public Object getFinalInput() {
            return null;
        }

        @Override
        public long getFieldCount(String fieldName) {
            return 0;
        }

        @Override
        public void close() {}
    }
}

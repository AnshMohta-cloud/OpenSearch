/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexableField;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.MappedFieldType;

import java.util.ArrayList;
import java.util.List;

/**
 * Tests that {@link LuceneDocumentInput} writes exactly ONE Lucene document per logical document.
 *
 * <p>Unlike vanilla OpenSearch, nested array elements do NOT become child documents. Every nested leaf
 * collapses into a MULTI-VALUED field on the single row document (one value per element, in parse order),
 * and no {@code _nested_path} term is ever written. This keeps the Lucene docId equal to the Parquet
 * rowId, which is what lets Lucene hand DataFusion a row-skipping bitset directly.
 *
 * <p>Dropping the element boundary is sound on this path because DataFusion re-evaluates the full query
 * over the Parquet {@code LIST<STRUCT>}: losing "which element matched" can only ever produce false
 * positives, never false negatives.
 */
public class LuceneNestedDocumentInputTests extends LucenePluginBaseTests {

    /** A flat doc produces one document holding its own leaf and no nested marker. */
    public void testFlatDocumentProducesSingleDocument() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(mockKeywordField("status"), "active");

        Document doc = input.getFinalInput();
        assertEquals(List.of("active"), valuesOf(doc, "status"));
        assertNoNestedPath(doc);
        assertEquals("balanced signals leave depth at zero", 0, input.nestedDepth());
    }

    /**
     * Two nested elements collapse into one multi-valued field on the row doc, in parse order. The root's
     * own leaf is unaffected and there is no {@code _nested_path}.
     */
    public void testNestedElementsCollapseIntoMultiValuedField() {
        MappedFieldType rootField = mockKeywordField("title");
        MappedFieldType commentAuthor = mockKeywordField("comments.author");

        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(rootField, "post-title");

        input.startNestedChild("comments");
        input.addField(commentAuthor, "alice");
        input.endNestedChild();

        input.startNestedChild("comments");
        input.addField(commentAuthor, "bob");
        input.endNestedChild();

        Document doc = input.getFinalInput();
        assertEquals("one value per element, in parse order", List.of("alice", "bob"), valuesOf(doc, "comments.author"));
        assertEquals(2, input.getFieldCount("comments.author"));
        assertEquals(List.of("post-title"), valuesOf(doc, "title"));
        assertNoNestedPath(doc);
        assertEquals(0, input.nestedDepth());
    }

    /**
     * Multi-level nesting: leaves from every level land on the same row doc, each multi-valued according
     * to how many elements contributed a value. Element identity is deliberately not preserved.
     */
    public void testMultiLevelNestedAllLeavesLandOnRowDoc() {
        MappedFieldType commentAuthor = mockKeywordField("comments.author");
        MappedFieldType replyText = mockKeywordField("comments.replies.text");

        LuceneDocumentInput input = new LuceneDocumentInput();

        // comment0 with one reply
        input.startNestedChild("comments");
        input.addField(commentAuthor, "alice");
        input.startNestedChild("comments.replies");
        input.addField(replyText, "nice");
        input.endNestedChild();
        input.endNestedChild();

        // comment1, no replies
        input.startNestedChild("comments");
        input.addField(commentAuthor, "bob");
        input.endNestedChild();

        Document doc = input.getFinalInput();
        assertEquals(List.of("alice", "bob"), valuesOf(doc, "comments.author"));
        assertEquals(List.of("nice"), valuesOf(doc, "comments.replies.text"));
        assertNoNestedPath(doc);
        assertEquals(0, input.nestedDepth());
    }

    /** A depth-3 chain still yields a single document carrying every level's leaf. */
    public void testThreeLevelNestedProducesSingleDocument() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.startNestedChild("comments");
        input.addField(mockKeywordField("comments.author"), "alice");
        input.startNestedChild("comments.replies");
        input.addField(mockKeywordField("comments.replies.text"), "r1");
        input.startNestedChild("comments.replies.reactions");
        input.addField(mockKeywordField("comments.replies.reactions.emoji"), "smile");
        assertEquals("three levels open", 3, input.nestedDepth());
        input.endNestedChild(); // reactions
        input.endNestedChild(); // replies
        input.endNestedChild(); // comments

        Document doc = input.getFinalInput();
        assertEquals(List.of("alice"), valuesOf(doc, "comments.author"));
        assertEquals(List.of("r1"), valuesOf(doc, "comments.replies.text"));
        assertEquals(List.of("smile"), valuesOf(doc, "comments.replies.reactions.emoji"));
        assertNoNestedPath(doc);
        assertEquals(0, input.nestedDepth());
    }

    /** An empty nested array emits no start/end signals, so the doc is simply flat. */
    public void testEmptyNestedArrayStillOneDocument() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(mockKeywordField("title"), "only-root");

        Document doc = input.getFinalInput();
        assertEquals(List.of("only-root"), valuesOf(doc, "title"));
        assertNoNestedPath(doc);
        assertEquals(0, input.nestedDepth());
    }

    /** Closing a nested child when none is open is a programming error and must fail fast. */
    public void testEndNestedChildWithoutOpenChildThrows() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        IllegalStateException e = expectThrows(IllegalStateException.class, input::endNestedChild);
        assertTrue(e.getMessage().contains("no open nested child"));
    }

    /**
     * {@code __row_id__} lands on the one and only document, so the Lucene docId ↔ Parquet rowId identity
     * holds without any per-child bookkeeping.
     */
    public void testSetRowIdWritesOnTheRowDocument() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.startNestedChild("comments");
        input.addField(mockKeywordField("comments.author"), "alice");
        input.endNestedChild();
        input.setRowId(DocumentInput.ROW_ID_FIELD, 7L);

        assertEquals(7L, input.getRowId());
        Document doc = input.getFinalInput();
        assertNotNull("the row doc carries __row_id__", doc.getField(DocumentInput.ROW_ID_FIELD));
        assertEquals(1, doc.getFields(DocumentInput.ROW_ID_FIELD).length);
    }

    /** Values of {@code field} on {@code doc}, in the order they were added. */
    private static List<String> valuesOf(Document doc, String field) {
        List<String> values = new ArrayList<>();
        for (IndexableField f : doc.getFields(field)) {
            values.add(f.stringValue());
        }
        return values;
    }

    /** No nested marker is ever written now that nested elements are not separate documents. */
    private static void assertNoNestedPath(Document doc) {
        assertNull("_nested_path must never be written", doc.getField(DocumentInput.NESTED_PATH_FIELD));
    }
}

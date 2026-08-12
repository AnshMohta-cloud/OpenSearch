/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableField;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.MappedFieldType;

import java.util.List;

/**
 * Tests the nested-block behavior of {@link LuceneDocumentInput}: the {@code startNestedChild} /
 * {@code endNestedChild} signals must reproduce vanilla OpenSearch's nested Lucene block, i.e. every
 * nested object materializes as its OWN child {@link Document} (carrying a {@code _nested_path} term
 * and its own leaf fields), laid out children-first / root-last in post-order for
 * {@code IndexWriter.addDocuments}.
 */
public class LuceneNestedDocumentInputTests extends LucenePluginBaseTests {

    /** (a) A flat doc emits no nested children: block == [root] and hasNestedChildren == false. */
    public void testFlatDocumentProducesSingleRootBlock() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(mockKeywordField("status"), "active");

        assertFalse("a flat doc must not report nested children", input.hasNestedChildren());

        List<Document> block = input.getDocumentBlock();
        assertEquals("flat block is just the root", 1, block.size());
        assertSame("the single block entry is the root document", input.getFinalInput(), block.get(0));
        assertNotNull("root keeps its own leaf field", block.get(0).getField("status"));
        assertNull("root never carries a _nested_path", block.get(0).getField(DocumentInput.NESTED_PATH_FIELD));
    }

    /**
     * (b) Single-level nested with two children: block == [child0, child1, root]. Each child carries a
     * {@code _nested_path} term equal to the path and its own leaf on the correct child; the root carries
     * neither the nested leaf nor a {@code _nested_path}.
     */
    public void testSingleLevelNestedTwoChildren() {
        MappedFieldType rootField = mockKeywordField("title");
        MappedFieldType commentAuthor = mockKeywordField("comments.author");

        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(rootField, "post-title"); // lands on root (no child open)

        input.startNestedChild("comments");
        input.addField(commentAuthor, "alice"); // lands on child0
        input.endNestedChild();

        input.startNestedChild("comments");
        input.addField(commentAuthor, "bob"); // lands on child1
        input.endNestedChild();

        assertTrue(input.hasNestedChildren());
        List<Document> block = input.getDocumentBlock();
        assertEquals("two children + root", 3, block.size());

        Document child0 = block.get(0);
        Document child1 = block.get(1);
        Document root = block.get(2);
        assertSame("root is last in the block", input.getFinalInput(), root);

        assertNestedPath(child0, "comments");
        assertNestedPath(child1, "comments");
        assertEquals("alice", child0.getField("comments.author").stringValue());
        assertEquals("bob", child1.getField("comments.author").stringValue());

        // Root carries only its own leaf, never the nested leaf nor a _nested_path.
        assertEquals("post-title", root.getField("title").stringValue());
        assertNull(root.getField("comments.author"));
        assertNull(root.getField(DocumentInput.NESTED_PATH_FIELD));
    }

    /**
     * (c) Multi-level nested (comments -> replies, depth 2/3) plus a sibling comment. Children must land
     * in POST-ORDER: a deeper child closes before its enclosing element, so the block is
     * [reply(depth-2), comment0(depth-1), comment1(depth-1), root]. Each child's {@code _nested_path} is
     * its full dotted path.
     */
    public void testMultiLevelNestedPostOrder() {
        MappedFieldType commentAuthor = mockKeywordField("comments.author");
        MappedFieldType replyText = mockKeywordField("comments.replies.text");

        LuceneDocumentInput input = new LuceneDocumentInput();

        // comment0 with one reply (depth 2)
        input.startNestedChild("comments");
        input.addField(commentAuthor, "alice");
        input.startNestedChild("comments.replies");
        input.addField(replyText, "nice");
        input.endNestedChild(); // reply closes first
        input.endNestedChild(); // then comment0

        // comment1, a flat sibling (depth 1)
        input.startNestedChild("comments");
        input.addField(commentAuthor, "bob");
        input.endNestedChild();

        List<Document> block = input.getDocumentBlock();
        assertEquals("reply + comment0 + comment1 + root", 4, block.size());

        // Post-order: deepest (reply) first, then its enclosing comment, then the sibling, root last.
        Document reply = block.get(0);
        Document comment0 = block.get(1);
        Document comment1 = block.get(2);
        Document root = block.get(3);

        assertNestedPath(reply, "comments.replies");
        assertEquals("nice", reply.getField("comments.replies.text").stringValue());
        assertNull("reply holds only its own leaf", reply.getField("comments.author"));

        assertNestedPath(comment0, "comments");
        assertEquals("alice", comment0.getField("comments.author").stringValue());
        assertNull("the reply leaf belongs to the reply child, not the comment", comment0.getField("comments.replies.text"));

        assertNestedPath(comment1, "comments");
        assertEquals("bob", comment1.getField("comments.author").stringValue());

        assertSame(input.getFinalInput(), root);
        assertNull(root.getField(DocumentInput.NESTED_PATH_FIELD));
    }

    /**
     * (c') Depth-3 chain (comments -> replies -> reactions): the block must be strictly deepest-first
     * [reaction, reply, comment, root].
     */
    public void testThreeLevelNestedPostOrder() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.startNestedChild("comments");
        input.addField(mockKeywordField("comments.author"), "alice");
        input.startNestedChild("comments.replies");
        input.addField(mockKeywordField("comments.replies.text"), "r1");
        input.startNestedChild("comments.replies.reactions");
        input.addField(mockKeywordField("comments.replies.reactions.emoji"), "smile");
        input.endNestedChild(); // reactions
        input.endNestedChild(); // replies
        input.endNestedChild(); // comments

        List<Document> block = input.getDocumentBlock();
        assertEquals(4, block.size());
        assertNestedPath(block.get(0), "comments.replies.reactions");
        assertNestedPath(block.get(1), "comments.replies");
        assertNestedPath(block.get(2), "comments");
        assertSame("root last", input.getFinalInput(), block.get(3));
        assertEquals("smile", block.get(0).getField("comments.replies.reactions.emoji").stringValue());
    }

    /**
     * (d) An empty nested array yields no child docs. The parser emits no start/end signals for zero
     * elements, so the document stays flat — block == [root], hasNestedChildren == false.
     */
    public void testEmptyNestedArrayProducesNoChildDocs() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.addField(mockKeywordField("title"), "only-root");
        // (no startNestedChild calls — an empty array contributes no elements)

        assertFalse(input.hasNestedChildren());
        List<Document> block = input.getDocumentBlock();
        assertEquals(1, block.size());
        assertSame(input.getFinalInput(), block.get(0));
    }

    /** (e) Closing a nested child when none is open is a programming error and must fail fast. */
    public void testEndNestedChildWithoutOpenChildThrows() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        IllegalStateException e = expectThrows(IllegalStateException.class, input::endNestedChild);
        assertTrue(e.getMessage().contains("no open nested child"));
    }

    /**
     * The row ID is a root-only marker: {@code setRowId} writes {@code __row_id__} onto the root document
     * only, never onto any nested child (children are correlated to their root via the block layout).
     */
    public void testSetRowIdWritesOnRootOnly() {
        LuceneDocumentInput input = new LuceneDocumentInput();
        input.startNestedChild("comments");
        input.addField(mockKeywordField("comments.author"), "alice");
        input.endNestedChild();
        input.setRowId(DocumentInput.ROW_ID_FIELD, 7L);

        assertEquals(7L, input.getRowId());
        List<Document> block = input.getDocumentBlock();
        Document child = block.get(0);
        Document root = block.get(1);
        assertNull("child must not carry __row_id__", child.getField(DocumentInput.ROW_ID_FIELD));
        assertNotNull("root carries __row_id__", root.getField(DocumentInput.ROW_ID_FIELD));
    }

    /** Asserts the doc carries a postings-only, not-stored {@code _nested_path} term equal to {@code expected}. */
    private static void assertNestedPath(Document doc, String expected) {
        IndexableField path = doc.getField(DocumentInput.NESTED_PATH_FIELD);
        assertNotNull("child must carry a _nested_path", path);
        assertEquals(expected, path.stringValue());
        assertFalse("_nested_path is not stored", path.fieldType().stored());
        assertNotEquals("_nested_path is indexed as a term", IndexOptions.NONE, path.fieldType().indexOptions());
    }
}

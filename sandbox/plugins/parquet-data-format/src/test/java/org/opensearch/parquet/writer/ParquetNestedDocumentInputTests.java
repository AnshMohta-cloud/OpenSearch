/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.writer;

import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.parquet.ParquetBaseTests;

import java.util.List;

/**
 * Tests the buffered {@code NestedChild} tree that {@link ParquetDocumentInput} builds from the
 * {@code startNestedChild}/{@code endNestedChild}/{@code addField} signal stream. Unlike the flat
 * collection path (which dedups on field type and rejects duplicates), fields inside a nested scope are
 * routed to the innermost open element with no dedup — different array elements legitimately repeat the
 * same field type.
 */
public class ParquetNestedDocumentInputTests extends ParquetBaseTests {

    /** A flat doc buffers no nested children. */
    public void testFlatDocumentHasNoNestedChildren() {
        ParquetDocumentInput input = new ParquetDocumentInput();
        input.addField(createNumberField("age", NumberFieldMapper.NumberType.INTEGER), 25);
        assertTrue("flat doc has no nested children", input.getNestedChildren().isEmpty());
    }

    /** Single-level nested with two elements: two top-level NestedChild, each with its own leaf value. */
    public void testSingleLevelNestedTwoElements() {
        MappedFieldType author = createKeywordField("comments.author");

        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(author, "alice");
        input.endNestedChild();
        input.startNestedChild("comments");
        input.addField(author, "bob");
        input.endNestedChild();

        List<ParquetDocumentInput.NestedChild> children = input.getNestedChildren();
        assertEquals(2, children.size());
        assertEquals("comments", children.get(0).path);
        assertEquals("comments", children.get(1).path);
        assertEquals(1, children.get(0).fields.size());
        assertEquals("alice", children.get(0).fields.get(0).getValue());
        assertEquals("bob", children.get(1).fields.get(0).getValue());
        assertTrue("single-level elements have no deeper children", children.get(0).children.isEmpty());
    }

    /**
     * Multi-level nested (comments -> replies): the inner element is attached to its enclosing element's
     * {@code children}, not to the top level. The top level holds only the comment.
     */
    public void testMultiLevelNestedTreeStructure() {
        MappedFieldType author = createKeywordField("comments.author");
        MappedFieldType replyText = createKeywordField("comments.replies.text");

        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(author, "alice");
        input.startNestedChild("comments.replies");
        input.addField(replyText, "nice");
        input.endNestedChild(); // reply -> attaches to comment
        input.endNestedChild(); // comment -> attaches to top level

        List<ParquetDocumentInput.NestedChild> top = input.getNestedChildren();
        assertEquals("only the comment is top-level", 1, top.size());
        ParquetDocumentInput.NestedChild comment = top.get(0);
        assertEquals("comments", comment.path);
        assertEquals("alice", comment.fields.get(0).getValue());

        assertEquals("the reply is nested inside the comment", 1, comment.children.size());
        ParquetDocumentInput.NestedChild reply = comment.children.get(0);
        assertEquals("comments.replies", reply.path);
        assertEquals("nice", reply.fields.get(0).getValue());
    }

    /** Three-level chain: comments -> replies -> reactions must nest one inside the other. */
    public void testThreeLevelNestedTreeStructure() {
        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.startNestedChild("comments.replies");
        input.startNestedChild("comments.replies.reactions");
        input.addField(createKeywordField("comments.replies.reactions.emoji"), "smile");
        input.endNestedChild();
        input.endNestedChild();
        input.endNestedChild();

        List<ParquetDocumentInput.NestedChild> top = input.getNestedChildren();
        assertEquals(1, top.size());
        ParquetDocumentInput.NestedChild reply = top.get(0).children.get(0);
        assertEquals("comments.replies", reply.path);
        ParquetDocumentInput.NestedChild reaction = reply.children.get(0);
        assertEquals("comments.replies.reactions", reaction.path);
        assertEquals("smile", reaction.fields.get(0).getValue());
    }

    /** Fields added while a child is open route to the innermost open element, not the flat collection. */
    public void testFieldsRouteToInnermostOpenElement() {
        MappedFieldType author = createKeywordField("comments.author");
        MappedFieldType replyText = createKeywordField("comments.replies.text");

        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(author, "alice"); // -> comment
        input.startNestedChild("comments.replies");
        input.addField(replyText, "r1"); // -> reply (innermost), NOT the comment
        input.endNestedChild();
        input.endNestedChild();

        ParquetDocumentInput.NestedChild comment = input.getNestedChildren().get(0);
        assertEquals(1, comment.fields.size());
        assertEquals("comments.author", comment.fields.get(0).getFieldType().name());
        ParquetDocumentInput.NestedChild reply = comment.children.get(0);
        assertEquals(1, reply.fields.size());
        assertEquals("comments.replies.text", reply.fields.get(0).getFieldType().name());
        // The flat collection stays empty — nested leaves never leak into it.
        assertEquals(0, input.getFieldCount("comments.author"));
        assertEquals(0, input.getFieldCount("comments.replies.text"));
    }

    /**
     * The dedup-bypass: the SAME field type object appearing in multiple elements must NOT throw
     * (the flat path would reject it with MapperParsingException). This is the core reason nested
     * routing bypasses the dedup set.
     */
    public void testRepeatedSameFieldTypeAcrossElementsDoesNotThrow() {
        MappedFieldType author = createKeywordField("comments.author");
        ParquetDocumentInput input = new ParquetDocumentInput();
        // Same MappedFieldType instance, five elements — no exception, each keeps its own value.
        for (int i = 0; i < 5; i++) {
            input.startNestedChild("comments");
            input.addField(author, "author-" + i);
            input.endNestedChild();
        }
        List<ParquetDocumentInput.NestedChild> children = input.getNestedChildren();
        assertEquals(5, children.size());
        for (int i = 0; i < 5; i++) {
            assertEquals("author-" + i, children.get(i).fields.get(0).getValue());
        }
    }

    /** The same field type may even repeat WITHIN one element (a scalar array inside a nested object). */
    public void testRepeatedSameFieldTypeWithinOneElementDoesNotThrow() {
        MappedFieldType tag = createKeywordField("comments.tags");
        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(tag, "x");
        input.addField(tag, "y");
        input.endNestedChild();
        assertEquals(2, input.getNestedChildren().get(0).fields.size());
    }

    /** close() clears the buffered nested tree (and the open stack) along with the flat collection. */
    public void testCloseClearsNestedState() {
        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(createKeywordField("comments.author"), "alice");
        input.endNestedChild();
        assertEquals(1, input.getNestedChildren().size());

        input.close();
        assertTrue("nested tree cleared on close", input.getNestedChildren().isEmpty());
    }

    /** After close(), the input is frozen: further nested signals must fail fast via ensureOpen(). */
    public void testStartNestedChildAfterCloseThrows() {
        ParquetDocumentInput input = new ParquetDocumentInput();
        input.close();
        expectThrows(IllegalStateException.class, () -> input.startNestedChild("comments"));
    }

    /** A metadata/leaf field added after all children close returns to the flat collection path. */
    public void testFieldAfterClosingChildReturnsToFlatCollection() {
        MappedFieldType author = createKeywordField("comments.author");
        NumberFieldMapper.NumberFieldType age = createNumberField("age", NumberFieldMapper.NumberType.INTEGER);

        ParquetDocumentInput input = new ParquetDocumentInput();
        input.startNestedChild("comments");
        input.addField(author, "alice");
        input.endNestedChild();
        input.addField(age, 30); // stack empty again -> flat collection

        assertEquals(1, input.getNestedChildren().size());
        assertEquals("root leaf lands in the flat collection", 1, input.getFieldCount("age"));
    }
}

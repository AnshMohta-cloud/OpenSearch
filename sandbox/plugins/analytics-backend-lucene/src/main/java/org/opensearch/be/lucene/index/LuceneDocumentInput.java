/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.be.lucene.LuceneFieldFactory;
import org.opensearch.be.lucene.LuceneFieldFactoryRegistry;
import org.opensearch.be.lucene.LucenePlugin;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.dataformat.FieldTypeCapabilities;
import org.opensearch.index.mapper.MappedFieldType;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Set;

/**
 * Lucene-specific {@link DocumentInput} that builds a Lucene {@link Document}.
 *
 * Field creation is delegated to a {@link LuceneFieldFactoryRegistry} which maps
 * OpenSearch field type names to {@link LuceneFieldFactory} instances. This makes
 * the set of supported field types extensible without modifying this class.
 *
 * Only field types registered in the registry are accepted. Attempting to add a field
 * of an unregistered type throws {@link IllegalArgumentException}.
 *
 * The row ID field is stored as a {@link SortedNumericDocValuesField} for efficient doc-value
 * access and compatibility with the {@code SortedNumericSortField}-based IndexSort,
 * maintaining 1:1 correspondence between Lucene doc IDs and Parquet row offsets.
 *
 * @opensearch.experimental
 */
@ExperimentalApi
public class LuceneDocumentInput implements DocumentInput<Document> {

    private final Document document;
    private final LuceneFieldFactoryRegistry fieldFactoryRegistry;
    private long rowId = -1L;

    // ── Nested block support ──
    // Vanilla OpenSearch materializes each nested object as its OWN Lucene document, laid out
    // children-first / root-last in one contiguous block (added via IndexWriter.addDocuments). We reproduce
    // that here from the parser's startNestedChild/endNestedChild signals:
    //   - `childDocStack` holds the currently-OPEN child docs (innermost on top). A field added while a
    //     child is open lands on the innermost open child (that child's own leaf), not the root.
    //   - `endNestedChild` pops the innermost open child and appends it to `childDocs`. Because inner
    //     children close before their enclosing element, `childDocs` ends up in POST-ORDER (descendants
    //     first, enclosing element after) — exactly vanilla's nested block order.
    //   - `getDocumentBlock()` returns [ childDocs (post-order)..., root ] with the root LAST.
    // A flat (non-nested) doc never calls startNestedChild, so childDocs stays empty and the writer uses
    // the single-document path unchanged.
    private final Deque<Document> childDocStack = new ArrayDeque<>();
    private final List<Document> childDocs = new ArrayList<>();

    /**
     * Creates a new LuceneDocumentInput with the default field factory registry.
     */
    public LuceneDocumentInput() {
        this(new LuceneFieldFactoryRegistry());
    }

    /**
     * Creates a new LuceneDocumentInput with a custom field factory registry.
     *
     * @param fieldFactoryRegistry the registry to use for field creation
     */
    public LuceneDocumentInput(LuceneFieldFactoryRegistry fieldFactoryRegistry) {
        this.document = new Document();
        this.fieldFactoryRegistry = fieldFactoryRegistry;
    }

    /**
     * Returns the built Lucene {@link Document} containing all added fields.
     *
     * @return the Lucene document
     */
    @Override
    public Document getFinalInput() {
        return document;
    }

    /**
     * Adds a field to the underlying Lucene document by looking up the appropriate
     * {@link LuceneFieldFactory} from the registry based on the field's type name.
     * <p>
     * The field is accepted only if OWNING_FORMAT owns at least one capability
     * for this field according to {@link MappedFieldType#getCapabilityMap()}. Fields with
     * an empty capability map (no format declared support) and fields owned by other
     * formats are silently skipped, mirroring the per-format self-filtering used by
     * {@code ParquetDocumentInput}.
     *
     * @param fieldType the OpenSearch mapped field type
     * @param value     the field value
     */
    @Override
    public void addField(MappedFieldType fieldType, Object value) {
        Set<FieldTypeCapabilities.Capability> capabilities = fieldType.getCapabilityMap().getOrDefault(LucenePlugin.DATA_FORMAT, Set.of());
        if (capabilities.isEmpty()) {
            // nothing to support on this format for this field.
            return;
        }
        if (value == null) {
            throw new IllegalArgumentException(
                "Field value must not be null for: " + fieldType.name() + " of type: " + fieldType.typeName()
            );
        }
        LuceneFieldFactory factory = fieldFactory(fieldType);
        if (factory == null) {
            // capabilities need to be supported but actual implementation to support lucene field type does not exist.
            throw new IllegalArgumentException(
                "Field: " + fieldType.name() + " requests capability: " + capabilities + " but does not have any factory to support"
            );
        }
        FieldType luceneFieldType = getFieldType(fieldType, capabilities);
        // Route to the innermost OPEN nested child if one is open, else the root document. This is what
        // puts a nested object's leaf fields on that object's own child doc (e.g. comments.author on the
        // comment child), while root/metadata fields — added outside any nested scope — land on the root.
        factory.addField(currentTarget(), fieldType, value, luceneFieldType);
    }

    /** The document currently receiving fields: the innermost open nested child, or the root if none. */
    private Document currentTarget() {
        return childDocStack.isEmpty() ? document : childDocStack.peek();
    }

    private static FieldType getFieldType(MappedFieldType fieldType, Set<FieldTypeCapabilities.Capability> capabilities) {
        FieldType luceneFieldType = null;
        if (fieldType.getTextSearchInfo() != null && fieldType.getTextSearchInfo().getLuceneFieldType() != null) {
            luceneFieldType = new FieldType(fieldType.getTextSearchInfo().getLuceneFieldType());
            if (!capabilities.contains(FieldTypeCapabilities.Capability.COLUMNAR_STORAGE)) {
                // Disable doc values even if core mappers have set it on lucene fields
                // once we introduce more frontend params, we can remove this check.
                luceneFieldType.setDocValuesType(DocValuesType.NONE);
            }
            luceneFieldType.setStored(false);
            luceneFieldType.setOmitNorms(true);
        }
        return luceneFieldType;
    }

    private LuceneFieldFactory fieldFactory(MappedFieldType fieldType) {
        if (fieldType == null) {
            throw new IllegalArgumentException("Field type and value must not be null");
        }
        return fieldFactoryRegistry.get(fieldType.typeName());
    }

    /**
     * Stores the row ID as a {@link SortedNumericDocValuesField} to maintain 1:1 correspondence
     * between Lucene doc IDs and Parquet row offsets.
     *
     * @param rowIdFieldName the name of the row ID field
     * @param rowId          the row ID value (0-based sequential within the writer)
     */
    @Override
    public void setRowId(String rowIdFieldName, long rowId) {
        document.add(new SortedNumericDocValuesField(rowIdFieldName, rowId));
        this.rowId = rowId;
    }

    /** Returns the row ID assigned via {@link #setRowId}, or {@code -1} if none. */
    public long getRowId() {
        return rowId;
    }

    /**
     * Opens a new nested child document for {@code nestedPath}. The child carries a {@code _nested_path}
     * term (its level marker, matching vanilla) and becomes the target for subsequent {@link #addField}
     * calls until its {@link #endNestedChild()}. Pushed onto the open-child stack so nesting composes to
     * arbitrary depth (a deeper startNestedChild opens a child of this child).
     */
    @Override
    public void startNestedChild(String nestedPath) {
        Document child = new Document();
        // Postings-only term identifying the nested level; not stored, not tokenized (StringField default).
        child.add(new StringField(DocumentInput.NESTED_PATH_FIELD, nestedPath, Field.Store.NO));
        childDocStack.push(child);
    }

    /**
     * Closes the innermost open nested child and appends it to the block. Because inner children close
     * before their enclosing element, {@code childDocs} accumulates in post-order (descendants first).
     */
    @Override
    public void endNestedChild() {
        if (childDocStack.isEmpty()) {
            throw new IllegalStateException("endNestedChild called with no open nested child");
        }
        childDocs.add(childDocStack.pop());
    }

    /** Whether this input produced any nested child docs (i.e. the doc must be written as a block). */
    public boolean hasNestedChildren() {
        return childDocs.isEmpty() == false;
    }

    /**
     * The full nested block to hand to {@code IndexWriter.addDocuments}: every child doc in post-order
     * (descendants first, enclosing element after), followed by the ROOT document last — the vanilla
     * nested block layout. For a flat doc (no nested children) this is just {@code [root]}.
     */
    public List<Document> getDocumentBlock() {
        List<Document> block = new ArrayList<>(childDocs.size() + 1);
        block.addAll(childDocs);
        block.add(document); // root is LAST
        return block;
    }

    @Override
    public long getFieldCount(String fieldName) {
        return document.getFields(fieldName).length;
    }

    /** No-op — this document input holds no closeable resources. */
    @Override
    public void close() {
        // No resources to release
    }
}

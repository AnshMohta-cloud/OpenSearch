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

    /** Name of the nested path field used to identify nested docs by their path. */
    private static final String NESTED_PATH_FIELD = "_nested_path";

    private final Document document;
    private final LuceneFieldFactoryRegistry fieldFactoryRegistry;
    private long rowId = -1L;

    // --- POC nested support: tracks child documents forming a block ---
    /**
     * Finished nested child docs in block order. Because children are appended when they
     * CLOSE (endNestedChild), multi-level nesting naturally lands post-order: inner replies
     * close before their enclosing comment, so replies precede the comment — matching the
     * vanilla OpenSearch nested block layout (descendants first, parent after, root last).
     */
    private final List<Document> childDocs = new ArrayList<>();
    /** Stack of currently-open child docs (innermost on top) for multi-level nesting. */
    private final ArrayDeque<Document> childDocStack = new ArrayDeque<>();

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
     * Returns the built Lucene {@link Document} — the ROOT document (last in the block).
     *
     * @return the Lucene document
     */
    @Override
    public Document getFinalInput() {
        return document;
    }

    /**
     * Returns the full block of documents for IndexWriter.addDocuments().
     * Order: all child docs in parse order (deepest first already by parse recursion),
     * then the root doc last. If no nested children exist, returns a single-element list.
     *
     * @return the ordered document block
     */
    public List<Document> getDocumentBlock() {
        if (childDocs.isEmpty()) {
            return List.of(document);
        }
        List<Document> block = new ArrayList<>(childDocs.size() + 1);
        block.addAll(childDocs);
        block.add(document); // root is LAST
        return block;
    }

    /**
     * Returns the number of Lucene documents this input will produce (children + 1 root).
     */
    public int getDocCount() {
        return childDocs.size() + 1;
    }

    /**
     * Returns whether this input has nested children.
     */
    public boolean hasNestedChildren() {
        return childDocs.isEmpty() == false;
    }

    // --- Nested boundary signals ---

    @Override
    public void startNestedChild(String nestedPath) {
        Document childDoc = new Document();
        // Add _nested_path field (keyword, indexed) so we can build per-path bitsets
        childDoc.add(new StringField(NESTED_PATH_FIELD, nestedPath, Field.Store.NO));
        childDocStack.push(childDoc);
    }

    @Override
    public void endNestedChild() {
        // Emit on close: inner children close before their enclosing element, so the
        // resulting childDocs order is post-order — the vanilla nested block layout.
        childDocs.add(childDocStack.pop());
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

        // POC: route to the innermost open child document if we're inside a nested child
        Document target = childDocStack.isEmpty() ? document : childDocStack.peek();
        factory.addField(target, fieldType, value, luceneFieldType);
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

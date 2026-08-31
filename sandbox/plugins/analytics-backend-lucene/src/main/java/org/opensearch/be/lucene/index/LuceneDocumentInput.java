/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.be.lucene.LuceneFieldFactory;
import org.opensearch.be.lucene.LuceneFieldFactoryRegistry;
import org.opensearch.be.lucene.LucenePlugin;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.dataformat.FieldTypeCapabilities;
import org.opensearch.index.mapper.MappedFieldType;

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

    // ── Nested handling: ONE Lucene doc per logical document (== one Parquet row) ──
    // Vanilla OpenSearch materializes every nested array element as its OWN Lucene document, written as a
    // contiguous block of (children..., root) via IndexWriter.addDocuments. This format deliberately does
    // NOT: the whole logical document becomes a SINGLE Lucene doc, and a nested leaf becomes a
    // MULTI-VALUED field on it — e.g. depts.teams.members.role = [lead, dev, lead], one value per element
    // in parse order. Because there is exactly one doc per row, the Lucene docId equals the Parquet rowId.
    //
    // Why dropping the element boundary is sound here: on the analytics path DataFusion evaluates the FULL
    // query over the Parquet LIST<STRUCT> and Lucene only supplies a row bitset for skipping. Losing
    // "which element matched" can therefore only ever produce FALSE POSITIVES, which DataFusion filters
    // out — never a false negative. A single-predicate filter is in fact exact, because a multi-valued
    // field answers "does SOME element have this value?" precisely.
    //
    // startNestedChild/endNestedChild are still delivered (the parser broadcasts them to every format),
    // but they only maintain a depth counter so unbalanced signals are still caught. No child Document and
    // no _nested_path term is ever created.
    private int nestedDepth = 0;

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
        // Every field lands on the single row document — including nested leaves, which therefore become
        // multi-valued (one value per array element, in parse order). The field name is already the
        // fully-qualified dotted path (e.g. depts.teams.members.role), so nothing needs renaming.
        factory.addField(document, fieldType, value, luceneFieldType);
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
     * Records entry into a nested array element. No child {@link Document} is created and no
     * {@code _nested_path} term is written — the element boundary matters only to the columnar format,
     * which starts a new {@code LIST<STRUCT>} element on this signal. Here it merely tracks depth so an
     * unbalanced {@link #endNestedChild()} is still detected.
     *
     * @param nestedPath the nested object's full path (unused; kept for interface compatibility)
     */
    @Override
    public void startNestedChild(String nestedPath) {
        nestedDepth++;
    }

    /** Records exit from a nested array element. */
    @Override
    public void endNestedChild() {
        if (nestedDepth == 0) {
            throw new IllegalStateException("endNestedChild called with no open nested child");
        }
        nestedDepth--;
    }

    /** Current nested nesting depth; {@code 0} outside any nested element. Visible for testing. */
    int nestedDepth() {
        return nestedDepth;
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

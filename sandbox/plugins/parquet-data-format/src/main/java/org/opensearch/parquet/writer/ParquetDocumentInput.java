/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.writer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.engine.dataformat.FieldTypeCapabilities;
import org.opensearch.index.engine.exec.PrimaryTermFieldType;
import org.opensearch.index.mapper.IdFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.SeqNoFieldMapper;
import org.opensearch.index.mapper.VersionFieldMapper;
import org.opensearch.parquet.ParquetDataFormatPlugin;

import java.util.AbstractMap;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Document input for the Parquet data format.
 *
 * <p>Implements {@link DocumentInput} to collect field-value pairs incrementally during
 * document indexing. Fields are stored as {@link FieldValuePair} objects and later transferred
 * to Arrow vectors by {@link org.opensearch.parquet.vsr.VSRManager#addDocument(ParquetDocumentInput)}.
 *
 * <p>Calling {@link #close()} clears all collected fields and resets the row ID,
 * allowing the instance to be discarded cleanly after use.
 */
public class ParquetDocumentInput implements DocumentInput<List<FieldValuePair>> {

    private static final Logger logger = LogManager.getLogger(ParquetDocumentInput.class);
    private final List<FieldValuePair> collectedFields = new ArrayList<>();
    // Keyed by field name, not field-type identity: within a single document parse each logical
    // field (including the derived-source `_ignored_source.*` companion) has a unique name, while
    // identity would silently miss a match if the parser ever handed back a fresh wrapper per array
    // element — degrading a multi_value field to last-value-wins or bypassing the scalar duplicate
    // guard. Name keying makes accumulation robust to that.
    private final Map<String, FieldValuePair> seen = new HashMap<>();
    private long rowId = -1;
    private boolean isClosed = false;
    // Nested support: children buffered hierarchically, in parse order.
    // A stack tracks currently-open elements so multi-level nesting (comments -> replies)
    // attaches inner elements to their enclosing element instead of losing them.
    private final List<NestedChild> topLevelChildren = new ArrayList<>();
    private final ArrayDeque<NestedChild> childStack = new ArrayDeque<>();
    // Map support: entries of map-typed fields (e.g. a flat_object's attributes) emitted at the document
    // root (not inside any nested element). Keyed by the map field's full name; each entry is one
    // (key,value) pair, preserved in parse order.
    private final LinkedHashMap<String, List<Map.Entry<String, Object>>> topLevelMapEntries = new LinkedHashMap<>();

    /**
     * One nested array element: its full dotted path (e.g. "comments"), its leaf field
     * values in parse order, any deeper nested elements it contains (e.g. replies), and any
     * map-typed fields (e.g. a flat_object {@code attributes}) buffered as key/value entries.
     */
    public static class NestedChild {
        public final String path;
        public final List<FieldValuePair> fields = new ArrayList<>();
        public final List<NestedChild> children = new ArrayList<>();
        // map field full name -> its (key,value) entries in parse order (one MAP<Utf8,Utf8> per key).
        public final LinkedHashMap<String, List<Map.Entry<String, Object>>> mapEntries = new LinkedHashMap<>();

        NestedChild(String path) {
            this.path = path;
        }
    }

    @Override
    public void startNestedChild(String nestedPath) {
        ensureOpen();
        childStack.push(new NestedChild(nestedPath));
    }

    @Override
    public void endNestedChild() {
        ensureOpen();
        NestedChild finished = childStack.pop();
        if (childStack.isEmpty()) {
            topLevelChildren.add(finished);
        } else {
            childStack.peek().children.add(finished);
        }
    }

    /** Returns the buffered top-level nested elements in parse order (POC). */
    public List<NestedChild> getNestedChildren() {
        return topLevelChildren;
    }

    @Override
    public void addMapEntry(MappedFieldType mapField, String key, Object value) {
        ensureOpen();
        Map.Entry<String, Object> entry = new AbstractMap.SimpleEntry<>(key, value);
        // Inside a nested element the map lives in that element's struct; otherwise it is a
        // document-root map column.
        LinkedHashMap<String, List<Map.Entry<String, Object>>> target = childStack.isEmpty()
            ? topLevelMapEntries
            : childStack.peek().mapEntries;
        target.computeIfAbsent(mapField.name(), k -> new ArrayList<>()).add(entry);
    }

    /** Returns the document-root map fields (name -&gt; entries), for MAP columns not inside a nested field. */
    public LinkedHashMap<String, List<Map.Entry<String, Object>>> getTopLevelMapEntries() {
        return topLevelMapEntries;
    }

    @Override
    public void addField(MappedFieldType fieldType, Object value) {
        ensureOpen();
        // POC: route fields inside a nested child to the innermost open element (no dedup —
        // different children legitimately repeat the same field type).
        if (childStack.isEmpty() == false) {
            childStack.peek().fields.add(new FieldValuePair(fieldType, value));
            return;
        }
        Set<FieldTypeCapabilities.Capability> capabilities = fieldType.getCapabilityMap()
            .getOrDefault(ParquetDataFormatPlugin.PARQUET_DATA_FORMAT, Set.of());
        if (capabilities.isEmpty() && fieldType != PrimaryTermFieldType.INSTANCE) {
            // nothing to support on this format for this field.
            logger.trace("Ignored to add field: {} {}", fieldType.name(), fieldType.getCapabilityMap());
            return;
        }
        FieldValuePair existing = seen.get(fieldType.name());
        if (existing == null) {
            // Fields declared `multi_value: true` in the mapping start out as a list of one so the
            // value shape reaching the VSR is the same whether the document had one value or several.
            // An explicit empty array (`"field": []`) is signalled by an empty List and seeds a
            // zero-value pair, so its LIST cell is written empty-but-non-null rather than null.
            final FieldValuePair pair;
            if (fieldType.isMultiValued()) {
                pair = value instanceof List<?> list && list.isEmpty()
                    ? FieldValuePair.emptyMultiValued(fieldType)
                    : FieldValuePair.multiValued(fieldType, value);
            } else {
                pair = new FieldValuePair(fieldType, value);
            }
            seen.put(fieldType.name(), pair);
            collectedFields.add(pair);
            return;
        }
        if (existing.isMultiValued() == false) {
            if (fieldType.isMultiValueSupported() && fieldType.isMultiValueAutoPromotionEnabled()) {
                existing.promoteToMultiValued(value);
                return;
            }
            String reason = fieldType.isMultiValueSupported()
                ? "the field is locked scalar by [multi_value: false]"
                : "the field type does not support automatic multi-value promotion";
            throw new MapperParsingException(
                "Cannot accept multiple values for field: [" + fieldType.name() + "] of type: [" + fieldType.typeName() + "]: " + reason
            );
        }
        existing.addValue(value);
    }

    @Override
    public void setRowId(String rowIdFieldName, long rowId) {
        ensureOpen();
        this.rowId = rowId;
    }

    @Override
    public List<FieldValuePair> getFinalInput() {
        if (!isClosed) {
            assert rowId >= 0 : "Row ID must be set before calling getFinalInput";
            // assertions for parquet primary
            // TODO: once parquet is supported in secondary mode, this assertion would change
            assert getFieldCount(IdFieldMapper.NAME) == 1;
            assert getFieldCount(SeqNoFieldMapper.NAME) == 1;
            assert getFieldCount(VersionFieldMapper.NAME) == 1;
            assert getFieldCount(SeqNoFieldMapper.PRIMARY_TERM_NAME) == 1;
        }
        return collectedFields;
    }

    @Override
    public long getFieldCount(String fieldName) {
        // Counts values, not entries: a multi-valued field is one entry holding N values, and
        // callers (single-value assertions below, the data-stream @timestamp check) mean values.
        return collectedFields.stream()
            .filter(fvp -> fvp.getFieldType().name().equals(fieldName))
            .mapToLong(FieldValuePair::valueCount)
            .sum();
    }

    @Override
    public void close() {
        isClosed = true;
        collectedFields.clear();
        seen.clear();
        topLevelChildren.clear();
        childStack.clear();
        topLevelMapEntries.clear();
        rowId = -1;
    }

    private void ensureOpen() {
        if (isClosed) {
            throw new IllegalStateException("Cannot add more fields to a frozen document input");
        }
    }

    /**
     * Returns the row ID assigned to this document.
     *
     * @return the row ID, or -1 if not set
     */
    public long getRowId() {
        return rowId;
    }
}

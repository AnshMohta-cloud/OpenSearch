/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.engine.dataformat;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.index.mapper.MappedFieldType;

/**
 * Represents a document input for adding fields and metadata to a writer.
 *
 * @param <T> the type of the final input representation
 *
 * @opensearch.experimental
 */
@ExperimentalApi
public interface DocumentInput<T> extends AutoCloseable {

    /** Standard field name for the row ID used to correlate documents across data formats. */
    String ROW_ID_FIELD = "__row_id__";

    /**
     * Standard field name written on each nested CHILD document identifying which nested level it belongs
     * to (the full dotted mapper path, e.g. {@code comments} or {@code comments.replies}). Mirrors vanilla
     * OpenSearch's {@code _nested_path}. Written only on child docs, never on the root.
     */
    String NESTED_PATH_FIELD = "_nested_path";

    /**
     * Gets the final input representation.
     *
     * @return the final input of type T
     */
    T getFinalInput();

    /**
     * Adds a field to the document.
     *
     * @param fieldType the mapped field type
     * @param value the field value
     */
    void addField(MappedFieldType fieldType, Object value);

    /**
     * Adds a row ID field to the document.
     *
     * @param rowIdFieldName the name of the row ID field
     * @param rowId the row ID value
     */
    void setRowId(String rowIdFieldName, long rowId);

    /**
     * Signals the start of a nested child object at {@code nestedPath} (the full dotted mapper path).
     * Subsequent {@link #addField} calls, until the matching {@link #endNestedChild()}, belong to this
     * nested child (or a deeper one). A format that materializes nested objects as their own documents
     * (e.g. the Lucene block layout) opens a new child document here; formats that flatten nested arrays
     * into columnar structures (e.g. Parquet) begin a new element. Nesting composes to arbitrary depth.
     *
     * <p>Default is a no-op so formats with no nested-block notion are unaffected and existing callers
     * that never emit these signals keep their current behavior.
     *
     * @param nestedPath the full dotted path of the nested object (e.g. {@code comments.replies})
     */
    default void startNestedChild(String nestedPath) {}

    /**
     * Signals the end of the innermost open nested child (the match to the most recent
     * {@link #startNestedChild(String)}). Inner children close before their enclosing element, so a format
     * that buffers children on close naturally lands them in post-order (descendants first, enclosing
     * element after) — the vanilla nested block order. Default is a no-op.
     */
    default void endNestedChild() {}

    /**
     * Given a field name, returns the number of values associated with that field in the document.
     * @param fieldName name of the field to lookup
     * @return count of field values
     */
    long getFieldCount(String fieldName);
}

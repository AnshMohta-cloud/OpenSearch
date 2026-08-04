/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.exec;

import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.util.Text;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Reconstructs an OpenSearch {@code _source} value from a DataFusion/arrow-rs Arrow vector, faithfully to the
 * mapping — the Arrow path counterpart of the Parquet-direct (FFI + Dremel) reconstruction in the
 * {@code parquet-data-format} codec. Given a nested field's {@code LIST<STRUCT<...>>} vector, it produces the
 * identical {@code List<Map<String,Object>>} that codec path produces, so a composite index can rebuild nested
 * {@code _source}/{@code inner_hits} from a DataFusion row-id fetch instead of hand-assembling Dremel levels.
 *
 * <p>Why a new serializer rather than {@link ArrowValues#toSourceValue}: the Arrow schema alone cannot recover the
 * OpenSearch type of a leaf — {@code ip} and {@code binary} are both Arrow {@code VarBinary}; {@code keyword} and
 * {@code text} are both {@code Utf8}; {@code date}/{@code date_nanos}/{@code long} are all 64-bit ints. Only the
 * mapping's {@link org.opensearch.index.mapper.MappedFieldType} disambiguates them and supplies date formats. This
 * serializer therefore takes a {@link TypeResolver} keyed by dotted field path (built up during descent, exactly as
 * the field name is the Parquet column address) and renders each leaf through the shared
 * {@link org.opensearch.index.mapper.SourceValueFormatters}, so both reconstruction paths cannot drift.
 *
 * <p>Fidelity/ordering contract (matches the codec path and the nested inner_hits offset requirement):
 * <ul>
 *   <li>Element order and duplicates within a {@code LIST} are preserved by iterating the {@code [start,end)}
 *       offset window (never sorted).</li>
 *   <li>A null leaf omits its key; an empty (or all-omitted) sub-array/object omits its key — matching the codec's
 *       "empty arrays produce no key" behavior.</li>
 *   <li>Arbitrary nesting depth: every {@link ListVector} child recurses, every {@link StructVector} recurses, and
 *       the dotted path grows by one segment per hop. No hardcoded depth.</li>
 *   <li>Numeric precision is preserved narrowly ({@code float}&rarr;{@code Float}, {@code byte}/{@code short}
 *       &rarr;{@code Integer}, {@code long}&rarr;{@code Long}), unlike {@link ArrowValues} which widens.</li>
 * </ul>
 */
public final class ArrowSourceSerializer {

    private ArrowSourceSerializer() {}

    /** Resolves the OpenSearch mapping value-formatter for a dotted leaf path (e.g. {@code items.host}). */
    @FunctionalInterface
    public interface TypeResolver {
        /**
         * The {@code _source} value formatter for the leaf at {@code dottedPath}, or {@code null} if the path is not
         * a known scalar leaf. Implementations typically delegate to
         * {@link org.opensearch.index.mapper.SourceValueFormatters#forField} using the field's
         * {@link org.opensearch.index.mapper.MappedFieldType}.
         */
        UnaryOperator<Object> formatterFor(String dottedPath);
    }

    /**
     * Reconstructs the {@code _source} array for a nested {@code LIST<STRUCT>} field at one row, byte-identical to
     * the codec's {@code readNestedArray(path, row)}.
     *
     * @param nested the Arrow {@link ListVector} for {@code path}
     * @param row    the row index within {@code nested}
     * @param path   the dotted nested root (e.g. {@code products} / {@code orgs})
     * @param types  resolves each leaf's mapping formatter by dotted path
     * @return the ordered element list (empty if the row's array is null/empty)
     */
    public static List<Map<String, Object>> toNestedSource(ListVector nested, int row, String path, TypeResolver types) {
        if (nested == null || nested.isNull(row)) {
            return List.of();
        }
        int start = nested.getElementStartIndex(row);
        int end = nested.getElementEndIndex(row);
        if (start >= end) {
            return List.of();
        }
        FieldVector data = nested.getDataVector();
        if (data instanceof StructVector == false) {
            // A nested field's element vector is always a struct; defensively return empty rather than misread.
            return List.of();
        }
        StructVector struct = (StructVector) data;
        List<Map<String, Object>> out = new ArrayList<>(end - start);
        for (int i = start; i < end; i++) {
            out.add(elementToMap(struct, i, path, types));
        }
        return out;
    }

    /** One struct element at {@code idx} → an ordered field map, recursing children with their dotted paths. */
    private static Map<String, Object> elementToMap(StructVector struct, int idx, String path, TypeResolver types) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (struct.isNull(idx)) {
            return map; // a null element occupies its offset as an empty object (keeps element alignment)
        }
        for (FieldVector child : struct.getChildrenFromFields()) {
            String name = child.getField().getName();
            Object v = toSourceValue(child, idx, path + "." + name, types);
            if (v != null) {
                map.put(name, v);
            }
        }
        return map;
    }

    /** One Arrow cell → its {@code _source} value: recurse for List/Struct, else render the scalar via the mapping. */
    private static Object toSourceValue(FieldVector vec, int idx, String dottedPath, TypeResolver types) {
        if (vec == null || vec.isNull(idx)) {
            return null;
        }
        if (vec instanceof ListVector) {
            List<Map<String, Object>> sub = toNestedSource((ListVector) vec, idx, dottedPath, types);
            return sub.isEmpty() ? null : sub; // omit empty sub-arrays (codec parity)
        }
        if (vec instanceof StructVector) {
            Map<String, Object> sub = elementToMap((StructVector) vec, idx, dottedPath, types);
            return sub.isEmpty() ? null : sub; // omit empty objects (codec parity)
        }
        Object raw = decodeScalar(vec, idx);
        if (raw == null) {
            return null;
        }
        UnaryOperator<Object> formatter = types.formatterFor(dottedPath);
        // No mapping formatter (unknown/unsupported leaf) → pass the decoded value through unchanged.
        return formatter == null ? raw : formatter.apply(raw);
    }

    /**
     * Decodes a scalar Arrow cell into the physical Java value the shared formatters expect, preserving precision
     * and the exact byte payload for BYTE_ARRAY-backed types:
     * <ul>
     *   <li>{@link VarCharVector} (keyword/text) &rarr; {@code byte[]} (UTF-8) — the formatter turns it into a String;</li>
     *   <li>{@link VarBinaryVector} (ip/binary) &rarr; raw {@code byte[]} — the formatter renders ip dotted / binary Base64;</li>
     *   <li>{@link TinyIntVector}/{@link SmallIntVector}/{@link IntVector} &rarr; {@code Integer};</li>
     *   <li>{@link BigIntVector} (long/date/date_nanos) &rarr; {@code Long} — the date formatter consumes the epoch;</li>
     *   <li>{@link Float4Vector} &rarr; {@code Float} (single precision kept), {@link Float8Vector} &rarr; {@code Double};</li>
     *   <li>{@link BitVector} &rarr; {@code Boolean}.</li>
     * </ul>
     * The {@code byte[]} shape for text/keyword mirrors the codec's BYTE_ARRAY leaf, where the shared formatter's
     * keyword/text case decodes UTF-8.
     */
    private static Object decodeScalar(FieldVector vec, int idx) {
        if (vec instanceof VarCharVector) {
            return ((VarCharVector) vec).get(idx); // byte[] (UTF-8); SourceValueFormatters keyword/text decodes it
        }
        if (vec instanceof VarBinaryVector) {
            return ((VarBinaryVector) vec).get(idx); // raw byte[] for ip (16-byte InetAddressPoint) / binary
        }
        if (vec instanceof TinyIntVector) {
            return (int) ((TinyIntVector) vec).get(idx);
        }
        if (vec instanceof SmallIntVector) {
            return (int) ((SmallIntVector) vec).get(idx);
        }
        if (vec instanceof IntVector) {
            return ((IntVector) vec).get(idx);
        }
        if (vec instanceof BigIntVector) {
            return ((BigIntVector) vec).get(idx);
        }
        if (vec instanceof Float4Vector) {
            return ((Float4Vector) vec).get(idx);
        }
        if (vec instanceof Float8Vector) {
            return ((Float8Vector) vec).get(idx);
        }
        if (vec instanceof BitVector) {
            return ((BitVector) vec).get(idx) != 0;
        }
        // Fallback for any vector not explicitly handled: unwrap Arrow Text, else pass the boxed object through.
        Object obj = vec.getObject(idx);
        return obj instanceof Text ? ((Text) obj).toString() : obj;
    }
}

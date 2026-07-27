/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.lucene.util.BytesRef;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure Dremel record-assembly for nested {@code LIST<STRUCT>} columns — the algorithm that turns a single
 * leaf's per-slot repetition/definition levels (plus its present values) into either a reconstructed nested
 * tree ({@link #assembleLeafInto}) or a per-element view at a chosen nesting level ({@link #expandElementsAtLevel}).
 *
 * <p>This class is deliberately free of any Parquet/DocValues I/O so the striped-assembly logic can be reasoned
 * about and unit-tested in isolation against hand-built level streams. All grouping/cardinality/null/empty-list
 * semantics are driven by the standard Dremel encoding:
 * <ul>
 *   <li><b>rep</b> (repetition level) names the deepest list that continued; a slot with {@code rep <= k} begins
 *       a NEW element at nesting level {@code k} (and resets the element indices of all deeper levels).</li>
 *   <li><b>def</b> (definition level) says how deeply the path is defined; an element at level {@code k} EXISTS in
 *       a slot iff {@code def >= levelThresholds[k-1]} — the schema-derived "element exists" threshold for that
 *       level (NOT a hardcoded per-level increment, so this generalizes to any LIST encoding). The leaf value
 *       itself is present iff {@code def == maxDef}.</li>
 * </ul>
 */
final class NestedDremel {

    private NestedDremel() {}

    /**
     * Deepest nesting level whose element exists in a slot with definition level {@code def}: the count of leading
     * levels whose ascending threshold is satisfied. {@code levelThresholds[k-1]} is the min def for a level-{@code k}
     * element to exist; thresholds are ascending, so the first unsatisfied level stops the chain.
     *
     * @param def            the slot's definition level
     * @param depth          the leaf's own nesting depth (only levels {@code 1..depth} are relevant here)
     * @param levelThresholds per-level element-exists thresholds (index {@code k-1} → level {@code k})
     * @return the number of existing ancestor levels (0 means the field is absent/null for this slot)
     */
    static int existingLevels(int def, int depth, int[] levelThresholds) {
        int existing = 0;
        for (int k = 1; k <= depth && k <= levelThresholds.length; k++) {
            if (def >= levelThresholds[k - 1]) {
                existing = k;
            } else {
                break;
            }
        }
        return existing;
    }

    /**
     * Assembles one leaf's level stream into {@code root}, the shared reconstructed array of the TOP-LEVEL nested
     * field. Mirrors the standard Dremel assembly: an index path {@code idx[1..depth]} advances by each slot's
     * repetition level; ancestor elements are created on demand; the scalar leaf value is placed on its level-{@code
     * depth} element only when fully defined. Multiple leaves of the same nested field call this against the same
     * {@code root} to co-assemble one tree (order-independent — elements are addressed by index path).
     *
     * @param root            the (mutable) element list being assembled into
     * @param fieldChain       relative chain from the nested root to this leaf; {@code chain[0..depth-2]} are the
     *                         nested container names, {@code chain[depth-1]} is the scalar field name
     * @param depth            {@code == fieldChain.length}; the list-level at which the value lives
     * @param rep              per-slot repetition levels
     * @param def              per-slot definition levels
     * @param maxDef           the column's max definition level (value present iff {@code def == maxDef})
     * @param levelThresholds  per-level element-exists thresholds
     * @param values           the present values in slot order — {@code Long} bits (numeric) or {@code byte[]}/
     *                         {@code null} (BYTE_ARRAY); consumed one per slot with {@code def == maxDef}
     * @param byteArray        whether the leaf is a BYTE_ARRAY column (values are {@code byte[]})
     * @param physical         physical type used to decode numeric bits to a JSON-friendly value
     * @param valueFormatter   applied to each decoded value before it is placed in the tree — lets the caller
     *                         render logical types faithfully (a {@code date} epoch-long → its date string, an
     *                         {@code ip}'s raw bytes → dotted form). For a BYTE_ARRAY leaf the formatter receives
     *                         the raw {@code byte[]}; for a numeric leaf it receives the physically-decoded value
     *                         (Integer/Long/Double/Boolean). Returning the input unchanged is the identity/default.
     *                         A {@code null} return drops the value (field omitted).
     */
    /**
     * Convenience overload with the DEFAULT value formatter: BYTE_ARRAY leaves render as UTF-8 {@code String}
     * and numeric leaves pass through unchanged. Used by callers that need no logical-type formatting (and by
     * the unit tests, so the assembly algorithm is exercised independently of any type rendering).
     */
    static void assembleLeafInto(
        List<Map<String, Object>> root,
        String[] fieldChain,
        int depth,
        int[] rep,
        int[] def,
        int maxDef,
        int[] levelThresholds,
        Object[] values,
        boolean byteArray,
        ParquetPhysicalType physical
    ) {
        java.util.function.UnaryOperator<Object> defaultFmt = byteArray
            ? v -> new String((byte[]) v, StandardCharsets.UTF_8)
            : java.util.function.UnaryOperator.identity();
        assembleLeafInto(root, fieldChain, depth, rep, def, maxDef, levelThresholds, values, byteArray, physical, defaultFmt);
    }

    static void assembleLeafInto(
        List<Map<String, Object>> root,
        String[] fieldChain,
        int depth,
        int[] rep,
        int[] def,
        int maxDef,
        int[] levelThresholds,
        Object[] values,
        boolean byteArray,
        ParquetPhysicalType physical,
        java.util.function.UnaryOperator<Object> valueFormatter
    ) {
        int[] idx = new int[depth + 1];
        int valueCursor = 0;
        for (int s = 0; s < rep.length; s++) {
            int r = rep[s];
            int d = def[s];
            if (s == 0) {
                java.util.Arrays.fill(idx, 0);
            } else {
                idx[r]++;
                for (int k = r + 1; k <= depth; k++) {
                    idx[k] = 0;
                }
            }

            int existing = existingLevels(d, depth, levelThresholds);
            boolean valuePresent = d == maxDef;
            if (existing < 1) {
                // Not even a level-1 element — the whole nested field is null/absent for this slot. A present
                // value is impossible here (def < maxDef), so no value cursor is consumed.
                continue;
            }

            List<Map<String, Object>> currentList = root;
            Map<String, Object> currentElem = null;
            for (int k = 1; k <= existing; k++) {
                currentElem = getOrCreateElement(currentList, idx[k]);
                if (k < existing) {
                    currentList = childListOf(currentElem, fieldChain[k - 1]);
                }
            }

            if (valuePresent) {
                // Physically decode: BYTE_ARRAY → raw bytes (formatter turns keyword/text into a String, ip/binary
                // into their logical form); numeric → the physical Java value. Then apply the caller's logical
                // formatter (identity by default).
                Object raw;
                if (byteArray) {
                    raw = values[valueCursor];  // byte[] or null
                } else {
                    raw = decodePrimitive(((Number) values[valueCursor]).longValue(), physical);
                }
                valueCursor++;
                Object value = raw == null ? null : valueFormatter.apply(raw);
                if (value != null && currentElem != null) {
                    currentElem.put(fieldChain[depth - 1], value);
                }
            }
        }
    }

    /** A per-element view at one nesting level: value + presence indexed by element offset. */
    record ElementValues(boolean[] present, long[] longs, BytesRef[] bytes) {}

    /**
     * Expands a leaf's level stream into per-element arrays indexed by the element offset at nesting {@code level}
     * (1-based; the field's owning list level). A NEW element at {@code level} begins at a slot whose repetition
     * level {@code rep <= level} AND whose definition level reaches {@code existThreshold} (the schema-derived
     * "element exists at this level" boundary). Elements that exist but whose leaf is null still occupy an offset
     * (so the child-doc offset lines up), just with {@code present=false}. Present values (def == maxDef) are
     * consumed in slot order; when several descendant values collapse into one element (a shallower {@code level}),
     * the FIRST is kept as the element's representative scalar.
     */
    static ElementValues expandElementsAtLevel(
        int[] rep,
        int[] def,
        int maxDef,
        int level,
        int existThreshold,
        Object[] values,
        boolean byteArray,
        ParquetPhysicalType physical
    ) {
        int elements = 0;
        for (int s = 0; s < rep.length; s++) {
            if (def[s] >= existThreshold && rep[s] <= level) {
                elements++;
            }
        }
        boolean[] present = new boolean[elements];
        long[] longs = byteArray ? null : new long[elements];
        BytesRef[] bytes = byteArray ? new BytesRef[elements] : null;
        int elemIdx = -1;
        int valueCursor = 0;
        for (int s = 0; s < rep.length; s++) {
            if (def[s] >= existThreshold && rep[s] <= level) {
                elemIdx++;
            }
            if (def[s] == maxDef) {
                if (elemIdx >= 0 && present[elemIdx] == false) {
                    if (byteArray) {
                        byte[] raw = (byte[]) values[valueCursor];
                        bytes[elemIdx] = raw == null ? null : new BytesRef(raw);
                        present[elemIdx] = raw != null;
                    } else {
                        longs[elemIdx] = decodePrimitiveToLong(((Number) values[valueCursor]).longValue(), physical);
                        present[elemIdx] = true;
                    }
                }
                valueCursor++;
            }
        }
        return new ElementValues(present, longs, bytes);
    }

    /** Number of elements at nesting {@code level} in a leaf's level stream (see {@link #expandElementsAtLevel}). */
    static int countElementsAtLevel(int[] rep, int[] def, int level, int existThreshold) {
        int elements = 0;
        for (int s = 0; s < rep.length; s++) {
            if (def[s] >= existThreshold && rep[s] <= level) {
                elements++;
            }
        }
        return elements;
    }

    /** Element {@code Map} at index {@code at}, appending empty maps so {@code list.size() == at+1}. */
    static Map<String, Object> getOrCreateElement(List<Map<String, Object>> list, int at) {
        while (list.size() <= at) {
            list.add(new LinkedHashMap<>());
        }
        return list.get(at);
    }

    /** Child sub-array under {@code containerName}, created (and stored) empty on first access. */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> childListOf(Map<String, Object> elem, String containerName) {
        Object existing = elem.get(containerName);
        if (existing instanceof List) {
            return (List<Map<String, Object>>) existing;
        }
        List<Map<String, Object>> created = new ArrayList<>();
        elem.put(containerName, created);
        return created;
    }

    /** Decodes a primitive value from its raw {@code long} bits per the child's Parquet physical type. */
    static Object decodePrimitive(long bits, ParquetPhysicalType physical) {
        return switch (physical) {
            case INT32 -> (int) bits;
            case INT64 -> bits;
            case FLOAT -> (double) Float.intBitsToFloat((int) bits);
            case DOUBLE -> Double.longBitsToDouble(bits);
            case BOOL -> bits != 0L;
            default -> bits;
        };
    }

    /**
     * The raw {@code long} form for a numeric DV: identity pass-through, because the native reader already
     * encodes bits exactly as the single-valued numeric DV path serves them (INT32 sign-extended, INT64 as-is,
     * FLOAT/DOUBLE raw IEEE-754 bits, BOOL 0/1).
     */
    static long decodePrimitiveToLong(long bits, ParquetPhysicalType physical) {
        return bits;
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Materializes a flat (top-level, single-valued) column's values for a whole primary-format file, so the
 * {@code parquet-data-format} codec can serve its Lucene doc-values accessors from a composite engine's analytical
 * backend (e.g. DataFusion) instead of the FFI decode path. A backend registers an implementation; the codec's
 * doc-values accessors look it up and, when the feature is enabled, materialize a field's column through it —
 * otherwise they fall back to their own FFI reader.
 *
 * <p>This is the query-phase counterpart to {@link NestedSourceReconstructor}: because a flat field's derived
 * {@code _source} is produced by the same doc-values accessors that serve range/aggregation/sort, routing those
 * accessors through one backend keeps {@code _source}, filtering, aggregation and sort reading a single source of
 * truth. Values are returned keyed by physical row position (0..rowCount-1); the codec applies its existing
 * docId→row translation.
 *
 * <p>Registration and lookup mirror {@link NestedSourceReconstructor} (a static holder in {@code :server}, the only
 * classloader shared by the codec and the backend plugin).
 */
public interface FlatColumnValueSource {

    /** Process-wide registry holding the single active source (or none). */
    AtomicReference<FlatColumnValueSource> REGISTRY = new AtomicReference<>();

    /** Registers {@code source} as the active implementation (last registration wins). */
    static void register(FlatColumnValueSource source) {
        REGISTRY.set(source);
    }

    /** The active source, or {@code null} if no backend has registered one. */
    static FlatColumnValueSource get() {
        return REGISTRY.get();
    }

    /** A materialized single-valued numeric column: {@code present[row]} gates {@code values[row]} (a {@code long}). */
    final class LongColumn {
        public final long[] values;
        public final boolean[] present;

        public LongColumn(long[] values, boolean[] present) {
            this.values = values;
            this.present = present;
        }
    }

    /** A materialized single-valued byte column: {@code present[row]} gates {@code values[row]} (raw bytes). */
    final class BytesColumn {
        public final byte[][] values;
        public final boolean[] present;

        public BytesColumn(byte[][] values, boolean[] present) {
            this.values = values;
            this.present = present;
        }
    }

    /**
     * Materializes the numeric {@code field} across the whole file as per-row {@code long} bits (the same bit
     * encoding the codec's single-valued numeric doc-values serve: INT32 sign-extended, INT64 as-is, FLOAT/DOUBLE
     * raw IEEE-754 bits, BOOL 0/1). Returns {@code null} if this source cannot serve the field.
     *
     * @param parquetFilePath absolute path of the file backing this segment
     * @param field           the flat field name
     * @param rowCount        the file's row count (materialized arrays are this length)
     */
    LongColumn materializeLong(String parquetFilePath, String field, int rowCount) throws IOException;

    /**
     * Materializes the byte-array {@code field} (keyword/text/ip/binary) across the whole file as per-row raw bytes
     * (unformatted; the caller renders/encodes as needed). Returns {@code null} if this source cannot serve the field.
     */
    BytesColumn materializeBytes(String parquetFilePath, String field, int rowCount) throws IOException;
}

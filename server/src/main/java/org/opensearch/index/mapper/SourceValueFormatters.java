/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import org.apache.lucene.document.InetAddressPoint;
import org.opensearch.common.network.InetAddresses;

import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Base64;
import java.util.Locale;
import java.util.function.UnaryOperator;

/**
 * Per-OpenSearch-type renderers that turn a physically-decoded columnar value into the exact form it takes in
 * a document's {@code _source}. This is the single source of truth for composite (pluggable data format)
 * derived-source reconstruction, shared by every reconstruction backend so they cannot drift byte-for-byte:
 * <ul>
 *   <li>the Parquet-direct (FFI + Dremel) reconstruction in the {@code parquet-data-format} codec, and</li>
 *   <li>the Arrow-batch reconstruction used when {@code _source} is rebuilt from a DataFusion fetch.</li>
 * </ul>
 *
 * <p>The rendering mirrors how each field mapper serializes a value into {@code _source}:
 * <ul>
 *   <li>{@code keyword}/{@code text}: raw UTF-8 bytes &rarr; {@link String}.</li>
 *   <li>{@code ip}: the fixed 16-byte {@link InetAddressPoint} encoding &rarr; canonical/dotted address
 *       (as {@code IpFieldMapper} renders it).</li>
 *   <li>{@code binary}: raw bytes &rarr; Base64 (the shape a {@code binary} field serializes as).</li>
 *   <li>{@code date}/{@code date_nanos}: epoch millis/nanos &rarr; the field's date string. The default ISO
 *       formatter matches the prior codec behavior; the {@link MappedFieldType}-aware factory instead honors a
 *       field's configured {@code format} where available (see {@link #forField}).</li>
 *   <li>numerics/{@code boolean}: already the correct Java value &rarr; identity.</li>
 * </ul>
 *
 * <p>Each returned {@link UnaryOperator} accepts the physically-decoded value (a {@code byte[]} for
 * BYTE_ARRAY-backed types; a boxed {@code Number}/{@code Boolean} for numeric types) and returns the
 * {@code _source}-ready value. A {@code null} input is passed straight through as {@code null}.
 */
public final class SourceValueFormatters {

    private SourceValueFormatters() {}

    /**
     * Default strict formatter for {@code date}/{@code date_nanos} leaves when a field's configured format is not
     * consulted. Kept byte-identical to the historical codec default ({@code yyyy-MM-dd'T'HH:mm:ss.SSS'Z'}, UTC).
     */
    public static final DateTimeFormatter DEFAULT_DATE_FORMATTER = DateTimeFormatter.ofPattern(
        "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
        Locale.ROOT
    );

    /**
     * The {@code _source} value formatter for a leaf of the given OpenSearch mapping {@code typeName}, using the
     * default date rendering. This is the exact behavior the Parquet codec's reconstruction relied on historically,
     * extracted verbatim so both reconstruction backends share one implementation.
     *
     * @param typeName the OpenSearch mapping type name (e.g. {@code keyword}, {@code ip}, {@code date})
     * @return a formatter mapping a physically-decoded value to its {@code _source} form
     */
    public static UnaryOperator<Object> forType(String typeName) {
        switch (typeName) {
            case "keyword":
            case "text":
                return v -> v == null ? null : new String((byte[]) v, StandardCharsets.UTF_8);
            case "ip":
                return SourceValueFormatters::renderIp;
            case "binary":
                return v -> v == null ? null : Base64.getEncoder().encodeToString((byte[]) v);
            case "date":
                return v -> renderEpochMillis(v, DEFAULT_DATE_FORMATTER);
            case "date_nanos":
                return v -> renderEpochNanos(v, DEFAULT_DATE_FORMATTER);
            default:
                return UnaryOperator.identity(); // numerics, boolean — already decoded
        }
    }

    /**
     * The {@code _source} value formatter for a specific mapped field. Identical to {@link #forType(String)} for all
     * types except that {@code date}/{@code date_nanos} render through the field's configured
     * {@link DateFieldMapper.DateFieldType#dateTimeFormatter() format} when the field is a date field, so a mapping
     * with a custom {@code format} reconstructs the same string vanilla stored. Falls back to
     * {@link #forType(String)} when the field type carries no usable date format.
     *
     * @param fieldType the mapped field type of the leaf (may be {@code null})
     * @return a formatter mapping a physically-decoded value to its {@code _source} form
     */
    public static UnaryOperator<Object> forField(MappedFieldType fieldType) {
        if (fieldType == null) {
            return UnaryOperator.identity();
        }
        if (fieldType instanceof DateFieldMapper.DateFieldType) {
            DateFieldMapper.DateFieldType dft = (DateFieldMapper.DateFieldType) fieldType;
            // Render exactly as DateFieldMapper's derived source does: the field's configured formatter applied to
            // the epoch value resolved through the field's own resolution (millis vs nanos), pinned to UTC.
            org.opensearch.common.time.DateFormatter formatter = dft.dateTimeFormatter();
            DateFieldMapper.Resolution resolution = dft.resolution();
            return v -> {
                if (v == null) {
                    return null;
                }
                long raw = ((Number) v).longValue();
                return formatter.format(resolution.toInstant(raw).atZone(ZoneOffset.UTC));
            };
        }
        return forType(fieldType.typeName());
    }

    private static Object renderIp(Object v) {
        if (v == null) {
            return null;
        }
        InetAddress addr = InetAddressPoint.decode((byte[]) v);
        return InetAddresses.toAddrString(addr);
    }

    private static Object renderEpochMillis(Object v, DateTimeFormatter formatter) {
        if (v == null) {
            return null;
        }
        long millis = ((Number) v).longValue();
        return formatter.format(Instant.ofEpochMilli(millis).atZone(ZoneOffset.UTC));
    }

    private static Object renderEpochNanos(Object v, DateTimeFormatter formatter) {
        if (v == null) {
            return null;
        }
        long nanos = ((Number) v).longValue();
        return formatter.format(Instant.ofEpochSecond(nanos / 1_000_000_000L, nanos % 1_000_000_000L).atZone(ZoneOffset.UTC));
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import org.apache.lucene.document.InetAddressPoint;
import org.opensearch.test.OpenSearchTestCase;

import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.util.function.UnaryOperator;

/**
 * Unit coverage for {@link SourceValueFormatters} — the shared per-type {@code _source} renderer used by both the
 * Parquet-direct and Arrow-batch derived-source reconstruction paths. These assertions pin the exact output form so
 * the two paths cannot drift, and lock the historical defaults the Parquet codec relied on.
 */
public class SourceValueFormattersTests extends OpenSearchTestCase {

    public void testKeywordAndTextRenderUtf8String() {
        byte[] bytes = "héllo".getBytes(StandardCharsets.UTF_8);
        assertEquals("héllo", SourceValueFormatters.forType("keyword").apply(bytes));
        assertEquals("héllo", SourceValueFormatters.forType("text").apply(bytes));
    }

    public void testIpRendersDottedFromInetAddressPointEncoding() throws Exception {
        byte[] encoded = InetAddressPoint.encode(InetAddress.getByName("10.0.0.255"));
        assertEquals("10.0.0.255", SourceValueFormatters.forType("ip").apply(encoded));

        byte[] v6 = InetAddressPoint.encode(InetAddress.getByName("2001:db8::1"));
        assertEquals("2001:db8::1", SourceValueFormatters.forType("ip").apply(v6));
    }

    public void testBinaryRendersBase64() {
        byte[] bytes = new byte[] { 0, 1, 2, (byte) 0xff };
        assertEquals(java.util.Base64.getEncoder().encodeToString(bytes), SourceValueFormatters.forType("binary").apply(bytes));
    }

    public void testDateRendersDefaultIsoMillisUtc() {
        // 2020-01-15T00:00:00.000Z == 1579046400000 epoch millis
        long epochMillis = 1579046400000L;
        assertEquals("2020-01-15T00:00:00.000Z", SourceValueFormatters.forType("date").apply(epochMillis));
    }

    public void testDateNanosRendersDefaultIsoMillisUtc() {
        // same instant expressed in epoch nanos
        long epochNanos = 1579046400000L * 1_000_000L;
        assertEquals("2020-01-15T00:00:00.000Z", SourceValueFormatters.forType("date_nanos").apply(epochNanos));
    }

    public void testNumericAndBooleanAreIdentity() {
        assertEquals(42, SourceValueFormatters.forType("integer").apply(42));
        assertEquals(9007199254740993L, SourceValueFormatters.forType("long").apply(9007199254740993L));
        assertEquals(1.5f, SourceValueFormatters.forType("float").apply(1.5f));
        assertEquals(1.5d, SourceValueFormatters.forType("double").apply(1.5d));
        assertEquals(Boolean.TRUE, SourceValueFormatters.forType("boolean").apply(Boolean.TRUE));
        assertEquals((byte) 7, SourceValueFormatters.forType("byte").apply((byte) 7));
        assertEquals((short) 300, SourceValueFormatters.forType("short").apply((short) 300));
    }

    public void testNullInputPassesThrough() {
        assertNull(SourceValueFormatters.forType("keyword").apply(null));
        assertNull(SourceValueFormatters.forType("ip").apply(null));
        assertNull(SourceValueFormatters.forType("binary").apply(null));
        assertNull(SourceValueFormatters.forType("date").apply(null));
        assertNull(SourceValueFormatters.forType("date_nanos").apply(null));
    }

    public void testForFieldDefaultDateMatchesForType() {
        DateFieldMapper.DateFieldType dft = new DateFieldMapper.DateFieldType("d");
        UnaryOperator<Object> byField = SourceValueFormatters.forField(dft);
        long epochMillis = 1579046400000L;
        // With the default date format the by-field renderer must agree with the by-type default.
        assertEquals("2020-01-15T00:00:00.000Z", byField.apply(epochMillis));
    }

    public void testForFieldHonorsCustomDateFormat() {
        DateFieldMapper.DateFieldType dft = new DateFieldMapper.DateFieldType(
            "d",
            org.opensearch.common.time.DateFormatter.forPattern("yyyy-MM-dd")
        );
        long epochMillis = 1579046400000L;
        // A field configured with a date-only format must reconstruct the date-only string (the divergence the
        // shared formatter fixes vs the hardcoded ISO-millis default).
        assertEquals("2020-01-15", SourceValueFormatters.forField(dft).apply(epochMillis));
    }

    public void testForFieldNullReturnsIdentity() {
        assertEquals(42, SourceValueFormatters.forField(null).apply(42));
    }
}

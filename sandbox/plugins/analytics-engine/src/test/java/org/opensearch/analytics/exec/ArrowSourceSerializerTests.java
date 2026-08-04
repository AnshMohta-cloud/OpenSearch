/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.exec;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.writer.BaseWriter.StructWriter;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.opensearch.test.OpenSearchTestCase;

import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Unit coverage for {@link ArrowSourceSerializer} over hand-built Arrow vectors — proving the mapping-driven
 * type fidelity (ip/binary/float/date), the nested recursion, element order + duplicates, and null/empty-array
 * omission, all without a cluster. The {@link ArrowSourceSerializer.TypeResolver} here returns the same
 * per-type renderers as the shared {@code SourceValueFormatters} (not importable on this module's test classpath),
 * so these assertions pin the exact {@code _source} forms the codec path also produces.
 */
public class ArrowSourceSerializerTests extends OpenSearchTestCase {

    // Mirrors SourceValueFormatters default date rendering.
    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.ROOT);

    /** A TypeResolver mapping a fixed set of dotted paths to the same renderers SourceValueFormatters uses. */
    private static ArrowSourceSerializer.TypeResolver resolver(Map<String, String> pathToType) {
        return dottedPath -> {
            String type = pathToType.get(dottedPath);
            if (type == null) {
                return null;
            }
            switch (type) {
                case "keyword":
                case "text":
                    return v -> v == null ? null : new String((byte[]) v, StandardCharsets.UTF_8);
                case "ip":
                    return v -> {
                        if (v == null) return null;
                        InetAddress a = org.apache.lucene.document.InetAddressPoint.decode((byte[]) v);
                        return org.opensearch.common.network.InetAddresses.toAddrString(a);
                    };
                case "binary":
                    return v -> v == null ? null : java.util.Base64.getEncoder().encodeToString((byte[]) v);
                case "date":
                    return v -> v == null ? null : DATE_FMT.format(Instant.ofEpochMilli(((Number) v).longValue()).atZone(ZoneOffset.UTC));
                default:
                    return UnaryOperator.identity();
            }
        };
    }

    private BufferAllocator allocator;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        allocator = new RootAllocator(Long.MAX_VALUE);
    }

    @Override
    public void tearDown() throws Exception {
        allocator.close();
        super.tearDown();
    }

    /**
     * Builds a nested LIST&lt;STRUCT&gt; like c3's specs and asserts full reconstruction, including
     * two elements under one row (duplicate-preserving, order-preserving).
     */
    public void testNestedListStructKeywordAndInt() {
        try (ListVector list = ListVector.empty("specs", allocator)) {
            UnionListWriter w = list.getWriter();
            StructWriter sw;

            w.setPosition(0);
            w.startList();
            // element 0: {k:"width", v:12}
            sw = w.struct();
            sw.start();
            writeVarChar(sw, "k", "width");
            sw.integer("v").writeInt(12);
            sw.end();
            // element 1: {k:"height", v:20}
            sw = w.struct();
            sw.start();
            writeVarChar(sw, "k", "height");
            sw.integer("v").writeInt(20);
            sw.end();
            w.endList();
            list.setValueCount(1);

            List<Map<String, Object>> out = ArrowSourceSerializer.toNestedSource(
                list,
                0,
                "specs",
                resolver(Map.of("specs.k", "keyword", "specs.v", "integer"))
            );

            assertEquals(2, out.size());
            assertEquals("width", out.get(0).get("k"));
            assertEquals(12, out.get(0).get("v"));
            assertEquals("height", out.get(1).get("k"));
            assertEquals(20, out.get(1).get("v"));
        }
    }

    /** ip → dotted, binary → Base64, keyword → String, date → ISO millis; proves mapping disambiguates VarBinary. */
    public void testTypeFidelityIpBinaryDate() throws Exception {
        byte[] ipEnc = org.apache.lucene.document.InetAddressPoint.encode(InetAddress.getByName("10.0.0.255"));
        byte[] bin = new byte[] { 1, 2, 3 };
        try (ListVector list = ListVector.empty("items", allocator)) {
            UnionListWriter w = list.getWriter();
            w.setPosition(0);
            w.startList();
            StructWriter sw = w.struct();
            sw.start();
            writeVarBinary(sw, "host", ipEnc);   // ip field
            writeVarBinary(sw, "blob", bin);      // binary field
            sw.bigInt("made").writeBigInt(1579046400000L); // date 2020-01-15
            sw.end();
            w.endList();
            list.setValueCount(1);

            List<Map<String, Object>> out = ArrowSourceSerializer.toNestedSource(
                list,
                0,
                "items",
                resolver(Map.of("items.host", "ip", "items.blob", "binary", "items.made", "date"))
            );

            assertEquals(1, out.size());
            assertEquals("10.0.0.255", out.get(0).get("host"));
            assertEquals(java.util.Base64.getEncoder().encodeToString(bin), out.get(0).get("blob"));
            assertEquals("2020-01-15T00:00:00.000Z", out.get(0).get("made"));
        }
    }

    /** float stays single-precision (not widened to double), matching vanilla NumberFieldMapper.FLOAT. */
    public void testFloatKeepsSinglePrecision() {
        try (ListVector list = ListVector.empty("items", allocator)) {
            UnionListWriter w = list.getWriter();
            w.setPosition(0);
            w.startList();
            StructWriter sw = w.struct();
            sw.start();
            sw.float4("weight").writeFloat4(1.6f);
            sw.end();
            w.endList();
            list.setValueCount(1);

            List<Map<String, Object>> out = ArrowSourceSerializer.toNestedSource(
                list,
                0,
                "items",
                resolver(Map.of("items.weight", "float"))
            );
            Object weight = out.get(0).get("weight");
            assertTrue("expected Float, got " + weight.getClass(), weight instanceof Float);
            assertEquals(1.6f, (Float) weight, 0.0f);
        }
    }

    /** A null row and an empty list both reconstruct as an empty element list (no crash, no phantom element). */
    public void testNullAndEmptyRowsYieldEmpty() {
        try (ListVector list = ListVector.empty("specs", allocator)) {
            UnionListWriter w = list.getWriter();
            // row 0: empty list
            w.setPosition(0);
            w.startList();
            w.endList();
            // row 1 left null (never written)
            list.setValueCount(2);

            ArrowSourceSerializer.TypeResolver r = resolver(Map.of("specs.k", "keyword"));
            assertTrue(ArrowSourceSerializer.toNestedSource(list, 0, "specs", r).isEmpty());
            assertTrue(ArrowSourceSerializer.toNestedSource(list, 1, "specs", r).isEmpty());
        }
    }

    /** An interior null leaf omits its key (matches the codec's "value present iff def==maxDef"). */
    public void testInteriorNullLeafOmitsKey() {
        try (ListVector list = ListVector.empty("specs", allocator)) {
            UnionListWriter w = list.getWriter();
            w.setPosition(0);
            w.startList();
            StructWriter sw = w.struct();
            sw.start();
            writeVarChar(sw, "k", "width");
            // v deliberately not written → null
            sw.end();
            w.endList();
            list.setValueCount(1);

            List<Map<String, Object>> out = ArrowSourceSerializer.toNestedSource(
                list,
                0,
                "specs",
                resolver(Map.of("specs.k", "keyword", "specs.v", "integer"))
            );
            assertEquals(1, out.size());
            assertEquals("width", out.get(0).get("k"));
            assertFalse("null leaf must be omitted", out.get(0).containsKey("v"));
        }
    }

    // ── fixture helpers ──

    private static void writeVarChar(StructWriter sw, String name, String value) {
        sw.varChar(name).writeVarChar(new org.apache.arrow.vector.util.Text(value));
    }

    private static void writeVarBinary(StructWriter sw, String name, byte[] value) {
        sw.varBinary(name).writeVarBinary(value, 0, value.length);
    }
}

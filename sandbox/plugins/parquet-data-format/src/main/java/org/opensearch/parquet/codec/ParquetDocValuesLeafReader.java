/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.codec;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;
import org.opensearch.common.lucene.search.Queries;
import org.opensearch.index.cache.bitset.BitsetFilterCache;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.DerivedSourceNestedNavigator;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.parquet.codec.cache.ColumnPageIndex;
import org.opensearch.parquet.codec.iter.ParquetDictionarySortedDocValues;
import org.opensearch.parquet.codec.iter.ParquetSortedDocValues;
import org.opensearch.parquet.codec.iter.ParquetUninvertedSortedDocValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A {@link FilterLeafReader} that serves doc values for Parquet-resident fields from a
 * {@link ParquetDocValuesProducer}, while delegating everything else to the underlying Lucene
 * leaf reader.
 *
 * <p>This is the read-time integration of the Parquet DocValues codec for the case where a field
 * is <b>Parquet-only</b> — i.e. it has no {@link FieldInfo} in the Lucene segment at all (the
 * composite engine's Lucene secondary writes only text/keyword inverted indexes plus the row-id
 * doc values; numeric fields like {@code age} live solely in Parquet). Lucene's
 * {@code PerFieldDocValuesFormat} cannot route to such a field because there is no segment
 * {@code FieldInfo} to carry the format name. This reader closes that gap by:
 *
 * <ol>
 *   <li><b>Synthesizing {@link FieldInfo}s</b> — for every mapped field that the Parquet codec
 *       supports and that is absent (or DV-less) in the delegate's {@link FieldInfos}, a synthetic
 *       {@code FieldInfo} with the appropriate {@link DocValuesType} (from {@link FieldTypeMapping})
 *       is added so OpenSearch's value-source layer believes the doc values exist and asks for
 *       them.</li>
 *   <li><b>Overriding the five DV accessors</b> — for those synthetic fields the iterators come
 *       from a per-segment {@link ParquetDocValuesProducer}; all other fields delegate to the
 *       underlying reader unchanged.</li>
 * </ol>
 *
 * <p>One producer is built lazily per segment and closed when this reader closes. Not shared
 * across segments.
 *
 * <p>Extends {@link SequentialStoredFieldsLeafReader} (rather than plain {@link FilterLeafReader})
 * so the fetch phase can retrieve {@code _source}/stored fields on {@code size > 0} queries. On a
 * Parquet-primary index derived source is force-enabled, so the reader stack is
 * {@code DerivedSourceLeafReader -> ParquetDocValuesLeafReader -> SegmentReader}. When the fetch
 * phase asks the outer {@code DerivedSourceLeafReader} for a sequential stored-fields reader, it
 * unwraps to this reader; a plain {@code FilterLeafReader} is neither a {@code CodecReader} nor a
 * {@code SequentialStoredFieldsLeafReader}, so the unwrap threw and only {@code size:0} worked.
 * As a {@code SequentialStoredFieldsLeafReader} this reader is transparent to that unwrap: it
 * passes the underlying segment's stored-fields reader straight through, and the derived-source
 * layer still synthesizes {@code _source} from doc values on top of it.
 */
public final class ParquetDocValuesLeafReader extends SequentialStoredFieldsLeafReader implements DerivedSourceNestedNavigator {

    private static final Logger logger = LogManager.getLogger(ParquetDocValuesLeafReader.class);

    private final MapperService mapperService;

    /**
     * Vanilla's node-level parent/child block-join bitset cache, or {@code null} when none is wired (e.g. a
     * unit-test reader). Reused for nested source assembly so we consume the SAME cached bitsets
     * {@code ToParentBlockJoinQuery} and inner_hits use, instead of building our own. See {@link #bits}.
     */
    private final BitsetFilterCache bitsetFilterCache;

    /** Lazily constructed Parquet producer for this segment; null until first DV access. */
    private ParquetDocValuesProducer producer;
    private boolean producerInitialized;

    /** Synthetic + real merged field infos, computed once. */
    private final FieldInfos mergedFieldInfos;

    /** Field name -> synthetic FieldInfo for Parquet-resident DV fields served by this reader. */
    private final Map<String, FieldInfo> parquetFields;

    /** Nested-leaf page→first-child-docId tilings ({@code firstDocIdOf}), built once per field. See {@link #nestedFirstDocIdOf}. */
    private final Map<String, int[]> nestedFirstDocIdOfByField = new HashMap<>();

    /** The segment read state used to build the producer (captured at construction). */
    private final SegmentReadState segmentReadState;

    /** Per-query stats accumulator shared across all leaves of one search; may be null in tests. */

    private ParquetDocValuesLeafReader(
        LeafReader in,
        MapperService mapperService,
        BitsetFilterCache bitsetFilterCache,
        SegmentReadState segmentReadState,
        Map<String, FieldInfo> parquetFields,
        FieldInfos mergedFieldInfos
    ) {
        super(in);
        this.mapperService = mapperService;
        this.bitsetFilterCache = bitsetFilterCache;
        this.segmentReadState = segmentReadState;
        this.parquetFields = parquetFields;
        this.mergedFieldInfos = mergedFieldInfos;
    }

    /**
     * Builds a {@link ParquetDocValuesLeafReader} for {@code in} if a Parquet file resolves for the
     * segment and the mapping declares at least one Parquet-codec-supported field that is missing
     * doc values in the Lucene segment. Otherwise returns {@code in} unwrapped.
     */
    public static LeafReader wrapIfApplicable(LeafReader in, MapperService mapperService, BitsetFilterCache bitsetFilterCache)
        throws IOException {
        SegmentReader segmentReader;
        try {
            segmentReader = Lucene.segmentReader(in);
        } catch (RuntimeException e) {
            // Not a segment-backed leaf (e.g. an in-memory test reader) — nothing to wrap.
            return in;
        }

        SegmentReadState state = new SegmentReadState(
            segmentReader.directory(),
            segmentReader.getSegmentInfo().info,
            segmentReader.getFieldInfos(),
            IOContext.DEFAULT
        );

        // Only proceed if a Parquet file exists for this segment.
        if (ParquetSegmentLayout.resolve(state) == null) {
            return in;
        }

        FieldInfos existing = in.getFieldInfos();
        Map<String, FieldInfo> parquetFields = new LinkedHashMap<>();
        List<FieldInfo> merged = new ArrayList<>();
        int maxNumber = -1;
        for (FieldInfo fi : existing) {
            merged.add(fi);
            maxNumber = Math.max(maxNumber, fi.number);
        }

        // Walk the mapping. For each field the Parquet codec supports whose doc values are NOT
        // already present in the Lucene segment, synthesize a FieldInfo with the mapped DV type.
        for (MappedFieldType mft : mapperService.fieldTypes()) {
            String name = mft.name();
            if (mapperService.isMetadataField(name)) {
                continue;
            }
            if (FieldTypeMapping.isSupported(mft.typeName()) == false) {
                continue;
            }
            FieldInfo realFi = existing.fieldInfo(name);
            if (realFi != null && realFi.getDocValuesType() != DocValuesType.NONE) {
                // Lucene already serves doc values for this field — leave it to the native reader.
                continue;
            }
            FieldTypeMapping.Mapping mapping = FieldTypeMapping.forType(mft.typeName());
            DocValuesType dvType = mapping.singleValued();
            FieldInfo synthetic = newDocValuesFieldInfo(name, ++maxNumber, dvType, skipIndexTypeFor(mapping));
            parquetFields.put(name, synthetic);
            // If a DV-less FieldInfo already exists for this field, replace it with the synthetic
            // one carrying the DV type; otherwise append.
            if (realFi != null) {
                merged.removeIf(fi -> fi.name.equals(name));
            }
            merged.add(synthetic);
        }

        if (parquetFields.isEmpty()) {
            // Nothing for us to serve — don't wrap.
            return in;
        }

        FieldInfos mergedInfos = new FieldInfos(merged.toArray(new FieldInfo[0]));
        return new ParquetDocValuesLeafReader(in, mapperService, bitsetFilterCache, state, parquetFields, mergedInfos);
    }

    /**
     * Skip-index declaration for a synthetic field: RANGE for integer-shaped columns whose
     * Parquet ColumnIndex min/max the producer can serve through a {@link DocValuesSkipper}
     * (raw-bits order == numeric order), NONE otherwise. Must stay in sync with
     * {@link ParquetDocValuesProducer#getSkipper}'s physical-type gate: declaring RANGE for a
     * field whose getSkipper returns null would break consumers that trust the declaration.
     */
    private static DocValuesSkipIndexType skipIndexTypeFor(FieldTypeMapping.Mapping mapping) {
        ParquetPhysicalType phys = mapping.physical();
        boolean skippable = phys == ParquetPhysicalType.INT32 || phys == ParquetPhysicalType.INT64 || phys == ParquetPhysicalType.BOOL;
        return skippable ? DocValuesSkipIndexType.RANGE : DocValuesSkipIndexType.NONE;
    }

    /** Builds a synthetic doc-values {@link FieldInfo} carrying the given DV type. */
    private static FieldInfo newDocValuesFieldInfo(String name, int number, DocValuesType dvType, DocValuesSkipIndexType skipType) {
        return new FieldInfo(
            name,
            number,
            false,                       // storeTermVector
            true,                        // omitNorms
            false,                       // storePayloads
            IndexOptions.NONE,           // not indexed via this reader
            dvType,
            skipType,
            -1,                          // dvGen
            new HashMap<>(),             // attributes (mutable, per FieldInfo contract)
            0,                           // pointDimensionCount
            0,                           // pointIndexDimensionCount
            0,                           // pointNumBytes
            0,                           // vectorDimension
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,                       // softDeletes
            false                        // isParentField
        );
    }

    private synchronized ParquetDocValuesProducer producer() throws IOException {
        if (producerInitialized == false) {
            producer = new ParquetDocValuesProducer(segmentReadState, mapperService);
            producerInitialized = true;
        }
        if (producer != null && producer.isClosed()) {
            // This wrapper outlived its request: a cache (fielddata, global ordinals) retained it
            // and is calling back after the search closed the request producer. Serve through the
            // segment-lifetime shared producer so cached consumers stay valid until the segment
            // itself closes — the contract every reader-keyed cache in Lucene/OpenSearch assumes.
            ParquetDocValuesProducer shared = SharedProducerRegistry.get(in.getCoreCacheHelper(), segmentReadState, mapperService);
            if (shared == null) {
                throw new IllegalStateException("doc values requested after the search closed and the segment has no core cache identity");
            }
            return shared;
        }
        return producer;
    }

    /** Whether the mapper types this field (or subfield) as keyword — values indexed verbatim. */
    private boolean isKeywordField(String field) {
        org.opensearch.index.mapper.MappedFieldType fieldType = mapperService.fieldType(field);
        return fieldType != null && "keyword".equals(fieldType.typeName());
    }

    /** Returns the synthetic FieldInfo if the given field is served from Parquet, else null. */
    private FieldInfo parquetFieldInfo(String field) {
        return parquetFields.get(field);
    }

    /**
     * Builds a {@link RowIdResolver} that translates this segment's {@code docId}s to Parquet row
     * positions by reading the underlying leaf's {@code __row_id__} doc values. Each codec iterator
     * needs its own resolver (its own {@code __row_id__} iterator), so this is called per DV accessor.
     * Falls back to identity when the segment has no {@code __row_id__} field.
     */
    private RowIdResolver newRowIdResolver() throws IOException {
        // Nested segment: docId != Parquet row in general. Only ROOT docs carry __row_id__ (children carry
        // only _nested_path — see LuceneWriter), and roots sit interleaved among their children, so the
        // identity shortcut below is unsafe. Flat/root fields are still read at ROOT docIds during _source
        // reconstruction (the hit is always a root), so translate each docId through its __row_id__ value.
        if (mapperService.documentMapper() != null && mapperService.documentMapper().hasNestedObjects()) {
            return rowIdResolverForNested();
        }
        // Flat segment: the write path GUARANTEES rowId == docId in every finished segment: row ids are
        // rewritten to sequential 0..maxDoc-1 after any sort/merge (SequentialRowIdProducer) and verified by
        // LuceneWriter.assertRowIdsSequential. So the per-doc __row_id__ lookup is pure waste here — it
        // re-reads a value that always equals the docId. Skip it: use the no-op IDENTITY resolver.
        //
        // Backed by an -ea assert that mirrors the writer's invariant; if a future write path ever
        // produced a non-identity segment, this trips in dev/test. (Costs nothing in prod.)
        assert assertRowIdsAreIdentity() : "non-identity __row_id__ segment reached read path; IDENTITY shortcut is unsafe here";
        return RowIdResolver.IDENTITY;
    }

    /** -ea-only check mirroring {@code LuceneWriter.assertRowIdsSequential}: every doc's __row_id__ == docId. */
    private boolean assertRowIdsAreIdentity() throws IOException {
        SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        if (rowId == null) {
            return true; // no row-id field => identity by definition
        }
        for (int docId = 0; docId < maxDoc(); docId++) {
            if (rowId.advanceExact(docId) == false) {
                return false;
            }
            if (rowId.nextValue() != docId) {
                return false;
            }
        }
        return true;
    }

    // ── Nested-leaf support ──
    // A field whose enclosing object is `nested` is stored as a repeated (LIST) Parquet leaf: one value per
    // nested child, addressed by (row, element-offset), not by docId. For such fields we build a
    // NestedElementResolver from two block-join bitsets — the ROOT (parent) bitset and this path's child
    // bitset — and route the DV getters to the producer's nested iterators. Both bitsets come from vanilla's
    // node-level BitsetFilterCache (the same instances ToParentBlockJoinQuery and inner_hits use), so nothing
    // is scanned or duplicated here; see bits(Query).

    /**
     * The Parquet PHYSICAL leaf path for an OpenSearch nested field name. Parquet materializes a nested
     * object as a {@code LIST<STRUCT>}, so the physical schema inserts the synthetic group levels
     * {@code list.element} at every nested-object boundary. Example (depth 2):
     * {@code orders.items.price} → {@code orders.list.element.items.list.element.price}.
     *
     * <p>Depth-independent: walks the dotted segments left-to-right, and after each prefix that is itself a
     * nested object appends {@code list.element} before the next segment. The native reader matches on this
     * physical {@code ColumnDescriptor.path()} string; the flat dotted name does not exist for nested leaves.
     */
    private String parquetPhysicalPath(String field) {
        String[] parts = field.split("\\.");
        StringBuilder physical = new StringBuilder(parts[0]);
        StringBuilder prefix = new StringBuilder(parts[0]);
        for (int i = 1; i < parts.length; i++) {
            ObjectMapper om = mapperService.getObjectMapper(prefix.toString());
            if (om != null && om.nested().isNested()) {
                physical.append(".list.element");
            }
            physical.append('.').append(parts[i]);
            prefix.append('.').append(parts[i]);
        }
        return physical.toString();
    }

    /**
     * The deepest {@code nested} {@link ObjectMapper} enclosing {@code field}, or {@code null} if the field
     * is not under any nested object. Depth-independent: walks the dotted path and picks the longest nested
     * prefix, so it works for {@code a.b} and {@code a.b.c.d.e} identically.
     */
    private ObjectMapper nestedParentOf(String field) {
        int dot = field.lastIndexOf('.');
        while (dot > 0) {
            String prefix = field.substring(0, dot);
            ObjectMapper om = mapperService.getObjectMapper(prefix);
            if (om != null && om.nested().isNested()) {
                return om;
            }
            dot = field.lastIndexOf('.', dot - 1);
        }
        return null;
    }

    /**
     * The per-segment block-join bitset for {@code filter}, taken from vanilla's node-level
     * {@link BitsetFilterCache} when one is wired — the SAME cached instance {@code ToParentBlockJoinQuery}
     * and inner_hits path resolution use, so no bitset is built or duplicated here. This wrapper's core cache
     * key delegates to the underlying segment ({@link #getCoreCacheHelper()}), so the lookup resolves to that
     * shared entry. Falls back to an uncached one-off build only when no cache is available (e.g. a unit-test
     * reader with no index cache). Never {@code null}: a filter matching nothing yields an empty set.
     */
    private BitSet bits(Query filter) throws IOException {
        boolean reuseCache = bitsetFilterCache != null;
        BitSet bitSet = reuseCache
            ? bitsetFilterCache.getBitSetProducer(filter).getBitSet(getContext())
            : BitsetFilterCache.bitsetFromQuery(filter, getContext());
        if (bitSet == null) {
            bitSet = new FixedBitSet(maxDoc());
        }
        // [DSL-TRACE] Reuse proof: with reuseCache=true this BitSet is served from vanilla's node-level
        // BitsetFilterCache. Compare bitsetId here against the "parentBitset ... bitsetId=" line
        // IndicesBitsetFilterCache logs when it BUILDS a bitset (query phase / inner_hits) — a matching id
        // proves the SAME cached instance is reused, not rebuilt. coreKey ties it to this segment.
        Object coreKey = in.getCoreCacheHelper() != null ? in.getCoreCacheHelper().getKey() : null;
        logger.info(
            "[DSL-TRACE] ParquetLeaf.bits: reuseCache={} filter={} maxDoc={} cardinality={} bitsetId={} coreKey={}",
            reuseCache,
            filter,
            maxDoc(),
            bitSet.cardinality(),
            System.identityHashCode(bitSet),
            System.identityHashCode(coreKey)
        );
        return bitSet;
    }

    // ── DerivedSourceNestedNavigator ──
    // Exposes the block-join bitsets so the mapper layer can reconstruct nested arrays for derived _source
    // without depending on this plugin. Both come from vanilla's cached BitsetFilterCache (see bits) — the
    // same instances ToParentBlockJoinQuery and inner_hits path resolution use; no new bitset is built here.

    @Override
    public BitSet parentDocs() throws IOException {
        // Roots carry _primary_term (written on roots only); newNonNestedFilter() == FieldExistsQuery(_primary_term).
        logger.info("[DSL-TRACE] ParquetLeaf.parentDocs() invoked (derived-source ROOT/parent bitset via vanilla cache)");
        return bits(Queries.newNonNestedFilter());
    }

    @Override
    public BitSet childDocs(Query nestedTypeFilter) throws IOException {
        logger.info(
            "[DSL-TRACE] ParquetLeaf.childDocs() invoked filter={} (derived-source child bitset via vanilla cache)",
            nestedTypeFilter
        );
        return bits(nestedTypeFilter);
    }

    /**
     * Builds a {@link NestedElementResolver} for a leaf under nested object {@code nestedParent}. Composes the
     * existing row resolver (child's enclosing root -> {@code __row_id__}) with the ROOT (parent) bitset and
     * this nested path's child bitset — both from vanilla's cached {@link BitsetFilterCache}.
     */
    private NestedElementResolver newNestedResolver(ObjectMapper nestedParent) throws IOException {
        logger.info(
            "[DSL-TRACE] ParquetLeaf.newNestedResolver: nestedParent={} childFilter={} (building nested DV resolver from cached bitsets)",
            nestedParent.name(),
            nestedParent.nestedTypeFilter()
        );
        return new NestedElementResolver(
            rowIdResolverForNested(),
            bits(Queries.newNonNestedFilter()),
            bits(nestedParent.nestedTypeFilter())
        );
    }

    /**
     * Row resolver used INSIDE the nested resolver to map a ROOT docId to its Parquet row. Unlike the flat
     * path, a nested segment is not docId==row overall, but roots still carry {@code __row_id__}; read it.
     */
    private RowIdResolver rowIdResolverForNested() throws IOException {
        SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        if (rowId == null) {
            return RowIdResolver.IDENTITY;
        }
        // Forward-only cursor over __row_id__; callers (the resolver) pass non-decreasing ROOT docIds.
        return docId -> {
            if (rowId.advanceExact(docId) == false) {
                throw new IllegalStateException("root docId " + docId + " has no __row_id__ value");
            }
            return rowId.nextValue();
        };
    }

    /**
     * The nested block-join docId tiling for {@code field}'s repeated Parquet column: {@code firstDocIdOf[p]}
     * is the child-docId at which page {@code p}'s first row's block begins, so page {@code p} owns doc range
     * {@code [firstDocIdOf[p], firstDocIdOf[p+1]-1]}. Length {@code pageCount + 1}; {@code [0] == 0} and
     * {@code [pageCount] == maxDoc}, strictly ascending. Cached per field (built once per segment).
     *
     * <p>Built by walking the ROOT (parent) block-join bitset once, in ascending root/row order: children
     * precede their root (root last), so a page that starts at row {@code r > 0} begins one doc past the ROOT
     * of row {@code r-1}, i.e. {@code firstDocIdOf[p] = ROOT_{firstRowOf(p)-1} + 1}. O(totalRows), reusing
     * vanilla's cached bitset (no new scan). The {@code __row_id__ == root-index} invariant is asserted under -ea.
     */
    private synchronized int[] nestedFirstDocIdOf(String field) throws IOException {
        int[] cached = nestedFirstDocIdOfByField.get(field);
        if (cached != null) {
            return cached;
        }
        ColumnPageIndex pageIndex = producer().repeatedPageIndex(parquetFieldInfo(field), parquetPhysicalPath(field));
        int pageCount = pageIndex.pageCount();
        int[] firstDocIdOf = new int[pageCount + 1];
        firstDocIdOf[0] = 0;                 // page 0 begins at row 0, whose block starts at docId 0
        firstDocIdOf[pageCount] = maxDoc();  // sentinel: one past the last doc (root of the last row)
        if (pageCount > 1) {
            BitSet rootBits = bits(Queries.newNonNestedFilter());
            DocIdSetIterator roots = new BitSetIterator(rootBits, rootBits.cardinality());
            int rootIndex = 0;
            int rootDoc = roots.nextDoc(); // ROOT of row 0 (root index 0)
            for (int p = 1; p < pageCount; p++) {
                long targetRow = pageIndex.firstRowOf(p) - 1; // page p begins one doc past this row's ROOT
                while (rootIndex < targetRow) {
                    rootDoc = roots.nextDoc();
                    rootIndex++;
                }
                assert rootDoc != DocIdSetIterator.NO_MORE_DOCS : "ran out of ROOT docs before row " + targetRow;
                firstDocIdOf[p] = rootDoc + 1;
                assert rowIdMatches(rootDoc, targetRow) : "ROOT docId " + rootDoc + " does not carry __row_id__ " + targetRow;
            }
        }
        nestedFirstDocIdOfByField.put(field, firstDocIdOf);
        return firstDocIdOf;
    }

    /** -ea check mirroring the write-path invariant: the ROOT at {@code rootDoc} carries {@code __row_id__ == expectedRow}. */
    private boolean rowIdMatches(int rootDoc, long expectedRow) throws IOException {
        SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        if (rowId == null) {
            return true; // no row-id field => identity by definition
        }
        return rowId.advanceExact(rootDoc) && rowId.nextValue() == expectedRow;
    }

    @Override
    public FieldInfos getFieldInfos() {
        return mergedFieldInfos;
    }

    @Override
    public NumericDocValues getNumericDocValues(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null && fi.getDocValuesType() == DocValuesType.NUMERIC) {
            RowIdResolver resolver = newRowIdResolver();
            NumericDocValues numeric = producer().getNumeric(fi);
            // IDENTITY (the guaranteed case — see newRowIdResolver) means docId == Parquet row, so the
            // remap wrapper is pure indirection: return the delegate, already in docId space, directly.
            return resolver == RowIdResolver.IDENTITY ? numeric : RowIdRemappingDocValues.numeric(numeric, resolver, maxDoc());
        }
        return in.getNumericDocValues(field);
    }

    @Override
    public SortedNumericDocValues getSortedNumericDocValues(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null) {
            // OpenSearch numeric value sources request SORTED_NUMERIC even for single-valued fields,
            // then call DocValues.unwrapSingleton(...) to take a leaner single-valued collector when
            // possible. We therefore serve single-valued numerics through the CACHED single-valued
            // iterator (producer().getNumeric → ParquetNumericDocValues → PageCache hot path), apply
            // the docId→row remapping at the numeric level, and wrap the result with
            // DocValues.singleton(...) so the returned value is a real SingletonSortedNumericDocValues
            // that unwrapSingleton(...) can detect. This wins on two layers: the PageCache (no per-doc
            // FFM call) and the aggregator's single-valued fast path.
            //
            // TODO(multi-value): this intentionally treats every numeric field as single-valued and so
            // breaks true multi-valued (array) numeric fields. Restore the repeated path for genuinely
            // multi-valued columns (e.g. branch on the Parquet column's repetition level) and return
            // RowIdRemappingDocValues.sortedNumeric(producer().getSortedNumeric(asSortedNumeric),
            // newRowIdResolver(), maxDoc()) for those.
            FieldInfo asNumeric = fi.getDocValuesType() == DocValuesType.NUMERIC
                ? fi
                : newDocValuesFieldInfo(field, fi.number, DocValuesType.NUMERIC, fi.docValuesSkipIndexType());
            ObjectMapper nestedParent = nestedParentOf(field);
            if (nestedParent != null) {
                // [SKIPPER-VERIFY] the Parquet DV value path for this nested numeric IS being built (once per
                // segment per query). Its presence — paired with the skipper's advance() log — distinguishes
                // "range query ran through Parquet doc-values" from "range query used the BKD/points path".
                logger.info(
                    "[SKIPPER-VERIFY] getSortedNumericDocValues NESTED field={} skipIndexType={}",
                    field,
                    fi.docValuesSkipIndexType()
                );
                // Nested leaf: read the child element at (row, offset) from the repeated Parquet column.
                // Already single-valued per child, so wrap with DocValues.singleton for unwrapSingleton.
                NumericDocValues nested = producer().getNestedNumeric(
                    asNumeric,
                    parquetPhysicalPath(field),
                    newNestedResolver(nestedParent)
                );
                return DocValues.singleton(nested);
            }
            NumericDocValues numeric = producer().getNumeric(asNumeric);
            RowIdResolver resolver = newRowIdResolver();
            NumericDocValues remapped = resolver == RowIdResolver.IDENTITY
                ? numeric
                : RowIdRemappingDocValues.numeric(numeric, resolver, maxDoc());
            return DocValues.singleton(remapped);
        }
        return in.getSortedNumericDocValues(field);
    }

    @Override
    public BinaryDocValues getBinaryDocValues(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null && fi.getDocValuesType() == DocValuesType.BINARY) {
            ObjectMapper nestedParent = nestedParentOf(field);
            if (nestedParent != null) {
                // Nested binary leaf: read the child's single element (raw bytes) at (row, offset) from the
                // repeated Parquet column — O(1), zero-copy, no ordinals (mirrors flat ParquetBinaryDocValues).
                return producer().getNestedBinary(fi, parquetPhysicalPath(field), newNestedResolver(nestedParent));
            }
            RowIdResolver resolver = newRowIdResolver();
            BinaryDocValues binary = producer().getBinary(fi);
            return resolver == RowIdResolver.IDENTITY ? binary : RowIdRemappingDocValues.binary(binary, resolver, maxDoc());
        }
        return in.getBinaryDocValues(field);
    }

    @Override
    public SortedDocValues getSortedDocValues(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null && fi.getDocValuesType() == DocValuesType.SORTED) {
            RowIdResolver resolver = newRowIdResolver();
            SortedDocValues sorted = withDictionaryOrdinals(field, producer().getSorted(fi));
            return resolver == RowIdResolver.IDENTITY ? sorted : RowIdRemappingDocValues.sorted(sorted, resolver, maxDoc());
        }
        return in.getSortedDocValues(field);
    }

    /**
     * Upgrades a streaming sorted iterator to fully contract-compliant segment ordinals when the
     * field's cardinality fits the dictionary budget. The sorted term dictionary is read from
     * the composite index's Lucene sidecar (O(distinct), cached per segment) — never from a row
     * scan. Above-budget fields keep the streaming iterator, whose global-ordinal operations
     * fail fast rather than materialize.
     */
    private SortedDocValues withDictionaryOrdinals(String field, SortedDocValues sorted) throws IOException {
        // Ordinal tiers rank Parquet VALUES against the Lucene sidecar's TERMS, which only
        // coincide for untokenized (keyword) fields. A text field's terms are analyzer tokens:
        // ranking values against tokens would produce silently wrong ordinals. Text fields stay
        // on the streaming iterator, whose global operations fail fast toward execution_hint:map.
        if (isKeywordField(field) == false) {
            return sorted;
        }
        if (sorted instanceof ParquetSortedDocValues streaming) {
            TermDictionary dictionary = TermDictionaryCache.get(
                in,
                field,
                ParquetDocValuesProducer.dictionaryMaxTerms(),
                ParquetDocValuesProducer.dictionaryCacheBytes()
            );
            if (dictionary != null) {
                return new ParquetDictionarySortedDocValues(streaming, dictionary);
            }
            // Above the dictionary budget: disk-backed uninverted ordinals (built once per
            // segment from the sidecar's postings, memory-mapped, working-set resident).
            long expectedNonNull = producer().nonNullRowCount(parquetFieldInfo(field));
            UninvertedOrdinals uninverted = UninvertedOrdinalsCache.get(in, segmentReadState.segmentInfo, field, expectedNonNull);
            if (uninverted != null) {
                return new ParquetUninvertedSortedDocValues(uninverted, streaming, maxDoc());
            }
        }
        return sorted;
    }

    @Override
    public SortedSetDocValues getSortedSetDocValues(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null) {
            // Mirror getSortedNumericDocValues: keyword value sources request SORTED_SET even for
            // single-valued fields, then call DocValues.unwrapSingleton(...). Serve single-valued
            // keywords through the single-valued ordinal-table iterator (producer().getSorted →
            // ParquetSortedDocValues), remap docId→row, and wrap with DocValues.singleton(...) so the
            // returned value is a real SingletonSortedSetDocValues that unwrapSingleton(...) detects.
            //
            // TODO(multi-value): intentionally treats every keyword field as single-valued and so breaks
            // true multi-valued (array) keyword fields. Restore the multi-valued path for genuinely
            // repeated columns and return RowIdRemappingDocValues.sortedSet(
            // producer().getSortedSet(asSortedSet), newRowIdResolver(), maxDoc()) for those.
            FieldInfo asSorted = fi.getDocValuesType() == DocValuesType.SORTED
                ? fi
                : newDocValuesFieldInfo(field, fi.number, DocValuesType.SORTED, fi.docValuesSkipIndexType());
            ObjectMapper nestedParent = nestedParentOf(field);
            if (nestedParent != null) {
                // Nested keyword leaf: read the child element at (row, offset) from the repeated Parquet
                // column, as streaming per-doc ordinals (upgraded to dictionary ordinals when in budget).
                SortedDocValues nested = withDictionaryOrdinals(
                    field,
                    producer().getNestedSorted(asSorted, parquetPhysicalPath(field), newNestedResolver(nestedParent))
                );
                return DocValues.singleton(nested);
            }
            SortedDocValues sorted = withDictionaryOrdinals(field, producer().getSorted(asSorted));
            RowIdResolver resolver = newRowIdResolver();
            SortedDocValues remapped = resolver == RowIdResolver.IDENTITY
                ? sorted
                : RowIdRemappingDocValues.sorted(sorted, resolver, maxDoc());
            return DocValues.singleton(remapped);
        }
        return in.getSortedSetDocValues(field);
    }

    @Override
    public DocValuesSkipper getDocValuesSkipper(String field) throws IOException {
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null) {
            boolean nested = nestedParentOf(field) != null;
            if (fi.docValuesSkipIndexType() == DocValuesSkipIndexType.NONE) {
                // [SKIPPER-VERIFY] Lucene asked for a skipper but this field carries no skip index -> none served.
                logger.info("[SKIPPER-VERIFY] getDocValuesSkipper field={} nested={} skipIndexType=NONE -> null", field, nested);
                return null;
            }
            // Nested leaf (repeated LIST): docId != row, so map each page's row range to a docId range
            // via the block-join tiling and serve a nested-aware skipper. The native reader now enumerates
            // a repeated column's real per-page min/max (record-aligned page emission in projected_pages),
            // so the per-page stats are trustworthy; getNestedSkipper opens the repeated reader by the
            // physical list.element path and reports no per-page density (forcing an inner per-doc check).
            if (nested) {
                DocValuesSkipper sk = producer().getNestedSkipper(fi, parquetPhysicalPath(field), nestedFirstDocIdOf(field));
                logger.info(
                    "[SKIPPER-VERIFY] getDocValuesSkipper field={} nested=true skipIndexType={} -> {}",
                    field,
                    fi.docValuesSkipIndexType(),
                    sk == null ? "null" : "NESTED skipper"
                );
                return sk;
            }
            // Doc IDs and Parquet rows coincide (IDENTITY resolver — see newRowIdResolver), so
            // page row ranges are directly valid as skipper doc ID intervals.
            DocValuesSkipper sk = producer().getSkipper(fi);
            logger.info(
                "[SKIPPER-VERIFY] getDocValuesSkipper field={} nested=false skipIndexType={} -> {}",
                field,
                fi.docValuesSkipIndexType(),
                sk == null ? "null" : "flat skipper"
            );
            return sk;
        }
        // [SKIPPER-VERIFY] non-parquet field -> delegate to the wrapped Lucene reader.
        DocValuesSkipper sk = in.getDocValuesSkipper(field);
        logger.info("[SKIPPER-VERIFY] getDocValuesSkipper field={} (non-parquet) -> {}", field, sk == null ? "null" : "delegate");
        return sk;
    }

    @Override
    protected void doClose() throws IOException {
        closeParquetResources();
        super.doClose();
    }

    /**
     * Releases resources owned by this wrapper without closing the underlying Lucene leaf.
     *
     * {@link FilterDirectoryReader} closes its wrapped directory, not the synthetic leaf
     * wrappers returned by its {@code SubReaderWrapper}. The request-scoped directory reader
     * therefore calls this method explicitly before closing its non-closing delegate.
     */
    void closeParquetResources() throws IOException {
        IOException first = null;
        try {
            if (producer != null) {
                producer.close();
            }
        } catch (IOException e) {
            first = e;
        }
        if (first != null) {
            throw first;
        }
    }

    /**
     * This reader serves no stored fields itself — it only overlays Parquet doc values. The
     * underlying segment reader holds the real stored fields, so return its sequential reader
     * unchanged. {@link SequentialStoredFieldsLeafReader#getSequentialStoredFieldsReader()} already
     * unwrapped {@code in} (a {@code CodecReader}/segment reader) down to {@code reader}; the
     * derived-source layer above wraps the result to synthesize {@code _source}.
     */
    @Override
    protected StoredFieldsReader doGetSequentialStoredFieldsReader(StoredFieldsReader reader) {
        return reader;
    }

    // Cache helpers must delegate to the underlying reader so query/segment caches stay coherent.
    @Override
    public CacheHelper getCoreCacheHelper() {
        // Full cache identity restored: filter cache, fielddata and global-ordinals caches all key
        // off this. Consumers cached beyond the request remain valid because producer() reroutes
        // post-close access to the segment-lifetime shared producer (SharedProducerRegistry).
        return in.getCoreCacheHelper();
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return in.getReaderCacheHelper();
    }
}

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
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
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
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.lucene.index.NestedSourceProvider;
import org.opensearch.parquet.bridge.ParquetColumnReader;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.parquet.codec.cache.QueryParquetStats;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
 */
public final class ParquetDocValuesLeafReader extends FilterLeafReader implements NestedSourceProvider {

    // [NESTED-TRACE] Verbose step-by-step tracing of the nested query/fetch pipeline for teaching/debugging.
    // Enable with: PUT /_cluster/settings {"transient":{"logger.org.opensearch.parquet.codec":"TRACE"}}
    private static final Logger TRACE = LogManager.getLogger(ParquetDocValuesLeafReader.class);

    private final MapperService mapperService;

    /**
     * Node-level kill switch for serving flat (top-level) doc-values — and therefore flat derived {@code _source},
     * range, aggregation and sort — from a registered {@link org.opensearch.index.mapper.FlatColumnValueSource}
     * (DataFusion) instead of the FFI decode path. Default off; enable with
     * {@code -Dopensearch.parquet.flat_docvalues.datafusion=true}. Interim full-column materialization; the
     * forward-cursor optimization supersedes it later.
     */
    private static final boolean FLAT_DATAFUSION_DV = Boolean.getBoolean("opensearch.parquet.flat_docvalues.datafusion");

    /** Lazily materialized flat columns (DataFusion path), keyed by field name; per-segment, single-threaded. */
    private final Map<String, org.opensearch.index.mapper.FlatColumnValueSource.LongColumn> flatLongColumns =
        new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<String, org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn> flatBytesColumns =
        new java.util.concurrent.ConcurrentHashMap<>();

    /** Lazily constructed Parquet producer for this segment; null until first DV access. */
    private ParquetDocValuesProducer producer;
    private boolean producerInitialized;

    /** Synthetic + real merged field infos, computed once. */
    private final FieldInfos mergedFieldInfos;

    /** Field name -> synthetic FieldInfo for Parquet-resident DV fields served by this reader. */
    private final Map<String, FieldInfo> parquetFields;

    /** The segment read state used to build the producer (captured at construction). */
    private final SegmentReadState segmentReadState;

    /** Per-query stats accumulator shared across all leaves of one search; may be null in tests. */
    private final QueryParquetStats queryStats;

    /** Cached docId==__row_id__ determination for this segment (null until first computed). A flat
     *  segment is identity; a nested block-join segment is not (child docs shift parent docIds). */
    private Boolean identitySegment;

    /**
     * Nested-child leaf fields served from the primary format's {@code LIST<STRUCT>} column, keyed by the
     * flat mapping field name (e.g. {@code comments.score}). These are NOT in {@link #parquetFields}: their
     * values are addressed by CHILD Lucene docId (block-join hidden docs), not by parent row, so they use a
     * distinct read path ({@link ChildFieldColumn}). Empty on a non-nested / classic index.
     */
    private final Map<String, ChildFieldInfo> childFields;

    /**
     * Cached materialized values per child field name; built lazily on first DV access. OpenSearch searches a
     * given segment on a single thread (leaf partitions are created per whole segment — see
     * {@code ContextIndexSearcher}), so a leaf-reader instance is not accessed concurrently today; the
     * {@link java.util.concurrent.ConcurrentHashMap} is defensive insurance so lazy population stays safe if
     * intra-segment slicing is ever enabled. Builds are pure functions of the segment, so a double-build under
     * a future race would be wasteful but not incorrect.
     */
    private final Map<String, ChildFieldColumn> childFieldColumns = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * The parquet element leaf column path + physical type for a nested-child field, plus its owning nested
     * path ({@code owningPath}, e.g. {@code orgs.divisions.teams.members}) and that path's nesting depth
     * ({@code listLevel}, 1-based: the Dremel list-level at which this field's elements repeat). The owning
     * path selects the per-path child-doc offset map; the list level drives multi-level Dremel expansion.
     */
    private record ChildFieldInfo(
        FieldInfo fieldInfo,
        String leafColumnPath,
        ParquetPhysicalType physical,
        String owningPath,
        int listLevel
    ) {}

    private ParquetDocValuesLeafReader(
        LeafReader in,
        MapperService mapperService,
        SegmentReadState segmentReadState,
        Map<String, FieldInfo> parquetFields,
        Map<String, ChildFieldInfo> childFields,
        FieldInfos mergedFieldInfos,
        QueryParquetStats queryStats
    ) {
        super(in);
        this.mapperService = mapperService;
        this.segmentReadState = segmentReadState;
        this.parquetFields = parquetFields;
        this.childFields = childFields;
        this.mergedFieldInfos = mergedFieldInfos;
        this.queryStats = queryStats;
    }

    /**
     * Builds a {@link ParquetDocValuesLeafReader} for {@code in} if a Parquet file resolves for the
     * segment and the mapping declares at least one Parquet-codec-supported field that is missing
     * doc values in the Lucene segment. Otherwise returns {@code in} unwrapped.
     */
    public static LeafReader wrapIfApplicable(LeafReader in, MapperService mapperService, QueryParquetStats queryStats)
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
        Map<String, ChildFieldInfo> childFields = new LinkedHashMap<>();
        List<FieldInfo> merged = new ArrayList<>();
        int maxNumber = -1;
        for (FieldInfo fi : existing) {
            merged.add(fi);
            maxNumber = Math.max(maxNumber, fi.number);
        }

        // The set of nested-object paths in this mapping (e.g. {"comments"}). Used to classify a leaf
        // field as a nested child and to build its columnar leaf path. Derived from the mapper, not
        // hardcoded, so it generalizes to any nested field(s).
        Set<String> nestedPaths = nestedObjectPaths(mapperService);

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

            // Nested-child leaf? Its values live in the LIST<STRUCT> column and are addressed by CHILD
            // docId, not parent row — route it to childFields with its element leaf column path. Only a
            // DIRECT leaf of a single nested level is handled here (deeper nesting is a follow-up); such a
            // field is left unexposed rather than mis-served.
            String owning = owningNestedPath(name, nestedPaths);
            if (owning != null) {
                // Nested-child leaf at ANY depth. Its Parquet column path inserts ".list.element." at EVERY
                // nested-path boundary between the root and the leaf (the standard 3-level LIST<STRUCT> encoding
                // nests one list per nested level), e.g. orgs.divisions.teams.members.age ->
                // orgs.list.element.divisions.list.element.teams.list.element.members.list.element.age. Building
                // it from the nested-path hierarchy (not a single owning+".list.element."+remainder hop)
                // generalizes to arbitrary nesting depth.
                String leafColumnPath = parquetChildLeafColumnPath(name, nestedPaths);
                // The list level is the number of nested-path boundaries between the root and this field —
                // i.e. how many LIST<STRUCT> levels its elements sit under (owning path's own depth in nested
                // paths). This is the Dremel list-level at which the field's elements repeat.
                int listLevel = nestedDepthOf(owning, nestedPaths);
                childFields.put(name, new ChildFieldInfo(synthetic, leafColumnPath, mapping.physical(), owning, listLevel));
                merged.removeIf(fi -> fi.name.equals(name));
                merged.add(synthetic);
                continue;
            }

            parquetFields.put(name, synthetic);
            // If a DV-less FieldInfo already exists for this field, replace it with the synthetic
            // one carrying the DV type; otherwise append.
            if (realFi != null) {
                merged.removeIf(fi -> fi.name.equals(name));
            }
            merged.add(synthetic);
        }

        if (parquetFields.isEmpty() && childFields.isEmpty()) {
            // Nothing for us to serve — don't wrap.
            return in;
        }

        FieldInfos mergedInfos = new FieldInfos(merged.toArray(new FieldInfo[0]));
        return new ParquetDocValuesLeafReader(in, mapperService, state, parquetFields, childFields, mergedInfos, queryStats);
    }

    /**
     * The set of nested-object field paths declared in the mapping (each a path whose object mapper is
     * {@code nested}). Derived from the mapper — generalizes to any number of nested fields at any path.
     */
    private static Set<String> nestedObjectPaths(MapperService mapperService) {
        Set<String> paths = new java.util.HashSet<>();
        for (org.opensearch.index.mapper.MappedFieldType mft : mapperService.fieldTypes()) {
            String name = mft.name();
            // Walk every dotted prefix of the field name; a prefix that resolves to a nested object mapper
            // is a nested path owning this field. (getObjectMapper returns null for non-object prefixes.)
            for (int dot = name.indexOf('.'); dot >= 0; dot = name.indexOf('.', dot + 1)) {
                String prefix = name.substring(0, dot);
                org.opensearch.index.mapper.ObjectMapper om = mapperService.getObjectMapper(prefix);
                if (om != null && om.nested().isNested()) {
                    paths.add(prefix);
                }
            }
        }
        return paths;
    }

    /**
     * The deepest nested path that strictly contains {@code name} (i.e. {@code name} starts with
     * {@code path + "."}), or {@code null} if none. Mirrors {@code ArrowSchemaBuilder.owningNestedPath}
     * so read-side child-field classification matches the write-side {@code LIST<STRUCT>} layout.
     */
    // package-private for unit testing (pure function of name + nested paths)
    static String owningNestedPath(String name, Set<String> nestedPaths) {
        String best = null;
        for (String path : nestedPaths) {
            if (name.length() > path.length() && name.startsWith(path) && name.charAt(path.length()) == '.') {
                if (best == null || path.length() > best.length()) {
                    best = path;
                }
            }
        }
        return best;
    }

    /**
     * Builds the physical Parquet column path for a nested-child leaf {@code name} at any depth, inserting
     * {@code ".list.element."} at every nested-path boundary between the root and the leaf. Mirrors the
     * write-side {@code ArrowSchemaBuilder.buildNestedListField}, which emits one {@code LIST<STRUCT>} level
     * per nested mapper and stores each field under its owning nested path's {@code element}.
     *
     * <p>Example: {@code orgs.divisions.teams.members.age} with nested paths {@code {orgs, orgs.divisions,
     * orgs.divisions.teams, orgs.divisions.teams.members}} →
     * {@code orgs.list.element.divisions.list.element.teams.list.element.members.list.element.age}.
     *
     * <p>Algorithm: collect every nested-path prefix of {@code name} in increasing length order (the chain of
     * list levels), then emit each successive segment joined by {@code ".list.element."}, with the leaf's own
     * name (the tail after the deepest nested path) appended last.
     */
    // package-private for unit testing (pure function of name + nested paths)
    static String parquetChildLeafColumnPath(String name, Set<String> nestedPaths) {
        // Ordered chain of nested-path prefixes of `name` (e.g. [orgs, orgs.divisions, ...]).
        List<String> chain = new ArrayList<>();
        for (String path : nestedPaths) {
            if (name.length() > path.length() && name.startsWith(path) && name.charAt(path.length()) == '.') {
                chain.add(path);
            }
        }
        chain.sort(java.util.Comparator.comparingInt(String::length));
        StringBuilder col = new StringBuilder();
        int consumed = 0; // characters of `name` already emitted (excluding the trailing dot)
        for (String path : chain) {
            // The segment of this nested level is the part of `path` after the previous level.
            String segment = path.substring(consumed == 0 ? 0 : consumed + 1);
            if (col.length() > 0) {
                col.append(".list.element.");
            }
            col.append(segment);
            consumed = path.length();
        }
        // The leaf field itself lives under the deepest nested path's element.
        String leaf = name.substring(consumed + 1);
        col.append(".list.element.").append(leaf);
        return col.toString();
    }

    /**
     * The nesting depth of a nested path {@code path} — the number of nested-path prefixes it has, inclusive
     * (e.g. {@code orgs} → 1, {@code orgs.divisions.teams.members} → 4). This is the Dremel list-level at which
     * that path's elements repeat.
     */
    // package-private for unit testing (pure function of path + nested paths)
    static int nestedDepthOf(String path, Set<String> nestedPaths) {
        int depth = 0;
        for (String p : nestedPaths) {
            if (path.equals(p) || (path.length() > p.length() && path.startsWith(p) && path.charAt(p.length()) == '.')) {
                depth++;
            }
        }
        return depth;
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
            producer.setQueryStats(queryStats);
            producerInitialized = true;
        }
        return producer;
    }

    /** Returns the synthetic FieldInfo if the given field is served from Parquet, else null. */
    private FieldInfo parquetFieldInfo(String field) {
        return parquetFields.get(field);
    }

    /**
     * Builds a {@link RowIdResolver} that translates this segment's {@code docId}s to Parquet row
     * positions by reading the underlying leaf's {@code __row_id__} doc values. Each codec iterator
     * needs its own resolver (its own {@code __row_id__} iterator), so this is called per DV accessor.
     *
     * <p>For a FLAT (non-nested) segment the write path guarantees {@code rowId == docId} — row ids are
     * rewritten to sequential 0..maxDoc-1 after any sort/merge (SequentialRowIdProducer) — so the
     * {@link RowIdResolver#IDENTITY} no-op is correct and the per-doc {@code __row_id__} lookup is skipped.
     *
     * <p>For a NESTED segment this invariant does NOT hold: the Lucene secondary block-adds N child docs
     * before each parent (children first, parent last), so a parent's {@code docId} is shifted past its
     * children while {@code __row_id__} stays the dense parent row index (parent docId 2 → __row_id__ 0
     * when it has 2 children). {@code docId != __row_id__}, so IDENTITY would read the WRONG Parquet row.
     * We therefore detect the non-identity case and return a doc-values-backed resolver that reads the
     * real {@code __row_id__} per doc. (Previously an {@code -ea} assert here crashed the node on any
     * nested segment reaching the read path.)
     */
    private RowIdResolver newRowIdResolver() throws IOException {
        if (isIdentitySegment()) {
            return RowIdResolver.IDENTITY; // flat segment (or no row-id field): docId == __row_id__, skip the lookup
        }
        // Non-identity (e.g. nested block-join) segment: translate docId → __row_id__ via doc values.
        // A fresh __row_id__ iterator per resolver (one resolver is bound to one codec DV iterator).
        // Forward-only and stateful, matching the codec iterators' ascending access (see RowIdResolver).
        final SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        return docId -> {
            if (rowId.advanceExact(docId) == false) {
                throw new IllegalStateException("document [" + docId + "] has no " + DocumentInput.ROW_ID_FIELD + " value");
            }
            return rowId.nextValue();
        };
    }

    /**
     * Per-nested-path child-doc → (root parquet row, element offset) mapping. Keyed by nested path (e.g.
     * {@code orgs.divisions.teams.members}). For a docId that is a child at that path, {@code parquetRow[d]} is
     * its ROOT's parquet row and {@code offset[d]} is that element's index within the root's flattened
     * {@code LIST<STRUCT>} column for the path (document/ingest order — the same order the Parquet column stores
     * its elements). {@code parquetRow[d] == -1} for any doc that is not a child at this path.
     *
     * <p>Why per-path: on deep nesting a root block interleaves child docs from ALL nested levels in post-order,
     * so a single flat block offset does not identify an element within a specific level's list. Grouping by the
     * child doc's {@code _nested_path} and counting in doc order within the root yields the correct per-level
     * element index that lines up with the Parquet column for that path.
     */
    // package-private for unit testing (see buildPathChildMaps)
    record PathChildMap(int[] parquetRow, int[] offset) {}

    /** Cached per-path child-doc maps for this segment; null until first built. */
    private Map<String, PathChildMap> pathChildMaps;

    /**
     * Builds (once, cached) a {@link PathChildMap} for every nested path present in the mapping, by walking the
     * block structure via {@code __row_id__} (root markers) and reading each child doc's {@code _nested_path}
     * term. Within each root block, child docs sharing a path are numbered 0,1,2,… in ascending docId (= ingest
     * post-order) order — matching the Parquet flattened-element order for that path.
     */
    private Map<String, PathChildMap> pathChildMaps() throws IOException {
        if (pathChildMaps != null) {
            return pathChildMaps;
        }
        int max = maxDoc();
        Set<String> paths = new java.util.HashSet<>();
        for (ChildFieldInfo info : childFields.values()) {
            paths.add(info.owningPath());
        }
        // Per-docId nested path (null = root/parent doc, which carries __row_id__).
        String[] docPath = readNestedPathPerDoc(max);
        // Per-docId root parquet row for parent docs, -1 for non-parents (children carry no __row_id__).
        long[] rootRowId = new long[max];
        java.util.Arrays.fill(rootRowId, -1L);
        SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        if (rowId != null) {
            for (int docId = 0; docId < max; docId++) {
                if (rowId.advanceExact(docId)) {
                    rootRowId[docId] = rowId.nextValue();
                }
            }
        }
        TRACE.trace("[NESTED-TRACE] pathChildMaps(): maxDoc={} paths={} — per-docId _nested_path={} , per-docId __row_id__(parents)={}",
            max, paths, java.util.Arrays.toString(docPath), java.util.Arrays.toString(rootRowId));
        pathChildMaps = buildPathChildMaps(max, docPath, rootRowId, paths);
        for (var e : pathChildMaps.entrySet()) {
            TRACE.trace("[NESTED-TRACE]   path '{}': docId->row {} , docId->elementOffset {}",
                e.getKey(), java.util.Arrays.toString(e.getValue().parquetRow()), java.util.Arrays.toString(e.getValue().offset()));
        }
        return pathChildMaps;
    }

    /**
     * Pure block-structure → per-path child map assignment (extracted for unit testing). For each root block
     * (a maximal run of child docs ending at a parent doc, identified by {@code rootRowId[docId] >= 0}), assigns
     * every child doc — grouped by its {@code _nested_path} — an ascending 0-based element offset within that
     * path's children of the block, and records the block's root parquet row. Child docs whose path is not in
     * {@code paths} (no exposed child field at that level) are skipped. This offset equals the Parquet flattened
     * element index for the path, because Lucene emits child docs in the same ingest (post-order) order Parquet
     * flattens elements.
     *
     * @param maxDoc     segment doc count
     * @param docPath    per-docId nested path ({@code null} for root/parent docs)
     * @param rootRowId  per-docId root parquet row for parent docs; {@code -1} for child (non-parent) docs
     * @param paths      the nested paths that have an exposed child field
     */
    static Map<String, PathChildMap> buildPathChildMaps(int maxDoc, String[] docPath, long[] rootRowId, Set<String> paths) {
        Map<String, int[]> rowByPath = new HashMap<>();
        Map<String, int[]> offByPath = new HashMap<>();
        for (String p : paths) {
            int[] rows = new int[maxDoc];
            int[] offs = new int[maxDoc];
            java.util.Arrays.fill(rows, -1);
            rowByPath.put(p, rows);
            offByPath.put(p, offs);
        }
        int prevParent = -1;
        for (int docId = 0; docId < maxDoc; docId++) {
            if (rootRowId[docId] < 0) {
                continue; // a child doc — handled when its enclosing parent is reached
            }
            int root = (int) rootRowId[docId];
            // Children of this root are docs (prevParent, docId), ascending (post-order). Count per path so each
            // path's k-th child in this block gets element offset k.
            Map<String, Integer> counter = new HashMap<>();
            for (int child = prevParent + 1; child < docId; child++) {
                String p = docPath[child];
                if (p == null || rowByPath.containsKey(p) == false) {
                    continue; // a level with no exposed child field — skip
                }
                int k = counter.merge(p, 1, Integer::sum) - 1;
                rowByPath.get(p)[child] = root;
                offByPath.get(p)[child] = k;
            }
            prevParent = docId;
        }
        Map<String, PathChildMap> maps = new HashMap<>();
        for (String p : paths) {
            maps.put(p, new PathChildMap(rowByPath.get(p), offByPath.get(p)));
        }
        return maps;
    }

    /**
     * Reads the {@code _nested_path} term of every doc in the segment (postings-only field written by the
     * engine on each nested child doc). Returns an array indexed by docId; entries are {@code null} for docs
     * with no {@code _nested_path} (root/parent docs). Uses the inverted index (the field is indexed, not stored
     * or doc-valued), iterating each term's postings once.
     */
    private String[] readNestedPathPerDoc(int max) throws IOException {
        String[] out = new String[max];
        org.apache.lucene.index.Terms terms = in.terms(org.opensearch.index.mapper.NestedPathFieldMapper.NAME);
        if (terms == null) {
            return out;
        }
        org.apache.lucene.index.TermsEnum te = terms.iterator();
        org.apache.lucene.index.PostingsEnum pe = null;
        org.apache.lucene.util.BytesRef term;
        while ((term = te.next()) != null) {
            String path = term.utf8ToString();
            pe = te.postings(pe, org.apache.lucene.index.PostingsEnum.NONE);
            for (int d = pe.nextDoc(); d != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS; d = pe.nextDoc()) {
                out[d] = path;
            }
        }
        return out;
    }

    /**
     * Materialized values of one nested-child field, keyed by child Lucene docId. Numeric fields populate
     * {@code longs} (raw per {@code ParquetPhysicalType}); {@code BYTE_ARRAY} fields populate {@code bytes}.
     * {@code present[d]} is true only on child docIds that have a value for this field.
     */
    private static final class ChildFieldColumn {
        final long[] longs;        // numeric values by docId (null for BYTE_ARRAY fields)
        final BytesRef[] bytes;    // string values by docId (null for numeric fields)
        final boolean[] present;   // has-value by docId

        ChildFieldColumn(long[] longs, BytesRef[] bytes, boolean[] present) {
            this.longs = longs;
            this.bytes = bytes;
            this.present = present;
        }
    }

    /**
     * DataFusion variant of {@link #childFieldColumn}: builds the same per-child-docId value arrays, but sources
     * each element's value from {@link org.opensearch.index.mapper.NestedSourceReconstructor#reconstructNestedLeaf}
     * (which reconstructs the owning path's ordered elements from the primary-format file) instead of the FFI
     * Dremel read. Scatter uses the SAME {@link PathChildMap} row+offset, so element {@code offset} here indexes
     * the reconstructor's flattened element list identically. Returns {@code null} to fall back to FFI.
     */
    private ChildFieldColumn datafusionChildColumn(
        String field,
        ChildFieldInfo info,
        PathChildMap map,
        int max,
        boolean isBytes,
        long[] longs,
        BytesRef[] bytes,
        boolean[] present
    ) throws IOException {
        org.opensearch.index.mapper.NestedSourceReconstructor rec = org.opensearch.index.mapper.NestedSourceReconstructor.get();
        if (rec == null) {
            return null;
        }
        String owning = info.owningPath();
        String leafName = field.substring(owning.length() + 1); // simple leaf name under the owning path
        String parquetFile = producer().parquetFilePath();
        // Per-row cache: reconstruct each distinct root row's owning-path leaf values once (child docIds of a
        // block share a row). Element k of the list = the k-th child of `owning` in the block = map.offset()[docId].
        java.util.Map<Integer, List<Object>> perRow = new java.util.HashMap<>();
        for (int docId = 0; docId < max; docId++) {
            int row = map.parquetRow()[docId];
            if (row < 0) {
                continue;
            }
            List<Object> elems = perRow.get(row);
            if (elems == null) {
                elems = rec.reconstructNestedLeaf(parquetFile, owning, leafName, row, mapperService);
                if (elems == null) {
                    return null; // reconstructor can't serve this — fall back to FFI wholesale
                }
                perRow.put(row, elems);
            }
            int off = map.offset()[docId];
            if (off < 0 || off >= elems.size()) {
                continue;
            }
            Object v = elems.get(off);
            if (v == null) {
                continue; // element exists but leaf absent — leave docId absent
            }
            if (isBytes) {
                byte[] raw = (v instanceof byte[]) ? (byte[]) v
                    : String.valueOf(v).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                bytes[docId] = new BytesRef(raw);
            } else {
                longs[docId] = childNumericBits(v, info.physical());
            }
            present[docId] = true;
        }
        return new ChildFieldColumn(longs, bytes, present);
    }

    /** Encodes a reconstructed nested-leaf raw value to the {@code long} bits the child numeric DV serves. */
    private static long childNumericBits(Object v, ParquetPhysicalType physical) {
        if (v instanceof Float f) {
            return Float.floatToRawIntBits(f) & 0xFFFF_FFFFL;
        }
        if (v instanceof Double d) {
            return Double.doubleToRawLongBits(d);
        }
        if (v instanceof Boolean b) {
            return b ? 1L : 0L;
        }
        // date/date_nanos leaves may come back from DataFusion as Arrow temporal objects rather than an epoch
        // long; the numeric DV (range/agg on a date) expects epoch millis, matching the FFI path.
        if (v instanceof java.time.LocalDateTime ldt) {
            return ldt.toInstant(java.time.ZoneOffset.UTC).toEpochMilli();
        }
        if (v instanceof java.time.Instant inst) {
            return inst.toEpochMilli();
        }
        if (v instanceof java.time.LocalDate ld) {
            return ld.atStartOfDay(java.time.ZoneOffset.UTC).toInstant().toEpochMilli();
        }
        return ((Number) v).longValue();
    }

    /**
     * Builds (once per field, cached) the {@link ChildFieldColumn} for a nested-child field: reads each
     * needed parent's repeated leaf column once and scatters element values to the child docIds of that
     * block. Numeric values are decoded per physical type; strings are kept as {@link BytesRef}.
     */
    private ChildFieldColumn childFieldColumn(String field) throws IOException {
        ChildFieldColumn cached = childFieldColumns.get(field);
        if (cached != null) {
            return cached;
        }
        ChildFieldInfo info = childFields.get(field);
        TRACE.trace("[NESTED-TRACE] childFieldColumn(field='{}'): owningPath='{}', parquetColumn='{}', listLevel={} — "
            + "building per-CHILD-docId value array from Parquet", field, info.owningPath(), info.leafColumnPath(), info.listLevel());
        PathChildMap map = pathChildMaps().get(info.owningPath());
        int max = maxDoc();
        boolean[] present = new boolean[max];
        boolean isBytes = info.physical() == ParquetPhysicalType.BYTE_ARRAY;
        long[] longs = isBytes ? null : new long[max];
        BytesRef[] bytes = isBytes ? new BytesRef[max] : null;
        if (map == null) {
            // No child docs at this path in the segment — leave everything absent.
            ChildFieldColumn empty = new ChildFieldColumn(longs, bytes, present);
            childFieldColumns.put(field, empty);
            return empty;
        }
        // Optional DataFusion path (query-phase nested range/agg): build the per-child-docId values from
        // reconstructed owning-path elements instead of the FFI Dremel read. Reuses the SAME PathChildMap
        // (row + per-path element offset), so only the value SOURCE differs — order/offset alignment is
        // identical (verified live at depth 3 and 4). Falls back to FFI on null/exception.
        if (FLAT_DATAFUSION_DV) {
            ChildFieldColumn df = datafusionChildColumn(field, info, map, max, isBytes, longs, bytes, present);
            if (df != null) {
                childFieldColumns.put(field, df);
                return df;
            }
        }

        ParquetColumnReader reader = producer().repeatedReaderFor(info.leafColumnPath(), info.physical());
        // The "element exists at this field's owning level" definition threshold, read from the actual schema
        // (no hardcoded per-level increment). thresholds[k-1] is the min def level at which a level-k element is
        // present; the field's owning level is info.listLevel().
        int[] thresholds = reader.levelDefThresholds();
        int existThreshold = info.listLevel() >= 1 && info.listLevel() <= thresholds.length
            ? thresholds[info.listLevel() - 1]
            : Integer.MAX_VALUE; // no such level in schema → no elements (defensive; shouldn't happen)
        // Read each root row's leaf ONCE with its Dremel levels and expand to a per-ELEMENT view at the field's
        // OWNING nesting level (info.listLevel): element k is the k-th element of the root's flattened path list,
        // counting ALL elements (including those whose leaf value is null) so the per-path element offset from the
        // child-doc map lines up exactly. A small per-row cache avoids re-reading when consecutive child docIds
        // share a root (they always do within a block).
        long lastRow = -1;
        long[] rowElemLongs = null;      // per-element value bits (numeric), index = element offset
        BytesRef[] rowElemBytes = null;  // per-element value (bytes), index = element offset
        boolean[] rowElemPresent = null; // per-element presence, index = element offset
        for (int docId = 0; docId < max; docId++) {
            int row = map.parquetRow()[docId];
            if (row < 0) {
                continue; // not a child doc at this path
            }
            if (row != lastRow) {
                ParquetColumnReader.LeafLevels levels = reader.readLeafLevelsAtRow(row, isBytes);
                int valueCount = isBytes ? (levels.bytes() == null ? 0 : levels.bytes().length) : levels.longs().length;
                Object[] values = new Object[valueCount];
                for (int i = 0; i < valueCount; i++) {
                    values[i] = isBytes ? levels.bytes()[i] : Long.valueOf(levels.longs()[i]);
                }
                TRACE.trace("[NESTED-TRACE]   read Parquet leaf at row {}: rep={} def={} maxDef={} presentValues={}",
                    row, java.util.Arrays.toString(levels.rep()), java.util.Arrays.toString(levels.def()), levels.maxDef(),
                    isBytes ? valueCount + " bytes" : java.util.Arrays.toString(levels.longs()));
                NestedDremel.ElementValues ev = NestedDremel.expandElementsAtLevel(
                    levels.rep(),
                    levels.def(),
                    levels.maxDef(),
                    info.listLevel(),
                    existThreshold,
                    values,
                    isBytes,
                    info.physical()
                );
                rowElemPresent = ev.present();
                rowElemLongs = ev.longs();
                rowElemBytes = ev.bytes();
                TRACE.trace("[NESTED-TRACE]   expandElementsAtLevel(level={}, existThreshold={}) -> per-element present={} "
                    + "(each element = one child doc of row {})", info.listLevel(), existThreshold,
                    java.util.Arrays.toString(rowElemPresent), row);
                lastRow = row;
            }
            int off = map.offset()[docId];
            if (off < rowElemPresent.length && rowElemPresent[off]) {
                if (isBytes) {
                    bytes[docId] = rowElemBytes[off];
                } else {
                    longs[docId] = rowElemLongs[off];
                }
                present[docId] = true;
                // Guard on isTraceEnabled(): the value renderer must not run when trace is off, because method
                // arguments are evaluated eagerly. bytes[] can hold non-UTF-8 payloads (ip = 16-byte
                // InetAddressPoint encoding, binary = arbitrary bytes), and utf8ToString() asserts valid UTF-8
                // (fatal under -ea). Use a byte-safe renderer so enabling trace on an ip/binary field is also safe.
                if (TRACE.isTraceEnabled()) {
                    TRACE.trace("[NESTED-TRACE]     docId {} (child) -> row {} element[{}] = {}", docId, row, off,
                        isBytes ? renderBytesRef(bytes[docId]) : Long.toString(longs[docId]));
                }
            }
        }
        ChildFieldColumn col = new ChildFieldColumn(longs, bytes, present);
        TRACE.trace("[NESTED-TRACE] childFieldColumn(field='{}') built: present child docIds={}", field, presentDocIds(present));
        childFieldColumns.put(field, col);
        return col;
    }

    /**
     * [NESTED-TRACE] helper: render a {@link BytesRef} for logging WITHOUT asserting UTF-8 validity. A child
     * field's bytes may be an ip (16-byte {@code InetAddressPoint} encoding) or arbitrary binary, neither of
     * which is valid UTF-8; {@link BytesRef#utf8ToString()} asserts and would be fatal under {@code -ea}. If the
     * bytes happen to be valid UTF-8 (keyword/text) we show the string; otherwise we fall back to the safe
     * hex-style {@link BytesRef#toString()}.
     */
    private static String renderBytesRef(BytesRef b) {
        if (b == null) {
            return "null";
        }
        try {
            return b.utf8ToString();
        } catch (AssertionError | RuntimeException notUtf8) {
            return b.toString(); // [xx xx ...] hex form — never asserts
        }
    }

    /** [NESTED-TRACE] helper: list the docIds where {@code present[d]} is true (for readable trace output). */
    private static String presentDocIds(boolean[] present) {
        StringBuilder sb = new StringBuilder("[");
        for (int d = 0; d < present.length; d++) {
            if (present[d]) {
                if (sb.length() > 1) sb.append(", ");
                sb.append(d);
            }
        }
        return sb.append("]").toString();
    }

    /** A single-valued {@link NumericDocValues} over a child field's per-docId {@code long} values. */
    private static NumericDocValues childNumericDocValues(ChildFieldColumn col, int maxDoc) {
        return new NumericDocValues() {
            private int doc = -1;

            @Override
            public long longValue() {
                return col.longs[doc];
            }

            @Override
            public boolean advanceExact(int target) {
                doc = target;
                return target < maxDoc && col.present[target];
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(int target) {
                for (int d = target; d < maxDoc; d++) {
                    if (col.present[d]) {
                        doc = d;
                        return d;
                    }
                }
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }

            @Override
            public long cost() {
                return maxDoc;
            }
        };
    }

    /**
     * A single-valued {@link SortedDocValues} over a child (keyword/ip) field's per-docId byte values.
     * Builds a per-field ordinal table (distinct terms sorted lexicographically) so term/prefix/range
     * queries over the child field resolve exactly as they would for a native keyword DV.
     */
    private static SortedDocValues childSortedDocValues(ChildFieldColumn col, int maxDoc) {
        // Distinct terms in sorted order → ordinal; per-doc ordinal (-1 = no value).
        java.util.TreeMap<BytesRef, Integer> termToOrd = new java.util.TreeMap<>();
        for (int d = 0; d < maxDoc; d++) {
            if (col.present[d]) {
                termToOrd.put(col.bytes[d], null);
            }
        }
        BytesRef[] ordToTerm = new BytesRef[termToOrd.size()];
        int ord = 0;
        for (Map.Entry<BytesRef, Integer> e : termToOrd.entrySet()) {
            e.setValue(ord);
            ordToTerm[ord] = e.getKey();
            ord++;
        }
        int[] docToOrd = new int[maxDoc];
        java.util.Arrays.fill(docToOrd, -1);
        for (int d = 0; d < maxDoc; d++) {
            if (col.present[d]) {
                docToOrd[d] = termToOrd.get(col.bytes[d]);
            }
        }
        final int valueCount = ordToTerm.length;
        return new SortedDocValues() {
            private int doc = -1;

            @Override
            public int ordValue() {
                return docToOrd[doc];
            }

            @Override
            public BytesRef lookupOrd(int o) {
                return ordToTerm[o];
            }

            @Override
            public int getValueCount() {
                return valueCount;
            }

            @Override
            public boolean advanceExact(int target) {
                doc = target;
                return target < maxDoc && docToOrd[target] >= 0;
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(int target) {
                for (int d = target; d < maxDoc; d++) {
                    if (docToOrd[d] >= 0) {
                        doc = d;
                        return d;
                    }
                }
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }

            @Override
            public long cost() {
                return maxDoc;
            }
        };
    }

    /**
     * {@link NestedSourceProvider} — reconstructs a nested {@code LIST<STRUCT>} array for derived source.
     * Resolves the parent's {@code docId} to its Parquet row (via {@code __row_id__}, same as every other
     * codec read) and delegates the columnar read + array assembly to {@link ParquetDocValuesProducer}.
     */
    @Override
    public List<Map<String, Object>> readNestedArray(String path, int docId) throws IOException {
        long parquetRow = newRowIdResolver().toRowId(docId);
        TRACE.trace("[NESTED-TRACE] readNestedArray(path='{}', parentDocId={}) -> resolved __row_id__/Parquet row={}"
            + " ; reconstructing the full nested array for _source/inner_hits", path, docId, parquetRow);
        List<Map<String, Object>> out = producer().readNestedArray(path, parquetRow);
        TRACE.trace("[NESTED-TRACE] readNestedArray(path='{}', row={}) -> reconstructed {} element(s): {}", path, parquetRow,
            out.size(), out);
        return out;
    }

    /** True if {@code __row_id__ == docId} for every doc (flat segment). A nested block-join segment
     *  returns false because child docs shift parent docIds past their dense __row_id__. Computed once
     *  per reader (a single O(maxDoc) scan) and cached, since it's checked per DV accessor. */
    private boolean isIdentitySegment() throws IOException {
        if (identitySegment != null) {
            return identitySegment;
        }
        boolean identity = true;
        SortedNumericDocValues rowId = in.getSortedNumericDocValues(DocumentInput.ROW_ID_FIELD);
        if (rowId != null) {
            for (int docId = 0; docId < maxDoc(); docId++) {
                if (rowId.advanceExact(docId) == false || rowId.nextValue() != docId) {
                    identity = false;
                    break;
                }
            }
        }
        identitySegment = identity;
        return identity;
    }

    @Override
    public FieldInfos getFieldInfos() {
        return mergedFieldInfos;
    }

    /**
     * Serves a flat numeric field from the registered {@link org.opensearch.index.mapper.FlatColumnValueSource}
     * (DataFusion) when the flag is on: materializes the whole column once (cached), then returns a
     * {@link NumericDocValues} indexed by docId. Flat segments are identity ({@code docId == __row_id__}), so the
     * materialized per-row array indexes directly by docId. Returns {@code null} to signal "not served here" so the
     * caller falls back to the FFI path.
     */
    private NumericDocValues datafusionFlatNumeric(String field) throws IOException {
        if (FLAT_DATAFUSION_DV == false || isIdentitySegment() == false) {
            return null; // only flat/identity segments; nested child fields use their own path
        }
        org.opensearch.index.mapper.FlatColumnValueSource src = org.opensearch.index.mapper.FlatColumnValueSource.get();
        if (src == null) {
            return null;
        }
        org.opensearch.index.mapper.FlatColumnValueSource.LongColumn col = flatLongColumns.get(field);
        if (col == null) {
            col = src.materializeLong(producer().parquetFilePath(), field, producer().parquetRowCount());
            if (col == null) {
                return null;
            }
            flatLongColumns.put(field, col);
        }
        final org.opensearch.index.mapper.FlatColumnValueSource.LongColumn c = col;
        return new NumericDocValues() {
            private int doc = -1;

            @Override
            public boolean advanceExact(int target) {
                doc = target;
                return target >= 0 && target < c.present.length && c.present[target];
            }

            @Override
            public long longValue() {
                return c.values[doc];
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(int target) {
                for (int i = target; i < c.present.length; i++) {
                    if (c.present[i]) {
                        doc = i;
                        return i;
                    }
                }
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }

            @Override
            public long cost() {
                return c.present.length;
            }
        };
    }

    @Override
    public NumericDocValues getNumericDocValues(String field) throws IOException {
        if (childFields.containsKey(field)) {
            // Nested-child numeric field: values keyed by CHILD docId (block-join hidden docs).
            return childNumericDocValues(childFieldColumn(field), maxDoc());
        }
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null && fi.getDocValuesType() == DocValuesType.NUMERIC) {
            NumericDocValues df = datafusionFlatNumeric(field);
            if (df != null) {
                return df;
            }
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
        if (childFields.containsKey(field)) {
            // Nested-child numeric field: single-valued per child doc, wrapped as SortedNumeric so the
            // range/term query's unwrapSingleton fast path applies. Keyed by CHILD docId.
            TRACE.trace("[NESTED-TRACE] getSortedNumericDocValues(child field='{}') -> serving numeric DV from Parquet"
                + " (this is the RANGE/agg path; e.g. range on comments.score)", field);
            return DocValues.singleton(childNumericDocValues(childFieldColumn(field), maxDoc()));
        }
        NumericDocValues dfNumeric = datafusionFlatNumeric(field);
        if (dfNumeric != null) {
            return DocValues.singleton(dfNumeric);
        }
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
            BinaryDocValues df = datafusionFlatBinary(field);
            if (df != null) {
                return df;
            }
            RowIdResolver resolver = newRowIdResolver();
            BinaryDocValues binary = producer().getBinary(fi);
            return resolver == RowIdResolver.IDENTITY ? binary : RowIdRemappingDocValues.binary(binary, resolver, maxDoc());
        }
        return in.getBinaryDocValues(field);
    }

    /**
     * Serves a flat BINARY field from the registered {@link org.opensearch.index.mapper.FlatColumnValueSource}
     * (DataFusion) when the flag is on. Same shape as {@link #datafusionFlatNumeric} but over raw bytes. Returns
     * {@code null} to fall back to FFI. NOTE: keyword/text/ip (SORTED/SORTED_SET) still use the FFI ordinal-table
     * path for now — the interim DataFusion flat path covers numeric + binary doc-values; ordinal-encoded keyword
     * columns join in the forward-cursor optimization pass.
     */
    private BinaryDocValues datafusionFlatBinary(String field) throws IOException {
        if (FLAT_DATAFUSION_DV == false || isIdentitySegment() == false) {
            return null;
        }
        org.opensearch.index.mapper.FlatColumnValueSource src = org.opensearch.index.mapper.FlatColumnValueSource.get();
        if (src == null) {
            return null;
        }
        org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn col = flatBytesColumns.get(field);
        if (col == null) {
            col = src.materializeBytes(producer().parquetFilePath(), field, producer().parquetRowCount());
            if (col == null) {
                return null;
            }
            flatBytesColumns.put(field, col);
        }
        final org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn c = col;
        return new BinaryDocValues() {
            private int doc = -1;

            @Override
            public org.apache.lucene.util.BytesRef binaryValue() {
                return new org.apache.lucene.util.BytesRef(c.values[doc]);
            }

            @Override
            public boolean advanceExact(int target) {
                doc = target;
                return target >= 0 && target < c.present.length && c.present[target];
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(int target) {
                for (int i = target; i < c.present.length; i++) {
                    if (c.present[i]) {
                        doc = i;
                        return i;
                    }
                }
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }

            @Override
            public long cost() {
                return c.present.length;
            }
        };
    }

    @Override
    public SortedDocValues getSortedDocValues(String field) throws IOException {
        if (childFields.containsKey(field)) {
            return childSortedDocValues(childFieldColumn(field), maxDoc());
        }
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null && fi.getDocValuesType() == DocValuesType.SORTED) {
            RowIdResolver resolver = newRowIdResolver();
            SortedDocValues sorted = producer().getSorted(fi);
            return resolver == RowIdResolver.IDENTITY ? sorted : RowIdRemappingDocValues.sorted(sorted, resolver, maxDoc());
        }
        return in.getSortedDocValues(field);
    }

    /**
     * Serves a flat keyword/text field's {@link SortedDocValues} from the registered
     * {@link org.opensearch.index.mapper.FlatColumnValueSource} (DataFusion) when the flag is on: materializes the
     * whole byte column once (cached), builds a sorted term dictionary + per-doc ordinal in Java (the ordinal
     * semantics the FFI {@code OrdinalTable} produces), and returns a docId-indexed {@link SortedDocValues}. Flat
     * segments are identity, so per-row values index directly by docId. Returns {@code null} to fall back to FFI.
     */
    private SortedDocValues datafusionFlatSorted(String field) throws IOException {
        if (FLAT_DATAFUSION_DV == false || isIdentitySegment() == false) {
            return null;
        }
        org.opensearch.index.mapper.FlatColumnValueSource src = org.opensearch.index.mapper.FlatColumnValueSource.get();
        if (src == null) {
            return null;
        }
        org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn col = flatBytesColumns.get(field);
        if (col == null) {
            col = src.materializeBytes(producer().parquetFilePath(), field, producer().parquetRowCount());
            if (col == null) {
                return null;
            }
            flatBytesColumns.put(field, col);
        }
        final org.opensearch.index.mapper.FlatColumnValueSource.BytesColumn c = col;
        // Build a sorted term dictionary (unique values, byte-ordered) and a per-doc ordinal, mirroring the FFI
        // OrdinalTable: ord -> term ascending, docId -> ord (or -1 absent).
        java.util.TreeMap<org.apache.lucene.util.BytesRef, Integer> dict = new java.util.TreeMap<>();
        for (int r = 0; r < c.present.length; r++) {
            if (c.present[r] && c.values[r] != null) {
                dict.putIfAbsent(new org.apache.lucene.util.BytesRef(c.values[r]), 0);
            }
        }
        final org.apache.lucene.util.BytesRef[] ordToTerm = new org.apache.lucene.util.BytesRef[dict.size()];
        int nextOrd = 0;
        for (java.util.Map.Entry<org.apache.lucene.util.BytesRef, Integer> e : dict.entrySet()) {
            e.setValue(nextOrd);
            ordToTerm[nextOrd] = e.getKey();
            nextOrd++;
        }
        final int[] docToOrd = new int[c.present.length];
        for (int r = 0; r < c.present.length; r++) {
            docToOrd[r] = (c.present[r] && c.values[r] != null)
                ? dict.get(new org.apache.lucene.util.BytesRef(c.values[r]))
                : -1;
        }
        return new SortedDocValues() {
            private int doc = -1;

            @Override
            public int ordValue() {
                return docToOrd[doc];
            }

            @Override
            public org.apache.lucene.util.BytesRef lookupOrd(int ord) {
                return ordToTerm[ord];
            }

            @Override
            public int getValueCount() {
                return ordToTerm.length;
            }

            @Override
            public boolean advanceExact(int target) {
                doc = target;
                return target >= 0 && target < docToOrd.length && docToOrd[target] >= 0;
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                return advance(doc + 1);
            }

            @Override
            public int advance(int target) {
                for (int i = target; i < docToOrd.length; i++) {
                    if (docToOrd[i] >= 0) {
                        doc = i;
                        return i;
                    }
                }
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }

            @Override
            public long cost() {
                return docToOrd.length;
            }
        };
    }

    @Override
    public SortedSetDocValues getSortedSetDocValues(String field) throws IOException {
        if (childFields.containsKey(field)) {
            // Nested-child keyword field: single-valued per child doc, wrapped as SortedSet so the term
            // query's unwrapSingleton fast path applies. Keyed by CHILD docId.
            TRACE.trace("[NESTED-TRACE] getSortedSetDocValues(child field='{}') -> serving keyword DV from Parquet"
                + " (doc-values path; used by keyword AGG/sort — NOT by an exact term query, which uses Lucene postings)", field);
            return DocValues.singleton(childSortedDocValues(childFieldColumn(field), maxDoc()));
        }
        SortedDocValues dfSorted = datafusionFlatSorted(field);
        if (dfSorted != null) {
            return DocValues.singleton(dfSorted);
        }
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
            SortedDocValues sorted = producer().getSorted(asSorted);
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
        if (childFields.containsKey(field)) {
            // Nested-child fields have no skip index: the parquet page min/max are per parent row, not per
            // child docId, so they can't bound a child-docId interval. Returning null makes the range query
            // fall back to a DocValues TwoPhaseIterator over childNumericDocValues — correct, no skip accel.
            return null;
        }
        FieldInfo fi = parquetFieldInfo(field);
        if (fi != null) {
            if (fi.docValuesSkipIndexType() == DocValuesSkipIndexType.NONE) {
                return null;
            }
            // The skip index's intervals are Parquet PAGE row ranges, and ParquetDocValuesSkipper.advance
            // treats the incoming target as a Parquet row (pageIndex.pageForRow(target)). That is only valid
            // when doc ID == Parquet row, i.e. on an IDENTITY (flat) segment. On a nested segment parent doc
            // IDs are shifted past their child docs (docId != row), so feeding a docId into row-space page
            // lookups would skip the wrong pages and drop matching parents. Return null there so the range
            // query falls back to a DocValues TwoPhaseIterator over the docId→row-remapped numeric DV
            // (correct, just without skip acceleration) — mirroring the child-field branch above.
            if (isIdentitySegment() == false) {
                return null;
            }
            return producer().getSkipper(fi);
        }
        return in.getDocValuesSkipper(field);
    }

    @Override
    protected void doClose() throws IOException {
        IOException first = null;
        try {
            if (producer != null) {
                producer.close();
            }
        } catch (IOException e) {
            first = e;
        }
        try {
            super.doClose();
        } catch (IOException e) {
            if (first == null) {
                first = e;
            }
        }
        if (first != null) {
            throw first;
        }
    }

    // Cache helpers must delegate to the underlying reader so query/segment caches stay coherent.
    @Override
    public CacheHelper getCoreCacheHelper() {
        return in.getCoreCacheHelper();
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return in.getReaderCacheHelper();
    }
}

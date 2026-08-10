/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene;

import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.be.lucene.index.LuceneDocumentInput;
import org.opensearch.index.mapper.NestedPathFieldMapper;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * The shared child ordinal for nested-predicate delegation: maps a nested-child Lucene docId to its
 * {@code (parquet root row, element offset)} — the coordinate BOTH engines key on when a nested predicate
 * is split across Lucene (keyword children) and Parquet/DataFusion (numeric/range children) and intersected
 * at CHILD grain before the block-join roll-up.
 *
 * <p>This is the delegation-path analog of the DSL-fetch path's {@code PathChildMap} (which lived inside the
 * parquet codec's {@code ParquetDocValuesLeafReader}). It is decoupled from that codec so the analytics
 * delegation path ({@link LuceneFilterDelegationHandle}) can build it directly from a Lucene {@link LeafReader}
 * using only two engine-written fields present on every composite nested segment:
 * <ul>
 *   <li>{@code _nested_path} ({@link NestedPathFieldMapper#NAME}) — an indexed (postings-only) term written on
 *       each nested child doc, identifying which nested level the child belongs to;</li>
 *   <li>{@code __row_id__} ({@link LuceneDocumentInput#ROW_ID_FIELD}) — a {@code SortedNumericDocValues} written
 *       only on ROOT/parent docs, carrying the Parquet row.</li>
 * </ul>
 *
 * <p><b>Per-path, arbitrary depth.</b> A root block interleaves child docs from ALL nested levels in ingest
 * (post-order) order, so a single flat block offset does not identify an element within one level's list.
 * Grouping child docs by their {@code _nested_path} and numbering them {@code 0,1,2,…} in ascending docId order
 * within each root yields the per-level element index that lines up with the Parquet flattened-element column
 * for that path. This holds because the writer emits child docs in the same ingest post-order Parquet flattens
 * elements (see {@code LuceneDocumentInput.getDocumentBlock} — children first, parent last).
 *
 * <p>Immutable and built once per (segment, needed-paths); safe to cache on the per-leaf translator.
 */
final class NestedChildOrdinalMap {

    /**
     * Per nested path, two arrays indexed by Lucene docId. For a docId that is a child at that path:
     * {@code parquetRow[d]} = its root's Parquet row, {@code offset[d]} = the element's index within that
     * root's flattened {@code LIST<STRUCT>} for the path. {@code parquetRow[d] == -1} otherwise.
     */
    record PathChildMap(int[] parquetRow, int[] offset) {}

    private final Map<String, PathChildMap> byPath;

    private NestedChildOrdinalMap(Map<String, PathChildMap> byPath) {
        this.byPath = byPath;
    }

    /** The paths this map covers (the nested paths that had a child predicate needing an ordinal). */
    Set<String> paths() {
        return byPath.keySet();
    }

    /**
     * The element offset of child {@code docId} within its root's list for {@code path}, or {@code -1} if
     * {@code docId} is not a child at {@code path}.
     */
    int offsetForDocId(String path, int docId) {
        PathChildMap m = byPath.get(path);
        return (m == null || docId < 0 || docId >= m.offset().length) ? -1 : (m.parquetRow()[docId] < 0 ? -1 : m.offset()[docId]);
    }

    /**
     * The root Parquet row of child {@code docId} for {@code path}, or {@code -1} if {@code docId} is not a
     * child at {@code path}.
     */
    int rowForDocId(String path, int docId) {
        PathChildMap m = byPath.get(path);
        return (m == null || docId < 0 || docId >= m.parquetRow().length) ? -1 : m.parquetRow()[docId];
    }

    /** Package-private accessor for the raw per-path map (used by tracing / tests). */
    PathChildMap pathMap(String path) {
        return byPath.get(path);
    }

    /**
     * Builds the child-ordinal map for {@code reader}, covering exactly {@code paths}. Reads the segment's
     * {@code _nested_path} postings and per-parent {@code __row_id__} once, then assigns per-path element
     * offsets via {@link #assign}. Returns an empty map (no coverage) if the segment is non-nested (no
     * {@code __row_id__}) or {@code paths} is empty.
     */
    static NestedChildOrdinalMap build(LeafReader reader, Set<String> paths) throws IOException {
        if (paths.isEmpty()) {
            return new NestedChildOrdinalMap(Map.of());
        }
        int maxDoc = reader.maxDoc();
        String[] docPath = readNestedPathPerDoc(reader, maxDoc);
        long[] rootRowId = new long[maxDoc];
        java.util.Arrays.fill(rootRowId, -1L);
        SortedNumericDocValues rowId = reader.getSortedNumericDocValues(LuceneDocumentInput.ROW_ID_FIELD);
        if (rowId != null) {
            for (int docId = rowId.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = rowId.nextDoc()) {
                rootRowId[docId] = rowId.nextValue();
            }
        }
        return new NestedChildOrdinalMap(assign(maxDoc, docPath, rootRowId, paths));
    }

    /**
     * Pure block-structure → per-path child-offset assignment (extracted for unit testing). For each root
     * block (a maximal run of child docs ending at a parent doc, identified by {@code rootRowId[docId] >= 0}),
     * numbers every child doc — grouped by its {@code _nested_path} — with an ascending 0-based element offset
     * within that path's children of the block, and records the block's root Parquet row.
     */
    static Map<String, PathChildMap> assign(int maxDoc, String[] docPath, long[] rootRowId, Set<String> paths) {
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
                continue; // a child doc — assigned when its enclosing parent is reached
            }
            int root = (int) rootRowId[docId];
            Map<String, Integer> counter = new HashMap<>();
            for (int child = prevParent + 1; child < docId; child++) {
                String p = docPath[child];
                if (p == null || rowByPath.containsKey(p) == false) {
                    continue; // a level with no requested child predicate — skip
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
     * Reads the {@code _nested_path} term of every doc via the inverted index (postings-only field). Returns
     * an array indexed by docId; entries are {@code null} for docs with no {@code _nested_path} (root/parent).
     */
    private static String[] readNestedPathPerDoc(LeafReader reader, int maxDoc) throws IOException {
        String[] out = new String[maxDoc];
        Terms terms = reader.terms(NestedPathFieldMapper.NAME);
        if (terms == null) {
            return out;
        }
        TermsEnum te = terms.iterator();
        PostingsEnum pe = null;
        BytesRef term;
        while ((term = te.next()) != null) {
            String path = term.utf8ToString();
            pe = te.postings(pe, PostingsEnum.NONE);
            for (int d = pe.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = pe.nextDoc()) {
                out[d] = path;
            }
        }
        return out;
    }
}

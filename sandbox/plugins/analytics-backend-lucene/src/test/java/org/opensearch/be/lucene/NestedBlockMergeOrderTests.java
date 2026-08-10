/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.lucene;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LogByteSizeMergePolicy;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.opensearch.be.lucene.index.LuceneWriter;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.NestedPathFieldMapper;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The "ordering hazard" guard for the child-grain nested-predicate split.
 *
 * <p>Flow A (the keyword→Lucene / range→DataFusion element-grain split) rests on ONE assumption:
 * <em>for each nested path, ascending child-docId order within a parent block equals the Arrow flattened
 * array order.</em> Lucene's {@link org.opensearch.be.lucene.NestedChildOrdinalMap} does not persist the
 * element offset — it RECOMPUTES it by counting child docs in docId order — so if a merge ever reordered a
 * parent's children (or scattered a block), the recomputed offset would silently diverge from the Arrow
 * flattened position and {@code childBase[row] + offset} would name the WRONG element. The {@code fallback}
 * safety net does NOT catch this: a reordered bitset is <em>present but wrong</em>, so the UDF trusts it.
 *
 * <p>Production defends against this with Lucene parent-block index sorting: the ingest writer
 * ({@link LuceneWriter}) and the merge writer ({@code LuceneCommitter}, secondary-composite path) both
 * configure {@code setIndexSort(SortedNumericSortField(__row_id__))} + {@code setParentField(__nested_parent)}
 * and add each row as an atomic block via {@code addDocuments}. This test reproduces that exact configuration,
 * forces a REAL multi-segment sorted merge whose sort order differs from arrival order (so blocks must move),
 * and asserts that after the merge:
 * <ol>
 *   <li>each parent's block is contiguous, children precede the parent (post-order), and within-block child
 *       order is preserved;</li>
 *   <li>{@link NestedChildOrdinalMap} recovers exactly the ingest {@code (row, offset)} for every child;</li>
 *   <li>the Part-2 identity {@code childBase[row] + offset == Arrow-flattened index} still holds on the
 *       post-merge segment;</li>
 *   <li>per-{@code _nested_path} offset isolation survives the merge for interleaved multi-level (nested-of-
 *       nested) blocks.</li>
 * </ol>
 */
public class NestedBlockMergeOrderTests extends OpenSearchTestCase {

    private static final String ROW_ID_FIELD = DocumentInput.ROW_ID_FIELD;      // "__row_id__"
    private static final String NESTED_PATH_FIELD = NestedPathFieldMapper.NAME; // "_nested_path"
    private static final String ORGS = "orgs";
    private static final String TEAMS = "orgs.teams";

    private Directory directory;
    private IndexWriter writer;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = new NIOFSDirectory(createTempDir());
        writer = new IndexWriter(directory, productionSecondaryConfig());
    }

    @Override
    public void tearDown() throws Exception {
        if (writer != null && writer.isOpen()) {
            writer.close();
        }
        if (directory != null) {
            directory.close();
        }
        super.tearDown();
    }

    /**
     * Mirrors {@code LuceneCommitter.createIndexWriterConfig}'s secondary-composite branch and
     * {@code LuceneWriter}'s ingest config: block-preserving index sort on {@code __row_id__} plus the
     * nested parent field, with {@link LogByteSizeMergePolicy} (the merge policy production uses when an
     * index sort is present).
     */
    private IndexWriterConfig productionSecondaryConfig() {
        IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
        iwc.setMergePolicy(new LogByteSizeMergePolicy());
        iwc.setIndexSort(new Sort(new SortedNumericSortField(ROW_ID_FIELD, SortField.Type.LONG)));
        iwc.setParentField(LuceneWriter.NESTED_PARENT_FIELD);
        return iwc;
    }

    /**
     * Core guard: single nested path, three rows whose ARRIVAL order (2, 0, 1) is not the sort order, split
     * across two segments so {@code forceMerge(1)} must relocate blocks. Proves physical order, offset/row
     * recovery, and the Part-2 flat-index identity all survive the sorted merge.
     */
    public void testChildDocOrderAndOffsetsSurviveSortedMerge() throws IOException {
        // Arrival order deliberately unsorted; two segments so the merge has real work to do.
        addNestedRow(writer, 2, ORGS, 3); // r2c0 r2c1 r2c2 P2
        addNestedRow(writer, 0, ORGS, 2); // r0c0 r0c1 P0
        writer.flush();                   // segment A: [row2 block, row0 block]
        addNestedRow(writer, 1, ORGS, 1); // r1c0 P1
        writer.flush();                   // segment B: [row1 block]

        // Guard: prove there really are >=2 segments to merge (else forceMerge is a no-op and this test would
        // degrade into a single-segment sorted-flush check rather than the multi-segment merge under test).
        try (DirectoryReader preMerge = DirectoryReader.open(writer)) {
            assertTrue("test must exercise a real multi-segment merge (>=2 segments pre-merge)", preMerge.leaves().size() >= 2);
        }

        writer.forceMerge(1);             // <-- the sorted, block-preserving merge under test

        try (DirectoryReader reader = DirectoryReader.open(writer)) {
            assertEquals("forceMerge(1) must leave exactly one segment", 1, reader.leaves().size());
            LeafReader leaf = reader.leaves().get(0).reader();

            // (1) Physical doc order after merge: blocks relocated as units and sorted by parent __row_id__
            //     (0,1,2), children BEFORE their parent (post-order), within-block child order preserved.
            List<String> actual = readLabels(leaf);
            List<String> expected = List.of("r0c0", "r0c1", "P0", "r1c0", "P1", "r2c0", "r2c1", "r2c2", "P2");
            assertEquals("post-merge physical doc order", expected, actual);
            // The merge really MOVED blocks: arrival had row2's block first, sorted output has row0's first.
            assertEquals("sorted merge must relocate row0's block to the front", "r0c0", actual.get(0));

            // (2) NestedChildOrdinalMap recovers the ingest (row, offset) for every child, and -1 for parents.
            NestedChildOrdinalMap map = NestedChildOrdinalMap.build(leaf, Set.of(ORGS));
            for (int d = 0; d < leaf.maxDoc(); d++) {
                String label = actual.get(d);
                if (label.charAt(0) == 'P') {
                    assertEquals("parent " + label + " must have no child offset", -1, map.offsetForDocId(ORGS, d));
                    assertEquals("parent " + label + " must have no child row", -1, map.rowForDocId(ORGS, d));
                    continue;
                }
                int ingestRow = label.charAt(1) - '0';
                int ingestOff = label.charAt(3) - '0';
                assertEquals("recovered row for " + label, ingestRow, map.rowForDocId(ORGS, d));
                assertEquals("recovered offset for " + label, ingestOff, map.offsetForDocId(ORGS, d));
            }

            // (3) Part-2 identity on POST-MERGE data: childBase[row] + offset == Arrow flattened index.
            //     Arrow flattens by sorted row order, ingest order within a row:
            //       flat:  r0c0=0 r0c1=1 | r1c0=2 | r2c0=3 r2c1=4 r2c2=5
            int[] childBase = { 0, 2, 3 }; // prefix sums of per-row child counts, in sorted row order
            Map<String, Integer> expectedFlat = Map.of("r0c0", 0, "r0c1", 1, "r1c0", 2, "r2c0", 3, "r2c1", 4, "r2c2", 5);
            for (int d = 0; d < leaf.maxDoc(); d++) {
                String label = actual.get(d);
                if (label.charAt(0) == 'P') {
                    continue;
                }
                int row = map.rowForDocId(ORGS, d);
                int off = map.offsetForDocId(ORGS, d);
                int flat = childBase[row] + off;
                assertEquals("childBase[row]+offset must equal Arrow flat index for " + label, (int) expectedFlat.get(label), flat);
            }
        }
    }

    /**
     * Deep guard: two levels ({@code orgs} + {@code orgs.teams}) interleaved within each block, rows arriving
     * unsorted across two segments. Proves the per-{@code _nested_path} offset counting still isolates each
     * level independently after a real sorted merge — the arbitrary-depth correctness guarantee.
     */
    public void testPerPathOffsetIsolationSurvivesMergeForNestedOfNested() throws IOException {
        // Interleave paths within a block (post-order style): a teams child, an orgs child, a teams child, ...
        // row 1 arrives first (unsorted) and in its own segment so the merge must reorder.
        addInterleavedRow(writer, 1); // r1T0 r1O0 P1
        writer.flush();
        addInterleavedRow(writer, 0); // r0T0 r0O0 r0T1 P0
        writer.flush();

        try (DirectoryReader preMerge = DirectoryReader.open(writer)) {
            assertTrue("test must exercise a real multi-segment merge (>=2 segments pre-merge)", preMerge.leaves().size() >= 2);
        }

        writer.forceMerge(1);

        try (DirectoryReader reader = DirectoryReader.open(writer)) {
            assertEquals(1, reader.leaves().size());
            LeafReader leaf = reader.leaves().get(0).reader();

            List<String> actual = readLabels(leaf);
            // Sorted by parent __row_id__: row0's block (3 children) then row1's block (2 children).
            assertEquals(List.of("r0T0", "r0O0", "r0T1", "P0", "r1T0", "r1O0", "P1"), actual);

            NestedChildOrdinalMap map = NestedChildOrdinalMap.build(leaf, Set.of(ORGS, TEAMS));
            for (int d = 0; d < leaf.maxDoc(); d++) {
                String label = actual.get(d);
                if (label.charAt(0) == 'P') {
                    continue;
                }
                int ingestRow = label.charAt(1) - '0';
                String path = label.charAt(2) == 'T' ? TEAMS : ORGS;
                int ingestOff = label.charAt(3) - '0';
                // The offset must be counted WITHIN the child's own path, ignoring interleaved other-path docs.
                assertEquals("row for " + label, ingestRow, map.rowForDocId(path, d));
                assertEquals("per-path offset for " + label, ingestOff, map.offsetForDocId(path, d));
                // And it must NOT be claimed by the other path.
                String otherPath = path.equals(TEAMS) ? ORGS : TEAMS;
                assertEquals(label + " must not belong to " + otherPath, -1, map.offsetForDocId(otherPath, d));
            }
        }
    }

    // ===== helpers =====

    /** Adds one nested row as an atomic block: {@code childCount} children of {@code path}, then the parent. */
    private void addNestedRow(IndexWriter w, long rowId, String path, int childCount) throws IOException {
        List<Document> block = new ArrayList<>(childCount + 1);
        for (int off = 0; off < childCount; off++) {
            Document child = new Document();
            child.add(new StringField(NESTED_PATH_FIELD, path, Field.Store.NO));
            child.add(new StoredField("label", "r" + rowId + "c" + off));
            block.add(child);
        }
        block.add(parentDoc(rowId, "P" + rowId));
        w.addDocuments(block); // parent is LAST — the setParentField contract
    }

    /**
     * Adds one row whose block interleaves two nested paths. row 0 gets 2 teams + 1 orgs interleaved
     * (T,O,T); row 1 gets 1 team + 1 orgs (T,O). Labels encode {@code r<row><T|O><offset-within-path>}.
     */
    private void addInterleavedRow(IndexWriter w, long rowId) throws IOException {
        List<Document> block = new ArrayList<>();
        if (rowId == 0) {
            block.add(childDoc(TEAMS, "r0T0"));
            block.add(childDoc(ORGS, "r0O0"));
            block.add(childDoc(TEAMS, "r0T1"));
        } else {
            block.add(childDoc(TEAMS, "r1T0"));
            block.add(childDoc(ORGS, "r1O0"));
        }
        block.add(parentDoc(rowId, "P" + rowId));
        w.addDocuments(block);
    }

    private Document childDoc(String path, String label) {
        Document child = new Document();
        child.add(new StringField(NESTED_PATH_FIELD, path, Field.Store.NO));
        child.add(new StoredField("label", label));
        return child;
    }

    private Document parentDoc(long rowId, String label) {
        Document parent = new Document();
        parent.add(new SortedNumericDocValuesField(ROW_ID_FIELD, rowId)); // only parents carry __row_id__
        parent.add(new StoredField("label", label));
        return parent;
    }

    private List<String> readLabels(LeafReader leaf) throws IOException {
        List<String> labels = new ArrayList<>(leaf.maxDoc());
        for (int d = 0; d < leaf.maxDoc(); d++) {
            labels.add(leaf.storedFields().document(d).get("label"));
        }
        return labels;
    }
}

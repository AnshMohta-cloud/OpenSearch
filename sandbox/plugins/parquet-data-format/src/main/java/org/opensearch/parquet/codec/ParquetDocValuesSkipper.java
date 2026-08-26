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
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.parquet.codec.cache.ColumnPageIndex;

import java.io.IOException;

/**
 * {@link DocValuesSkipper} backed by the Parquet ColumnIndex (per-page min/max/null-count),
 * exposed through the already-loaded {@link ColumnPageIndex}.
 *
 * <p>This is the query-level complement of the codec's internal skips: Layer 3/4 make a
 * requested page cheap to reach, while this skipper lets Lucene's range machinery avoid
 * requesting excluded pages at all — zero decode, zero FFM, zero iteration for any page whose
 * [min, max] does not intersect the query range.
 *
 * <p>Single level: level 0 intervals are Parquet pages (~20k rows).
 *
 * <p><b>Flat mode</b> (no {@code firstDocIdOf} loader): the Row ID = Doc ID invariant makes each
 * page's row range directly usable as a Lucene doc-ID range ({@code firstRowOf(p) ..
 * firstRowOf(p)+numRowsOf(p)-1}), and per-page null counts give an exact per-page value density.
 *
 * <p><b>Nested mode</b> (a {@code firstDocIdOf} loader is supplied): the column is a repeated (LIST) leaf,
 * so a page holds the ELEMENTS of a contiguous run of top-level rows, and one logical row is a block-join
 * block of {@code {children..., ROOT}} Lucene docs — docId != row. The caller therefore supplies a loader
 * that lazily builds {@code firstDocIdOf} on the first {@link #advance(int)}: the child-docId at which each
 * page's first row's block begins. Consecutive
 * entries tile the doc space with no gaps, so page {@code p} owns doc range
 * {@code [firstDocIdOf[p], firstDocIdOf[p+1]-1]}. Per-page value min/max still read correctly (they
 * are element bounds, so the range test — the actual page-pruning win — still applies), but per-page
 * null counts mix element and row units, so {@code docCount} is reported as 0 (both per level and
 * globally). Reporting 0 makes Lucene's "whole range matches" fast path ({@code docCount == span})
 * never fire, forcing an inner per-doc value check for every candidate page — exactly what nested
 * needs, since a page's doc range also spans roots and other-path children that must not be blindly
 * matched. See {@code ParquetDocValuesLeafReader#nestedFirstDocIdOf}.
 *
 * <p>Pages with unknown min/max carry the sentinel ({@code Long.MIN_VALUE}, {@code Long.MAX_VALUE})
 * from the native page-index load, so they intersect every query range and are never wrongly skipped.
 *
 * <p>Only served for integer-shaped columns (long/int/date/boolean): their raw-bits value
 * order matches numeric order. Float/double doc values are raw IEEE-754 bits whose order
 * diverges for negatives, so the producer declines to build a skipper for them (see
 * {@link ParquetDocValuesProducer#getSkipper}).
 */
public final class ParquetDocValuesSkipper extends DocValuesSkipper {

    private static final Logger logger = LogManager.getLogger(ParquetDocValuesSkipper.class);

    /**
     * [SKIPPER-VERIFY] instrumentation (kept intentionally): a monotonic per-instance id so a per-query
     * log slice can attribute page advances to the specific segment skipper that produced them.
     */
    private static final java.util.concurrent.atomic.AtomicInteger INSTANCE_SEQ = new java.util.concurrent.atomic.AtomicInteger();

    private final int instanceId = INSTANCE_SEQ.incrementAndGet();

    private final ColumnPageIndex pageIndex;
    private final int maxDoc;
    private final long globalMin;
    private final long globalMax;
    private final int globalDocCount;

    /** True in nested mode: the docId axis is block-join child-docId space, not Parquet rows. */
    private final boolean nested;

    /**
     * Nested mode only ({@code null} for flat): lazily supplies {@link #firstDocIdOf} on the first
     * {@link #advance(int)}. Deferring the build keeps {@link #minValue()}/{@link #maxValue()} — which
     * Lucene's {@code SortedNumericDocValuesRangeQuery.rewrite()} probes for a whole-range short-circuit —
     * answerable from per-page stats alone, so a query that rewrites to MatchNoDocs never triggers the
     * O(totalRows) parent-bitset walk that builds the tiling. See {@code ParquetDocValuesLeafReader#nestedFirstDocIdOf}.
     */
    private final IntArrayLoader firstDocIdOfLoader;

    /**
     * Nested mode only ({@code null} for flat, and {@code null} until the first {@link #advance(int)}
     * materializes it via {@link #firstDocIdOfLoader}): the child-docId at which each page's first row's
     * block-join block begins, length {@code pageCount + 1}. {@code firstDocIdOf[0] == 0},
     * {@code firstDocIdOf[pageCount] == maxDoc}, strictly ascending; page {@code p} owns doc range
     * {@code [firstDocIdOf[p], firstDocIdOf[p+1]-1]}. Built by the leaf reader from the ROOT block-join
     * bitset (see {@code ParquetDocValuesLeafReader#nestedFirstDocIdOf}).
     */
    private int[] firstDocIdOf;

    /** Current page index, -1 before the first advance, pageCount when exhausted. */
    private int page = -1;

    public ParquetDocValuesSkipper(ColumnPageIndex pageIndex, int maxDoc) {
        this(pageIndex, maxDoc, null);
    }

    /**
     * @param firstDocIdOfLoader nested-mode loader that lazily builds the page→first-child-docId tiling
     *                           (length {@code pageCount + 1}) on the first {@link #advance(int)}, or
     *                           {@code null} for a flat column where docId == row. Passing a loader rather
     *                           than a built array is what defers the O(totalRows) parent-bitset walk past
     *                           the rewrite phase — see {@link #firstDocIdOfLoader}.
     */
    public ParquetDocValuesSkipper(ColumnPageIndex pageIndex, int maxDoc, IntArrayLoader firstDocIdOfLoader) {
        this.pageIndex = pageIndex;
        this.maxDoc = maxDoc;
        this.firstDocIdOfLoader = firstDocIdOfLoader;
        this.nested = firstDocIdOfLoader != null;
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        long withValue = 0;
        boolean hasComparableValue = false;
        for (int p = 0; p < pageIndex.pageCount(); p++) {
            // Flat: skip provably all-null pages (exact per-page null counts). Nested: null counts are in
            // element units while numRowsOf is in row units, so isAllNulls is unreliable — include every
            // page, yielding a safe-wide global that never wrongly excludes the field. Note this reads only
            // per-page stats (already loaded), never the firstDocIdOf tiling — so construction stays cheap.
            boolean include = nested || pageIndex.isAllNulls(p) == false;
            if (include) {
                min = Math.min(min, pageIndex.minOf(p));
                max = Math.max(max, pageIndex.maxOf(p));
                hasComparableValue = true;
            }
            withValue += pageDocCount(pageIndex, p);
        }
        this.globalMin = hasComparableValue ? min : Long.MIN_VALUE;
        this.globalMax = hasComparableValue ? max : Long.MAX_VALUE;
        this.globalDocCount = (int) withValue;
        if (logger.isInfoEnabled()) {
            StringBuilder perPage = new StringBuilder();
            for (int p = 0; p < pageIndex.pageCount(); p++) {
                perPage.append(p == 0 ? "" : " ")
                    .append("p")
                    .append(p)
                    .append(":[")
                    .append(pageIndex.minOf(p))
                    .append(",")
                    .append(pageIndex.maxOf(p))
                    .append("]rows")
                    .append(pageIndex.firstRowOf(p))
                    .append("+")
                    .append(pageIndex.numRowsOf(p));
            }
            // firstDocIdOf is built lazily on the first advance() in nested mode, so it is not yet available.
            logger.info(
                "[SKIPPER-VERIFY] create #{} nested={} pageCount={} gmin={} gmax={} perPage=({}) firstDocIdOf=deferred",
                instanceId,
                nested(),
                pageIndex.pageCount(),
                globalMin,
                globalMax,
                perPage
            );
        }
    }

    /**
     * Lazily supplies the nested {@code firstDocIdOf} tiling on the first {@link #advance(int)}. Kept as a
     * loader (not a pre-built array) so constructing the skipper — and answering {@link #minValue()}/
     * {@link #maxValue()} during Lucene's rewrite — never forces the O(totalRows) parent-bitset walk.
     */
    @FunctionalInterface
    public interface IntArrayLoader {
        int[] load() throws IOException;
    }

    /** True in nested mode: the docId axis is block-join child-docId space, not Parquet rows. */
    private boolean nested() {
        return nested;
    }

    /**
     * Documents with a value in page {@code p}. When the page's null count is unknown (-1) this
     * UNDER-claims (0): consumers use docCount for density checks (all-docs-have-values fast
     * paths), where overclaiming would produce wrong results and underclaiming merely disables
     * an optimization.
     */
    private static long pageDocCount(ColumnPageIndex pageIndex, int p) {
        long nulls = pageIndex.nullCountOf(p);
        return nulls < 0 ? 0 : pageIndex.numRowsOf(p) - nulls;
    }

    @Override
    public void advance(int target) throws IOException {
        // Nested: materialize the page→child-docId tiling on first use. Deferring to scan time (here) is
        // Fix A — Lucene's rewrite phase only probes minValue()/maxValue() (answered from per-page stats),
        // so a query that rewrites to MatchNoDocs never triggers the O(totalRows) parent-bitset walk. The
        // loader itself is segment-cached, so this build runs at most once per segment. Flat: no-op.
        if (nested && firstDocIdOf == null) {
            firstDocIdOf = firstDocIdOfLoader.load();
        }
        if (target >= maxDoc) {
            page = pageIndex.pageCount();
        } else if (nested) {
            page = pageForDocId(target);
        } else {
            page = pageIndex.pageForRow(target);
        }
        // [SKIPPER-VERIFY] runtime proof Lucene's range iterator is actively consulting this skipper.
        if (logger.isInfoEnabled()) {
            if (exhausted()) {
                logger.info("[SKIPPER-VERIFY] adv #{} nested={} target={} -> EXHAUSTED", instanceId, nested(), target);
            } else {
                logger.info(
                    "[SKIPPER-VERIFY] adv #{} nested={} target={} -> page={} docRange=[{},{}] valRange=[{},{}] docCount={}",
                    instanceId,
                    nested(),
                    target,
                    page,
                    minDocID(0),
                    maxDocID(0),
                    minValue(0),
                    maxValue(0),
                    docCount(0)
                );
            }
        }
    }

    private boolean exhausted() {
        return page >= pageIndex.pageCount();
    }

    @Override
    public int numLevels() {
        return 1;
    }

    @Override
    public int minDocID(int level) {
        if (page < 0) {
            return -1;
        }
        if (exhausted()) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        return nested() ? firstDocIdOf[page] : (int) pageIndex.firstRowOf(page);
    }

    @Override
    public int maxDocID(int level) {
        if (page < 0) {
            return -1;
        }
        if (exhausted()) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        return nested() ? firstDocIdOf[page + 1] - 1 : (int) (pageIndex.firstRowOf(page) + pageIndex.numRowsOf(page) - 1);
    }

    @Override
    public long minValue(int level) {
        return pageIndex.minOf(page);
    }

    @Override
    public long maxValue(int level) {
        return pageIndex.maxOf(page);
    }

    @Override
    public int docCount(int level) {
        // Nested: per-page null counts mix element/row units, so report no density and force Lucene's
        // inner per-doc value check (never the blind "whole range matches" fast path). See class javadoc.
        // TODO(nested-doccount): deliberate perf trade-off, NOT a bug. Returning 0 disables Lucene's
        // unsound bulk-accept ("YES") fast path in block-join space, where a page's doc range spans
        // child+ROOT+sibling docs. Revisit later only as an optimization (nested-aware bulk-accept), not
        // a correctness fix. Full rationale + candidate fixes: nested-doccount-zero-divergence.md.
        return nested() ? 0 : (int) pageDocCount(pageIndex, page);
    }

    @Override
    public long minValue() {
        return globalMin;
    }

    @Override
    public long maxValue() {
        return globalMax;
    }

    @Override
    public int docCount() {
        // Nested: no reliable field-wide value density (see docCount(int)); 0 matches the prior
        // null-skipper behavior for FieldExistsQuery.count(), so this is not a regression.
        // TODO(nested-doccount): see docCount(int) above and nested-doccount-zero-divergence.md — revisit
        // only as a perf optimization, not a correctness fix.
        return nested() ? 0 : globalDocCount;
    }

    /**
     * Binary search over the nested {@code firstDocIdOf} tiling: the page owning child {@code docId},
     * i.e. the highest page {@code p} with {@code firstDocIdOf[p] <= docId}. Callers pass
     * {@code 0 <= docId < maxDoc}, so the result is always a real page in {@code [0, pageCount-1]}.
     */
    private int pageForDocId(int docId) {
        int lo = 0;
        int hi = firstDocIdOf.length - 2; // last real page index (pageCount-1); [pageCount] is the maxDoc sentinel
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (firstDocIdOf[mid] <= docId) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return hi;
    }
}

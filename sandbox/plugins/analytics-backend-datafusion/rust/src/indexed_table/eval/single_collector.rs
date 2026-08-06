/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Single-collector evaluator — one backend collector plus DataFusion for
//! residual predicates.
//!
//! When the filter has exactly one `index_filter(...)` call AND'd with
//! (possibly zero, one, or many) parquet-native predicates, this evaluator
//! runs. Per RG:
//!
//! 1. Call the single collector → bitset.
//! 2. Apply page pruning (AND/OR mode depending on how the query combined them).
//! 3. Hand the bitset offsets to `IndexedStream` as a RowSelection.
//! 4. `on_batch_mask` returns `None` — DataFusion's
//!    `with_predicate(residual).with_pushdown_filters(true)` applies the
//!    residual predicates during decode, so indices stay aligned and no
//!    post-filtering is needed.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

use datafusion::arrow::array::{Array, AsArray, BooleanArray};
use datafusion::arrow::record_batch::RecordBatch;
use native_bridge_common::log_debug;
use roaring::RoaringBitmap;

use super::{PrefetchedRg, RowGroupBitsetSource};
use crate::indexed_table::ffm_callbacks::{create_provider, FfmSegmentCollector, ProviderHandle};
use crate::indexed_table::index::RowGroupDocsCollector;
use crate::indexed_table::page_pruner::{PagePruneMetrics, PagePruner, StatsPruneTree};
use crate::indexed_table::row_selection::{
    bitmap_to_packed_bits, packed_bits_to_boolean_array, row_selection_to_bitmap, PositionMap,
};
use crate::udf::nested_any_match_expr::evaluate_nested_with_lucene;
use datafusion::parquet::file::metadata::ParquetMetaData;
use datafusion::physical_optimizer::pruning::PruningPredicate;
use std::time::Instant;

/// Re-exported from parent module for backward compatibility.
pub use super::CollectorCallStrategy;
use crate::indexed_table::stream::RowGroupInfo;

/// TODO(phase-99): hardcoded selectivity threshold for opportunistic peer consultation.
/// Replaced by a cluster setting plumbed through `WireConfigSnapshot` and
/// `DatafusionQueryConfig` in the very last phase, after Phase 7 OR/NOT support and
/// everything else. Until then, performance-delegated leaves consult the peer when DF
/// page-pruning kept more than 5% of an RG.
const HARDCODED_SELECTIVITY_THRESHOLD: f64 = 0.05;

/// Builds delegated-backend collectors for performance-delegated leaves. Production impl
/// wraps `FfmSegmentCollector::create` (Java/Lucene round-trip); fuzz tests inject a
/// mock that replays a pre-computed bitset without an FFM call.
///
/// `context_id` is the per-query identifier passed through to every FFM upcall so Java
/// can route each callback to the correct per-query handle and tracker.
///
/// TODO: extend this factory to also build the *correctness* collector currently passed
/// in pre-built by `indexed_executor.rs`. Today delegated-backend (perf-delegated)
/// collectors are built inside this evaluator while correctness collectors are built
/// upstream — that asymmetry should go once we have more than one delegated backend
/// (DSL, vector, etc.) and the executor wants a single place to plug them in.
pub trait DelegatedBackendCollectorFactory: Send + Sync + std::fmt::Debug {
    fn create(
        &self,
        context_id: i64,
        provider_key: i32,
        writer_generation: i64,
        doc_min: i32,
        doc_max: i32,
    ) -> Result<Arc<dyn RowGroupDocsCollector>, String>;
}

/// Production factory: delegates to `FfmSegmentCollector::create`, which round-trips
/// to Java via FFM to build a Lucene-backed collector.
#[derive(Debug)]
pub struct FfmDelegatedBackendCollectorFactory;

impl DelegatedBackendCollectorFactory for FfmDelegatedBackendCollectorFactory {
    fn create(
        &self,
        context_id: i64,
        provider_key: i32,
        writer_generation: i64,
        doc_min: i32,
        doc_max: i32,
    ) -> Result<Arc<dyn RowGroupDocsCollector>, String> {
        let collector = FfmSegmentCollector::create(
            context_id,
            provider_key,
            writer_generation,
            doc_min,
            doc_max,
        )?;
        Ok(Arc::new(collector) as Arc<dyn RowGroupDocsCollector>)
    }
}

/// Per-RG state the evaluator keeps for refinement. In row-granular
/// mode parquet narrowed fully via `with_predicate` + `RowSelection`
/// and nothing is needed here. In block-granular mode we need the
/// Collector candidate bitmap to build a post-decode mask.
///
/// `mask_buffer` is the candidate bitmap in Arrow's native LSB-first bit
/// layout, wrapped as a refcounted `Buffer`. Sharing an `Arc<Buffer>` lets
/// `on_batch_mask` and `build_mask` build zero-copy `BooleanBuffer`
/// views via `BooleanBuffer::new(buf.clone(), bit_offset, bit_len)`.
/// Length of the underlying buffer covers `mask_len` bits (= rg_num_rows).
struct SingleCollectorState {
    candidates: RoaringBitmap,
    mask_buffer: datafusion::arrow::buffer::Buffer,
    mask_len: usize,
    /// Number of rows in this RG (== `mask_len`). Kept explicitly so the child-grain split's
    /// `on_batch_mask` can size the per-RG `child_base` vector without re-deriving it.
    rg_num_rows: usize,
    /// RG's first row in absolute (segment-relative) doc space. The child collect scans the parent-row
    /// window `[rg_first_row, rg_first_row + rg_num_rows)`. `None` when no child split is active.
    rg_first_row: i64,
}

/// One Lucene-delegated keyword clause of a child-grain nested split. `clause_idx` is the position
/// referenced by the residual JSON's `{"lucene": clause_idx}` node; `annotation_id` keys the lazily-created
/// peer provider (a child-scoped Lucene query — see `NestedAnyMatchChildSerializer`).
#[derive(Debug, Clone)]
pub struct ChildClause {
    pub clause_idx: usize,
    pub annotation_id: i32,
}

/// State for the child-grain nested-predicate split, attached to a `SingleCollectorEvaluator` when the
/// residual is a `nested_any_match_expr` whose keyword conjunct(s) were routed to Lucene at element grain.
///
/// When present, `on_batch_mask` evaluates the nested predicate against the decoded `LIST<STRUCT>` column
/// with each `{"lucene": i}` node's per-element verdict supplied by clause `i`'s Lucene child collect —
/// intersecting the keyword (Lucene) and range/other (DataFusion) clauses at the SAME element before the
/// ∃ roll-up to parents. `None` (the common case) → the evaluator behaves exactly as before.
#[derive(Debug, Clone)]
pub struct ChildSplitState {
    /// NAME of the `LIST<STRUCT>` nested-array column the predicate ranges over. Resolved to the delivered
    /// batch's column position by NAME at eval time — the `Column` index carried in the physical plan is a
    /// FULL-TABLE-schema index, which does not match a projected batch's column order (same reason
    /// `remap_expr_to_batch` remaps residual columns by name).
    pub array_col_name: String,
    /// The `nested_any_match_expr` JSON (with `{"lucene": i}` holes, each carrying a `"fallback"`).
    pub expr_json: String,
    /// One entry per Lucene-delegated clause, sorted ascending by `clause_idx` so `clause_bits[i]`
    /// aligns with the JSON's `{"lucene": i}` reference.
    pub clauses: Vec<ChildClause>,
    /// Lazy per-child-clause provider locks, keyed by `annotation_id` (one `OnceLock` each). Separate from
    /// the evaluator's parent-grain `performance_provider_locks` so a child-scoped query is NEVER consulted
    /// at parent grain in `prefetch_rg`. Query-scoped (shared across per-(segment×chunk) evaluators via
    /// `Arc::clone`) so each child provider is created once per (query × annotation_id).
    pub provider_locks: Arc<HashMap<i32, Arc<OnceLock<ProviderHandle>>>>,
}

/// Evaluator holding one collector and applying per-RG page pruning.
///
/// Always AND-intersects the collector bitmap with page pruning. The
/// `BitsetMode::Or` branch that previously existed was never emitted by
/// the classifier (reserved for a future `OR(Collector, predicates)`
/// extension) and has been removed; an OR-between-Collector-and-predicates
/// shape routes to the multi-filter tree path today.
pub struct SingleCollectorEvaluator {
    /// Always-call collector for correctness-delegated predicates. `None` when
    /// the query has only performance-delegated leaves (no peer call required
    /// upfront — see `performance_provider_locks`).
    collector: Option<Arc<dyn RowGroupDocsCollector>>,
    page_pruner: Arc<PagePruner>,
    /// Residual pruning predicate: the non-Collector portion of the
    /// top-level AND, translated to a `PruningPredicate`. `None` means
    /// no residual predicate applies (nothing to prune with).
    pruning_predicate: Option<Arc<PruningPredicate>>,
    /// Raw residual expression (non-Collector children of the top-level
    /// AND, converted to a single `PhysicalExpr`).
    ///
    /// Used in two modes:
    ///
    /// - **Row-granular** (`min_skip_run = 1`): the same expression is
    ///   stashed on `IndexedTableConfig.pushdown_predicate` and handed
    ///   to parquet's `with_predicate` for decode-time filtering.
    ///   Combined with the Collector-bitmap `RowSelection`, parquet
    ///   delivers exact `Collector ∧ residual` rows. `on_batch_mask`
    ///   returns `None` (nothing left to do).
    ///
    /// - **Block-granular** (`min_skip_run > 1`): pushdown is OFF
    ///   (alignment risk with coalesced selection). `on_batch_mask`
    ///   evaluates this expression against the decoded batch and
    ///   AND-combines with the Collector bitmap mask to produce the
    ///   exact result.
    residual_expr: Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>,
    /// Counters recorded by `page_pruner.prune_rg`. Built from the
    /// stream's `PartitionMetrics` at evaluator construction.
    page_prune_metrics: Option<PagePruneMetrics>,
    /// Incremented once per `prefetch_rg` call (once per RG) — the
    /// Collector path always performs one FFM round-trip to Java.
    ffm_collector_calls: Option<datafusion::physical_plan::metrics::Count>,
    call_strategy: CollectorCallStrategy,
    /// Lazy `ProviderHandle` cache, one per performance-delegated annotation_id.
    /// Empty when the query has no performance-delegated leaves. Populated by
    /// the factory at query setup; lookups + `OnceLock` init happen ONLY when
    /// `should_consult_lucene` decides DF's own pruning wasn't selective enough
    /// for an RG. Drop releases the Lucene Weight via `releaseProvider`.
    ///
    /// The HashMap is **query-scoped** (shared across all per-(segment×chunk)
    /// evaluators of a single query via `Arc::clone`), so threads racing to fill
    /// a slot do so once per (query × annotation_id) — not per chunk.
    performance_provider_locks: Arc<HashMap<i32, Arc<OnceLock<ProviderHandle>>>>,
    /// Writer generation identifying the segment this evaluator was bound to at
    /// factory time. Captured so `prefetch_rg` can build a per-call
    /// `FfmSegmentCollector` lazily without re-deriving the segment from
    /// `RowGroupInfo` (which doesn't carry it).
    writer_generation: i64,
    /// Builds the per-RG delegated-backend collector when the gate fires. Production
    /// wires `FfmDelegatedBackendCollectorFactory`; fuzz tests inject a mock that
    /// replays a pre-computed bitset without an FFM call.
    delegated_backend_collector_factory: Arc<dyn DelegatedBackendCollectorFactory>,
    /// Per-query context identifier passed through every FFM upcall so Java can route
    /// each callback to the correct per-query `FilterDelegationHandle` and tracker.
    context_id: i64,
    /// Bloom filter pruning config. None = disabled.
    bloom_config: Option<BloomConfig>,
    /// Precomputed per-RG/subtree match status from RG-level column stats.
    stats_prune_tree: Option<Arc<StatsPruneTree>>,
    /// Reverse map: absolute RG index → position in `rg_can_match` vectors.
    rg_index_to_pos: HashMap<usize, usize>,
    /// Child-grain nested split. `Some` when the residual is a `nested_any_match_expr` with keyword
    /// clause(s) routed to Lucene at element grain; the child clauses' peer providers live in
    /// `performance_provider_locks` (keyed by their annotation_id) and are consulted at CHILD grain in
    /// `on_batch_mask`, NOT at parent grain in `prefetch_rg`. `None` → non-nested behavior, unchanged.
    child_split: Option<ChildSplitState>,
}

/// Resources needed for per-RG bloom filter pruning.
pub struct BloomConfig {
    pub store: Arc<dyn object_store::ObjectStore>,
    pub object_path: object_store::path::Path,
    pub metadata: Arc<ParquetMetaData>,
    pub arrow_schema: Arc<datafusion::arrow::datatypes::Schema>,
    pub io_handle: tokio::runtime::Handle,
    pub rg_bloom_pruned: Option<datafusion::physical_plan::metrics::Count>,
    pub bloom_filter_eval_time: Option<datafusion::physical_plan::metrics::Time>,
}

impl SingleCollectorEvaluator {
    pub fn new(
        collector: Option<Arc<dyn RowGroupDocsCollector>>,
        page_pruner: Arc<PagePruner>,
        pruning_predicate: Option<Arc<PruningPredicate>>,
        residual_expr: Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>,
        page_prune_metrics: Option<PagePruneMetrics>,
        ffm_collector_calls: Option<datafusion::physical_plan::metrics::Count>,
        call_strategy: CollectorCallStrategy,
        performance_provider_locks: Arc<HashMap<i32, Arc<OnceLock<ProviderHandle>>>>,
        writer_generation: i64,
        delegated_backend_collector_factory: Arc<dyn DelegatedBackendCollectorFactory>,
        context_id: i64,
        bloom_config: Option<BloomConfig>,
        stats_prune_tree: Option<Arc<StatsPruneTree>>,
        rg_index_to_pos: HashMap<usize, usize>,
        child_split: Option<ChildSplitState>,
    ) -> Self {
        Self {
            collector,
            page_pruner,
            pruning_predicate,
            residual_expr,
            page_prune_metrics,
            ffm_collector_calls,
            call_strategy,
            performance_provider_locks,
            writer_generation,
            delegated_backend_collector_factory,
            context_id,
            bloom_config,
            stats_prune_tree,
            rg_index_to_pos,
            child_split,
        }
    }
}

/// Per-RG decision: should we consult the peer backend?
///
/// Pure function. Inputs: post-page-prune surviving ranges, RG row count, and the
/// configured selectivity threshold. The function says "consult" when DF kept
/// MORE than `threshold` of the RG (page pruning wasn't selective enough — peer
/// might narrow further); "skip" when DF already squeezed it below threshold
/// (peer call would be wasted work).
///
/// `page_ranges == None` means there's no usable PruningPredicate for the
/// residual at all (e.g. text-column predicate with no parquet stats) — DF
/// can't help, so consult.
fn should_consult_lucene(
    page_ranges: &Option<Vec<(i32, i32)>>,
    rg: &RowGroupInfo,
    threshold: f64,
) -> bool {
    let surviving_rows = match page_ranges {
        None => rg.num_rows as i64,
        Some(ranges) => ranges.iter().map(|(lo, hi)| (hi - lo) as i64).sum::<i64>(),
    };
    if rg.num_rows == 0 {
        return false;
    }
    let surviving_fraction = surviving_rows as f64 / rg.num_rows as f64;
    surviving_fraction > threshold
}

impl SingleCollectorEvaluator {
    /// Evaluate a child-grain nested split against one delivered batch. Returns the per-parent-row
    /// BooleanArray (length `batch_len`) that the `nested_any_match_expr` predicate produces once each
    /// `{"lucene": i}` node is fed clause `i`'s per-element Lucene verdict.
    ///
    /// The coordinate system is owned HERE, not by Java: `child_base[p]` is the batch-flattened element
    /// index at which parent RG-row `p`'s elements begin (from the decoded LIST `value_offsets`), so the
    /// child bitset Java returns is directly indexable by the residual UDF's element index — correct under
    /// every `PositionMap` (Identity / Bitmap / Runs) and multi-batch RG split. A parent row not delivered
    /// in this batch keeps `child_base[p] == -1`, and Java skips any child whose row maps to `-1`.
    /// Build the correctness-Collector candidate mask over the delivered rows (the same mask the non-child
    /// `on_batch_mask` path AND-combines with its residual). Bit `i` is set iff delivered row `i`'s RG
    /// position is a candidate in `state.mask_buffer`. Shared by the child-split and non-child paths so a
    /// correctness Collector's row narrowing (e.g. a text `match()` parent predicate delegated to Lucene)
    /// is honored under the child split too.
    fn collector_mask_for_batch(
        &self,
        state: &SingleCollectorState,
        position_map: &PositionMap,
        batch_offset: usize,
        batch_len: usize,
    ) -> Result<BooleanArray, String> {
        Ok(match position_map {
            PositionMap::Identity { .. } => {
                let bb = datafusion::arrow::buffer::BooleanBuffer::new(
                    state.mask_buffer.clone(),
                    batch_offset,
                    batch_len,
                );
                BooleanArray::new(bb, None)
            }
            PositionMap::Bitmap { .. } => {
                BooleanArray::new(datafusion::arrow::buffer::BooleanBuffer::new_set(batch_len), None)
            }
            PositionMap::Runs { .. } => {
                let words = batch_len.div_ceil(64);
                let mut out = vec![0u64; words];
                let src_bytes = state.mask_buffer.as_slice();
                for i in 0..batch_len {
                    let delivered_idx = batch_offset + i;
                    let rg_pos = position_map.rg_position(delivered_idx).ok_or_else(|| {
                        format!("SingleCollectorEvaluator: delivered_idx {} out of range", delivered_idx)
                    })?;
                    let hit =
                        rg_pos < state.mask_len && (src_bytes[rg_pos >> 3] >> (rg_pos & 7)) & 1 == 1;
                    if hit {
                        out[i >> 6] |= 1u64 << (i & 63);
                    }
                }
                packed_bits_to_boolean_array(out, batch_len)
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_child_split(
        &self,
        child_split: &ChildSplitState,
        state: &SingleCollectorState,
        position_map: &PositionMap,
        batch_offset: usize,
        batch_len: usize,
        batch: &RecordBatch,
    ) -> Result<BooleanArray, String> {
        // Resolve the LIST<STRUCT> column by NAME against the DELIVERED batch's schema. The plan's Column
        // index is a full-table-schema index and does not match a projected batch's column order.
        let array_idx = batch.schema().index_of(&child_split.array_col_name).map_err(|_| {
            format!(
                "child-split: array column '{}' not found in batch schema {:?}",
                child_split.array_col_name,
                batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>()
            )
        })?;
        let array = batch.column(array_idx).clone();
        let list = array.as_list_opt::<i32>().ok_or_else(|| {
            format!(
                "child-split: column '{}' is not a List, got {:?}",
                child_split.array_col_name,
                array.data_type()
            )
        })?;
        let value_offsets = list.value_offsets();
        // value_offsets has batch_len+1 entries; the last is the flattened element count for this batch.
        let total_children = *value_offsets.last().map(|&o| o).get_or_insert(0) as usize;

        // Build child_base over the WHOLE RG (rows not in this batch stay -1). value_offsets[d] is the
        // element start of delivered row d; map d → its RG position and record the base there.
        let mut child_base = vec![-1i32; state.rg_num_rows];
        for d in 0..batch_len {
            let p = position_map.rg_position(batch_offset + d).ok_or_else(|| {
                format!(
                    "child-split: delivered row {} (batch_offset {}) out of PositionMap range",
                    d, batch_offset
                )
            })?;
            if p >= state.rg_num_rows {
                return Err(format!(
                    "child-split: RG position {} >= rg_num_rows {}",
                    p, state.rg_num_rows
                ));
            }
            child_base[p] = value_offsets[d];
        }

        // Parent-row window for the child scan: the whole RG in absolute doc space.
        let min_doc = state.rg_first_row as i32;
        let max_doc = (state.rg_first_row + state.rg_num_rows as i64) as i32;

        // For each Lucene-delegated clause (ascending clause_idx), collect its per-element bits. The
        // provider is created lazily and shared across batches/RGs of the same query via the query-scoped
        // OnceLock map; the collector is per-(segment, RG window). clause_bits[i] aligns with clause_idx i
        // because `clauses` is sorted ascending — the classifier guarantees a dense 0..N clause set.
        let mut clause_bits: Vec<BooleanArray> = Vec::with_capacity(child_split.clauses.len());
        for (expected_idx, clause) in child_split.clauses.iter().enumerate() {
            if clause.clause_idx != expected_idx {
                return Err(format!(
                    "child-split: clauses not densely sorted — expected clause_idx {}, got {}",
                    expected_idx, clause.clause_idx
                ));
            }
            let lock = child_split
                .provider_locks
                .get(&clause.annotation_id)
                .ok_or_else(|| {
                    format!(
                        "child-split: no provider lock for child clause annotation_id={}",
                        clause.annotation_id
                    )
                })?;
            let context_id = self.context_id;
            let annotation_id = clause.annotation_id;
            let provider = lock.get_or_init(|| {
                create_provider(context_id, annotation_id).expect("create_provider FFM upcall failed")
            });
            let collector = self
                .delegated_backend_collector_factory
                .create(context_id, provider.key(), self.writer_generation, min_doc, max_doc)
                .map_err(|e| {
                    format!(
                        "child-split: collector create (annotation_id={}, provider={}, writer_generation={}): {}",
                        annotation_id,
                        provider.key(),
                        self.writer_generation,
                        e
                    )
                })?;
            let words =
                collector.collect_child_docs_batch(min_doc, max_doc, &child_base, total_children)?;
            if let Some(ref c) = self.ffm_collector_calls {
                c.add(1);
            }
            clause_bits.push(packed_bits_to_boolean_array(words, total_children));
        }

        // Evaluate the nested predicate with the Lucene per-element verdicts wired into the {"lucene":i}
        // nodes. The ∃-over-elements roll-up (and element correlation) lives inside the UDF.
        evaluate_nested_with_lucene(&array, &child_split.expr_json, &clause_bits)
            .map_err(|e| format!("child-split: evaluate_nested_with_lucene: {}", e))
    }
}

impl RowGroupBitsetSource for SingleCollectorEvaluator {
    fn prefetch_rg(
        &self,
        rg: &RowGroupInfo,
        min_doc: i32,
        max_doc: i32,
    ) -> Result<Option<PrefetchedRg>, String> {
        let t = Instant::now();

        // RG-level early-exit: precomputed from column stats at construction.
        if let Some(ref spt) = self.stats_prune_tree {
            if let Some(&pos) = self.rg_index_to_pos.get(&rg.index) {
                if let Some(&false) = spt.rg_can_match.get(pos) {
                    native_bridge_common::log_debug!(
                        "SingleCollector: skipping RG {} — pruned by RG-level stats",
                        rg.index
                    );
                    return Ok(None);
                }
            }
        }

        // Page-prune to discover which row ranges survive.
        let page_ranges: Option<Vec<(i32, i32)>> = self.pruning_predicate.as_ref().and_then(|pp| {
            self.page_pruner
                .prune_rg(pp, rg.index, self.page_prune_metrics.as_ref())
                .map(|sel| {
                    let mut ranges = Vec::new();
                    let mut rg_pos: i64 = 0;
                    for s in sel.iter() {
                        if s.skip {
                            rg_pos += s.row_count as i64;
                        } else {
                            let abs_min = min_doc + rg_pos as i32;
                            let abs_max = min_doc + rg_pos as i32 + s.row_count as i32;
                            ranges.push((abs_min, abs_max));
                            rg_pos += s.row_count as i64;
                        }
                    }
                    ranges
                })
        });

        // All pages pruned by stats → skip bloom + collector entirely.
        if let Some(ref ranges) = page_ranges {
            if ranges.is_empty() {
                return Ok(None);
            }
        }

        // Bloom filter pruning: runs after page pruning (free) but before
        // the expensive FFM collector call. Uses the IO runtime handle from
        // the RuntimeManager to drive the async object-store read.
        if let (Some(bloom), Some(pp)) = (&self.bloom_config, &self.pruning_predicate) {
            let _timer = bloom.bloom_filter_eval_time.as_ref().map(|t| t.timer());
            let pruned =
                bloom
                    .io_handle
                    .block_on(crate::indexed_table::bloom_pruner::bloom_prune_rg(
                        &*bloom.store,
                        &bloom.object_path,
                        &bloom.metadata,
                        &bloom.arrow_schema,
                        rg.index,
                        pp.as_ref(),
                    ));
            if pruned {
                if let Some(ref c) = bloom.rg_bloom_pruned {
                    c.add(1);
                }
                return Ok(None);
            }
        }

        // Build candidates either from the always-call correctness collector OR, when
        // the query is performance-only (no Collector leaves), from the page-pruned
        // universe. Performance leaves are AND'd in below if the selectivity gate fires.
        let mut candidates = match self.collector.as_ref() {
            Some(collector) => {
                // Dispatch collector call strategy.
                let call_ranges: Vec<(i32, i32)> = match self.call_strategy {
                    CollectorCallStrategy::FullRange => vec![(min_doc, max_doc)],
                    CollectorCallStrategy::TightenOuterBounds => match &page_ranges {
                        Some(r) if r.is_empty() => return Ok(None),
                        Some(r) => vec![(r.first().unwrap().0, r.last().unwrap().1)],
                        None => vec![(min_doc, max_doc)],
                    },
                    CollectorCallStrategy::PageRangeSplit => match &page_ranges {
                        Some(r) if r.is_empty() => return Ok(None),
                        Some(r) => r.clone(),
                        None => vec![(min_doc, max_doc)],
                    },
                };

                // Call collector for each range, merge into one RG-relative bitmap.
                let mut bm = RoaringBitmap::new();
                for (r_min, r_max) in &call_ranges {
                    let bitset = collector
                        .collect_packed_u64_bitset(*r_min, *r_max)
                        .map_err(|e| {
                            format!(
                                "collector.collect_packed_u64_bitset(rg={}, [{}, {})): {}",
                                rg.index, r_min, r_max, e
                            )
                        })?;
                    if let Some(ref c) = self.ffm_collector_calls {
                        c.add(1);
                    }
                    let offset = (*r_min as i64 - rg.first_row) as u32;
                    let num_docs = (*r_max - *r_min) as u32;
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(bitset.as_ptr() as *const u8, bitset.len() * 8)
                    };
                    let mut chunk = RoaringBitmap::from_lsb0_bytes(offset, bytes);
                    let upper = offset.saturating_add(num_docs);
                    if upper < u32::MAX {
                        chunk.remove_range(upper..);
                    }
                    bm |= chunk;
                }

                // For FullRange and TightenOuterBounds, AND with page bitmap
                // to remove rows in dead pages that the collector scanned.
                if self.call_strategy != CollectorCallStrategy::PageRangeSplit {
                    if let Some(ref ranges) = page_ranges {
                        let mut allowed = RoaringBitmap::new();
                        for (r_min, r_max) in ranges {
                            let lo = (*r_min as i64 - rg.first_row) as u32;
                            let hi = (*r_max as i64 - rg.first_row) as u32;
                            allowed.insert_range(lo..hi);
                        }
                        bm &= allowed;
                    }
                }
                bm
            }
            None => {
                // Performance-only query. Seed candidates with the page-pruned universe
                // (or the full RG if no PruningPredicate). The opportunistic peer branch
                // below may narrow further; otherwise DF's pushdown filter handles the
                // residual at decode time.
                let mut bm = RoaringBitmap::new();
                match &page_ranges {
                    Some(r) if r.is_empty() => return Ok(None),
                    Some(r) => {
                        for (r_min, r_max) in r {
                            let lo = (*r_min as i64 - rg.first_row) as u32;
                            let hi = (*r_max as i64 - rg.first_row) as u32;
                            bm.insert_range(lo..hi);
                        }
                    }
                    None => {
                        bm.insert_range(0..rg.num_rows as u32);
                    }
                }
                bm
            }
        };

        // Opportunistic peer consultation for performance-delegated leaves. Only fires
        // when DF page-pruning kept more than the configured fraction of the RG —
        // skipping the FFM round-trip when DF was already selective. Lazy: lock the
        // map only if the gate fires; create the provider only once per query × leaf.
        // TODO(d3): consult ALL performance leaves whose gate fires and AND their
        // bitsets. Today we consult the first leaf only — sufficient for AND-only
        // single-call demo. Multi-leaf intersection is part of D3 follow-up.
        if !self.performance_provider_locks.is_empty()
            && should_consult_lucene(&page_ranges, rg, HARDCODED_SELECTIVITY_THRESHOLD)
        {
            // Pick the smallest annotation_id deterministically so logs/tests are stable.
            // Avoids the Vec/sort allocation in the common single-leaf case.
            let annotation_id = *self
                .performance_provider_locks
                .keys()
                .min()
                .expect("performance_provider_locks is non-empty (just checked)");
            // Per-RG debug log — `format!` runs unconditionally regardless of log level
            // (the level filter happens on the Java side). Commented out to avoid
            // per-RG allocation. Re-enable locally for debugging.
            // log_debug!(
            //     "[scf-rust] consulting peer for performance leaf rg={} writer_generation={} range=[{},{}) annotation_id={}",
            //     rg.index, self.writer_generation, min_doc, max_doc, annotation_id
            // );
            let lock = self
                .performance_provider_locks
                .get(&annotation_id)
                .expect("annotation_id was just pulled from the map's keys");
            let context_id = self.context_id;
            let mut just_initialized = false;
            let provider = lock.get_or_init(|| {
                just_initialized = true;
                create_provider(context_id, annotation_id)
                    .expect("create_provider FFM upcall failed")
            });
            if just_initialized {
                log_debug!(
                    "[scf-rust] lazy provider initialized context_id={} annotation_id={} provider_key={}",
                    context_id, annotation_id, provider.key()
                );
            }

            let collector = self
                .delegated_backend_collector_factory
                .create(context_id, provider.key(), self.writer_generation, min_doc, max_doc)
                .map_err(|e| {
                    format!(
                        "DelegatedBackendCollectorFactory::create(context_id={}, provider={}, writer_generation={}, doc_range=[{},{})): {}",
                        context_id,
                        provider.key(),
                        self.writer_generation,
                        min_doc,
                        max_doc,
                        e
                    )
                })?;
            let bitset = collector
                .collect_packed_u64_bitset(min_doc, max_doc)
                .map_err(|e| {
                    format!(
                        "delegated-backend collector.collect_packed_u64_bitset(rg={}, [{}, {})): {}",
                        rg.index, min_doc, max_doc, e
                    )
                })?;
            if let Some(ref c) = self.ffm_collector_calls {
                c.add(1);
            }
            let offset = (min_doc as i64 - rg.first_row) as u32;
            let num_docs = (max_doc - min_doc) as u32;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(bitset.as_ptr() as *const u8, bitset.len() * 8)
            };
            let mut peer_bm = RoaringBitmap::from_lsb0_bytes(offset, bytes);
            let upper = offset.saturating_add(num_docs);
            if upper < u32::MAX {
                peer_bm.remove_range(upper..);
            }
            // Per-RG debug log — see note above on `format!` cost. Re-enable locally for debugging.
            // let candidates_before = candidates.len();
            // let peer_card = peer_bm.len();
            candidates &= peer_bm;
            // log_debug!(
            //     "[scf-rust] peer bitset intersected rg={} writer_generation={} candidates_before={} peer_cardinality={} candidates_after={}",
            //     rg.index, self.writer_generation, candidates_before, peer_card, candidates.len()
            // );
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        // Materialise the final RG-relative bitmap as an Arrow `Buffer`
        // in Arrow's native LSB-first layout. This is the ONLY
        // representation the hot paths (`on_batch_mask`, `build_mask`)
        // need; they construct zero-copy `BooleanBuffer` views via
        // `BooleanBuffer::new(buf.clone(), bit_offset, bit_len)`.
        let mask_len = rg.num_rows as usize;
        let packed_bits = bitmap_to_packed_bits(&candidates, mask_len as u32);
        let mask_buffer = datafusion::arrow::buffer::Buffer::from_vec(packed_bits);
        Ok(Some(PrefetchedRg {
            candidates: candidates.clone(),
            eval_nanos: t.elapsed().as_nanos() as u64,
            context: Box::new(SingleCollectorState {
                candidates,
                mask_buffer: mask_buffer.clone(),
                mask_len,
                rg_num_rows: rg.num_rows as usize,
                rg_first_row: rg.first_row,
            }),
            mask_buffer: Some(mask_buffer),
        }))
    }

    fn on_batch_mask(
        &self,
        rg_state: &dyn std::any::Any,
        _rg_first_row: i64,
        position_map: &PositionMap,
        batch_offset: usize,
        batch_len: usize,
        batch: &RecordBatch,
    ) -> Result<Option<BooleanArray>, String> {
        // No child split AND no residual → no post-decode work; the stream's current_mask handles
        // Collector narrowing. Return before touching rg_state (some callers pass a placeholder state on
        // this no-work path).
        if self.child_split.is_none() && self.residual_expr.is_none() {
            return Ok(None);
        }

        let state = rg_state
            .downcast_ref::<SingleCollectorState>()
            .ok_or_else(|| {
                "SingleCollectorEvaluator: rg_state is not SingleCollectorState".to_string()
            })?;

        // Child-grain nested split: evaluate the nested predicate against the decoded LIST<STRUCT>
        // column, feeding each {"lucene": i} node its clause's per-element Lucene verdict. This is the
        // authoritative, element-correlated filter — it REPLACES the collector-mask ∧ residual combine
        // (the child peers were kept OUT of `residual_expr` by the classifier; any non-nested conjuncts
        // that remain in `residual_expr` are still AND'd in below).
        if let Some(ref child_split) = self.child_split {
            let nested_mask = self.evaluate_child_split(
                child_split,
                state,
                position_map,
                batch_offset,
                batch_len,
                batch,
            )?;
            // AND the correctness-Collector candidate mask: when a genuine correctness Collector coexists
            // (e.g. a text `match()` parent predicate delegated to Lucene, which is NOT in residual_expr),
            // its row narrowing lives only in state.mask_buffer and must still apply. For a performance-only
            // query the candidates are the page-pruned/full universe, so this AND is a safe no-op.
            let mut combined =
                self.collector_mask_for_batch(state, position_map, batch_offset, batch_len)?;
            combined = datafusion::arrow::compute::kernels::boolean::and_kleene(&combined, &nested_mask)
                .map_err(|e| format!("SingleCollectorEvaluator child-split: and_kleene(collector): {}", e))?;
            // AND any non-nested residual conjuncts (e.g. a parent-column predicate co-located in the
            // same top-level AND) still carried in residual_expr.
            if let Some(ref residual) = self.residual_expr {
                let residual_mask =
                    super::eval_helpers::evaluate_residual(residual, batch, batch_len)?;
                combined = datafusion::arrow::compute::kernels::boolean::and_kleene(&combined, &residual_mask)
                    .map_err(|e| format!("SingleCollectorEvaluator child-split: and_kleene(residual): {}", e))?;
            }
            return Ok(Some(combined));
        }

        // No residual → no post-decode work. Stream's current_mask
        // (if built) handles Collector narrowing.
        let Some(ref residual) = self.residual_expr else {
            return Ok(None);
        };

        // Build the Collector candidate mask over delivered rows via PositionMap (shared with the
        // child-split path). Zero allocation for Identity, at most one small packed Vec<u64> for Runs.
        let collector_mask =
            self.collector_mask_for_batch(state, position_map, batch_offset, batch_len)?;

        // Evaluate residual against the batch.
        let residual_mask = super::eval_helpers::evaluate_residual(residual, batch, batch_len)?;

        // AND with kleene semantics (NULL → exclude).
        let combined = datafusion::arrow::compute::kernels::boolean::and_kleene(
            &collector_mask,
            &residual_mask,
        )
        .map_err(|e| format!("SingleCollectorEvaluator: and_kleene: {}", e))?;
        Ok(Some(combined))
    }

    /// When we have a residual to apply in `on_batch_mask`, pushdown
    /// must be OFF in **block-granular mode** because we use
    /// `PositionMap` to look up RG positions over the full delivered
    /// rowset — pushdown would drop rows and misalign. In
    /// **row-granular mode** (`min_skip_run == 1`), pushdown is safe
    /// and desirable: parquet applies the residual in lockstep with
    /// decoding, `on_batch_mask` returns `None`, and output is
    /// exact. But the evaluator doesn't know min_skip_run — the
    /// stream does. The stream guards this via its
    /// `alignment_risk = min_skip_run != 1 && needs_row_mask()`
    /// check plus `forbid_parquet_pushdown`. We return `false` here
    /// and rely on `needs_row_mask = true` (default when residual is
    /// present) to trigger the stream's alignment guard in block
    /// mode; in row-granular mode that guard is inactive and
    /// pushdown proceeds.
    fn forbid_parquet_pushdown(&self) -> bool {
        false
    }

    /// Stream's `current_mask` construction consults this. When
    /// residual is set, we return `true` so the stream knows our
    /// `on_batch_mask` uses PositionMap (alignment risk) — this flag
    /// flips the stream's `alignment_risk` computation which
    /// suppresses pushdown in block-granular mode. In row-granular
    /// mode (min_skip_run == 1) the stream ignores this flag's
    /// pushdown impact and pushes anyway (which is what we want:
    /// parquet applies residual during decode of already-narrowed
    /// rowset, on_batch_mask returns None below).
    ///
    /// Without residual, we return `true` too — stream builds
    /// `current_mask` from Collector bitmap to narrow post-decode
    /// (legacy path for SingleCollector without a residual wasn't
    /// used in production but kept for defensive correctness).
    fn needs_row_mask(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::parquet::arrow::arrow_reader::ArrowReaderMetadata;
    use datafusion::parquet::arrow::arrow_reader::ArrowReaderOptions;
    use datafusion::parquet::arrow::ArrowWriter;
    use std::fmt;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// Stub collector: returns a pre-defined set of doc IDs, encoded into
    /// the bitset the trait contract requires.
    #[derive(Debug)]
    struct StubCollector {
        docs: Vec<i32>,
    }

    impl RowGroupDocsCollector for StubCollector {
        fn collect_packed_u64_bitset(
            &self,
            min_doc: i32,
            max_doc: i32,
        ) -> Result<Vec<u64>, String> {
            let span = (max_doc - min_doc) as usize;
            let mut bitset = vec![0u64; (span + 63) / 64];
            for &doc in &self.docs {
                if doc >= min_doc && doc < max_doc {
                    let idx = (doc - min_doc) as usize;
                    bitset[idx / 64] |= 1u64 << (idx % 64);
                }
            }
            Ok(bitset)
        }
    }

    /// Child-grain stub collector: given a set of matching `(rg_row, element_offset)` pairs (what a
    /// Lucene child-scoped query would match), it consults the caller-supplied `child_base` exactly as
    /// the real FFM collector does — setting bit `child_base[row - min_doc] + offset` — so the test
    /// exercises the REAL coordinate math (`child_base` built from `value_offsets` under a non-Identity
    /// PositionMap), not a shortcut. Rows whose `child_base` entry is `-1` (not in this batch) are skipped.
    #[derive(Debug)]
    struct ChildStubCollector {
        /// (rg_row, element_offset) pairs the child-scoped query matches.
        matches: Vec<(i32, i32)>,
    }

    impl RowGroupDocsCollector for ChildStubCollector {
        fn collect_packed_u64_bitset(&self, _min: i32, _max: i32) -> Result<Vec<u64>, String> {
            Err("ChildStubCollector is child-grain only".into())
        }
        fn collect_child_docs_batch(
            &self,
            min_doc: i32,
            max_doc: i32,
            child_base: &[i32],
            total_children: usize,
        ) -> Result<Vec<u64>, String> {
            let mut bits = vec![0u64; total_children.div_ceil(64)];
            for &(row, off) in &self.matches {
                if row < min_doc || row >= max_doc {
                    continue;
                }
                let base = child_base[(row - min_doc) as usize];
                if base < 0 {
                    continue; // row not delivered in this batch
                }
                let child_id = (base + off) as usize;
                if child_id < total_children {
                    bits[child_id / 64] |= 1u64 << (child_id % 64);
                }
            }
            Ok(bits)
        }
    }

    #[derive(Debug)]
    struct ChildStubFactory {
        matches: Vec<(i32, i32)>,
    }

    impl DelegatedBackendCollectorFactory for ChildStubFactory {
        fn create(
            &self,
            _context_id: i64,
            _provider_key: i32,
            _writer_generation: i64,
            _doc_min: i32,
            _doc_max: i32,
        ) -> Result<Arc<dyn RowGroupDocsCollector>, String> {
            Ok(Arc::new(ChildStubCollector {
                matches: self.matches.clone(),
            }) as Arc<dyn RowGroupDocsCollector>)
        }
    }

    fn minimal_page_pruner() -> Arc<PagePruner> {
        // Build a 1-row-group parquet with no filters — page pruner becomes a no-op
        // (filter_row_ids returns input, candidate_row_ids returns [first_row, first_row+num_rows)).
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = datafusion::arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(datafusion::arrow::array::Int32Array::from(
                vec![0i32; 8],
            ))],
        )
        .unwrap();
        let tmp = NamedTempFile::new().unwrap();
        {
            let mut writer =
                ArrowWriter::try_new(tmp.reopen().unwrap(), schema.clone(), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        let file = tmp.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index(true);
        let meta = ArrowReaderMetadata::load(&file, options).unwrap();
        let pruner = PagePruner::new(meta.schema(), meta.metadata().clone());
        Arc::new(pruner)
    }

    #[test]
    fn path_b_and_mode_collects_docs_and_returns_offsets() {
        let collector = Arc::new(StubCollector {
            docs: vec![0, 3, 7],
        }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );

        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        let prefetched = eval.prefetch_rg(&rg, 0, 8).unwrap().expect("has matches");
        let got: Vec<u32> = prefetched.candidates.iter().collect();
        assert_eq!(got, vec![0u32, 3, 7]);
    }

    #[test]
    fn on_batch_mask_returns_none_for_path_b() {
        let collector = Arc::new(StubCollector { docs: vec![0] }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = datafusion::arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![Arc::new(datafusion::arrow::array::Int32Array::from(vec![
                1, 2, 3,
            ]))],
        )
        .unwrap();
        // Empty position map is fine; SingleCollectorEvaluator ignores it.
        let pm = PositionMap::from_selection(
            &datafusion::parquet::arrow::arrow_reader::RowSelection::from(Vec::<
                datafusion::parquet::arrow::arrow_reader::RowSelector,
            >::new()),
        );
        assert!(eval
            .on_batch_mask(&(), 0, &pm, 0, 3, &batch)
            .unwrap()
            .is_none());
    }

    #[test]
    fn single_collector_needs_row_mask() {
        // SingleCollectorEvaluator returns None from on_batch_mask, so
        // IndexedStream must build current_mask from candidate offsets
        // (it's the only post-decode filter we have on this path).
        let collector = Arc::new(StubCollector { docs: vec![0] }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );
        assert!(eval.needs_row_mask());
    }

    #[test]
    fn empty_match_returns_none() {
        let collector = Arc::new(StubCollector { docs: vec![] }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );
        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        assert!(eval.prefetch_rg(&rg, 0, 8).unwrap().is_none());
    }

    #[test]
    fn empty_pruning_predicates_leave_collector_unchanged() {
        // With no pruning predicates, the evaluator is a pass-through for
        // the collector bitmap: every doc the collector returns remains a
        // candidate. (Contrast with the old BitsetMode::Or path, which
        // would have unioned with page-pruner-derived "anything-allowed"
        // row IDs — semantics that were never wired up in production.)
        let collector = Arc::new(StubCollector {
            docs: vec![0, 3, 7],
        }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );

        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        let prefetched = eval.prefetch_rg(&rg, 0, 8).unwrap().expect("has matches");
        let got: Vec<u32> = prefetched.candidates.iter().collect();
        assert_eq!(got, vec![0u32, 3, 7]);
    }

    #[test]
    fn stats_prune_tree_skips_rg_when_false() {
        let collector = Arc::new(StubCollector {
            docs: vec![0, 3, 7],
        }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let spt = StatsPruneTree {
            rg_can_match: vec![false],
            children: vec![],
        };
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            Some(Arc::new(spt)),
            HashMap::from([(0, 0)]),
            None,
        );
        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        assert!(eval.prefetch_rg(&rg, 0, 8).unwrap().is_none());
    }

    #[test]
    fn stats_prune_tree_allows_rg_when_true() {
        let collector = Arc::new(StubCollector {
            docs: vec![0, 3, 7],
        }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let spt = StatsPruneTree {
            rg_can_match: vec![true],
            children: vec![],
        };
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            Some(Arc::new(spt)),
            HashMap::from([(0, 0)]),
            None,
        );
        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        let prefetched = eval
            .prefetch_rg(&rg, 0, 8)
            .unwrap()
            .expect("should have matches");
        let got: Vec<u32> = prefetched.candidates.iter().collect();
        assert_eq!(got, vec![0u32, 3, 7]);
    }

    #[test]
    fn stats_prune_tree_none_does_not_prune() {
        let collector =
            Arc::new(StubCollector { docs: vec![1, 5] }) as Arc<dyn RowGroupDocsCollector>;
        let pruner = minimal_page_pruner();
        let eval = SingleCollectorEvaluator::new(
            Some(collector),
            pruner,
            None,
            None,
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()),
            0,
            Arc::new(FfmDelegatedBackendCollectorFactory),
            0,
            None,
            None,
            HashMap::new(),
            None,
        );
        let rg = RowGroupInfo {
            index: 0,
            first_row: 0,
            num_rows: 8,
        };
        let prefetched = eval
            .prefetch_rg(&rg, 0, 8)
            .unwrap()
            .expect("should have matches");
        let got: Vec<u32> = prefetched.candidates.iter().collect();
        assert_eq!(got, vec![1u32, 5]);
    }

    // ── Child-grain split coordinate math ─────────────────────────────

    use datafusion::arrow::array::{ArrayRef, Int64Array, ListArray, RecordBatch, StringArray, StructArray};
    use datafusion::arrow::buffer::OffsetBuffer;
    use datafusion::arrow::datatypes::Fields;

    /// One `LIST<STRUCT{author:Utf8, score:Int64}>` column wrapped in a RecordBatch named "comments".
    fn comments_batch(rows: &[Vec<(&str, i64)>]) -> RecordBatch {
        let mut authors: Vec<Option<String>> = Vec::new();
        let mut scores: Vec<Option<i64>> = Vec::new();
        let mut offsets: Vec<i32> = vec![0];
        let mut acc = 0i32;
        for row in rows {
            for (a, s) in row {
                authors.push(Some((*a).to_string()));
                scores.push(Some(*s));
            }
            acc += row.len() as i32;
            offsets.push(acc);
        }
        let sfields: Fields = Fields::from(vec![
            Field::new("author", DataType::Utf8, true),
            Field::new("score", DataType::Int64, true),
        ]);
        let struct_array = StructArray::new(
            sfields.clone(),
            vec![
                Arc::new(StringArray::from(authors)) as ArrayRef,
                Arc::new(Int64Array::from(scores)) as ArrayRef,
            ],
            None,
        );
        let list_field = Arc::new(Field::new("item", DataType::Struct(sfields), true));
        let list = ListArray::new(
            list_field.clone(),
            OffsetBuffer::new(offsets.into()),
            Arc::new(struct_array),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "comments",
            DataType::List(list_field),
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(list) as ArrayRef]).unwrap()
    }

    fn split_json() -> String {
        // comments.author='alice' (routed to Lucene, clause 0) AND comments.score>50 (native).
        r#"{"op":"AND","args":[
             {"lucene":0,"fallback":{"op":"=","args":[{"field":"author"},{"lit":"alice"}]}},
             {"op":">","args":[{"field":"score"},{"lit":50}]}
           ]}"#
        .to_string()
    }

    fn child_split_eval(matches: Vec<(i32, i32)>) -> SingleCollectorEvaluator {
        let annotation_id = 7;
        // Child provider locks live on ChildSplitState. Pre-seed the lock so no FFM createProvider upcall
        // happens in the unit test.
        let mut child_locks = HashMap::new();
        let lock: Arc<OnceLock<ProviderHandle>> = Arc::new(OnceLock::new());
        lock.set(ProviderHandle::new_for_test(0)).ok();
        child_locks.insert(annotation_id, lock);
        SingleCollectorEvaluator::new(
            None,
            minimal_page_pruner(),
            None,
            None, // no non-nested residual
            None,
            None,
            CollectorCallStrategy::FullRange,
            Arc::new(HashMap::new()), // parent-grain perf locks: none (child peers excluded)
            0,
            Arc::new(ChildStubFactory { matches }),
            0,
            None,
            None,
            HashMap::new(),
            Some(ChildSplitState {
                array_col_name: "comments".to_string(),
                expr_json: split_json(),
                clauses: vec![ChildClause {
                    clause_idx: 0,
                    annotation_id,
                }],
                provider_locks: Arc::new(child_locks),
            }),
        )
    }

    fn state_for(rg_num_rows: usize) -> SingleCollectorState {
        // No correctness Collector → candidates are the full universe (every row a candidate), which is
        // what prefetch_rg seeds for a performance-only query. The collector_mask must then be all-true so
        // the child-split AND is a no-op.
        state_with_candidates(rg_num_rows, &(0..rg_num_rows).collect::<Vec<_>>())
    }

    /// State whose candidate mask has exactly `candidate_rows` set (RG-relative positions).
    fn state_with_candidates(rg_num_rows: usize, candidate_rows: &[usize]) -> SingleCollectorState {
        let mut candidates = RoaringBitmap::new();
        for &r in candidate_rows {
            candidates.insert(r as u32);
        }
        let packed = bitmap_to_packed_bits(&candidates, rg_num_rows as u32);
        SingleCollectorState {
            candidates,
            mask_buffer: datafusion::arrow::buffer::Buffer::from_vec(packed),
            mask_len: rg_num_rows,
            rg_num_rows,
            rg_first_row: 0,
        }
    }

    #[test]
    fn child_split_identity_full_rg() {
        // 2 parents delivered in full (Identity). P0: alice@40, bob@90 (alice element is 40, not >50 →
        // no single element satisfies both). P1: alice@70 (satisfies both). Lucene matches author=alice
        // at (row0,off0) and (row1,off0). Expected [false, true].
        let batch = comments_batch(&[vec![("alice", 40), ("bob", 90)], vec![("alice", 70)]]);
        let eval = child_split_eval(vec![(0, 0), (1, 0)]);
        let state = state_for(2);
        let pm = PositionMap::Identity { delivered_count: 2 };
        let mask = eval
            .evaluate_child_split(eval.child_split.as_ref().unwrap(), &state, &pm, 0, 2, &batch)
            .unwrap();
        assert_eq!(mask, BooleanArray::from(vec![false, true]));
    }

    #[test]
    fn child_split_runs_delivered_subset_preserves_correlation() {
        // The RG has 4 parent rows but a co-delegated `status='x'` pruned it so parquet delivers only
        // rows 1 and 3 (a Runs PositionMap). This is the exact case where value_offsets (batch-local)
        // diverge from a whole-RG child prefix sum — the redesign's raison d'être.
        //
        // Delivered batch (2 rows): d0 = RG row 1 = [alice@70]; d1 = RG row 3 = [bob@30, alice@80].
        // value_offsets = [0, 1, 3]. child_base must be [-1, 0, -1, 1] (RG rows 0,2 absent).
        // Lucene matches author=alice at RG (row1, off0) and (row3, off1). For row1: alice@70 satisfies
        // (alice AND >50) → true. For row3: alice is off1 with score 80 → element (alice,80) satisfies →
        // true. Expected mask over DELIVERED rows [true, true].
        let batch = comments_batch(&[vec![("alice", 70)], vec![("bob", 30), ("alice", 80)]]);
        let eval = child_split_eval(vec![(1, 0), (3, 1)]);
        let state = state_for(4);
        // Runs: delivered d0 → rg 1, d1 → rg 3.
        let pm = PositionMap::Runs {
            runs: vec![(1, 0, 1), (3, 1, 1)],
            delivered_count: 2,
        };
        let mask = eval
            .evaluate_child_split(eval.child_split.as_ref().unwrap(), &state, &pm, 0, 2, &batch)
            .unwrap();
        assert_eq!(mask, BooleanArray::from(vec![true, true]));
    }

    #[test]
    fn child_split_runs_wrong_element_offset_would_break_correlation() {
        // Same delivered subset, but Lucene claims author=alice at RG (row3, off0) — that's bob@30, NOT
        // alice. If child_base were miscomputed the intersection would silently pass; with correct
        // coordinates, element off0 of row3 is (bob,30) which fails score>50, so row3 → false.
        // Row1 still true. Expected [true, false].
        let batch = comments_batch(&[vec![("alice", 70)], vec![("bob", 30), ("alice", 80)]]);
        let eval = child_split_eval(vec![(1, 0), (3, 0)]);
        let state = state_for(4);
        let pm = PositionMap::Runs {
            runs: vec![(1, 0, 1), (3, 1, 1)],
            delivered_count: 2,
        };
        let mask = eval
            .evaluate_child_split(eval.child_split.as_ref().unwrap(), &state, &pm, 0, 2, &batch)
            .unwrap();
        assert_eq!(mask, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn child_split_on_batch_mask_honors_collector_candidates() {
        // Full on_batch_mask path (not just evaluate_child_split): a correctness Collector coexists and
        // narrowed candidates to rows {0} only (e.g. a text match() parent predicate delegated to Lucene,
        // which is NOT in residual_expr). Even though BOTH P0 and P1 satisfy the nested predicate
        // element-wise, the collector_mask must exclude P1. Expected mask [true, false].
        let batch = comments_batch(&[vec![("alice", 70)], vec![("alice", 90)]]);
        let eval = child_split_eval(vec![(0, 0), (1, 0)]); // Lucene says alice at both rows' elem 0
        // candidates = {row 0} only → collector narrowed P1 out.
        let state = state_with_candidates(2, &[0]);
        let pm = PositionMap::Identity { delivered_count: 2 };
        let mask = eval
            .on_batch_mask(&state as &dyn std::any::Any, 0, &pm, 0, 2, &batch)
            .unwrap()
            .expect("child split produces a mask");
        assert_eq!(mask, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn child_split_on_batch_mask_full_universe_is_noop() {
        // No correctness Collector → full-universe candidates → collector_mask all-true → the nested
        // predicate alone decides. Same corpus as the Identity test: P0 no (alice@40 not>50), P1 yes.
        let batch = comments_batch(&[vec![("alice", 40), ("bob", 90)], vec![("alice", 70)]]);
        let eval = child_split_eval(vec![(0, 0), (1, 0)]);
        let state = state_for(2); // full universe
        let pm = PositionMap::Identity { delivered_count: 2 };
        let mask = eval
            .on_batch_mask(&state as &dyn std::any::Any, 0, &pm, 0, 2, &batch)
            .unwrap()
            .expect("child split produces a mask");
        assert_eq!(mask, BooleanArray::from(vec![false, true]));
    }

    // Keep the `fmt` import used
    #[allow(dead_code)]
    fn _use(_: &dyn fmt::Debug) {}
}

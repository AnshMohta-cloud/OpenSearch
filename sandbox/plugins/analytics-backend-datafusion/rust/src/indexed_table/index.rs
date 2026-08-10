/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Core index traits — the contract between the index and the query engine.
//!
//! These traits are all `IndexedExec`/`IndexedStream` need. How the searcher
//! was created (FFM upcall to Java, in-process native index, test stub) is
//! irrelevant here.
//!
//! ```text
//! ShardSearcher (shard-scoped compiled query — once per shard)
//!   └── RowGroupDocsCollector (per-segment matcher — once per segment)
//!          └── collect_packed_u64_bitset(range) → Vec<u64>
//! ```

use std::fmt::Debug;
use std::sync::Arc;

/// A collector that retrieves matching doc IDs as a packed bitset for a row
/// group's doc-id range within a segment.
///
/// May be called multiple times with increasing ranges (forward-only iteration).
///
/// # Bit layout contract
///
/// [`collect_packed_u64_bitset`](Self::collect_packed_u64_bitset) returns a
/// word-packed bitset matching Lucene's `FixedBitSet.getBits()` exactly:
///
/// - Word `j` covers the 64 doc-id-relative positions `j*64 .. (j+1)*64`.
/// - Bit `i` of word `j` (i.e. `word & (1u64 << i) != 0`) represents the
///   doc at relative position `j*64 + i`, i.e. absolute doc ID
///   `min_doc + j*64 + i`.
/// - Length is `ceil((max_doc - min_doc) / 64)` words. The last word may
///   have unused high bits set past `max_doc - min_doc`; consumers MUST
///   clamp by relative position before using a bit.
///
/// # Empty-range contract
///
/// If `max_doc <= min_doc`, implementations MUST return `Ok(Vec::new())`
/// (zero-length bitset). This is a no-op case and must not error. Callers
/// rely on this — e.g. `IndexedStream` skips filter-bitset fetch on empty
/// row groups by calling with `max_doc == min_doc`.
pub trait RowGroupDocsCollector: Send + Sync + Debug {
    fn collect_packed_u64_bitset(&self, min_doc: i32, max_doc: i32) -> Result<Vec<u64>, String>;

    /// CHILD-grain collect for the nested-predicate split: for each nested-child doc the collector's
    /// (child-scoped) query matches in the parent-row window `[min_doc, max_doc)`, set bit
    /// `child_base[row - min_doc] + element_offset`. `child_base` has length `max_doc - min_doc`; a `-1`
    /// entry marks a parent row not present in the caller's current batch (skip it). The returned bitset
    /// spans `total_children` bits in the CALLER'S element coordinate space (built from the decoded Arrow
    /// LIST `value_offsets`), so it is directly indexable by the residual UDF's flattened element index.
    ///
    /// Default: unsupported. Only the FFM-backed collector implements it (the real Lucene child scan);
    /// mock/stub collectors used off the child-split path never call it.
    fn collect_child_docs_batch(
        &self,
        _min_doc: i32,
        _max_doc: i32,
        _child_base: &[i32],
        _total_children: usize,
    ) -> Result<Vec<u64>, String> {
        Err("collect_child_docs_batch not supported by this collector".to_string())
    }
}

/// A searcher scoped to a single shard (index), created once per query.
///
/// Represents a shard-scoped compiled form of the query — typically expensive
/// to build (parses query, compiles automata / prepares iterators, etc.) but
/// cheap to bind to individual segments via [`collector`].
pub trait ShardSearcher: Send + Sync + Debug {
    /// Number of segments in this shard.
    fn segment_count(&self) -> usize;

    /// Max doc ID for a specific segment, identified by writer generation.
    fn segment_max_doc(&self, writer_generation: i64) -> Result<i64, String>;

    /// Create a collector for a specific segment and doc ID range.
    ///
    /// The collector only returns docs in `[doc_min, doc_max)`. One collector
    /// per segment per query, cheap to construct from the shard-scoped
    /// compiled query this searcher represents.
    fn collector(
        &self,
        writer_generation: i64,
        doc_min: i32,
        doc_max: i32,
    ) -> Result<Arc<dyn RowGroupDocsCollector>, String>;
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! FFM upcall surface for index-filter providers and collectors.
//!
//! Five callback slots, populated once at startup by
//! `df_register_filter_tree_callbacks` (see `ffm.rs`):
//!
//! - `createProvider(contextId, annotationId) -> providerKey|-1`
//! - `createCollector(contextId, providerKey, writerGeneration, minDoc, maxDoc) -> collectorKey|-1`
//! - `collectDocs(contextId, collectorKey, minDoc, maxDoc, outBuf, outWordCap) -> wordsWritten|-1`
//! - `releaseCollector(contextId, collectorKey)`
//! - `releaseProvider(contextId, providerKey)`
//!
//! One OPTIONAL sixth slot, populated by a SEPARATE additive registration
//! `df_register_child_collect_callback` (kept separate so the primary 5-arg
//! registration ABI never changes — a node whose Java side predates the child
//! split simply leaves this slot null and the child-grain nested path errors
//! gracefully while everything else, including non-nested delegation, boots and
//! runs unaffected):
//!
//! - `collectChildDocs(contextId, collectorKey, minDoc, maxDoc, childBase, totalChildren, outBuf, outWordCap) -> wordsWritten|-1`
//!   The child-grain sibling of `collectDocs`: sets bits by CHILD-ELEMENT ordinal
//!   (`childBase[row-minDoc] + elementOffset`) instead of by parent row, so the
//!   returned bitset lands directly in the caller's batch-flattened element space.
//!   `childBase` is a `*const i32` of length `maxDoc-minDoc` supplied by the
//!   caller (built from the decoded Arrow LIST `value_offsets`); a `-1` entry
//!   marks a parent row not present in the current batch (Java must skip it).
//!
//! `ProviderHandle` and `FfmSegmentCollector` are the lifetime wrappers —
//! they call the release callbacks on drop.
//!
//! The `context_id` is the per-query identifier (from `QueryTrackingContext::context_id()`)
//! that Java uses to route each callback to the correct per-query handle and tracker,
//! eliminating the global-singleton concurrency bug when multiple queries run in parallel.

use std::sync::atomic::{AtomicPtr, Ordering};

use super::index::RowGroupDocsCollector;

// ── Callback signatures ───────────────────────────────────────────────

type CreateProviderFn = unsafe extern "C" fn(i64, i32) -> i32;
type ReleaseProviderFn = unsafe extern "C" fn(i64, i32);
/// `(context_id, provider_key, writer_generation, doc_min, doc_max) -> collector_key | -1`.
///
/// `writer_generation` is the stable per-segment identifier.
/// `context_id` routes the upcall to the correct per-query Java handle.
type CreateCollectorFn = unsafe extern "C" fn(i64, i32, i64, i32, i32) -> i32;
type CollectDocsFn = unsafe extern "C" fn(i64, i32, i32, i32, *mut u64, i64) -> i64;
type ReleaseCollectorFn = unsafe extern "C" fn(i64, i32);
/// `(context_id, collector_key, min_doc, max_doc, child_base, total_children, out, out_word_cap) -> words_written | -1`.
///
/// Child-grain sibling of `CollectDocsFn`. `child_base` is a `*const i32` of length
/// `max_doc - min_doc`; `child_base[row - min_doc]` is the batch-flattened element index of
/// parent row `row`'s first element (or `-1` if that row isn't in the current batch, which
/// Java skips). Bits are set at `child_base[row-min_doc] + element_offset`. The output bitset
/// spans `total_children` bits.
type CollectChildDocsFn =
    unsafe extern "C" fn(i64, i32, i32, i32, *const i32, i64, *mut u64, i64) -> i64;

static CREATE_PROVIDER: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static RELEASE_PROVIDER: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static CREATE_COLLECTOR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static COLLECT_DOCS: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static RELEASE_COLLECTOR: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
/// Optional sixth slot — see `df_register_child_collect_callback`. Null until Java
/// registers it; the child-grain nested split is the only consumer.
static COLLECT_CHILD_DOCS: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Registered by Java at startup. Stores function pointers into atomic
/// slots. Each call to this entry replaces the slots wholesale.
///
/// Not annotated `#[ffm_safe]` because that macro is specific to the
/// `-> i64` error-pointer convention. We use a manual `catch_unwind`
/// instead, though the body (atomic stores) can't realistically panic.
#[no_mangle]
pub unsafe extern "C" fn df_register_filter_tree_callbacks(
    create_provider: CreateProviderFn,
    release_provider: ReleaseProviderFn,
    create_collector: CreateCollectorFn,
    collect_docs: CollectDocsFn,
    release_collector: ReleaseCollectorFn,
) {
    // catch_unwind is defense-in-depth: atomic stores shouldn't panic,
    // but if they ever did (e.g. allocator OOM if we grew the atomics),
    // unwinding across the FFM boundary is UB. Swallow the panic
    // silently — there's no way to report it back to Java for a
    // `-> ()` function.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CREATE_PROVIDER.store(create_provider as *mut (), Ordering::Release);
        RELEASE_PROVIDER.store(release_provider as *mut (), Ordering::Release);
        CREATE_COLLECTOR.store(create_collector as *mut (), Ordering::Release);
        COLLECT_DOCS.store(collect_docs as *mut (), Ordering::Release);
        RELEASE_COLLECTOR.store(release_collector as *mut (), Ordering::Release);
    }));
}

/// Registers the OPTIONAL child-grain collect callback. Kept as a SEPARATE entry point from
/// `df_register_filter_tree_callbacks` so the primary 5-callback registration ABI is untouched:
/// an older Java side that never calls this leaves `COLLECT_CHILD_DOCS` null, the node boots
/// normally, and only the child-grain nested split (which checks the slot and errors cleanly if
/// unset) is unavailable. Same manual `catch_unwind` rationale as the primary registration.
#[no_mangle]
pub unsafe extern "C" fn df_register_child_collect_callback(collect_child_docs: CollectChildDocsFn) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        COLLECT_CHILD_DOCS.store(collect_child_docs as *mut (), Ordering::Release);
    }));
}

fn load_create_provider() -> Result<CreateProviderFn, String> {
    let p = CREATE_PROVIDER.load(Ordering::Acquire);
    if p.is_null() {
        return Err("FilterTree callbacks not registered".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut (), CreateProviderFn>(p) })
}
fn load_release_provider() -> Option<ReleaseProviderFn> {
    let p = RELEASE_PROVIDER.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute::<*mut (), ReleaseProviderFn>(p) })
    }
}
fn load_create_collector() -> Result<CreateCollectorFn, String> {
    let p = CREATE_COLLECTOR.load(Ordering::Acquire);
    if p.is_null() {
        return Err("FilterTree callbacks not registered".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut (), CreateCollectorFn>(p) })
}
fn load_collect_docs() -> Result<CollectDocsFn, String> {
    let p = COLLECT_DOCS.load(Ordering::Acquire);
    if p.is_null() {
        return Err("FilterTree callbacks not registered".into());
    }
    Ok(unsafe { std::mem::transmute::<*mut (), CollectDocsFn>(p) })
}
fn load_release_collector() -> Option<ReleaseCollectorFn> {
    let p = RELEASE_COLLECTOR.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute::<*mut (), ReleaseCollectorFn>(p) })
    }
}
fn load_collect_child_docs() -> Result<CollectChildDocsFn, String> {
    let p = COLLECT_CHILD_DOCS.load(Ordering::Acquire);
    if p.is_null() {
        return Err(
            "child-grain collect callback not registered (Java side predates the nested child \
             split, or df_register_child_collect_callback was never called)"
                .into(),
        );
    }
    Ok(unsafe { std::mem::transmute::<*mut (), CollectChildDocsFn>(p) })
}

// ── ProviderHandle — owns `releaseProvider` on drop ───────────────────

/// Returned from `create_provider`. Drop releases the provider.
pub struct ProviderHandle {
    context_id: i64,
    key: i32,
}

impl ProviderHandle {
    pub fn key(&self) -> i32 {
        self.key
    }

    /// Test-only ctor: manufacture a handle with a chosen key without going through
    /// the FFM `createProvider` upcall. Drop is a no-op when the FFM `releaseProvider`
    /// callback isn't registered, which is always the case in unit/fuzz tests.
    #[cfg(test)]
    pub fn new_for_test(key: i32) -> Self {
        ProviderHandle { context_id: 0, key }
    }
}

impl std::fmt::Debug for ProviderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderHandle")
            .field("context_id", &self.context_id)
            .field("key", &self.key)
            .finish()
    }
}

impl Drop for ProviderHandle {
    fn drop(&mut self) {
        if let Some(release) = load_release_provider() {
            unsafe { release(self.context_id, self.key) };
        }
    }
}

/// Create a provider by annotation ID by upcalling Java.
///
/// `context_id` is the per-query identifier used by Java to route this upcall
/// to the correct per-query `FilterDelegationHandle`.
pub fn create_provider(context_id: i64, annotation_id: i32) -> Result<ProviderHandle, String> {
    let create = load_create_provider()?;
    let key = unsafe { create(context_id, annotation_id) };
    if key < 0 {
        return Err(format!(
            "createProvider failed: context_id={} annotation_id={} -> {}",
            context_id, annotation_id, key
        ));
    }
    Ok(ProviderHandle { context_id, key })
}

// ── FfmSegmentCollector — owns `releaseCollector` on drop ─────────────

#[derive(Debug)]
pub struct FfmSegmentCollector {
    context_id: i64,
    key: i32,
}

impl FfmSegmentCollector {
    /// Ask Java for a collector keyed by `provider_key` for the given segment/doc range.
    ///
    /// `context_id` is the per-query identifier used by Java to route this upcall
    /// to the correct per-query `FilterDelegationHandle`.
    /// `writer_generation` identifies the segment.
    pub fn create(
        context_id: i64,
        provider_key: i32,
        writer_generation: i64,
        doc_min: i32,
        doc_max: i32,
    ) -> Result<Self, String> {
        let create = load_create_collector()?;
        let key = unsafe {
            create(
                context_id,
                provider_key,
                writer_generation,
                doc_min,
                doc_max,
            )
        };
        if key < 0 {
            return Err(format!(
                "createCollector(context_id={}, provider={}, writer_generation={}) failed: {}",
                context_id, provider_key, writer_generation, key
            ));
        }
        Ok(FfmSegmentCollector { context_id, key })
    }
}

impl RowGroupDocsCollector for FfmSegmentCollector {
    /// Child-grain collect for the nested predicate split: returns a packed u64 bitset in the CALLER'S
    /// element coordinate space, dimensioned by `total_children`. For each Lucene child doc the scorer
    /// matches in `[min_doc, max_doc)` (parent-row window), Java sets bit `child_base[row - min_doc] +
    /// element_offset`, where `child_base` is supplied BY THE CALLER — built from the decoded Arrow LIST
    /// `value_offsets` of the current batch, so the returned bits are directly indexable by the UDF's
    /// batch-flattened element index (`elem_idx`). This is what makes the intersection correct under every
    /// PositionMap (Identity / Bitmap / Runs) and multi-batch RG split: the caller, not Java, owns the
    /// element numbering, and it always matches the array it will evaluate the residual against.
    ///
    /// `child_base` has length `max_doc - min_doc`; an entry of `-1` marks a parent row not present in the
    /// current batch (Java skips it). Uses the optional sixth FFM callback (`load_collect_child_docs`),
    /// which returns a clear error if the Java side never registered it.
    fn collect_child_docs_batch(
        &self,
        min_doc: i32,
        max_doc: i32,
        child_base: &[i32],
        total_children: usize,
    ) -> Result<Vec<u64>, String> {
        if max_doc <= min_doc || total_children == 0 {
            return Ok(Vec::new());
        }
        let span = (max_doc - min_doc) as usize;
        if child_base.len() != span {
            return Err(format!(
                "collect_child_docs_batch: child_base.len()={} != row span={} ([{},{}))",
                child_base.len(),
                span,
                min_doc,
                max_doc
            ));
        }
        let word_count = total_children.div_ceil(64);
        let mut buf = vec![0u64; word_count];
        let collect_fn = load_collect_child_docs()?;
        let n = unsafe {
            collect_fn(
                self.context_id,
                self.key,
                min_doc,
                max_doc,
                child_base.as_ptr(),
                total_children as i64,
                buf.as_mut_ptr(),
                word_count as i64,
            )
        };
        if n < 0 {
            return Err(format!(
                "collectChildDocs(context_id={}, key={}) failed: {}",
                self.context_id, self.key, n
            ));
        }
        let n = n as usize;
        if n > word_count {
            return Err(format!(
                "collectChildDocs(context_id={}, key={}) reported wordsWritten={} > capacity={}; \
                 callback contract violated (possible heap overflow)",
                self.context_id, self.key, n, word_count,
            ));
        }
        buf.truncate(n);
        Ok(buf)
    }

    fn collect_packed_u64_bitset(&self, min_doc: i32, max_doc: i32) -> Result<Vec<u64>, String> {
        if max_doc <= min_doc {
            return Ok(Vec::new());
        }
        let span = (max_doc - min_doc) as usize;
        let word_count = span.div_ceil(64);
        let mut buf = vec![0u64; word_count];
        let collect_fn = load_collect_docs()?;
        let n = unsafe {
            collect_fn(
                self.context_id,
                self.key,
                min_doc,
                max_doc,
                buf.as_mut_ptr(),
                word_count as i64,
            )
        };
        if n < 0 {
            return Err(format!(
                "collectDocs(context_id={}, key={}) failed: {}",
                self.context_id, self.key, n
            ));
        }
        // Defensive: the Java callback is contracted to return
        // `wordsWritten <= outWordCap`. If it lied, the buffer already
        // overflowed, but truncating won't recover the clobbered heap.
        // Detect the violation and fail loudly so the Java callback bug
        // is surfaced before downstream code consumes the tainted bitset.
        let n = n as usize;
        if n > word_count {
            return Err(format!(
                "collectDocs(context_id={}, key={}) reported wordsWritten={} > capacity={}; \
                 callback contract violated (possible heap overflow)",
                self.context_id, self.key, n, word_count,
            ));
        }
        buf.truncate(n);
        Ok(buf)
    }
}

impl Drop for FfmSegmentCollector {
    fn drop(&mut self) {
        if let Some(release) = load_release_collector() {
            unsafe { release(self.context_id, self.key) };
        }
    }
}

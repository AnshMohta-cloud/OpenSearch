/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Forward-only Parquet doc-values cursor over DataFusion I/O and Arrow decoding.
//!
//! DataFusion supplies metadata, page indexes, and remote object-store/cache I/O.
//! Local shard files use Arrow's synchronous `ChunkReader` directly. Arrow owns
//! page traversal, skipping, decompression, definition-level expansion, and batch
//! construction. This module selects the byte source and exposes the FFM lifecycle.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Float32Array, Float64Array,
    Int32Array, Int64Array, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray,
    StringArray, StringViewArray, StructArray,
};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use datafusion::execution::cache::cache_manager::FileMetadataCache;
use datafusion::execution::cache::DefaultFilesMetadataCache;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource_parquet::ParquetForwardBatchReaderFactory;
use liquid_cache_datafusion::{LiquidForwardBatchReader, LiquidForwardReaderConfig};
use native_bridge_common::ffm_safe;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use parquet::arrow::ProjectionMask;
use parquet::basic::Type as PhysicalType;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use tokio::runtime::{Builder, Runtime};

use crate::cache::metadata_cache::MutexFileMetadataCache;
use crate::cache::page_index::load_scoped_page_index_cols;
use crate::indexed_table::parquet_bridge::{
    load_parquet_metadata_with_meta, CachedMetadataReaderFactory, ReadIoStats,
};
use datafusion::datasource::physical_plan::parquet::ParquetFileReaderFactory;

/// Hard ceiling for the adaptive decode window. Mirrored by
/// `DataFusionColumnReader.MAX_BATCH_ROWS` and the upper bound of the
/// `parquet.docvalues.initial_batch_size` setting on the Java side; keep the
/// three in sync.
const MAX_BATCH_SIZE: usize = 8192;
const RC_OK: i64 = 0;
const RC_OVERFLOW: i64 = 1;
const RC_EOF: i64 = 2;
const MINMAX_UNKNOWN: (i64, i64) = (i64::MIN, i64::MAX);

static NEXT_HANDLE: AtomicI64 = AtomicI64::new(0);
static CURSORS: Lazy<DashMap<i64, Arc<Mutex<DocValuesCursor>>>> = Lazy::new(DashMap::new);

const DIAGNOSTIC_FIELD_COUNT: usize = 15;

#[derive(Default)]
struct CursorDiagnostics {
    enabled: AtomicBool,
    opens: AtomicU64,
    batch_calls: AtomicU64,
    sequential_batches: AtomicU64,
    sparse_batches: AtomicU64,
    decoded_rows: AtomicU64,
    skipped_rows: AtomicU64,
    overflow_probes: AtomicU64,
    range_reads: AtomicU64,
    range_bytes: AtomicU64,
    io_ns: AtomicU64,
    page_samples: AtomicU64,
    page_rows_total: AtomicU64,
    page_rows_min: AtomicU64,
    page_rows_max: AtomicU64,
}

impl CursorDiagnostics {
    fn reset(&self) {
        self.opens.store(0, Ordering::Relaxed);
        self.batch_calls.store(0, Ordering::Relaxed);
        self.sequential_batches.store(0, Ordering::Relaxed);
        self.sparse_batches.store(0, Ordering::Relaxed);
        self.decoded_rows.store(0, Ordering::Relaxed);
        self.skipped_rows.store(0, Ordering::Relaxed);
        self.overflow_probes.store(0, Ordering::Relaxed);
        self.range_reads.store(0, Ordering::Relaxed);
        self.range_bytes.store(0, Ordering::Relaxed);
        self.io_ns.store(0, Ordering::Relaxed);
        self.page_samples.store(0, Ordering::Relaxed);
        self.page_rows_total.store(0, Ordering::Relaxed);
        self.page_rows_min.store(u64::MAX, Ordering::Relaxed);
        self.page_rows_max.store(0, Ordering::Relaxed);
        self.enabled.store(true, Ordering::Release);
    }

    fn snapshot_and_disable(&self) -> [i64; DIAGNOSTIC_FIELD_COUNT] {
        self.enabled.store(false, Ordering::Release);
        let page_samples = self.page_samples.load(Ordering::Relaxed);
        let page_rows_min = if page_samples == 0 {
            0
        } else {
            self.page_rows_min.load(Ordering::Relaxed)
        };
        [
            self.opens.load(Ordering::Relaxed) as i64,
            self.batch_calls.load(Ordering::Relaxed) as i64,
            self.sequential_batches.load(Ordering::Relaxed) as i64,
            self.sparse_batches.load(Ordering::Relaxed) as i64,
            self.decoded_rows.load(Ordering::Relaxed) as i64,
            self.skipped_rows.load(Ordering::Relaxed) as i64,
            self.overflow_probes.load(Ordering::Relaxed) as i64,
            self.range_reads.load(Ordering::Relaxed) as i64,
            self.range_bytes.load(Ordering::Relaxed) as i64,
            self.io_ns.load(Ordering::Relaxed) as i64,
            page_samples as i64,
            self.page_rows_total.load(Ordering::Relaxed) as i64,
            page_rows_min as i64,
            self.page_rows_max.load(Ordering::Relaxed) as i64,
            CURSORS.len() as i64,
        ]
    }
}

static DIAGNOSTICS: Lazy<CursorDiagnostics> = Lazy::new(CursorDiagnostics::default);

/// File path to the DataFusion shard store that owns it. Weak references avoid
/// extending a shard view's lifetime.
static STORES: Lazy<DashMap<String, Weak<dyn ObjectStore>>> = Lazy::new(DashMap::new);

/// The node's DataFusion file-metadata cache, registered with the global runtime.
static METADATA_CACHE: RwLock<Option<Weak<dyn FileMetadataCache>>> = RwLock::new(None);

static FALLBACK_METADATA_CACHE: Lazy<Arc<MutexFileMetadataCache>> = Lazy::new(|| {
    Arc::new(MutexFileMetadataCache::new(DefaultFilesMetadataCache::new(
        64 * 1024 * 1024,
    )))
});

static FALLBACK_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(
        Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("df-docvalues-io")
            .enable_all()
            .build()
            .expect("failed to build doc-values fallback runtime"),
    )
});

pub fn register_metadata_cache(cache: Arc<dyn FileMetadataCache>) {
    *METADATA_CACHE.write() = Some(Arc::downgrade(&cache));
}

pub fn register_store(object_metas: &[ObjectMeta], store: Arc<dyn ObjectStore>) {
    STORES.retain(|_, weak| weak.strong_count() > 0);
    let weak = Arc::downgrade(&store);
    for meta in object_metas {
        STORES.insert(normalize_path(meta.location.as_ref()), Weak::clone(&weak));
    }
}

fn normalize_path(path: &str) -> String {
    path.strip_prefix("file://")
        .unwrap_or(path)
        .trim_start_matches('/')
        .to_string()
}

fn registered_store(filename: &str, location: &ObjectPath) -> Option<Arc<dyn ObjectStore>> {
    for key in [normalize_path(filename), normalize_path(location.as_ref())] {
        if let Some(entry) = STORES.get(&key) {
            if let Some(store) = entry.value().upgrade() {
                return Some(store);
            }
            drop(entry);
            STORES.remove(&key);
        }
    }
    None
}

fn metadata_cache() -> Arc<dyn FileMetadataCache> {
    METADATA_CACHE
        .read()
        .as_ref()
        .and_then(Weak::upgrade)
        .unwrap_or_else(|| Arc::clone(&FALLBACK_METADATA_CACHE) as Arc<dyn FileMetadataCache>)
}

fn io_runtime() -> Arc<Runtime> {
    crate::ffm::try_get_rt_manager()
        .map(|manager| Arc::clone(&manager.io_runtime))
        .unwrap_or_else(|| Arc::clone(&FALLBACK_RUNTIME))
}

fn liquid_cache() -> Option<liquid_cache_datafusion::LiquidCacheParquetRef> {
    crate::liquid_cache::LiquidOnlyRuntime::is_enabled_globally()
        .then(crate::liquid_cache::LiquidOnlyRuntime::cache_ref_globally)
        .flatten()
}

struct DocValuesCursor {
    reader: LiquidForwardBatchReader,
    physical_type: PhysicalType,
    repeated: bool,
    row_count: i64,
    initial_batch_size: usize,
    batch_size: usize,
    has_decoded_batch: bool,
    pending_batch: Option<(i64, RecordBatch)>,
    /// Keeps the most recently borrowed-out batch's buffers alive. Java reads
    /// the exported pointers until its next call on this cursor, so the array
    /// must outlive exactly one call cycle; each export replaces the previous.
    borrowed_batch: Option<ArrayRef>,
    stats: Arc<ReadIoStats>,
}

impl DocValuesCursor {
    async fn open(
        filename: &str,
        column: &str,
        batch_size: usize,
        store_override: Option<Arc<dyn ObjectStore>>,
        location_override: Option<ObjectPath>,
        runtime: Arc<Runtime>,
    ) -> Result<Self, String> {
        let location = location_override.unwrap_or_else(|| ObjectPath::from(filename));
        let store = store_override
            .or_else(|| registered_store(filename, &location))
            .unwrap_or_else(|| Arc::new(LocalFileSystem::new()));

        let object_meta = store
            .head(&location)
            .await
            .map_err(|e| format!("df_docvalues: object-store head {location}: {e}"))?;
        let (arrow_schema, file_size, footer) = load_parquet_metadata_with_meta(
            Arc::clone(&store),
            &location,
            object_meta,
            metadata_cache(),
        )
        .await?;

        let schema = footer.file_metadata().schema_descr();
        let leaf_idx = (0..schema.num_columns())
            .find(|&idx| {
                let descriptor = schema.column(idx);
                descriptor.name() == column
                    || descriptor.path().string() == column
                    || descriptor
                        .path()
                        .parts()
                        .first()
                        .is_some_and(|root| root == column)
            })
            .ok_or_else(|| format!("df_docvalues: column '{column}' not found in {filename}"))?;
        let descriptor = schema.column(leaf_idx);
        let root_column = descriptor
            .path()
            .parts()
            .first()
            .ok_or_else(|| format!("df_docvalues: column '{column}' has no root path"))?;
        let root_column_id = arrow_schema.index_of(root_column).map_err(|e| {
            format!("df_docvalues: root column '{root_column}' missing from Arrow schema: {e}")
        })?;
        let physical_type = descriptor.physical_type();
        if !matches!(
            physical_type,
            PhysicalType::INT32
                | PhysicalType::INT64
                | PhysicalType::FLOAT
                | PhysicalType::DOUBLE
                | PhysicalType::BOOLEAN
                | PhysicalType::BYTE_ARRAY
        ) {
            return Err(format!(
                "df_docvalues: unsupported physical type {physical_type:?} for column '{column}'"
            ));
        }

        let metadata =
            load_scoped_page_index_cols(&store, &location, &footer, &[leaf_idx], &[leaf_idx])
                .await
                .ok_or_else(|| {
                    format!(
                        "df_docvalues: no page index available for column '{column}' in {filename}"
                    )
                })?;
        let stats = Arc::new(ReadIoStats::default());
        let projection =
            ProjectionMask::leaves(metadata.file_metadata().schema_descr(), [leaf_idx]);
        let batch_size = batch_size.clamp(1, MAX_BATCH_SIZE);
        let reader_factory: Arc<dyn ParquetFileReaderFactory> = Arc::new(
            CachedMetadataReaderFactory::new(store, Arc::clone(&metadata), Arc::clone(&stats)),
        );
        let factory = Arc::new(
            ParquetForwardBatchReaderFactory::new(
                reader_factory,
                PartitionedFile::new(location.to_string(), file_size),
                metadata,
                projection,
                batch_size,
                Arc::clone(&runtime),
            )
            .with_local_file_if_exists(filename),
        );
        let reader = LiquidForwardBatchReader::try_new(
            factory,
            LiquidForwardReaderConfig {
                cache: liquid_cache(),
                file_path: normalize_path(filename),
                file_schema: arrow_schema,
                root_column_id,
                runtime,
                // Fixed-width cache hits may serve up to the Java buffer
                // capacity (one probe absorbs the page remainder); the Java
                // side always reserves MAX_BATCH_SIZE rows.
                hit_serve_limit: MAX_BATCH_SIZE,
            },
        )
        .map_err(|e| format!("df_docvalues: build forward reader for {filename}: {e}"))?;
        let row_count = reader.row_count() as i64;
        let repeated = reader.is_repeated();

        Ok(Self {
            reader,
            physical_type,
            repeated,
            row_count,
            initial_batch_size: batch_size,
            batch_size,
            has_decoded_batch: false,
            pending_batch: None,
            borrowed_batch: None,
            stats,
        })
    }

    fn planned_batch(&self, target_row: i64) -> Result<(usize, usize), String> {
        if target_row < 0 || target_row >= self.row_count {
            return Err(format!("df_docvalues: row {target_row} out of range"));
        }
        let target_row = target_row as usize;
        let position = self.reader.position();
        // Additive-increase / multiplicative-decrease window policy. A forward skip no
        // larger than the current window still indicates page-scale dense access (e.g. a
        // 10-30% selective filter that touches every page), so the window keeps growing.
        // Only a jump beyond the window decays it, and it halves rather than resetting so
        // mixed access patterns don't pay a full re-ramp after every jump.
        let dense = self.has_decoded_batch
            && target_row >= position
            && target_row - position <= self.batch_size;
        let window = if dense {
            self.batch_size.saturating_mul(2).min(MAX_BATCH_SIZE)
        } else if self.has_decoded_batch {
            (self.batch_size / 2).max(self.initial_batch_size)
        } else {
            self.batch_size
        };
        let page_rows = self
            .reader
            .page_row_count(target_row)
            .map_err(|e| format!("df_docvalues: find page at row {target_row}: {e}"))?;
        let page_remaining = self
            .reader
            .rows_remaining_in_page(target_row)
            .map_err(|e| format!("df_docvalues: find page end at row {target_row}: {e}"))?;
        let window = window.min(page_rows);
        Ok((window, window.min(page_remaining)))
    }

    fn next_batch(&mut self, target_row: i64) -> Result<RecordBatch, String> {
        let (window, rows) = self.planned_batch(target_row)?;
        let position = self.reader.position();
        let sequential = self.has_decoded_batch && target_row as usize == position;
        let diagnostics_enabled = DIAGNOSTICS.enabled.load(Ordering::Acquire);
        let diagnostics_before = if diagnostics_enabled {
            Some((
                self.stats.count.load(Ordering::Relaxed),
                self.stats.bytes.load(Ordering::Relaxed),
                self.stats.total_ns.load(Ordering::Relaxed),
            ))
        } else {
            None
        };
        let page_rows = if diagnostics_enabled {
            Some(
                self.reader
                    .page_row_count(target_row as usize)
                    .map_err(|e| format!("df_docvalues: find page at row {target_row}: {e}"))?,
            )
        } else {
            None
        };
        let batch = self
            .reader
            .read_batch_at(target_row as usize, rows)
            .map_err(|e| format!("df_docvalues: read row {target_row}: {e}"))?
            .ok_or_else(|| format!("df_docvalues: reader exhausted before row {target_row}"))?;
        self.batch_size = window;
        self.has_decoded_batch = true;
        if let Some(page_rows) = page_rows {
            DIAGNOSTICS.batch_calls.fetch_add(1, Ordering::Relaxed);
            if sequential {
                DIAGNOSTICS
                    .sequential_batches
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                DIAGNOSTICS.sparse_batches.fetch_add(1, Ordering::Relaxed);
            }
            DIAGNOSTICS
                .decoded_rows
                .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
            DIAGNOSTICS.skipped_rows.fetch_add(
                (target_row as usize).saturating_sub(position) as u64,
                Ordering::Relaxed,
            );
            let (range_reads_before, range_bytes_before, io_ns_before) =
                diagnostics_before.expect("diagnostic baselines missing");
            DIAGNOSTICS.range_reads.fetch_add(
                self.stats
                    .count
                    .load(Ordering::Relaxed)
                    .saturating_sub(range_reads_before),
                Ordering::Relaxed,
            );
            DIAGNOSTICS.range_bytes.fetch_add(
                self.stats
                    .bytes
                    .load(Ordering::Relaxed)
                    .saturating_sub(range_bytes_before),
                Ordering::Relaxed,
            );
            DIAGNOSTICS.io_ns.fetch_add(
                self.stats
                    .total_ns
                    .load(Ordering::Relaxed)
                    .saturating_sub(io_ns_before),
                Ordering::Relaxed,
            );
            DIAGNOSTICS.page_samples.fetch_add(1, Ordering::Relaxed);
            DIAGNOSTICS
                .page_rows_total
                .fetch_add(page_rows as u64, Ordering::Relaxed);
            DIAGNOSTICS
                .page_rows_min
                .fetch_min(page_rows as u64, Ordering::Relaxed);
            DIAGNOSTICS
                .page_rows_max
                .fetch_max(page_rows as u64, Ordering::Relaxed);
        }
        Ok(batch)
    }
}

fn page_min_max(index: Option<&ColumnIndexMetaData>, page_index: usize) -> (i64, i64) {
    match index {
        Some(ColumnIndexMetaData::INT32(index)) => {
            match (index.min_value(page_index), index.max_value(page_index)) {
                (Some(min), Some(max)) => (*min as i64, *max as i64),
                _ => MINMAX_UNKNOWN,
            }
        }
        Some(ColumnIndexMetaData::INT64(index)) => {
            match (index.min_value(page_index), index.max_value(page_index)) {
                (Some(min), Some(max)) => (*min, *max),
                _ => MINMAX_UNKNOWN,
            }
        }
        Some(ColumnIndexMetaData::BOOLEAN(index)) => {
            match (index.min_value(page_index), index.max_value(page_index)) {
                (Some(min), Some(max)) => (i64::from(*min), i64::from(*max)),
                _ => MINMAX_UNKNOWN,
            }
        }
        _ => MINMAX_UNKNOWN,
    }
}

/// Copies an Arrow primitive array directly into the caller-owned FFM buffers.
///
/// This intentionally avoids a normalized `Vec<i64>` and presence `Vec<i64>`:
/// Java already owns buffers in exactly that representation.
unsafe fn copy_array_out(
    array: &dyn Array,
    physical_type: PhysicalType,
    out_value_buf: *mut u8,
    out_presence_bitset: *mut i64,
) -> Result<(), String> {
    write_presence(array, out_presence_bitset);
    copy_array_values_out(array, physical_type, out_value_buf)
}

unsafe fn copy_array_values_out(
    array: &dyn Array,
    physical_type: PhysicalType,
    out_value_buf: *mut u8,
) -> Result<(), String> {
    let rows = array.len();
    match physical_type {
        PhysicalType::FLOAT => {
            let array = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    format!("df_docvalues: expected Float32, got {}", array.data_type())
                })?;
            if array.null_count() == 0 {
                for (idx, value) in array.values().iter().enumerate() {
                    write_i64(out_value_buf, idx, value.to_bits() as i64);
                }
            } else {
                for idx in 0..rows {
                    let value = if array.is_valid(idx) {
                        array.value(idx).to_bits() as i64
                    } else {
                        0
                    };
                    write_i64(out_value_buf, idx, value);
                }
            }
        }
        PhysicalType::DOUBLE => {
            let array = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    format!("df_docvalues: expected Float64, got {}", array.data_type())
                })?;
            if array.null_count() == 0 {
                for (idx, value) in array.values().iter().enumerate() {
                    write_i64(out_value_buf, idx, value.to_bits() as i64);
                }
            } else {
                for idx in 0..rows {
                    let value = if array.is_valid(idx) {
                        array.value(idx).to_bits() as i64
                    } else {
                        0
                    };
                    write_i64(out_value_buf, idx, value);
                }
            }
        }
        PhysicalType::BOOLEAN => {
            let array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    format!("df_docvalues: expected Boolean, got {}", array.data_type())
                })?;
            if array.null_count() == 0 {
                for idx in 0..rows {
                    write_i64(out_value_buf, idx, i64::from(array.value(idx)));
                }
            } else {
                for idx in 0..rows {
                    let value = if array.is_valid(idx) && array.value(idx) {
                        1
                    } else {
                        0
                    };
                    write_i64(out_value_buf, idx, value);
                }
            }
        }
        PhysicalType::INT32 => {
            if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
                if array.null_count() == 0 {
                    for (idx, value) in array.values().iter().enumerate() {
                        write_i64(out_value_buf, idx, *value as i64);
                    }
                } else {
                    for idx in 0..rows {
                        let value = if array.is_valid(idx) {
                            array.value(idx) as i64
                        } else {
                            0
                        };
                        write_i64(out_value_buf, idx, value);
                    }
                }
            } else {
                copy_casted_integer(array, out_value_buf)?;
            }
        }
        PhysicalType::INT64 => {
            if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
                copy_int64(array, out_value_buf);
            } else {
                copy_casted_integer(array, out_value_buf)?;
            }
        }
        other => {
            return Err(format!("df_docvalues: unsupported physical type {other:?}"));
        }
    }

    Ok(())
}

unsafe fn copy_casted_integer(array: &dyn Array, out_value_buf: *mut u8) -> Result<(), String> {
    let casted = cast(array, &DataType::Int64)
        .map_err(|e| format!("df_docvalues: cast {} to Int64: {e}", array.data_type()))?;
    let array = casted
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to Int64 returned a different type");
    copy_int64(array, out_value_buf);
    Ok(())
}

unsafe fn copy_int64(array: &Int64Array, out_value_buf: *mut u8) {
    let rows = array.len();
    if array.null_count() == 0 && array.offset() == 0 {
        std::ptr::copy_nonoverlapping(
            array.values().as_ptr() as *const u8,
            out_value_buf,
            rows * std::mem::size_of::<i64>(),
        );
    } else {
        for idx in 0..rows {
            let value = if array.is_valid(idx) {
                array.value(idx)
            } else {
                0
            };
            write_i64(out_value_buf, idx, value);
        }
    }
}

unsafe fn write_i64(out_value_buf: *mut u8, idx: usize, value: i64) {
    std::ptr::copy_nonoverlapping(
        &value as *const i64 as *const u8,
        out_value_buf.add(idx * std::mem::size_of::<i64>()),
        std::mem::size_of::<i64>(),
    );
}

unsafe fn write_presence(array: &dyn Array, out_presence_bitset: *mut i64) {
    let rows = array.len();
    let words = rows.div_ceil(64);
    if array.null_count() == 0 {
        // Every word is fully overwritten below; no zero-fill needed.
        for word in 0..words {
            let remaining = rows - word * 64;
            *out_presence_bitset.add(word) = if remaining >= 64 {
                -1
            } else {
                ((1u64 << remaining) - 1) as i64
            };
        }
        return;
    }
    std::ptr::write_bytes(
        out_presence_bitset as *mut u8,
        0,
        words * std::mem::size_of::<i64>(),
    );

    if array.offset() == 0 {
        if let Some(nulls) = array.nulls() {
            if nulls.inner().offset() == 0 {
                let bytes = rows.div_ceil(8);
                std::ptr::copy_nonoverlapping(
                    nulls.inner().values().as_ptr(),
                    out_presence_bitset as *mut u8,
                    bytes,
                );
                if rows % 64 != 0 {
                    *out_presence_bitset.add(words - 1) &= ((1u64 << (rows % 64)) - 1) as i64;
                }
                return;
            }
        }
    }

    for idx in 0..rows {
        if array.is_valid(idx) {
            *out_presence_bitset.add(idx / 64) |= 1i64 << (idx % 64);
        }
    }
}

fn binary_value_at(array: &dyn Array, index: usize) -> Result<&[u8], String> {
    if let Some(array) = array.as_any().downcast_ref::<BinaryArray>() {
        return Ok(array.value(index));
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(array.value(index));
    }
    if let Some(array) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(array.value(index));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(array.value(index).as_bytes());
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(array.value(index).as_bytes());
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(array.value(index).as_bytes());
    }
    Err(format!(
        "df_docvalues: expected binary or string Arrow array, got {}",
        array.data_type()
    ))
}

fn binary_value_bytes(array: &dyn Array) -> Result<usize, String> {
    let mut bytes = 0usize;
    for index in 0..array.len() {
        if array.is_valid(index) {
            bytes = bytes
                .checked_add(binary_value_at(array, index)?.len())
                .ok_or_else(|| "df_docvalues: binary output length overflow".to_string())?;
        }
    }
    if bytes > i32::MAX as usize {
        return Err(format!(
            "df_docvalues: binary batch requires {bytes} bytes, exceeding i32 offsets"
        ));
    }
    Ok(bytes)
}

unsafe fn copy_binary_array_out(
    array: &dyn Array,
    out_value_buf: *mut u8,
    out_byte_offsets: *mut i32,
    out_presence_bitset: *mut i64,
) -> Result<(), String> {
    write_presence(array, out_presence_bitset);
    copy_binary_values_out(array, out_value_buf, out_byte_offsets)
}

unsafe fn copy_binary_values_out(
    array: &dyn Array,
    out_value_buf: *mut u8,
    out_byte_offsets: *mut i32,
) -> Result<(), String> {
    *out_byte_offsets = 0;
    let mut offset = 0usize;
    for index in 0..array.len() {
        if array.is_valid(index) {
            let value = binary_value_at(array, index)?;
            std::ptr::copy_nonoverlapping(value.as_ptr(), out_value_buf.add(offset), value.len());
            offset += value.len();
        }
        *out_byte_offsets.add(index + 1) = offset as i32;
    }
    Ok(())
}

/// Descends one list level: returns `(child_values, absolute_element_offsets)`. Handles List and
/// LargeList (i32/i64 offsets, normalized to i64). Errors on any other array type.
fn list_level(array: &dyn Array) -> Result<(ArrayRef, Vec<i64>), String> {
    if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        let offs = a.value_offsets();
        return Ok((a.values().clone(), offs.iter().map(|&o| o as i64).collect()));
    }
    if let Some(a) = array.as_any().downcast_ref::<LargeListArray>() {
        let offs = a.value_offsets();
        return Ok((a.values().clone(), offs.to_vec()));
    }
    Err(format!(
        "df_docvalues: expected List or LargeList Arrow array, got {}",
        array.data_type()
    ))
}

/// Flattens a nested repeated leaf to `(leaf_values, top_row_offsets)`.
///
/// A nested leaf column projects to `List<Struct<...more List/Struct...leaf>>` (any depth). This
/// descends to the primitive/binary leaf, composing the list-level offsets so that
/// `top_row_offsets[i]..top_row_offsets[i+1]` are the leaf-value indices belonging to top-level row `i`
/// — exactly what the Java side (NestedElementResolver) indexes with (row, offsetInRow). Depth-independent
/// (a loop over levels). Cost is O(total list boundaries), pure offset arithmetic + one final `values`
/// slice — no per-element materialization, so latency tracks the flat scalar path.
///
/// Rules per level, starting at the top `List`:
///   * `List`  → record boundaries, compose with the running top-row offsets, descend into values.
///   * `Struct`→ descend into its single (leaf-projected) child; struct adds no repetition, offsets unchanged.
///   * leaf    → stop; slice the leaf to the covered element range and rebase offsets to 0.
fn flatten_repeated_leaf(array: &dyn Array) -> Result<(ArrayRef, Vec<i64>), String> {
    // Top level MUST be a list (a repeated column). Its per-top-row boundaries seed the composition.
    let (mut current, mut top_offsets) = list_level(array)?;

    loop {
        match current.data_type() {
            DataType::List(_) | DataType::LargeList(_) => {
                // Compose: each top-row boundary maps through this inner list's offsets. The inner
                // offsets are absolute into the inner child; remap every top boundary to the child index.
                let (child, inner_offsets) = list_level(current.as_ref())?;
                top_offsets = top_offsets
                    .iter()
                    .map(|&b| inner_offsets[b as usize])
                    .collect();
                current = child;
            }
            DataType::Struct(_) => {
                // Leaf projection prunes sibling struct children, so the struct carries exactly the child
                // on the path to our leaf. Descend into it; a struct is not repeated, so offsets are unchanged.
                let s = current
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| "df_docvalues: struct downcast failed".to_string())?;
                if s.num_columns() != 1 {
                    return Err(format!(
                        "df_docvalues: expected leaf-projected struct with 1 child, got {}",
                        s.num_columns()
                    ));
                }
                current = s.column(0).clone();
            }
            _ => {
                // Primitive/binary leaf reached. Slice to the covered element range and rebase offsets.
                let start = *top_offsets.first().unwrap_or(&0);
                let end = *top_offsets.last().unwrap_or(&start);
                let leaf = current.slice(start as usize, (end - start) as usize);
                let rebased = top_offsets.iter().map(|&o| o - start).collect();
                return Ok((leaf, rebased));
            }
        }
    }
}

fn repeated_value_count(array: &dyn Array) -> Result<usize, String> {
    let (leaf, _) = flatten_repeated_leaf(array)?;
    Ok(leaf.len())
}

fn repeated_values(array: &dyn Array) -> Result<ArrayRef, String> {
    let (leaf, _) = flatten_repeated_leaf(array)?;
    Ok(leaf)
}

unsafe fn write_repeated_offsets(
    array: &dyn Array,
    out_row_offsets: *mut i32,
) -> Result<(), String> {
    let (_, top_offsets) = flatten_repeated_leaf(array)?;
    for (idx, offset) in top_offsets.iter().enumerate() {
        *out_row_offsets.add(idx) = i32::try_from(*offset)
            .map_err(|_| "df_docvalues: repeated row offset exceeds i32".to_string())?;
    }
    Ok(())
}

/// Shared entry-point prologue: resolves a live cursor handle.
fn cursor_for(handle: i64, fn_name: &str) -> Result<Arc<Mutex<DocValuesCursor>>, String> {
    CURSORS
        .get(&handle)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| format!("{fn_name}: unknown handle {handle}"))
}

/// Each of the four batch outputs is valid for exactly one (repeated, binary) column shape.
fn check_column_shape(
    cursor: &DocValuesCursor,
    want_repeated: bool,
    want_binary: bool,
    fn_name: &str,
) -> Result<(), String> {
    let is_binary = cursor.physical_type == PhysicalType::BYTE_ARRAY;
    if cursor.repeated != want_repeated || is_binary != want_binary {
        return Err(format!(
            "{fn_name}: column is repeated={}, binary={is_binary}; use the batch output matching that shape",
            cursor.repeated
        ));
    }
    Ok(())
}

/// Validates `target_row`, returning `true` when the cursor is exactly at end-of-column.
fn at_eof(cursor: &DocValuesCursor, target_row: i64, fn_name: &str) -> Result<bool, String> {
    if target_row == cursor.row_count {
        return Ok(true);
    }
    if target_row < 0 || target_row > cursor.row_count {
        return Err(format!("{fn_name}: row {target_row} out of range"));
    }
    Ok(false)
}

/// Takes the batch staged by a previous overflow probe for `target_row`, or decodes a new one.
/// Staging guarantees variable-width values are never decompressed or decoded twice across the
/// caller's grow-and-retry.
fn take_pending_or_decode(
    cursor: &mut DocValuesCursor,
    target_row: i64,
    fn_name: &str,
) -> Result<RecordBatch, String> {
    match cursor.pending_batch.take() {
        Some((pending_target, batch)) if pending_target == target_row => Ok(batch),
        Some(pending) => {
            cursor.pending_batch = Some(pending);
            Err(format!(
                "{fn_name}: row {target_row} requested while another batch awaits retry"
            ))
        }
        None => cursor.next_batch(target_row),
    }
}

/// Stages `batch` for the caller's retry and reports the overflow.
fn stage_overflow(cursor: &mut DocValuesCursor, target_row: i64, batch: RecordBatch) -> i64 {
    cursor.pending_batch = Some((target_row, batch));
    note_overflow_probe();
    RC_OVERFLOW
}

fn note_overflow_probe() {
    if DIAGNOSTICS.enabled.load(Ordering::Acquire) {
        DIAGNOSTICS.overflow_probes.fetch_add(1, Ordering::Relaxed);
    }
}

/// Writes `value` through a nullable out-parameter.
unsafe fn write_out(ptr: *mut i64, value: i64) {
    if !ptr.is_null() {
        *ptr = value;
    }
}

/// Java-side interpretation of a borrowed values buffer. Mirrors the
/// `KIND_*` constants in `PageCache.java`; keep in sync.
const BORROW_KIND_LONG: i64 = 1; // i64 / u64 / f64 raw bits, 8 bytes per row
const BORROW_KIND_INT: i64 = 2; // i32 / date32, sign-extended, 4 bytes per row
const BORROW_KIND_UINT_BITS: i64 = 3; // u32 / f32 raw bits, zero-extended, 4 bytes per row
const BORROW_KIND_SHORT: i64 = 4; // i16, sign-extended, 2 bytes per row
const BORROW_KIND_USHORT: i64 = 5; // u16, zero-extended, 2 bytes per row
const BORROW_KIND_BYTE: i64 = 6; // i8, sign-extended, 1 byte per row
const BORROW_KIND_UBYTE: i64 = 7; // u8, zero-extended, 1 byte per row

struct BorrowedBuffers {
    values_addr: usize,
    validity_addr: usize,
    validity_bit_offset: usize,
    kind: i64,
}

/// Exposes an Arrow primitive array's buffers for zero-copy reads from Java.
///
/// Sparse readers touch a handful of rows per served batch; copying — and for
/// narrow integers like `Int16`, running a whole-array cast kernel — per probe
/// dominated the warm profile. Instead Java reads the Arrow values buffer and
/// validity bitmap in place, widening per accessed row: O(rows accessed), not
/// O(rows served). Keyed on the Arrow type (the decode output), not the
/// Parquet physical type: Parquet INT32 arrives as Int8/Int16/Int32/Date32
/// depending on the logical type. Boolean stays on the copy path (bit-packed)
/// as do binary and repeated shapes.
fn borrowable_buffers(array: &dyn Array, _physical: PhysicalType) -> Option<BorrowedBuffers> {
    use arrow::datatypes::DataType as DT;
    let (kind, width) = match array.data_type() {
        DT::Int64 | DT::UInt64 | DT::Float64 | DT::Date64 | DT::Timestamp(_, _) => {
            (BORROW_KIND_LONG, 8usize)
        }
        DT::Int32 | DT::Date32 | DT::Time32(_) => (BORROW_KIND_INT, 4),
        DT::UInt32 | DT::Float32 => (BORROW_KIND_UINT_BITS, 4),
        DT::Int16 => (BORROW_KIND_SHORT, 2),
        DT::UInt16 => (BORROW_KIND_USHORT, 2),
        DT::Int8 => (BORROW_KIND_BYTE, 1),
        DT::UInt8 => (BORROW_KIND_UBYTE, 1),
        _ => return None,
    };
    debug_assert_eq!(array.data_type().primitive_width(), Some(width));
    let data = array.to_data();
    let buffer = data.buffers().first()?;
    let values_addr = buffer.as_ptr() as usize + data.offset() * width;
    let (validity_addr, validity_bit_offset) = match data.nulls() {
        None => (0, 0),
        Some(nulls) => (nulls.buffer().as_ptr() as usize, nulls.offset()),
    };
    Some(BorrowedBuffers {
        values_addr,
        validity_addr,
        validity_bit_offset,
        kind,
    })
}

fn open(filename: &str, column: &str, initial_batch_size: usize) -> Result<i64, String> {
    let runtime = io_runtime();
    let cursor = runtime.block_on(DocValuesCursor::open(
        filename,
        column,
        initial_batch_size,
        None,
        None,
        Arc::clone(&runtime),
    ))?;
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
    CURSORS.insert(handle, Arc::new(Mutex::new(cursor)));
    if DIAGNOSTICS.enabled.load(Ordering::Acquire) {
        DIAGNOSTICS.opens.fetch_add(1, Ordering::Relaxed);
    }
    Ok(handle)
}

unsafe fn str_from_raw<'a>(ptr: *const u8, len: i64) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null string pointer".to_string());
    }
    if len < 0 {
        return Err(format!("negative string length: {len}"));
    }
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len as usize))
        .map_err(|e| format!("invalid UTF-8: {e}"))
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_open_iter(
    file_ptr: *const u8,
    file_len: i64,
    column_ptr: *const u8,
    column_len: i64,
    initial_batch_size: i64,
) -> i64 {
    let filename =
        str_from_raw(file_ptr, file_len).map_err(|e| format!("parquet_df_open_iter file: {e}"))?;
    let column = str_from_raw(column_ptr, column_len)
        .map_err(|e| format!("parquet_df_open_iter column: {e}"))?;
    if initial_batch_size <= 0 || initial_batch_size > MAX_BATCH_SIZE as i64 {
        return Err(format!(
            "parquet_df_open_iter: initial batch size {initial_batch_size} outside 1..={MAX_BATCH_SIZE}"
        ));
    }
    open(filename, column, initial_batch_size as usize)
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_close_iter(handle: i64) -> i64 {
    CURSORS.remove(&handle);
    Ok(RC_OK)
}

/// Rewinds a cursor to row zero without tearing it down: only the inner
/// Parquet decoder is rebuilt; metadata, page index, cache registrations and
/// column handles are retained. Concurrent segment-search slices interleave
/// reads over shared per-field cursors, so backward seeks are routine — a
/// full close/open cycle per seek dominated cold sparse profiles.
#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_reset_iter(handle: i64) -> i64 {
    let cursor = cursor_for(handle, "parquet_df_reset_iter")?;
    let mut cursor = cursor.lock();
    cursor
        .reader
        .reset()
        .map_err(|e| format!("parquet_df_reset_iter: {e}"))?;
    cursor.batch_size = cursor.initial_batch_size;
    cursor.has_decoded_batch = false;
    cursor.pending_batch = None;
    cursor.borrowed_batch = None;
    Ok(RC_OK)
}

#[no_mangle]
pub extern "C" fn parquet_df_open_iter_count() -> i64 {
    CURSORS.len() as i64
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_row_count(handle: i64) -> i64 {
    let cursor = CURSORS
        .get(&handle)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| format!("parquet_df_row_count: unknown handle {handle}"))?;
    let row_count = cursor.lock().row_count;
    Ok(row_count)
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_page_count(handle: i64) -> i64 {
    let cursor = CURSORS
        .get(&handle)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| format!("parquet_df_page_count: unknown handle {handle}"))?;
    let page_count = cursor.lock().reader.pages().len() as i64;
    Ok(page_count)
}

/// Copies the retained reader's scoped OffsetIndex and ColumnIndex into
/// caller-owned parallel arrays for Lucene's DocValuesSkipper.
#[ffm_safe]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn parquet_df_page_index(
    handle: i64,
    out_first_row: *mut i64,
    out_file_offset: *mut i64,
    out_compressed_size: *mut i32,
    out_null_count: *mut i64,
    out_min_long: *mut i64,
    out_max_long: *mut i64,
    out_buf_capacity: i64,
    out_actual_pages: *mut i64,
) -> i64 {
    let cursor = CURSORS
        .get(&handle)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| format!("parquet_df_page_index: unknown handle {handle}"))?;
    let cursor = cursor.lock();
    let pages = cursor.reader.pages();
    if !out_actual_pages.is_null() {
        *out_actual_pages = pages.len() as i64;
    }
    if out_buf_capacity < pages.len() as i64 {
        return Ok(RC_OVERFLOW);
    }

    let metadata = cursor.reader.metadata();
    let leaf = cursor.reader.projected_leaf_column();
    for (idx, page) in pages.iter().enumerate() {
        if !out_first_row.is_null() {
            *out_first_row.add(idx) = page.first_row as i64;
        }
        if !out_file_offset.is_null() {
            *out_file_offset.add(idx) = page.file_offset;
        }
        if !out_compressed_size.is_null() {
            *out_compressed_size.add(idx) = page.compressed_size;
        }
        if !out_null_count.is_null() {
            *out_null_count.add(idx) = page.null_count.unwrap_or(-1);
        }

        let column_index = metadata
            .column_index()
            .and_then(|index| index.get(page.row_group_index))
            .and_then(|row_group| row_group.get(leaf))
            .filter(|index| {
                !matches!(index, ColumnIndexMetaData::NONE)
                    && page.page_index < index.num_pages() as usize
            });
        let (min, max) = page_min_max(column_index, page.page_index);
        if !out_min_long.is_null() {
            *out_min_long.add(idx) = min;
        }
        if !out_max_long.is_null() {
            *out_max_long.add(idx) = max;
        }
    }
    Ok(RC_OK)
}

#[no_mangle]
pub extern "C" fn parquet_df_diagnostics_reset() -> i64 {
    DIAGNOSTICS.reset();
    RC_OK
}

#[ffm_safe]
#[no_mangle]
pub unsafe extern "C" fn parquet_df_diagnostics_snapshot(
    out_stats: *mut i64,
    out_stats_cap: i64,
) -> i64 {
    if out_stats.is_null() {
        return Err("parquet_df_diagnostics_snapshot: null output pointer".to_string());
    }
    if out_stats_cap < DIAGNOSTIC_FIELD_COUNT as i64 {
        return Err(format!(
            "parquet_df_diagnostics_snapshot: capacity {out_stats_cap} is smaller than {DIAGNOSTIC_FIELD_COUNT}"
        ));
    }
    let snapshot = DIAGNOSTICS.snapshot_and_disable();
    std::ptr::copy_nonoverlapping(snapshot.as_ptr(), out_stats, snapshot.len());
    Ok(RC_OK)
}

/// Advance to `target_row`, skip intermediate pages through Arrow's retained
/// reader, and copy one standard Arrow batch into the caller's buffers.
#[ffm_safe]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn parquet_df_next_batch(
    handle: i64,
    target_row: i64,
    out_first_row: *mut i64,
    out_last_row: *mut i64,
    out_value_buf: *mut u8,
    out_value_buf_cap: i64,
    out_value_actual_len: *mut i64,
    out_presence_bitset: *mut i64,
    out_presence_bits_cap: i64,
    out_values_addr: *mut i64,
    out_validity_addr: *mut i64,
    out_validity_bit_offset: *mut i64,
    out_value_kind: *mut i64,
) -> i64 {
    const FN: &str = "parquet_df_next_batch";
    let cursor = cursor_for(handle, FN)?;
    let mut cursor = cursor.lock();
    check_column_shape(&cursor, false, false, FN)?;
    if at_eof(&cursor, target_row, FN)? {
        return Ok(RC_EOF);
    }

    // Decode first: a page-grid cache hit may serve more rows than the
    // planned window (up to MAX_BATCH_SIZE), so sizes are known only after
    // the read. The batch is staged across an overflow retry.
    let batch = take_pending_or_decode(&mut cursor, target_row, FN)?;
    let rows = batch.num_rows();
    if rows == 0 || rows > MAX_BATCH_SIZE {
        return Err(format!("{FN}: Arrow returned {rows} rows"));
    }
    write_out(out_first_row, target_row);
    write_out(out_last_row, target_row + rows as i64 - 1);

    // Zero-copy fast path: hand Java the Arrow buffers directly. The array is
    // retained on the cursor so the pointers stay valid until Java's next
    // call on this handle (Java swaps its resident batch before that call).
    let array = batch.column(0);
    if let Some(borrow) = borrowable_buffers(array.as_ref(), cursor.physical_type) {
        write_out(out_value_actual_len, 0);
        write_out(out_values_addr, borrow.values_addr as i64);
        write_out(out_validity_addr, borrow.validity_addr as i64);
        write_out(out_validity_bit_offset, borrow.validity_bit_offset as i64);
        write_out(out_value_kind, borrow.kind);
        cursor.borrowed_batch = Some(Arc::clone(array));
        return Ok(RC_OK);
    }

    // Copy fallback (boolean): widen into the caller's pooled buffers.
    write_out(out_values_addr, 0);
    write_out(out_validity_addr, 0);
    write_out(out_validity_bit_offset, 0);
    write_out(out_value_kind, 0);
    let value_bytes = (rows * std::mem::size_of::<i64>()) as i64;
    let presence_words = rows.div_ceil(64) as i64;
    write_out(out_value_actual_len, value_bytes);
    if out_value_buf.is_null()
        || out_presence_bitset.is_null()
        || out_value_buf_cap < value_bytes
        || out_presence_bits_cap < presence_words
    {
        return Ok(stage_overflow(&mut cursor, target_row, batch));
    }
    copy_array_out(
        batch.column(0).as_ref(),
        cursor.physical_type,
        out_value_buf,
        out_presence_bitset,
    )?;
    Ok(RC_OK)
}

/// Binary counterpart of [`parquet_df_next_batch`]. A decoded batch is staged
/// across an overflow retry so variable-width values are never decompressed or
/// decoded twice.
#[ffm_safe]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn parquet_df_next_binary_batch(
    handle: i64,
    target_row: i64,
    out_first_row: *mut i64,
    out_last_row: *mut i64,
    out_value_buf: *mut u8,
    out_value_buf_cap: i64,
    out_value_actual_len: *mut i64,
    out_byte_offsets: *mut i32,
    out_byte_offsets_cap: i64,
    out_presence_bitset: *mut i64,
    out_presence_bits_cap: i64,
) -> i64 {
    const FN: &str = "parquet_df_next_binary_batch";
    let cursor = cursor_for(handle, FN)?;
    let mut cursor = cursor.lock();
    check_column_shape(&cursor, false, true, FN)?;
    if at_eof(&cursor, target_row, FN)? {
        return Ok(RC_EOF);
    }

    let batch = take_pending_or_decode(&mut cursor, target_row, FN)?;
    let rows = batch.num_rows();
    let array = batch.column(0).as_ref();
    let value_bytes = binary_value_bytes(array)?;
    let presence_words = rows.div_ceil(64);

    write_out(out_first_row, target_row);
    write_out(out_last_row, target_row + rows as i64 - 1);
    write_out(out_value_actual_len, value_bytes as i64);

    if out_value_buf.is_null()
        || out_byte_offsets.is_null()
        || out_presence_bitset.is_null()
        || out_value_buf_cap < value_bytes as i64
        || out_byte_offsets_cap < rows as i64 + 1
        || out_presence_bits_cap < presence_words as i64
    {
        return Ok(stage_overflow(&mut cursor, target_row, batch));
    }

    copy_binary_array_out(array, out_value_buf, out_byte_offsets, out_presence_bitset)?;
    Ok(RC_OK)
}

/// Repeated primitive counterpart of [`parquet_df_next_batch`]. Arrow owns the
/// list decoding; this adapter copies the flattened child values and normalized
/// row-to-value offsets into caller-owned buffers.
#[ffm_safe]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn parquet_df_next_repeated_batch(
    handle: i64,
    target_row: i64,
    out_first_row: *mut i64,
    out_last_row: *mut i64,
    out_values: *mut u8,
    out_values_cap: i64,
    out_value_count: *mut i64,
    out_row_offsets: *mut i32,
    out_row_offsets_cap: i64,
) -> i64 {
    const FN: &str = "parquet_df_next_repeated_batch";
    let cursor = cursor_for(handle, FN)?;
    let mut cursor = cursor.lock();
    check_column_shape(&cursor, true, false, FN)?;
    if at_eof(&cursor, target_row, FN)? {
        return Ok(RC_EOF);
    }

    let batch = take_pending_or_decode(&mut cursor, target_row, FN)?;
    let rows = batch.num_rows();
    let array = batch.column(0).as_ref();
    let value_count = repeated_value_count(array)?;

    write_out(out_first_row, target_row);
    write_out(out_last_row, target_row + rows as i64 - 1);
    write_out(out_value_count, value_count as i64);

    if out_values.is_null()
        || out_row_offsets.is_null()
        || out_values_cap < value_count as i64
        || out_row_offsets_cap < rows as i64 + 1
    {
        return Ok(stage_overflow(&mut cursor, target_row, batch));
    }

    let values = repeated_values(array)?;
    if values.null_count() != 0 {
        return Err(format!(
            "{FN}: null list elements are not valid DocValues"
        ));
    }
    write_repeated_offsets(array, out_row_offsets)?;
    copy_array_values_out(values.as_ref(), cursor.physical_type, out_values)?;
    Ok(RC_OK)
}

/// Repeated binary counterpart of [`parquet_df_next_binary_batch`].
#[ffm_safe]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn parquet_df_next_repeated_binary_batch(
    handle: i64,
    target_row: i64,
    out_first_row: *mut i64,
    out_last_row: *mut i64,
    out_value_buf: *mut u8,
    out_value_buf_cap: i64,
    out_value_actual_len: *mut i64,
    out_element_offsets: *mut i32,
    out_element_offsets_cap: i64,
    out_value_count: *mut i64,
    out_row_offsets: *mut i32,
    out_row_offsets_cap: i64,
) -> i64 {
    const FN: &str = "parquet_df_next_repeated_binary_batch";
    let cursor = cursor_for(handle, FN)?;
    let mut cursor = cursor.lock();
    check_column_shape(&cursor, true, true, FN)?;
    if at_eof(&cursor, target_row, FN)? {
        return Ok(RC_EOF);
    }

    let batch = take_pending_or_decode(&mut cursor, target_row, FN)?;
    let rows = batch.num_rows();
    let array = batch.column(0).as_ref();
    let values = repeated_values(array)?;
    if values.null_count() != 0 {
        return Err(format!(
            "{FN}: null list elements are not valid DocValues"
        ));
    }
    let value_count = values.len();
    let value_bytes = binary_value_bytes(values.as_ref())?;

    write_out(out_first_row, target_row);
    write_out(out_last_row, target_row + rows as i64 - 1);
    write_out(out_value_actual_len, value_bytes as i64);
    write_out(out_value_count, value_count as i64);

    if out_value_buf.is_null()
        || out_element_offsets.is_null()
        || out_row_offsets.is_null()
        || out_value_buf_cap < value_bytes as i64
        || out_element_offsets_cap < value_count as i64 + 1
        || out_row_offsets_cap < rows as i64 + 1
    {
        return Ok(stage_overflow(&mut cursor, target_row, batch));
    }

    write_repeated_offsets(array, out_row_offsets)?;
    copy_binary_values_out(values.as_ref(), out_value_buf, out_element_offsets)?;
    Ok(RC_OK)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use arrow::array::{
        builder::{ListBuilder, StringBuilder},
        types::Int64Type,
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, ListArray,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use tempfile::NamedTempFile;

    use super::*;

    const ROWS_PER_PAGE: usize = 64;

    fn parquet_fixture_with_page_rows(row_groups: usize, rows_per_page: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_row_count_limit(rows_per_page)
            .set_write_batch_size(rows_per_page)
            .set_max_row_group_row_count(Some(rows_per_page * 8))
            .build();
        let mut writer =
            ArrowWriter::try_new(Cursor::new(Vec::new()), Arc::clone(&schema), Some(props))
                .unwrap();
        let row_count = row_groups * rows_per_page * 8;
        let values = (0..row_count).map(|value| value as i64).collect::<Vec<_>>();
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values)) as ArrayRef])
                .unwrap();
        writer.write(&batch).unwrap();
        Bytes::from(writer.into_inner().unwrap().into_inner())
    }

    fn parquet_fixture_with_all_null_page(rows_per_page: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_row_count_limit(rows_per_page)
            .set_write_batch_size(rows_per_page)
            .set_max_row_group_row_count(Some(rows_per_page * 4))
            .build();
        let mut writer =
            ArrowWriter::try_new(Cursor::new(Vec::new()), Arc::clone(&schema), Some(props))
                .unwrap();
        let values = (0..rows_per_page * 2)
            .map(|row| (row >= rows_per_page).then_some(row as i64))
            .collect::<Vec<_>>();
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values)) as ArrayRef])
                .unwrap();
        writer.write(&batch).unwrap();
        Bytes::from(writer.into_inner().unwrap().into_inner())
    }

    fn parquet_binary_fixture(rows_per_page: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_row_count_limit(rows_per_page)
            .set_write_batch_size(rows_per_page)
            .build();
        let mut writer =
            ArrowWriter::try_new(Cursor::new(Vec::new()), Arc::clone(&schema), Some(props))
                .unwrap();
        let values = StringArray::from(vec![Some("alpha"), None, Some("z"), Some("omega")]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
        writer.write(&batch).unwrap();
        Bytes::from(writer.into_inner().unwrap().into_inner())
    }

    fn parquet_repeated_numeric_fixture() -> Bytes {
        let values = ListArray::from_iter_primitive::<Int64Type, _, _>([
            Some(vec![Some(3), Some(1)]),
            None,
            Some(vec![]),
            Some(vec![Some(8), Some(5), Some(8)]),
        ]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            values.data_type().clone(),
            true,
        )]));
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_row_count_limit(2)
            .set_write_batch_size(2)
            .build();
        let mut writer =
            ArrowWriter::try_new(Cursor::new(Vec::new()), Arc::clone(&schema), Some(props))
                .unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
        writer.write(&batch).unwrap();
        Bytes::from(writer.into_inner().unwrap().into_inner())
    }

    fn parquet_repeated_binary_fixture() -> Bytes {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("beta");
        builder.values().append_value("alpha");
        builder.append(true);
        builder.append(false);
        builder.append(true);
        builder.values().append_value("omega");
        builder.append(true);
        let values = builder.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            values.data_type().clone(),
            true,
        )]));
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_data_page_row_count_limit(2)
            .set_write_batch_size(2)
            .build();
        let mut writer =
            ArrowWriter::try_new(Cursor::new(Vec::new()), Arc::clone(&schema), Some(props))
                .unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
        writer.write(&batch).unwrap();
        Bytes::from(writer.into_inner().unwrap().into_inner())
    }

    fn open_parquet_fixture(bytes: Bytes, batch_size: usize) -> (DocValuesCursor, Arc<Runtime>) {
        let runtime = Arc::new(Builder::new_current_thread().enable_all().build().unwrap());
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectPath::from(format!(
            "doc-values-cursor-{}.parquet",
            NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
        ));
        runtime
            .block_on(store.put(&location, bytes.into()))
            .unwrap();
        let cursor = runtime
            .block_on(DocValuesCursor::open(
                location.as_ref(),
                "value",
                batch_size,
                Some(store),
                Some(location.clone()),
                Arc::clone(&runtime),
            ))
            .unwrap();
        (cursor, runtime)
    }

    fn open_fixture_with_page_rows(
        row_groups: usize,
        rows_per_page: usize,
        batch_size: usize,
    ) -> (DocValuesCursor, Arc<Runtime>) {
        open_parquet_fixture(
            parquet_fixture_with_page_rows(row_groups, rows_per_page),
            batch_size,
        )
    }

    fn open_fixture(row_groups: usize, batch_size: usize) -> (DocValuesCursor, Arc<Runtime>) {
        open_fixture_with_page_rows(row_groups, ROWS_PER_PAGE, batch_size)
    }

    fn int64_values(batch: &RecordBatch) -> Vec<i64> {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn copied_output(array: &dyn Array, physical_type: PhysicalType) -> (Vec<i64>, Vec<i64>) {
        let mut values = vec![i64::MIN; array.len()];
        let mut presence = vec![-1; array.len().div_ceil(64)];
        unsafe {
            copy_array_out(
                array,
                physical_type,
                values.as_mut_ptr() as *mut u8,
                presence.as_mut_ptr(),
            )
            .unwrap();
        }
        (values, presence)
    }

    #[test]
    fn direct_output_matches_java_numeric_layout() {
        let int32 = Int32Array::from(vec![Some(-7), None, Some(42)]);
        assert_eq!(
            copied_output(&int32, PhysicalType::INT32),
            (vec![-7, 0, 42], vec![0b101])
        );

        let int64 = Int64Array::from(vec![Some(11), None, Some(-19)]);
        assert_eq!(
            copied_output(&int64, PhysicalType::INT64),
            (vec![11, 0, -19], vec![0b101])
        );

        let float32 = Float32Array::from(vec![Some(1.5), None, Some(-2.25)]);
        assert_eq!(
            copied_output(&float32, PhysicalType::FLOAT),
            (
                vec![1.5f32.to_bits() as i64, 0, (-2.25f32).to_bits() as i64],
                vec![0b101]
            )
        );

        let float64 = Float64Array::from(vec![Some(3.5), None, Some(-9.25)]);
        assert_eq!(
            copied_output(&float64, PhysicalType::DOUBLE),
            (
                vec![3.5f64.to_bits() as i64, 0, (-9.25f64).to_bits() as i64],
                vec![0b101]
            )
        );

        let boolean = BooleanArray::from(vec![Some(true), None, Some(false)]);
        assert_eq!(
            copied_output(&boolean, PhysicalType::BOOLEAN),
            (vec![1, 0, 0], vec![0b101])
        );
    }

    #[test]
    fn direct_output_handles_sliced_validity_bitmaps() {
        let base = Int64Array::from(vec![Some(99), Some(7), None, Some(-2)]);
        let sliced = base.slice(1, 3);
        assert_eq!(
            copied_output(&sliced, PhysicalType::INT64),
            (vec![7, 0, -2], vec![0b101])
        );
    }

    #[test]
    fn direct_binary_output_matches_java_page_layout() {
        let array = StringArray::from(vec![Some("alpha"), None, Some("z")]);
        let required = binary_value_bytes(&array).unwrap();
        let mut values = vec![0u8; required];
        let mut offsets = vec![-1i32; array.len() + 1];
        let mut presence = vec![-1i64; array.len().div_ceil(64)];
        unsafe {
            copy_binary_array_out(
                &array,
                values.as_mut_ptr(),
                offsets.as_mut_ptr(),
                presence.as_mut_ptr(),
            )
            .unwrap();
        }
        assert_eq!(values, b"alphaz");
        assert_eq!(offsets, vec![0, 5, 5, 6]);
        assert_eq!(presence, vec![0b101]);
    }

    #[test]
    fn retained_binary_reader_uses_arrow_batch_output() {
        let (mut cursor, _runtime) = open_parquet_fixture(parquet_binary_fixture(4), 4);
        assert_eq!(cursor.physical_type, PhysicalType::BYTE_ARRAY);
        let batch = cursor.next_batch(0).unwrap();
        let array = batch.column(0).as_ref();
        let required = binary_value_bytes(array).unwrap();
        let mut values = vec![0u8; required];
        let mut offsets = vec![-1i32; batch.num_rows() + 1];
        let mut presence = vec![-1i64; batch.num_rows().div_ceil(64)];
        unsafe {
            copy_binary_array_out(
                array,
                values.as_mut_ptr(),
                offsets.as_mut_ptr(),
                presence.as_mut_ptr(),
            )
            .unwrap();
        }
        assert_eq!(values, b"alphazomega");
        assert_eq!(offsets, vec![0, 5, 5, 6, 11]);
        assert_eq!(presence, vec![0b1101]);
    }

    #[test]
    fn retained_repeated_numeric_reader_uses_list_offsets() {
        let (mut cursor, _runtime) = open_parquet_fixture(parquet_repeated_numeric_fixture(), 4);
        assert!(cursor.repeated);
        let batch = cursor.next_batch(0).unwrap();
        let array = batch.column(0).as_ref();
        assert_eq!(repeated_value_count(array).unwrap(), 5);
        let values = repeated_values(array).unwrap();
        assert_eq!(
            values
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[3, 1, 8, 5, 8]
        );
        let mut row_offsets = vec![-1; batch.num_rows() + 1];
        unsafe {
            write_repeated_offsets(array, row_offsets.as_mut_ptr()).unwrap();
        }
        assert_eq!(row_offsets, vec![0, 2, 2, 2, 5]);
    }

    #[test]
    fn retained_repeated_binary_reader_uses_two_level_offsets() {
        let (mut cursor, _runtime) = open_parquet_fixture(parquet_repeated_binary_fixture(), 4);
        assert!(cursor.repeated);
        let batch = cursor.next_batch(0).unwrap();
        let array = batch.column(0).as_ref();
        let values = repeated_values(array).unwrap();
        let required = binary_value_bytes(values.as_ref()).unwrap();
        let mut bytes = vec![0u8; required];
        let mut element_offsets = vec![-1; values.len() + 1];
        let mut row_offsets = vec![-1; batch.num_rows() + 1];
        unsafe {
            write_repeated_offsets(array, row_offsets.as_mut_ptr()).unwrap();
            copy_binary_values_out(
                values.as_ref(),
                bytes.as_mut_ptr(),
                element_offsets.as_mut_ptr(),
            )
            .unwrap();
        }
        assert_eq!(bytes, b"betaalphaomega");
        assert_eq!(element_offsets, vec![0, 4, 9, 14]);
        assert_eq!(row_offsets, vec![0, 2, 2, 2, 3]);
    }

    #[test]
    fn forward_jump_uses_arrow_skip_without_fetching_intermediate_pages() {
        let (mut cursor, _runtime) = open_fixture(1, 8);
        let stats = Arc::clone(&cursor.stats);

        let first = cursor.next_batch(0).unwrap();
        assert_eq!(int64_values(&first), (0..8).collect::<Vec<_>>());
        assert_eq!(stats.count.load(Ordering::Relaxed), 1);

        let target = (ROWS_PER_PAGE * 5 + 7) as i64;
        let jumped = cursor.next_batch(target).unwrap();
        assert_eq!(
            int64_values(&jumped),
            (target..target + 8).collect::<Vec<_>>()
        );
        assert_eq!(
            stats.count.load(Ordering::Relaxed),
            2,
            "Arrow should fetch only the first and target data pages"
        );
    }

    #[test]
    fn all_null_page_is_skipped_without_fetch_or_decode() {
        let (mut cursor, _runtime) =
            open_parquet_fixture(parquet_fixture_with_all_null_page(ROWS_PER_PAGE), 8);
        assert!(cursor.reader.pages()[0].all_null);

        let reads_before = cursor.stats.count.load(Ordering::Relaxed);
        let first = cursor.next_batch(17).unwrap();
        assert_eq!(first.num_rows(), 8);
        assert_eq!(first.column(0).null_count(), 8);
        assert_eq!(cursor.reader.position(), 25);
        assert_eq!(
            cursor.stats.count.load(Ordering::Relaxed),
            reads_before,
            "all-null page must be skipped from OffsetIndex metadata"
        );

        let second = cursor.next_batch(25).unwrap();
        assert_eq!(second.num_rows(), 16);
        assert_eq!(second.column(0).null_count(), 16);
        assert_eq!(
            cursor.stats.count.load(Ordering::Relaxed),
            reads_before,
            "bounded synthetic batches must not revisit the skipped page"
        );

        let non_null = cursor.next_batch(ROWS_PER_PAGE as i64).unwrap();
        assert_eq!(non_null.column(0).null_count(), 0);
        assert!(
            cursor.stats.count.load(Ordering::Relaxed) > reads_before,
            "the following non-null page should be fetched"
        );
    }

    #[test]
    fn local_file_cursor_reuses_retained_descriptor_for_page_reads() {
        let runtime = Arc::new(Builder::new_current_thread().enable_all().build().unwrap());
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&parquet_fixture_with_page_rows(1, ROWS_PER_PAGE))
            .unwrap();
        file.flush().unwrap();
        let filename = file.path().to_str().unwrap();
        let mut cursor = runtime
            .block_on(DocValuesCursor::open(
                filename,
                "value",
                8,
                None,
                None,
                Arc::clone(&runtime),
            ))
            .unwrap();
        let stats = Arc::clone(&cursor.stats);

        assert_eq!(
            int64_values(&cursor.next_batch(0).unwrap()),
            (0..8).collect::<Vec<_>>()
        );
        let target = (ROWS_PER_PAGE * 5 + 7) as i64;
        assert_eq!(
            int64_values(&cursor.next_batch(target).unwrap()),
            (target..target + 8).collect::<Vec<_>>()
        );
        assert_eq!(
            stats.count.load(Ordering::Relaxed),
            0,
            "local page reads must use the retained synchronous file descriptor"
        );
    }

    #[test]
    fn retained_arrow_reader_crosses_row_groups_and_rejects_backward_seeks() {
        let (mut cursor, _runtime) = open_fixture(2, 8);
        let second_rg = (ROWS_PER_PAGE * 8) as i64;

        let batch = cursor.next_batch(second_rg + 11).unwrap();
        assert_eq!(
            int64_values(&batch),
            (second_rg + 11..second_rg + 19).collect::<Vec<_>>()
        );
        let error = cursor.next_batch(second_rg - 1).unwrap_err();
        assert!(error.contains("backward seek"), "{error}");
    }

    #[test]
    fn adaptive_batches_grow_and_stop_at_page_boundaries() {
        let (mut cursor, _runtime) = open_fixture(1, 8);

        assert_eq!(cursor.next_batch(0).unwrap().num_rows(), 8);
        assert_eq!(cursor.next_batch(8).unwrap().num_rows(), 16);
        assert_eq!(cursor.next_batch(24).unwrap().num_rows(), 32);
        assert_eq!(
            cursor.next_batch(56).unwrap().num_rows(),
            8,
            "a batch must not decode across a data-page boundary"
        );
    }

    #[test]
    fn dense_access_retains_the_grown_window_across_pages() {
        let (mut cursor, _runtime) = open_fixture(1, 8);

        assert_eq!(cursor.next_batch(0).unwrap().num_rows(), 8);
        assert_eq!(cursor.next_batch(8).unwrap().num_rows(), 16);
        assert_eq!(cursor.next_batch(24).unwrap().num_rows(), 32);
        assert_eq!(cursor.next_batch(56).unwrap().num_rows(), 8);
        assert_eq!(
            cursor.next_batch(64).unwrap().num_rows(),
            ROWS_PER_PAGE,
            "a boundary-clamped read must not reset dense-scan growth"
        );
    }

    #[test]
    fn small_forward_skip_keeps_growing_the_window() {
        let (mut cursor, _runtime) = open_fixture(1, 8);

        assert_eq!(cursor.next_batch(0).unwrap().num_rows(), 8);
        // Skips no larger than the current window are page-scale dense access
        // (e.g. a moderately selective filter) and must not stall the ramp-up.
        assert_eq!(cursor.next_batch(11).unwrap().num_rows(), 16);
        assert_eq!(cursor.next_batch(27).unwrap().num_rows(), 32);
    }

    #[test]
    fn large_jump_halves_the_window_instead_of_resetting() {
        let (mut cursor, _runtime) = open_fixture(2, 8);

        assert_eq!(cursor.next_batch(0).unwrap().num_rows(), 8);
        assert_eq!(cursor.next_batch(8).unwrap().num_rows(), 16);
        assert_eq!(cursor.next_batch(24).unwrap().num_rows(), 32);

        // Decay is multiplicative (32 → 16), not a reset to the initial window.
        let jump = (ROWS_PER_PAGE * 3 + 5) as i64;
        assert_eq!(cursor.next_batch(jump).unwrap().num_rows(), 16);

        // Repeated large jumps keep halving and bottom out at the initial window.
        let jump = (ROWS_PER_PAGE * 6 + 3) as i64;
        assert_eq!(cursor.next_batch(jump).unwrap().num_rows(), 8);
        let jump = (ROWS_PER_PAGE * 9 + 1) as i64;
        assert_eq!(cursor.next_batch(jump).unwrap().num_rows(), 8);
    }

    #[test]
    fn adaptive_window_is_capped_at_max_batch_size() {
        let rows_per_page = MAX_BATCH_SIZE * 2;
        let (mut cursor, _runtime) =
            open_fixture_with_page_rows(1, rows_per_page, MAX_BATCH_SIZE / 2);

        assert_eq!(cursor.next_batch(0).unwrap().num_rows(), MAX_BATCH_SIZE / 2);
        assert_eq!(
            cursor
                .next_batch((MAX_BATCH_SIZE / 2) as i64)
                .unwrap()
                .num_rows(),
            MAX_BATCH_SIZE
        );
        assert_eq!(
            cursor
                .next_batch((MAX_BATCH_SIZE + MAX_BATCH_SIZE / 2) as i64)
                .unwrap()
                .num_rows(),
            MAX_BATCH_SIZE / 2,
            "the max-sized window is clamped at the page boundary"
        );
        assert_eq!(
            cursor.next_batch(rows_per_page as i64).unwrap().num_rows(),
            MAX_BATCH_SIZE,
            "the adaptive window must not grow beyond the configured maximum"
        );
    }
}

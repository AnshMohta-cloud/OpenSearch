/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Predicate-column name → parquet leaf-index resolution.
//!
//! Resolution is done against the file's OWN schema (derived from the footer)
//! rather than the shared table schema to ensure correct leaf indices under
//! schema evolution (see [`resolve_predicate_parquet_columns`] for details).

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use datafusion::parquet::file::metadata::ParquetMetaData;
use parquet::arrow::parquet_to_arrow_schema;

/// Map the query's predicate-column names to **this file's** parquet leaf
/// indices, resolving against the file's OWN schema so the indices are correct
/// even when the file is missing columns (schema evolution).
///
/// # Why the file's own schema, not the shared table schema
///
/// `StatisticsConverter`/`parquet_column` map a column by finding its position in
/// the supplied arrow schema and then matching that position to a parquet leaf
/// (`get_column_root_idx`). The table schema is the **union** of all
/// files' columns [N]; a given file may physically contain fewer[M] (e.g.
/// the merged file has M leaves — the absent columns are all-null and not
/// written). Resolving against the N-field union therefore maps a column to the
/// WRONG leaf in a M-leaf file. We would then build
/// the scoped ColumnIndex/OffsetIndex at the wrong leaf and leave the real one an
/// empty placeholder — and DataFusion's pruner, which resolves against the file's
/// physical schema, reads the real leaf and panics on the empty `page_locations`
/// (`statistics.rs` `page_locations.last().unwrap()`).
///
/// Deriving the arrow schema from the file footer (`parquet_to_arrow_schema`)
/// gives a 1:1 field↔leaf correspondence for that file, so the resolved index
/// matches what DataFusion dereferences. Columns absent from the file are skipped.
pub fn resolve_predicate_parquet_columns(
    _arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    predicate_column_names: &[String],
) -> Vec<usize> {
    let parquet_schema = metadata.file_metadata().schema_descr();
    // Per-file arrow schema: 1:1 with this file's parquet leaves, so a column's
    // arrow position maps to its true leaf. (The passed `_arrow_schema` is the
    // union table schema and is intentionally NOT used for index resolution —
    // see the doc comment.)
    let file_arrow_schema = match parquet_to_arrow_schema(
        parquet_schema,
        metadata.file_metadata().key_value_metadata(),
    ) {
        Ok(s) => Arc::new(s),
        // If we can't derive the file schema (malformed footer, unsupported type),
        // return empty. Empty is the safe conservative choice:  the caller skips the
        // scoped load and falls back to footer-only.
        Err(_) => return vec![],
    };
    resolve_with_schema(&file_arrow_schema, metadata, predicate_column_names)
}

/// Resolve TWO name-sets (e.g. predicate columns and projection columns) against
/// the same file in one pass. Deriving the per-file arrow schema
/// (`parquet_to_arrow_schema`) is the dominant cost of name→leaf resolution on
/// wide schemas (it rebuilds the whole file's Schema); the two callers in the
/// indexed setup loop previously each rebuilt it, so doing it once here removes a
/// full schema reconstruction per file per query. Pure refactor — each returned
/// Vec is identical to calling `resolve_predicate_parquet_columns` separately.
pub fn resolve_predicate_parquet_columns_pair(
    _union_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    predicate_col_names: &[String],
    projection_col_names: &[String],
) -> (Vec<usize>, Vec<usize>) {
    let parquet_schema = metadata.file_metadata().schema_descr();
    match parquet_to_arrow_schema(
        parquet_schema,
        metadata.file_metadata().key_value_metadata(),
    ) {
        Ok(s) => {
            let file_arrow_schema = Arc::new(s);
            (
                resolve_with_schema(&file_arrow_schema, metadata, predicate_col_names),
                resolve_with_schema(&file_arrow_schema, metadata, projection_col_names),
            )
        }
        // Can't derive the file schema — return empty for both sets.
        Err(_) => (vec![], vec![]),
    }
}

/// Resolve predicate column names → parquet leaf indices against a specific arrow
/// schema, via the same `StatisticsConverter` mapping DataFusion's pruner uses.
pub(super) fn resolve_with_schema(
    arrow_schema: &SchemaRef,
    metadata: &ParquetMetaData,
    predicate_column_names: &[String],
) -> Vec<usize> {
    let parquet_schema = metadata.file_metadata().schema_descr();
    let mut set = HashSet::new();
    for name in predicate_column_names {
        if let Ok(conv) = StatisticsConverter::try_new(name, arrow_schema, parquet_schema) {
            if let Some(idx) = conv.parquet_column_index() {
                set.insert(idx);
                continue;
            }
        }
        // `parquet_column_index()` (arrow-rs `parquet_column`) returns None for a NESTED
        // field (LIST/STRUCT/MAP) because a nested column maps to MULTIPLE parquet leaves,
        // not one. Expand it here: find the field's root index in the arrow schema, then
        // collect every parquet leaf whose column-root-index matches. Without this, a nested
        // column resolves to zero leaves, is dropped from the scoped page-index's real-column
        // set, and gets a single-page placeholder OffsetIndex — which then corrupts a real
        // multi-page read of that column ("output too small for the decompressed data").
        if let Some((root_idx, field)) = arrow_schema.fields().find(name) {
            if field.data_type().is_nested() {
                let mut leaves = Vec::new();
                for leaf in 0..parquet_schema.num_columns() {
                    if parquet_schema.get_column_root_idx(leaf) == root_idx {
                        set.insert(leaf);
                        leaves.push(leaf);
                    }
                }
                log::info!(
                    "[NESTED-POC] scoped-page-index: expanded nested column '{}' (root_idx={}) to parquet leaves {:?} \
                     (POC fix: arrow-rs parquet_column() returns None for nested, so we resolve all leaves ourselves — \
                     without this the leaves get a placeholder OffsetIndex and a real read fails with 'output too small')",
                    name,
                    root_idx,
                    leaves
                );
            }
        }
    }
    set.into_iter().collect()
}

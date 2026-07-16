/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import java.util.List;

import org.apache.calcite.rel.type.RelDataType;

/**
 * [NESTED-POC] Describes a hand-authored N1-rewritten nested query, enough for the DataFusion
 * fragment convertor to assemble the Substrait plan. The general shape is:
 *
 * <pre>
 *   LEFT  = Scan(index)                              // intact parent rows (all columns)
 *   RIGHT = Scan(index) -> UNNEST(unnestColumn)
 *                       -> Filter(filterField &gt; threshold)
 *                       -> distinct(groupByColumn)   // matching parent row-ids
 *   LEFT SEMI JOIN on groupByColumn                  // keep intact parents that matched
 *   -> Project(projection)                           // return whatever the user asked
 * </pre>
 *
 * The semi-join back to the intact scan is what lets the query return arbitrary output (e.g.
 * {@code select *}, {@code select title, views}) rather than just the row-id: UNNEST destroys the
 * nested array, so the intact parent columns must be recovered by joining the matched ids back.
 *
 * <p>Why a descriptor and not the finished bytes: the Substrait proto is assembled with isthmus /
 * {@code io.substrait.proto} classes, which live in the analytics-backend-datafusion module — not
 * in the test-ppl-frontend where the query is recognised. So the front-end passes this lightweight,
 * Calcite-only descriptor (carried across the transport thread hop on
 * {@code QueryRequestContext#n1Descriptor()} — see {@link NestedPocOverride}) and the
 * convertor builds the bytes. {@code baseRowType} is the row type of the plain {@code source=index}
 * scan, used to emit the ReadRel's base schema so it matches what DataFusion infers from Parquet.
 *
 * <p>{@code projection} is the list of parent columns to return (from {@code baseRowType}); an empty
 * list means "all parent columns" ({@code select *}). Only parent (scalar / nested-array) columns
 * are projectable — the filter's exploded child fields are gone after the semi-join back.
 *
 * <p>This is POC scaffolding standing in for the real customer-query -> N1 rewrite.
 *
 * @opensearch.internal
 */
public record N1Descriptor(
    String indexName,
    String unnestColumn,
    String filterStructField,
    int threshold,
    String groupByColumn,
    List<String> projection,
    RelDataType baseRowType
) {}

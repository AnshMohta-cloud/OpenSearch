/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics;

import org.apache.calcite.rel.type.RelDataType;

/**
 * [NESTED-POC] Describes a hand-authored N1-rewritten nested query, enough for the DataFusion
 * fragment convertor to assemble the Substrait plan
 * {@code Scan -> UNNEST(unnestColumn) -> Filter(filterField > threshold) -> distinct(groupByColumn)}.
 *
 * <p>Why a descriptor and not the finished bytes: the Substrait proto is assembled with isthmus /
 * {@code io.substrait.proto} classes, which live in the analytics-backend-datafusion module — not
 * in the test-ppl-frontend where the query is recognised. So the front-end passes this lightweight,
 * Calcite-only descriptor (carried across the transport thread hop on
 * {@code QueryRequestContext#n1Descriptor()} — see {@link NestedPocOverride}) and the
 * convertor builds the bytes. {@code baseRowType} is the row type of the plain {@code source=index}
 * scan, used to emit the ReadRel's base schema so it matches what DataFusion infers from Parquet.
 *
 * <p>This is POC scaffolding standing in for the real customer-query -> N1 rewrite. It intentionally
 * models only the one predicate shape the POC demonstrates ({@code nestedColumn.field > threshold}).
 *
 * @opensearch.internal
 */
public record N1Descriptor(
    String indexName,
    String unnestColumn,
    String filterStructField,
    int threshold,
    String groupByColumn,
    RelDataType baseRowType
) {}

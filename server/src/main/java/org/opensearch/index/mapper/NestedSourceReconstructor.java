/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.index.mapper;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Reconstructs a nested field's {@code _source} array for one document by reading the primary data-format's
 * columnar file directly, as an alternative to the Parquet-direct (FFI + Dremel) reconstruction that the
 * {@code parquet-data-format} codec performs today. A composite engine's analytical backend (e.g. DataFusion)
 * registers an implementation here; the codec's derived-source path looks it up and, when the feature is enabled,
 * delegates to it — otherwise it falls back to its own reconstruction.
 *
 * <p>Why a static registry rather than dependency injection: the codec's {@code ParquetDocValuesProducer} is a
 * Lucene component with no service-injection point, and the analytical backend lives in a sibling plugin
 * classloader. The two share only the {@code :server} classloader, so a static registry in {@code :server} is the
 * established idiom for the codec to reach a backend-provided capability at runtime (mirrors
 * {@code DataFormatStatsProviderRegistry}). Registration happens once, at plugin construction.
 *
 * <p>Keying by the concrete {@code parquetFilePath} (which the codec already resolves per segment, bound to the
 * search's pinned catalog snapshot) means the implementation reads exactly the same file the query phase saw — so
 * there is no cross-phase snapshot-alignment concern; the file identity carries it.
 */
public interface NestedSourceReconstructor {

    /** Process-wide registry holding the single active reconstructor (or none). */
    AtomicReference<NestedSourceReconstructor> REGISTRY = new AtomicReference<>();

    /** Registers {@code reconstructor} as the active implementation (last registration wins). */
    static void register(NestedSourceReconstructor reconstructor) {
        REGISTRY.set(reconstructor);
    }

    /** The active reconstructor, or {@code null} if no backend has registered one. */
    static NestedSourceReconstructor get() {
        return REGISTRY.get();
    }

    /**
     * Reconstructs the ordered element array of the nested field {@code path} for the row {@code rowId} within the
     * given primary-format file. Element order and duplicates MUST be preserved (the nested {@code inner_hits}
     * offset chain indexes into this array by position); a null/absent leaf omits its key; an empty sub-array is
     * omitted — matching the codec's own reconstruction contract so the two paths are byte-identical.
     *
     * @param parquetFilePath absolute path of the primary-format columnar file backing this segment
     * @param writerGeneration the file's writer generation (as the codec resolved it)
     * @param path             the nested field's full path (e.g. {@code products.variants.specs})
     * @param rowId            the document's row id within the file ({@code __row_id__})
     * @param mapperService    the shard's mapping, used to render each leaf faithfully by its OpenSearch type
     * @return the reconstructed element list, or {@code null} if this reconstructor cannot serve the request
     *         (the caller then falls back to its own reconstruction)
     */
    List<Map<String, Object>> reconstructNested(
        String parquetFilePath,
        long writerGeneration,
        String path,
        long rowId,
        MapperService mapperService
    ) throws IOException;

    /**
     * Reconstructs, for one row, the ordered per-element values of a single nested-child leaf field — the values
     * a nested range/aggregation reads, one per child element of {@code owningPath} in flattened document order.
     * This is the query-phase counterpart of {@link #reconstructNested}: the codec scatters these values onto child
     * Lucene docIds using its {@code _nested_path} element offsets (which are the same flatten order, so element
     * {@code k} here is the k-th child of {@code owningPath} in the block).
     *
     * <p>Each returned entry is the leaf's physically-decoded value for that element, or {@code null} if the element
     * exists but the leaf is absent (so the returned list length equals the element count and indices line up with
     * the child-doc offsets). Returns {@code null} if this reconstructor cannot serve the request.
     *
     * @param parquetFilePath absolute path of the primary-format columnar file backing this segment
     * @param owningPath      the leaf's owning nested path (e.g. {@code products.variants.specs})
     * @param leafName        the leaf's simple field name under {@code owningPath} (e.g. {@code v})
     * @param rowId           the document's row id within the file ({@code __row_id__})
     * @param mapperService   the shard's mapping
     * @return ordered per-element leaf values (nulls preserved), or {@code null} if unservable
     */
    List<Object> reconstructNestedLeaf(
        String parquetFilePath,
        String owningPath,
        String leafName,
        long rowId,
        MapperService mapperService
    ) throws IOException;
}

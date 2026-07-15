/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import com.google.protobuf.Any;
import io.substrait.isthmus.TypeConverter;
import io.substrait.proto.AggregateRel;
import io.substrait.proto.Expression;
import io.substrait.proto.ExtensionSingleRel;
import io.substrait.proto.FilterRel;
import io.substrait.proto.FunctionArgument;
import io.substrait.proto.NamedStruct;
import io.substrait.proto.Plan;
import io.substrait.proto.PlanRel;
import io.substrait.proto.ReadRel;
import io.substrait.proto.Rel;
import io.substrait.proto.RelCommon;
import io.substrait.proto.RelRoot;
import io.substrait.proto.SimpleExtensionDeclaration;
import io.substrait.proto.Type;
import io.substrait.type.proto.TypeProtoConverter;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.N1Descriptor;

import java.util.List;

/**
 * [NESTED-POC] Hand-assembles the Substrait plan for an N1-rewritten nested predicate query,
 * bypassing isthmus (which cannot emit relational UNNEST).
 *
 * <p>For {@code where <unnestColumn>.<field> > <threshold>} it builds:
 * <pre>
 *   ReadRel(index, base_schema)                       // the parent rows, comments = LIST&lt;STRUCT&gt;
 *     -> ExtensionSingleRel(detail.type_url="unnest:&lt;col&gt;")   // our unnest marker; Rust consumer
 *                                                               //   turns it into LogicalPlan::Unnest
 *     -> FilterRel( gt( struct_field(col, field), literal(threshold) ) )
 *     -> AggregateRel( group by &lt;groupByColumn&gt; )     // distinct parent row-ids
 * </pre>
 *
 * <p>The `gt` function is declared once in {@code Plan.extensions}; the Rust consumer resolves it by
 * name via the function registry. Field references are positional against the ReadRel row type —
 * unnesting the LIST changes {@code comments}' type (LIST&lt;STRUCT&gt; -> STRUCT) but not its column
 * position, so the same index is valid for the struct-field reference in the filter. The trailing
 * Aggregate on {@code __row_id__} both de-duplicates to parent rows AND ensures the plan bytes carry
 * the {@code __row_id__} needle that routes execution to the unnest-aware indexed executor.
 *
 * <p>POC scaffolding standing in for the real customer-query -> N1 rewrite. Grep: NESTED-POC.
 */
final class N1SubstraitBuilder {

    private static final Logger LOGGER = LogManager.getLogger(N1SubstraitBuilder.class);

    /** Anchor for the `gt` comparison function declared in Plan.extensions. */
    private static final int GT_FUNCTION_ANCHOR = 1;

    private N1SubstraitBuilder() {}

    static byte[] build(N1Descriptor d, TypeProtoConverter typeProtoConverter) {
        RelDataType rowType = d.baseRowType();
        List<RelDataTypeField> fields = rowType.getFieldList();

        // Positional index of the nested column in the scan row type.
        int unnestColIdx = indexOf(fields, d.unnestColumn());
        if (unnestColIdx < 0) {
            throw new IllegalStateException("[NESTED-POC] unnest column '" + d.unnestColumn() + "' not in row type " + rowType.getFieldNames());
        }
        // Struct-field index of the filtered field within the nested column's element struct.
        RelDataType structType = fields.get(unnestColIdx).getType().getComponentType() != null
            ? fields.get(unnestColIdx).getType().getComponentType()
            : fields.get(unnestColIdx).getType();
        int structFieldIdx = structFieldIndex(structType, d.filterStructField());
        if (structFieldIdx < 0) {
            throw new IllegalStateException(
                "[NESTED-POC] struct field '" + d.filterStructField() + "' not in nested column '" + d.unnestColumn() + "'"
            );
        }
        int structFieldCount = structType.getFieldList().size();

        // The Filter/Aggregate sit ABOVE a double UNNEST of the nested column (list->struct, then
        // struct->top-level fields — see UnnestConsumer). DataFusion's unnest expands the struct
        // IN PLACE: the nested column at `unnestColIdx` is replaced by its `structFieldCount`
        // fields (named `col.field`), and every column after it shifts right by structFieldCount-1.
        // So field references above the unnest are positional against this POST-UNNEST layout, NOT
        // the ReadRel layout. (Verified empirically with a DataFusion unnest probe.)
        //   filtered field  ->  unnestColIdx + structFieldIdx        (a top-level column now)
        //   __row_id__       ->  originalRowIdIdx + (structFieldCount - 1)
        // __row_id__ is a PHYSICAL parquet column appended after the user fields (ArrowSchemaBuilder
        // writes it after them); it is not in the logical Calcite row type, so its pre-unnest index
        // is fields.size(). DataFusion widens the scan to the physical parquet schema so it resolves.
        int postUnnestFilterFieldIdx = unnestColIdx + structFieldIdx;
        int originalRowIdIdx = fields.size();
        int postUnnestGroupByIdx = originalRowIdIdx + (structFieldCount - 1);

        // --- ReadRel: the parent-row scan, with a base schema matching what DataFusion infers ---
        // Start from the user-field NamedStruct isthmus derives, then append the physical
        // __row_id__ (BIGINT) column so the group-by can reference it positionally.
        NamedStruct userSchema = TypeConverter.DEFAULT.toNamedStruct(rowType).toProto(typeProtoConverter);
        NamedStruct baseSchema = userSchema.toBuilder()
            .addNames(d.groupByColumn())
            .setStruct(userSchema.getStruct().toBuilder().addTypes(nullableI64()).build())
            .build();
        ReadRel readRel = ReadRel.newBuilder()
            .setCommon(directCommon())
            .setNamedTable(ReadRel.NamedTable.newBuilder().addNames(d.indexName()).build())
            .setBaseSchema(baseSchema)
            .build();
        Rel scan = Rel.newBuilder().setRead(readRel).build();

        // --- ExtensionSingleRel: our unnest marker over the scan (Rust consumer -> LogicalPlan::Unnest) ---
        Rel unnest = Rel.newBuilder()
            .setExtensionSingle(
                ExtensionSingleRel.newBuilder()
                    .setCommon(directCommon())
                    .setInput(scan)
                    .setDetail(Any.newBuilder().setTypeUrl("unnest:" + d.unnestColumn()).build())
                    .build()
            )
            .build();

        // --- FilterRel: gt( <col>.<field> , threshold ) ---
        // After the double unnest, <col>.<field> is a TOP-LEVEL column (single-level StructField
        // reference, no child) — the only shape DataFusion's Substrait consumer accepts.
        Expression fieldRef = fieldReference(postUnnestFilterFieldIdx);
        Expression literal = Expression.newBuilder()
            .setLiteral(Expression.Literal.newBuilder().setI32(d.threshold()).setNullable(true).build())
            .build();
        Expression gt = Expression.newBuilder()
            .setScalarFunction(
                Expression.ScalarFunction.newBuilder()
                    .setFunctionReference(GT_FUNCTION_ANCHOR)
                    .setOutputType(nullableBool())
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldRef).build())
                    .addArguments(FunctionArgument.newBuilder().setValue(literal).build())
                    .build()
            )
            .build();
        Rel filter = Rel.newBuilder()
            .setFilter(FilterRel.newBuilder().setCommon(directCommon()).setInput(unnest).setCondition(gt).build())
            .build();

        // --- AggregateRel: group by __row_id__ (distinct parents; also carries the routing needle) ---
        Rel aggregate = Rel.newBuilder()
            .setAggregate(
                AggregateRel.newBuilder()
                    .setCommon(directCommon())
                    .setInput(filter)
                    .addGroupings(
                        AggregateRel.Grouping.newBuilder()
                            .addGroupingExpressions(fieldReference(postUnnestGroupByIdx))
                            .build()
                    )
                    .build()
            )
            .build();

        // --- Wrap in a plan; declare the `gt` function extension ---
        RelRoot root = RelRoot.newBuilder().setInput(aggregate).addNames(d.groupByColumn()).build();
        Plan plan = Plan.newBuilder()
            .addExtensions(
                SimpleExtensionDeclaration.newBuilder()
                    .setExtensionFunction(
                        SimpleExtensionDeclaration.ExtensionFunction.newBuilder()
                            .setFunctionAnchor(GT_FUNCTION_ANCHOR)
                            .setName("gt")
                            .build()
                    )
                    .build()
            )
            .addRelations(PlanRel.newBuilder().setRoot(root).build())
            .build();

        byte[] bytes = SubstraitPlanProtoRewriter.rewrite(plan).toByteArray();
        LOGGER.info(
            "[NESTED-POC] N1SubstraitBuilder: index='{}' unnestCol='{}'(scanIdx {}) filter '{}.{}' -> postUnnestIdx {} > {}, "
                + "groupBy '{}' -> postUnnestIdx {} ({} struct fields) -> {} bytes",
            d.indexName(),
            d.unnestColumn(),
            unnestColIdx,
            d.unnestColumn(),
            d.filterStructField(),
            postUnnestFilterFieldIdx,
            d.threshold(),
            d.groupByColumn(),
            postUnnestGroupByIdx,
            structFieldCount,
            bytes.length
        );
        return bytes;
    }

    private static RelCommon directCommon() {
        return RelCommon.newBuilder().setDirect(RelCommon.Direct.newBuilder().build()).build();
    }

    private static Type nullableBool() {
        return Type.newBuilder()
            .setBool(Type.Boolean.newBuilder().setNullability(Type.Nullability.NULLABILITY_NULLABLE).build())
            .build();
    }

    private static Type nullableI64() {
        return Type.newBuilder()
            .setI64(Type.I64.newBuilder().setNullability(Type.Nullability.NULLABILITY_NULLABLE).build())
            .build();
    }

    /** A top-level (positional) column reference: root -> struct field {@code idx}. */
    private static Expression fieldReference(int idx) {
        return Expression.newBuilder()
            .setSelection(
                Expression.FieldReference.newBuilder()
                    .setRootReference(Expression.FieldReference.RootReference.newBuilder().build())
                    .setDirectReference(
                        Expression.ReferenceSegment.newBuilder()
                            .setStructField(Expression.ReferenceSegment.StructField.newBuilder().setField(idx).build())
                            .build()
                    )
                    .build()
            )
            .build();
    }

    private static int indexOf(List<RelDataTypeField> fields, String name) {
        for (int i = 0; i < fields.size(); i++) {
            if (fields.get(i).getName().equals(name)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Index of {@code fieldName} within the element struct of a nested (ARRAY&lt;ROW&gt;) column type.
     * The column type is ARRAY whose component is a ROW; we search the ROW's fields.
     */
    private static int structFieldIndex(RelDataType nestedColumnType, String fieldName) {
        RelDataType structType = nestedColumnType.getComponentType() != null ? nestedColumnType.getComponentType() : nestedColumnType;
        List<RelDataTypeField> structFields = structType.getFieldList();
        for (int i = 0; i < structFields.size(); i++) {
            if (structFields.get(i).getName().equals(fieldName)) {
                return i;
            }
        }
        return -1;
    }
}

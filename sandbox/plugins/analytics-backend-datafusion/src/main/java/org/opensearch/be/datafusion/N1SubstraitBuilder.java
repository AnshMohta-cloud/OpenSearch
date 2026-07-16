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
import io.substrait.proto.JoinRel;
import io.substrait.proto.NamedStruct;
import io.substrait.proto.Plan;
import io.substrait.proto.PlanRel;
import io.substrait.proto.ProjectRel;
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

import java.util.ArrayList;
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

    /** Anchor for the `equal` function (semi-join condition) declared in Plan.extensions. */
    private static final int EQUAL_FUNCTION_ANCHOR = 2;

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

        // --- AggregateRel: group by __row_id__ → the distinct MATCHING PARENT row-ids (RIGHT branch) ---
        // Output schema of this branch = [__row_id__] (one column).
        Rel matchingRowIds = Rel.newBuilder()
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

        // --- LEFT branch: a second, INTACT scan of the same table (all parent columns incl. the
        // whole comments array + __row_id__). This is what lets us return arbitrary output: UNNEST
        // destroyed the array on the RIGHT branch, so the intact parent rows are recovered here and
        // filtered to the matching ids via the semi-join. Same base schema as the RIGHT scan. ---
        int leftFieldCount = fields.size() + 1; // user fields + appended __row_id__
        int leftRowIdIdx = fields.size(); // __row_id__ position in the intact scan
        Rel intactScan = Rel.newBuilder()
            .setRead(
                ReadRel.newBuilder()
                    .setCommon(directCommon())
                    .setNamedTable(ReadRel.NamedTable.newBuilder().addNames(d.indexName()).build())
                    .setBaseSchema(baseSchema)
                    .build()
            )
            .build();

        // --- JoinRel (LEFT SEMI): keep intact-scan rows whose __row_id__ is in the matching set.
        // The join condition references the CONCATENATED left++right schema: left.__row_id__ at
        // leftRowIdIdx, right.__row_id__ at leftFieldCount + 0 (right branch outputs just [__row_id__]).
        Expression joinCond = Expression.newBuilder()
            .setScalarFunction(
                Expression.ScalarFunction.newBuilder()
                    .setFunctionReference(EQUAL_FUNCTION_ANCHOR)
                    .setOutputType(nullableBool())
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldReference(leftRowIdIdx)).build())
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldReference(leftFieldCount)).build())
                    .build()
            )
            .build();
        Rel semiJoin = Rel.newBuilder()
            .setJoin(
                JoinRel.newBuilder()
                    .setCommon(directCommon())
                    .setLeft(intactScan)
                    .setRight(matchingRowIds)
                    .setType(JoinRel.JoinType.JOIN_TYPE_LEFT_SEMI)
                    .setExpression(joinCond)
                    .build()
            )
            .build();

        // --- ProjectRel: subset to exactly the requested output columns (TIGHT output schema).
        // A LEFT SEMI join outputs the LEFT (intact) columns in order: [user fields..., __row_id__]
        // (leftFieldCount total). We add a ProjectRel selecting the requested columns. Substrait's
        // ProjectRel outputs (all input cols ++ its expressions), so to emit ONLY our selected columns
        // we set RelCommon.emit output_mapping to the expression outputs (offsets leftFieldCount ..
        // leftFieldCount+projCount). This matches how the engine natively subsets columns; DataFusion
        // hard-fails if RelRoot.names count != produced width, so the output MUST be tight. Empty
        // projection = select * = all left user columns (the whole comments array included). ---
        List<Integer> selectedLeftIdx = new ArrayList<>();
        List<String> outputNames = new ArrayList<>();
        if (d.projection().isEmpty()) {
            for (int i = 0; i < fields.size(); i++) {
                selectedLeftIdx.add(i);
                outputNames.add(fields.get(i).getName());
            }
        } else {
            for (String col : d.projection()) {
                int idx = col.equals(d.groupByColumn()) ? leftRowIdIdx : indexOf(fields, col);
                if (idx < 0) {
                    throw new IllegalStateException("[NESTED-POC] projection column '" + col + "' not in scan schema " + rowType.getFieldNames());
                }
                selectedLeftIdx.add(idx);
                outputNames.add(col);
            }
        }
        ProjectRel.Builder projectBuilder = ProjectRel.newBuilder().setInput(semiJoin);
        RelCommon.Emit.Builder emit = RelCommon.Emit.newBuilder();
        for (int i = 0; i < selectedLeftIdx.size(); i++) {
            projectBuilder.addExpressions(fieldReference(selectedLeftIdx.get(i)));
            // ProjectRel output = [leftFieldCount input cols] ++ [our expressions]; select the expr outputs.
            emit.addOutputMapping(leftFieldCount + i);
        }
        projectBuilder.setCommon(RelCommon.newBuilder().setEmit(emit.build()).build());
        Rel project = Rel.newBuilder().setProject(projectBuilder.build()).build();

        // --- Wrap in a plan; declare the `gt` and `equal` function extensions ---
        // RelRoot.names = exactly the tight output columns (depth-first; these are scalars here, so
        // the flattened list equals the top-level names). A nested column in `select *` would need
        // its struct children appended depth-first — handled by flattenOutputNames on the normal path;
        // for the POC the projected columns are scalars or the whole array is emitted as one name.
        RelRoot root = RelRoot.newBuilder().setInput(project).addAllNames(flattenNames(outputNames, selectedLeftIdx, fields, d.groupByColumn())).build();
        Plan plan = Plan.newBuilder()
            .addExtensions(
                SimpleExtensionDeclaration.newBuilder()
                    .setExtensionFunction(
                        SimpleExtensionDeclaration.ExtensionFunction.newBuilder().setFunctionAnchor(GT_FUNCTION_ANCHOR).setName("gt").build()
                    )
                    .build()
            )
            .addExtensions(
                SimpleExtensionDeclaration.newBuilder()
                    .setExtensionFunction(
                        SimpleExtensionDeclaration.ExtensionFunction.newBuilder().setFunctionAnchor(EQUAL_FUNCTION_ANCHOR).setName("equal").build()
                    )
                    .build()
            )
            .addRelations(PlanRel.newBuilder().setRoot(root).build())
            .build();

        Plan finalPlan = SubstraitPlanProtoRewriter.rewrite(plan);
        byte[] bytes = finalPlan.toByteArray();
        LOGGER.info(
            "[NESTED-POC] N1SubstraitBuilder: index='{}' filter '{}.{}'(postUnnestIdx {}) > {}, groupBy '{}'(postUnnestIdx {}), "
                + "semi-join back on '{}', output columns {} -> {} bytes",
            d.indexName(),
            d.unnestColumn(),
            d.filterStructField(),
            postUnnestFilterFieldIdx,
            d.threshold(),
            d.groupByColumn(),
            postUnnestGroupByIdx,
            d.groupByColumn(),
            outputNames,
            bytes.length
        );
        // [NESTED-POC] Full readable hand-built N1 Substrait plan shipped to the data node — shows the
        // Read → ExtensionSingle(unnest) → Filter → Aggregate → (intact Read) → LEFT SEMI Join →
        // Project(emit) rel tree + gt/equal extensions. This is the exact wire content the Rust
        // unnest-aware consumer receives. Grep: NESTED-POC.
        LOGGER.info("[NESTED-POC] Substrait plan ({} bytes) [hand-built N1 path]:\n{}", bytes.length, finalPlan);
        return bytes;
    }

    /**
     * Depth-first flattened RelRoot.names for the output columns. Scalar columns contribute their
     * own name; the nested unnest column (ARRAY&lt;ROW&gt;) contributes its own name followed, depth-first,
     * by its struct child names (list element recurses without consuming a name for the collection).
     * DataFusion's make_renamed_schema requires the flattened name count to equal the produced width.
     */
    private static List<String> flattenNames(
        List<String> outputNames,
        List<Integer> selectedLeftIdx,
        List<RelDataTypeField> fields,
        String groupByColumn
    ) {
        List<String> flat = new ArrayList<>();
        for (int i = 0; i < outputNames.size(); i++) {
            String name = outputNames.get(i);
            flat.add(name);
            int leftIdx = selectedLeftIdx.get(i);
            if (leftIdx < fields.size()) {
                RelDataType t = fields.get(leftIdx).getType();
                RelDataType struct = t.getComponentType() != null ? t.getComponentType() : (t.isStruct() ? t : null);
                if (struct != null) {
                    // ARRAY<ROW> or ROW column: append the struct child names depth-first.
                    for (RelDataTypeField child : struct.getFieldList()) {
                        flat.add(child.getName());
                    }
                }
            }
        }
        return flat;
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

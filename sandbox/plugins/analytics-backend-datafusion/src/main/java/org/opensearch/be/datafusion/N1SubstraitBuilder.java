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
import org.opensearch.analytics.N1Predicate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    /** `equal` function name (used for the semi-join condition + `=` comparisons). */
    private static final String EQUAL_FN = "equal";

    private N1SubstraitBuilder() {}

    static byte[] build(N1Descriptor d, TypeProtoConverter typeProtoConverter) {
        RelDataType rowType = d.baseRowType();
        List<RelDataTypeField> fields = rowType.getFieldList();

        // Simulate the POST-UNNEST column layout. DataFusion unnests each path level twice
        // (list->struct, then struct expands IN PLACE into `level.child` columns — verified). We
        // reproduce that on the Calcite types to get the exact flat column-name list the Filter/
        // Aggregate see, so field references (which are POSITIONAL in Substrait) land correctly for
        // ANY nesting depth. __row_id__ is a physical parquet column appended after the user fields
        // (not in the Calcite row type); we track it as the last column.
        List<String> layout = new ArrayList<>();
        List<RelDataType> layoutTypes = new ArrayList<>(); // parallel: element/leaf type per column (null for scalars)
        for (RelDataTypeField f : fields) {
            layout.add(f.getName());
            layoutTypes.add(f.getType());
        }
        layout.add(d.groupByColumn()); // __row_id__
        layoutTypes.add(null);

        for (String level : d.unnestPath()) {
            int at = layout.indexOf(level);
            if (at < 0) {
                throw new IllegalStateException("[NESTED-POC] unnest level '" + level + "' not present in layout " + layout);
            }
            RelDataType colType = layoutTypes.get(at);
            RelDataType elem = colType != null && colType.getComponentType() != null ? colType.getComponentType() : colType;
            if (elem == null || !elem.isStruct()) {
                throw new IllegalStateException("[NESTED-POC] unnest level '" + level + "' is not a LIST<STRUCT> (type " + colType + ")");
            }
            // Replace the level column IN PLACE with its struct children named `level.child`.
            layout.remove(at);
            layoutTypes.remove(at);
            List<RelDataTypeField> children = elem.getFieldList();
            for (int i = 0; i < children.size(); i++) {
                RelDataTypeField child = children.get(i);
                layout.add(at + i, level + "." + child.getName());
                layoutTypes.add(at + i, child.getType());
            }
        }

        String deepestLevel = d.unnestPath().get(d.unnestPath().size() - 1);
        // Group-by (__row_id__) index in the final layout.
        int postUnnestGroupByIdx = layout.indexOf(d.groupByColumn());
        if (postUnnestGroupByIdx < 0) {
            throw new IllegalStateException("[NESTED-POC] group-by '" + d.groupByColumn() + "' vanished from layout " + layout);
        }

        // Function-anchor allocator: each distinct scalar function used (comparisons, and/or, plus
        // the join's `equal`) is declared once in Plan.extensions with a stable anchor.
        java.util.LinkedHashMap<String, Integer> fnAnchors = new java.util.LinkedHashMap<>();
        fnAnchors.put(EQUAL_FN, 1); // reserve anchor 1 for the semi-join equality

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

        // --- ExtensionSingleRel: our unnest marker over the scan. The tag carries the full path
        // (comma-separated), and the Rust consumer unnests each level twice (Rust: UnnestConsumer). ---
        String pathSpec = String.join(",", d.unnestPath());
        Rel unnest = Rel.newBuilder()
            .setExtensionSingle(
                ExtensionSingleRel.newBuilder()
                    .setCommon(directCommon())
                    .setInput(scan)
                    .setDetail(Any.newBuilder().setTypeUrl("unnest:" + pathSpec).build())
                    .build()
            )
            .build();

        // --- METRIC aggregate (avg/sum/min/max over a CHILD field) — a DIFFERENT, simpler plan shape
        // than count(). Per HLD §4.3 row 1, `stats avg(comments.score)` is a metric over the
        // unnested (+ optionally filtered) CHILD rows: `SELECT AVG(c.score) FROM blogs b,
        // UNNEST(b.comments) AS c [WHERE <pred>]`. No semi-join back to parents (that's the
        // reverse_nested / count-of-distinct-parents shape). So we short-circuit here: Read → unnest
        // → [filter] → Aggregate(measure(childField)). __row_id__ still rides in base_schema so the
        // plan routes to the unnest-aware indexed executor. ---
        if (d.aggregate() != null && !d.aggregate().isCountStar()) {
            return buildChildMetric(d, layout, layoutTypes, deepestLevel, unnest, fnAnchors);
        }

        // --- FilterRel: the general predicate tree. Each predicate field `f` is a leaf of the deepest
        // level, resolving to the post-unnest column `<deepestLevel>.<f>` (looked up by name in the
        // simulated layout → its positional index). ---
        Expression predicateExpr = buildPredicate(d.predicate(), deepestLevel, layout, fnAnchors);
        Rel filter = Rel.newBuilder()
            .setFilter(FilterRel.newBuilder().setCommon(directCommon()).setInput(unnest).setCondition(predicateExpr).build())
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
                    .setFunctionReference(fnAnchors.get(EQUAL_FN))
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

        // --- Optional AGGREGATE on top (e.g. `| stats count()`). count() of matching docs = count of
        // the distinct matching PARENTS (mirrors vanilla reverse_nested), i.e. a count(*) over the
        // semi-join+project result. Output = a single row, one column named per PPL convention. ---
        Rel outputRel = project;
        List<String> rootNames = flattenNames(outputNames, selectedLeftIdx, fields, d.groupByColumn());
        if (d.aggregate() != null) {
            org.opensearch.analytics.N1Aggregate agg = d.aggregate();
            if (!agg.isCountStar()) {
                throw new IllegalStateException("[NESTED-POC] only count() aggregate is supported so far, got: " + agg.fn());
            }
            int countAnchor = anchorFor(agg.fn().substraitName(), fnAnchors);
            AggregateRel.Measure countMeasure = AggregateRel.Measure.newBuilder()
                .setMeasure(
                    io.substrait.proto.AggregateFunction.newBuilder()
                        .setFunctionReference(countAnchor)
                        .setPhase(io.substrait.proto.AggregationPhase.AGGREGATION_PHASE_INITIAL_TO_RESULT)
                        .setInvocation(io.substrait.proto.AggregateFunction.AggregationInvocation.AGGREGATION_INVOCATION_ALL)
                        .setOutputType(nullableI64())
                        // count() with NO arguments → count(*) (DataFusion special-cases this).
                        .build()
                )
                .build();
            outputRel = Rel.newBuilder()
                .setAggregate(
                    AggregateRel.newBuilder().setCommon(directCommon()).setInput(project).addMeasures(countMeasure).build()
                )
                .build();
            rootNames = List.of(agg.outputColumn());
        }

        // --- Wrap in a plan; declare the scalar+aggregate function extensions ---
        // RelRoot.names = exactly the tight output columns (depth-first for nested; scalar otherwise).
        RelRoot root = RelRoot.newBuilder().setInput(outputRel).addAllNames(rootNames).build();
        // --- Declare every scalar function used (comparisons, and/or, join equal) with its anchor ---
        Plan.Builder planBuilder = Plan.newBuilder();
        for (Map.Entry<String, Integer> e : fnAnchors.entrySet()) {
            planBuilder.addExtensions(
                SimpleExtensionDeclaration.newBuilder()
                    .setExtensionFunction(
                        SimpleExtensionDeclaration.ExtensionFunction.newBuilder().setFunctionAnchor(e.getValue()).setName(e.getKey()).build()
                    )
                    .build()
            );
        }
        Plan plan = planBuilder.addRelations(PlanRel.newBuilder().setRoot(root).build()).build();

        Plan finalPlan = SubstraitPlanProtoRewriter.rewrite(plan);
        byte[] bytes = finalPlan.toByteArray();
        LOGGER.info(
            "[NESTED-POC] N1SubstraitBuilder: index='{}' unnestPath {}, post-unnest layout {}, predicate {}, "
                + "groupBy '{}'(postUnnestIdx {}), semi-join back on '{}', output columns {}, functions {} -> {} bytes",
            d.indexName(),
            d.unnestPath(),
            layout,
            d.predicate(),
            d.groupByColumn(),
            postUnnestGroupByIdx,
            d.groupByColumn(),
            outputNames,
            fnAnchors.keySet(),
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
     * [NESTED-POC] Build the plan for a metric aggregate (avg/sum/min/max) over a nested CHILD field.
     *
     * <p>This is the second nested-aggregate shape (HLD §4.3 row 1): a metric over the matched CHILD
     * elements, NOT a count of distinct parents. Plan:
     * <pre>
     *   Read(index, base_schema incl. __row_id__)
     *     -> ExtensionSingleRel(unnest:&lt;path&gt;)          // child rows, one per deepest element
     *     -> [FilterRel(predicate)]                       // optional WHERE over child fields
     *     -> AggregateRel( measure = avg/sum/min/max( &lt;deepestLevel&gt;.&lt;argField&gt; ) )   // GLOBAL, no grouping
     * </pre>
     *
     * <p>No semi-join and no group-by: the metric is computed directly over the (filtered) child rows,
     * exactly like {@code SELECT AVG(c.score) FROM blogs b, UNNEST(b.comments) AS c WHERE ...}. The
     * argument column {@code <deepestLevel>.<argField>} is resolved positionally against the SAME
     * simulated post-unnest {@code layout} the predicate uses, so it works at any nesting depth. Output
     * type: {@code fp64} for avg (and for sum/min/max over a fp64 arg); otherwise the argument's own
     * type (i64 promotion for integer sum, matching DataFusion). {@code __row_id__} stays in the base
     * schema so the {@code has_row_id} routing gate still selects the unnest-aware indexed executor.
     */
    private static byte[] buildChildMetric(
        N1Descriptor d,
        List<String> layout,
        List<RelDataType> layoutTypes,
        String deepestLevel,
        Rel unnest,
        java.util.Map<String, Integer> fnAnchors
    ) {
        org.opensearch.analytics.N1Aggregate agg = d.aggregate();

        // Optional predicate over the child rows (a bare `stats avg(x)` with no where has predicate==null).
        Rel input = unnest;
        if (d.predicate() != null) {
            Expression predicateExpr = buildPredicate(d.predicate(), deepestLevel, layout, fnAnchors);
            input = Rel.newBuilder()
                .setFilter(FilterRel.newBuilder().setCommon(directCommon()).setInput(unnest).setCondition(predicateExpr).build())
                .build();
        }

        // Build the measure. count() (no argField) → count(*) (no argument); else fn(argCol).
        int fnAnchor = anchorFor(agg.fn().substraitName(), fnAnchors);
        io.substrait.proto.AggregateFunction.Builder measureFn = io.substrait.proto.AggregateFunction.newBuilder()
            .setFunctionReference(fnAnchor)
            .setPhase(io.substrait.proto.AggregationPhase.AGGREGATION_PHASE_INITIAL_TO_RESULT)
            .setInvocation(io.substrait.proto.AggregateFunction.AggregationInvocation.AGGREGATION_INVOCATION_ALL);
        Type measureOutputType;
        if (agg.argField() == null) {
            // Grouped count() → count of child elements per group (count(*), no argument, i64).
            if (!agg.hasGroupBy()) {
                throw new IllegalStateException("[NESTED-POC] ungrouped count() should use the reverse_nested path, not buildChildMetric");
            }
            measureOutputType = nullableI64();
            measureFn.setOutputType(measureOutputType);
        } else {
            String argColumn = deepestLevel + "." + agg.argField();
            int argIdx = layout.indexOf(argColumn);
            if (argIdx < 0) {
                throw new IllegalStateException("[NESTED-POC] metric arg '" + argColumn + "' not in post-unnest layout " + layout);
            }
            measureOutputType = metricOutputType(agg.fn(), layoutTypes.get(argIdx));
            measureFn.setOutputType(measureOutputType).addArguments(FunctionArgument.newBuilder().setValue(fieldReference(argIdx)).build());
        }
        AggregateRel.Measure measure = AggregateRel.Measure.newBuilder().setMeasure(measureFn.build()).build();

        // GROUP BY a child dimension: <deepestLevel>.<groupByField>. Substrait emits grouping keys
        // FIRST then measures, so the output layout is [groupKey, measure]. Ungrouped → one row,
        // [measure] only.
        List<String> rootNames;
        int groupKeyIdx = -1;
        if (agg.hasGroupBy()) {
            String groupCol = deepestLevel + "." + agg.groupByField();
            groupKeyIdx = layout.indexOf(groupCol);
            if (groupKeyIdx < 0) {
                throw new IllegalStateException("[NESTED-POC] group-by field '" + groupCol + "' not in post-unnest layout " + layout);
            }
            // DataFusion's unnest preserves a row for parents whose array is empty/absent (Wayne's
            // empty variants, the Empty doc's no products) — the child columns come out NULL. Vanilla
            // `nested`->`terms` aggregation has NO child docs for those parents, so it emits no NULL
            // bucket. Match that: drop rows whose group key is NULL before grouping (unless a WHERE
            // already filtered them out — harmless then). is_not_null(groupCol).
            int notNullAnchor = anchorFor("is_not_null", fnAnchors);
            Expression notNull = Expression.newBuilder()
                .setScalarFunction(
                    Expression.ScalarFunction.newBuilder()
                        .setFunctionReference(notNullAnchor)
                        .setOutputType(nullableBool())
                        .addArguments(FunctionArgument.newBuilder().setValue(fieldReference(groupKeyIdx)).build())
                        .build()
                )
                .build();
            input = Rel.newBuilder()
                .setFilter(FilterRel.newBuilder().setCommon(directCommon()).setInput(input).setCondition(notNull).build())
                .build();
        }

        AggregateRel.Builder aggRel = AggregateRel.newBuilder().setCommon(directCommon()).setInput(input).addMeasures(measure);
        if (agg.hasGroupBy()) {
            aggRel.addGroupings(AggregateRel.Grouping.newBuilder().addGroupingExpressions(fieldReference(groupKeyIdx)).build());
            rootNames = List.of(agg.groupByOutputColumn(), agg.outputColumn());
        } else {
            rootNames = List.of(agg.outputColumn());
        }
        Rel outputRel = Rel.newBuilder().setAggregate(aggRel.build()).build();

        // HAVING: filter the grouped aggregate result on the measure. In the aggregate's output the
        // group key is field 0 and the measure is field 1; compare field 1 <havingOp> <literal>.
        if (agg.hasHaving()) {
            if (!agg.hasGroupBy()) {
                throw new IllegalStateException("[NESTED-POC] HAVING without GROUP BY is unsupported");
            }
            int havingAnchor = anchorFor(agg.havingOp().substraitName(), fnAnchors);
            Expression havingCond = Expression.newBuilder()
                .setScalarFunction(
                    Expression.ScalarFunction.newBuilder()
                        .setFunctionReference(havingAnchor)
                        .setOutputType(nullableBool())
                        .addArguments(FunctionArgument.newBuilder().setValue(fieldReference(1)).build())
                        .addArguments(FunctionArgument.newBuilder().setValue(literalOf(agg.havingValue())).build())
                        .build()
                )
                .build();
            outputRel = Rel.newBuilder()
                .setFilter(FilterRel.newBuilder().setCommon(directCommon()).setInput(outputRel).setCondition(havingCond).build())
                .build();
        }

        RelRoot root = RelRoot.newBuilder().setInput(outputRel).addAllNames(rootNames).build();
        Plan.Builder planBuilder = Plan.newBuilder();
        for (Map.Entry<String, Integer> e : fnAnchors.entrySet()) {
            planBuilder.addExtensions(
                SimpleExtensionDeclaration.newBuilder()
                    .setExtensionFunction(
                        SimpleExtensionDeclaration.ExtensionFunction.newBuilder().setFunctionAnchor(e.getValue()).setName(e.getKey()).build()
                    )
                    .build()
            );
        }
        Plan plan = planBuilder.addRelations(PlanRel.newBuilder().setRoot(root).build()).build();
        Plan finalPlan = SubstraitPlanProtoRewriter.rewrite(plan);
        byte[] bytes = finalPlan.toByteArray();
        LOGGER.info(
            "[NESTED-POC] N1SubstraitBuilder (CHILD METRIC): index='{}' unnestPath {}, metric {}(argField={}), "
                + "groupBy={}, having={} {}, predicate {}, output columns {}, functions {} -> {} bytes",
            d.indexName(),
            d.unnestPath(),
            agg.fn().substraitName(),
            agg.argField(),
            agg.groupByField(),
            agg.hasHaving() ? agg.havingOp() : "none",
            agg.hasHaving() ? agg.havingValue() : "",
            d.predicate(),
            agg.outputColumns(),
            fnAnchors.keySet(),
            bytes.length
        );
        LOGGER.info("[NESTED-POC] Substrait plan ({} bytes) [hand-built N1 child-metric path]:\n{}", bytes.length, finalPlan);
        return bytes;
    }

    /**
     * Output type of a metric aggregate. {@code avg} is always {@code fp64}. {@code sum} of an integer
     * arg promotes to {@code i64} (DataFusion semantics); {@code sum} of a floating arg stays
     * {@code fp64}. {@code min}/{@code max} preserve the argument's type. Falls back to {@code fp64}
     * when the argument type is unknown. All nullable (an empty child set yields NULL).
     */
    private static Type metricOutputType(org.opensearch.analytics.N1Aggregate.Fn fn, RelDataType argType) {
        boolean floating = isFloating(argType);
        switch (fn) {
            case AVG:
                return nullableFp64();
            case SUM:
                return floating ? nullableFp64() : nullableI64();
            case MIN:
            case MAX:
                if (argType == null) {
                    return nullableFp64();
                }
                if (isCharacter(argType)) {
                    // min/max of a keyword/text field preserves the string type.
                    return nullableString();
                }
                return floating ? nullableFp64() : nullableI64();
            default:
                throw new IllegalStateException("[NESTED-POC] not a metric aggregate: " + fn);
        }
    }

    private static boolean isFloating(RelDataType t) {
        if (t == null) {
            return false;
        }
        switch (t.getSqlTypeName()) {
            case DOUBLE:
            case FLOAT:
            case REAL:
            case DECIMAL:
                return true;
            default:
                return false;
        }
    }

    private static Type nullableFp64() {
        return Type.newBuilder()
            .setFp64(Type.FP64.newBuilder().setNullability(Type.Nullability.NULLABILITY_NULLABLE).build())
            .build();
    }

    private static Type nullableString() {
        return Type.newBuilder()
            .setString(Type.String.newBuilder().setNullability(Type.Nullability.NULLABILITY_NULLABLE).build())
            .build();
    }

    /** True for character (keyword/text → VARCHAR/CHAR) argument types. */
    private static boolean isCharacter(RelDataType t) {
        if (t == null) {
            return false;
        }
        switch (t.getSqlTypeName()) {
            case VARCHAR:
            case CHAR:
                return true;
            default:
                return false;
        }
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

    /**
     * [NESTED-POC] Recursively build the Substrait filter expression for a predicate tree. A
     * comparison becomes a scalar function (equal/gt/lt/...) over the post-unnest struct-field column
     * and a literal; AND/OR become the `and`/`or` scalar functions over their children. Each distinct
     * function name is assigned a stable anchor in {@code fnAnchors} (declared once in Plan.extensions).
     * The comparison field resolves to the post-unnest column {@code <deepestLevel>.<field>}, looked
     * up by name in the simulated {@code layout} to get its positional index.
     */
    private static Expression buildPredicate(
        N1Predicate pred,
        String deepestLevel,
        List<String> layout,
        java.util.Map<String, Integer> fnAnchors
    ) {
        if (pred instanceof N1Predicate.Comparison c) {
            // Predicate fields are leaves of the deepest unnested level → post-unnest column
            // `<deepestLevel>.<field>` (e.g. comments.score, or comments.replies.reactions.by).
            String colName = deepestLevel + "." + c.field();
            int idx = layout.indexOf(colName);
            if (idx < 0) {
                throw new IllegalStateException("[NESTED-POC] predicate field '" + colName + "' not in post-unnest layout " + layout);
            }
            Expression fieldRef = fieldReference(idx);
            Expression literal = literalOf(c.value());
            int anchor = anchorFor(c.op().substraitName(), fnAnchors);
            return Expression.newBuilder()
                .setScalarFunction(
                    Expression.ScalarFunction.newBuilder()
                        .setFunctionReference(anchor)
                        .setOutputType(nullableBool())
                        .addArguments(FunctionArgument.newBuilder().setValue(fieldRef).build())
                        .addArguments(FunctionArgument.newBuilder().setValue(literal).build())
                        .build()
                )
                .build();
        }
        if (pred instanceof N1Predicate.And a) {
            return combine("and", a.children(), deepestLevel, layout, fnAnchors);
        }
        if (pred instanceof N1Predicate.Or o) {
            return combine("or", o.children(), deepestLevel, layout, fnAnchors);
        }
        throw new IllegalStateException("[NESTED-POC] unknown predicate node: " + pred);
    }

    /** Fold a list of child predicates into a left-deep chain of the given boolean function (and/or). */
    private static Expression combine(
        String boolFn,
        List<N1Predicate> children,
        String deepestLevel,
        List<String> layout,
        java.util.Map<String, Integer> fnAnchors
    ) {
        if (children.isEmpty()) {
            throw new IllegalStateException("[NESTED-POC] " + boolFn + " with no children");
        }
        Expression acc = buildPredicate(children.get(0), deepestLevel, layout, fnAnchors);
        int anchor = anchorFor(boolFn, fnAnchors);
        for (int i = 1; i < children.size(); i++) {
            Expression next = buildPredicate(children.get(i), deepestLevel, layout, fnAnchors);
            acc = Expression.newBuilder()
                .setScalarFunction(
                    Expression.ScalarFunction.newBuilder()
                        .setFunctionReference(anchor)
                        .setOutputType(nullableBool())
                        .addArguments(FunctionArgument.newBuilder().setValue(acc).build())
                        .addArguments(FunctionArgument.newBuilder().setValue(next).build())
                        .build()
                )
                .build();
        }
        return acc;
    }

    /** Allocate (or reuse) a stable function anchor for a Substrait function name. */
    private static int anchorFor(String fnName, java.util.Map<String, Integer> fnAnchors) {
        return fnAnchors.computeIfAbsent(fnName, k -> fnAnchors.size() + 1);
    }

    /** Build a Substrait nullable literal for an Integer/Long/Double/String/Boolean value. */
    private static Expression literalOf(Object value) {
        Expression.Literal.Builder lit = Expression.Literal.newBuilder().setNullable(true);
        if (value instanceof Integer i) {
            lit.setI32(i);
        } else if (value instanceof Long l) {
            lit.setI64(l);
        } else if (value instanceof Double db) {
            lit.setFp64(db);
        } else if (value instanceof Boolean b) {
            lit.setBoolean(b);
        } else if (value instanceof String s) {
            lit.setString(s);
        } else {
            throw new IllegalStateException("[NESTED-POC] unsupported literal type: " + (value == null ? "null" : value.getClass()));
        }
        return Expression.newBuilder().setLiteral(lit.build()).build();
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

}

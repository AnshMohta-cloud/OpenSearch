/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import io.substrait.proto.AggregateFunction;
import io.substrait.proto.AggregateRel;
import io.substrait.proto.Expression;
import io.substrait.proto.FunctionArgument;
import io.substrait.proto.NamedStruct;
import io.substrait.proto.Plan;
import io.substrait.proto.PlanRel;
import io.substrait.proto.ProjectRel;
import io.substrait.proto.ReadRel;
import io.substrait.proto.Rel;
import io.substrait.proto.RelCommon;
import io.substrait.proto.RelRoot;
import io.substrait.proto.Type;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * [NESTED] Parent-dedup post-pass for the generic {@code expand} path — restores vanilla
 * {@code nested()} semantics for the PARENT-RETURNING {@code count} shape.
 *
 * <p>PPL {@code expand} is a flatten: {@code expand products | where rating>4 | stats count()}
 * emits one row per matching child, so a parent with several matching children is counted several
 * times. Vanilla OpenSearch counts distinct matching PARENT documents (reverse_nested). This pass
 * rewrites {@code count()} to {@code count(DISTINCT __row_id__)} where {@code __row_id__} is the
 * physical parent-doc identity.
 *
 * <p>{@code __row_id__} is physical-only (never in the Calcite schema, so isthmus never emits it).
 * The mechanism (the same group-by-{@code __row_id__} distinct-parent trick the former hand-built
 * path used, now applied generically as a proto post-pass):
 * <ol>
 *   <li>Append {@code __row_id__} (i64) to the unnest read's {@code base_schema}. This both makes the
 *       physical scan materialise it (the {@code requestsRowIds} routing, tripped by the unnest,
 *       computes the shard-global value) and lets the reshape consumer carry it through. The Rust
 *       reshape reorders {@code __row_id__} to the LAST output column (see {@code build_reshaping_unnest}),
 *       so it sits at index {@code W} — the Calcite post-unnest width — strictly after every column
 *       isthmus already indexed. Every existing positional field reference upstream is undisturbed.</li>
 *   <li>Point the top {@code count} {@code Aggregate} directly at the {@code Filter}/reshape input
 *       (the intermediate parent-column {@code Project} is bypassed — {@code count()} references no
 *       input column) and rewrite the measure to {@code count(DISTINCT $W)}.</li>
 * </ol>
 *
 * <p>{@code W} is read from the intermediate {@code Project}'s emit {@code output_mapping[0]} (=
 * the project's input field count = the Calcite post-unnest width = {@code __row_id__}'s post-reshape
 * tail index), so no index is hardcoded.
 *
 * <p>The <b>docs</b> shape (parent projection returning duplicate parents) is a separate follow-up
 * and is intentionally left untouched here. Child-aggregate shapes (metric {@code avg(price)}; group
 * {@code count() by color}) are also untouched — they aggregate over child rows and are already
 * correct.
 *
 * @opensearch.internal
 */
final class NestedParentDedupRewriter {

    private static final Logger LOGGER = LogManager.getLogger(NestedParentDedupRewriter.class);

    static final String ROW_ID = "__row_id__";
    private static final String UNNEST_RESHAPE_PREFIX = "unnest_reshape:";

    private NestedParentDedupRewriter() {}

    /**
     * Returns a copy of {@code plan} with parent-dedup applied when the plan is the parent-returning
     * {@code count} nested shape; otherwise returns {@code plan} unchanged.
     */
    static Plan rewrite(Plan plan) {
        if (plan.getRelationsCount() != 1) {
            return plan; // multi-root plans are not the single-fragment nested shape we target
        }
        PlanRel pr = plan.getRelations(0);
        if (pr.getRelTypeCase() != PlanRel.RelTypeCase.ROOT || pr.getRoot().hasInput() == false) {
            return plan;
        }
        RelRoot root = pr.getRoot();
        Rel top = root.getInput();

        // Only act when the plan contains our unnest marker somewhere below.
        if (containsUnnestReshape(top) == false) {
            return plan;
        }

        // Function anchor↔name resolution (read-only). Substrait carries aggregate function names in
        // Plan.extensions as anchor→name; the Rel layer only holds the numeric function_reference. The
        // generic parent-measure dedup needs to know whether a measure is sum/min/max/count to build the
        // correct outer combiner, so build the map once here and thread it down.
        FnCatalog fns = FnCatalog.from(plan);

        Rel rewritten = tryDedup(top, fns);
        if (rewritten == top) {
            return plan;
        }
        RelRoot newRoot = root.toBuilder().setInput(rewritten).build();
        return plan.toBuilder().setRelations(0, pr.toBuilder().setRoot(newRoot).build()).build();
    }

    /**
     * Dispatch on the top rel shape. {@code Aggregate(count)} → count-dedup; {@code Project} of
     * parent-only columns → docs-dedup (distinct parents); recurse through {@code Fetch}/{@code Sort}.
     */
    private static Rel tryDedup(Rel top, FnCatalog fns) {
        switch (top.getRelTypeCase()) {
            case AGGREGATE: {
                // First the argless-count() special case (distinct-parent count via group-by-__row_id__).
                Rel counted = tryCountDedup(top);
                if (counted != top) return counted;
                // Then the general parent-grain measure case: sum/min/max/count(parentCol) — and, since
                // avg/stddev were already reduced to sum+count upstream, avg(parentCol) too.
                return tryParentMeasureDedup(top, fns);
            }
            case PROJECT: {
                // Parent-only projection (`fields title`) → docs dedup (distinct parents).
                Rel docs = tryDocsDedup(top);
                if (docs != top) return docs;
                // avg(parentCol) reduces (AggregateReduceFunctionsRule) to a scalar Project computing
                // divide(sum,count) OVER an Aggregate. The Project itself is grain-neutral (it divides two
                // already-computed scalars); recurse so the underlying parent-measure Aggregate is deduped.
                ProjectRel pr = top.getProject();
                if (pr.hasInput() && pr.getInput().getRelTypeCase() == Rel.RelTypeCase.AGGREGATE) {
                    Rel inner = tryDedup(pr.getInput(), fns);
                    return inner == pr.getInput()
                        ? top
                        : top.toBuilder().setProject(pr.toBuilder().setInput(inner).build()).build();
                }
                return top;
            }
            case FETCH: {
                if (top.getFetch().hasInput() == false) return top;
                Rel inner = tryDedup(top.getFetch().getInput(), fns);
                return inner == top.getFetch().getInput()
                    ? top
                    : top.toBuilder().setFetch(top.getFetch().toBuilder().setInput(inner).build()).build();
            }
            case SORT: {
                if (top.getSort().hasInput() == false) return top;
                Rel inner = tryDedup(top.getSort().getInput(), fns);
                return inner == top.getSort().getInput()
                    ? top
                    : top.toBuilder().setSort(top.getSort().toBuilder().setInput(inner).build()).build();
            }
            default:
                return top;
        }
    }

    /**
     * docs shape: a top {@code Project} of PARENT-only columns over a nested-filter unnest chain
     * (e.g. {@code where comments.score>4 | fields title}). PPL {@code expand} flattens, so the raw
     * result has one row per matching child; vanilla nested returns each matching PARENT once. Restore
     * that by deduping on parent identity ({@code __row_id__}) before the projection.
     *
     * <p>Mechanism: thread {@code __row_id__} up the projection's input chain (see {@link #threadRowId}),
     * then insert an {@code Aggregate} grouping by {@code [__row_id__, ...projected-parent-cols]} — one
     * row per distinct parent (the parent columns are functionally dependent on {@code __row_id__}, so
     * grouping by them too is harmless and keeps them in the aggregate output) — and finally re-project
     * the parent columns in their original order (dropping {@code __row_id__}).
     *
     * <p>Applies ONLY when every projected expression is a bare parent-column field ref that resolves to
     * a column at-or-before the pre-row_id width (i.e. an ORIGINAL column, never an exploded child). A
     * projection that includes a child field ({@code fields title, comments.author}) is a genuine
     * per-child flatten and must NOT be deduped — left unchanged.
     */
    private static Rel tryDocsDedup(Rel projRel) {
        ProjectRel p = projRel.getProject();
        if (p.hasInput() == false) return projRel;
        // The projected outputs must be exactly bare field refs (isthmus `fields` shape with an emit).
        if (p.hasCommon() == false || p.getCommon().hasEmit() == false) {
            LOGGER.debug("[NESTED] tryDocsDedup skip: project has no emit (hasCommon={})", p.hasCommon());
            return projRel;
        }

        // Only dedup a PARENT-ONLY projection. A projection that reads an exploded-child column
        // (`fields title, comments.author`) is a genuine per-child flatten — must NOT be deduped. The
        // parent (scan) columns occupy indices [0, scanWidth) of the reshape input; exploded children are
        // appended at scanWidth..W-1. So: dedup iff every projected column index is < scanWidth.
        Integer scanWidth = scanColumnCount(p.getInput());
        if (scanWidth == null) {
            LOGGER.debug("[NESTED] tryDocsDedup skip: scanWidth null (no reshape read found beneath project)");
            return projRel;
        }
        // Each projected output must be dedup-safe: either a bare PARENT field ref (idx < scanWidth) or a
        // LITERAL. A literal arises when a keyword `=` predicate is constant-folded into the projection
        // (e.g. `where title='x' | fields title` → project emits CAST('x') instead of a field ref); it is
        // trivially parent-scalar (same value for every row of a parent), so it stays dedup-eligible.
        // A field ref at idx >= scanWidth is an EXPLODED child column → genuine per-child flatten, no dedup.
        for (Expression e : p.getExpressionsList()) {
            Integer idx = fieldIndexOf(e);
            boolean constant = isConstantExpr(e);
            if ((idx == null && constant == false) || (idx != null && idx >= scanWidth)) {
                LOGGER.debug("[NESTED] tryDocsDedup skip: projected expr not parent-scalar (idx={} constant={} scanWidth={})", idx, constant, scanWidth);
                return projRel;   // non-trivial or child ref → leave alone
            }
        }

        RowIdChain chain = threadRowId(p.getInput());
        if (chain == null) {
            LOGGER.debug("[NESTED] tryDocsDedup skip: threadRowId returned null (no reshape chain)");
            return projRel;   // no reshape beneath → not our shape
        }
        int w = chain.rowIdIndex;             // __row_id__ index in chain.rel's output (its tail)

        // Group by the field-ref parent columns (shifted for the inserted __row_id__) + __row_id__ itself.
        // Literal projections carry no grouping column — they're constant per parent — and are re-emitted
        // as literals in the output projection. Track, per output position, whether it's a grouped
        // field-ref (and which grouping slot) or a literal (carried verbatim).
        AggregateRel.Grouping.Builder grouping = AggregateRel.Grouping.newBuilder();
        java.util.List<Integer> groupSlotOfOutput = new java.util.ArrayList<>();  // grouping slot per output col, -1 if literal
        int groupSlot = 0;
        for (Expression e : p.getExpressionsList()) {
            Integer idx = fieldIndexOf(e);
            if (idx != null) {
                int shifted = idx >= w ? idx + 1 : idx;      // shift for the inserted __row_id__
                grouping.addGroupingExpressions(fieldRef(shifted));
                groupSlotOfOutput.add(groupSlot++);
            } else {
                groupSlotOfOutput.add(-1);                    // literal — no grouping column
            }
        }
        int rowIdSlot = groupSlot;                            // __row_id__ occupies the last grouping slot
        grouping.addGroupingExpressions(fieldRef(w));
        AggregateRel dedupAgg = AggregateRel.newBuilder().setInput(chain.rel).addGroupings(grouping.build()).build();
        Rel dedupAggRel = Rel.newBuilder().setAggregate(dedupAgg).build();

        // Substrait Aggregate output = grouping columns in declared order = [fieldGroupCols..., __row_id__].
        // Re-project each original output: a grouped field-ref maps to its grouping-slot position; a literal
        // is re-emitted verbatim. Emit preserves the original output order and drops the trailing __row_id__.
        ProjectRel.Builder outProj = ProjectRel.newBuilder().setInput(dedupAggRel);
        RelCommon.Emit.Builder emit = RelCommon.Emit.newBuilder();
        int groupWidth = rowIdSlot + 1;                       // grouping cols (fields + __row_id__), then measures
        int appended = 0;                                     // project appends its exprs after the agg's group cols
        java.util.List<Expression> outputs = p.getExpressionsList();
        for (int i = 0; i < outputs.size(); i++) {
            int slot = groupSlotOfOutput.get(i);
            if (slot >= 0) {
                outProj.addExpressions(fieldRef(slot));       // read the grouped field from its agg slot
            } else {
                outProj.addExpressions(outputs.get(i));       // literal — carry verbatim
            }
            emit.addOutputMapping(groupWidth + appended++);
        }
        outProj.setCommon(RelCommon.newBuilder().setEmit(emit.build()).build());
        LOGGER.info("[NESTED] parent-dedup(docs): group-by [{} field cols + __row_id__@{}] ({} literal cols carried) then re-project",
            groupSlot, w, outputs.size() - groupSlot);
        return Rel.newBuilder().setProject(outProj.build()).build();
    }

    /**
     * count shape: {@code Aggregate(groupings=[], measures=[count()])} whose input is a parent-column
     * {@code Project} over the unnest chain. Append {@code __row_id__} to the read, and insert a
     * {@code Aggregate(group by [__row_id__])} beneath the (untouched) {@code count()} so it collapses
     * to one row per matching PARENT before counting.
     *
     * <p>Why group-by rather than {@code count(DISTINCT __row_id__)}: this ungrouped count is split by
     * the CBO across a producer/reduce exchange, and the reduce derives the producer's output schema
     * with the combine-partial-final optimizer pass DISABLED. A {@code count(DISTINCT)} in that mode
     * stays a PARTIAL distinct-aggregate whose state is {@code List(Int64)} — mismatching the reduce's
     * scalar {@code i64 count()} input and failing at the exchange. A plain {@code count()} over rows
     * already deduped by a group-by decomposes normally (partial-count + merge), which is the proven
     * group-by-{@code __row_id__}-then-plain-count shape.
     */
    private static Rel tryCountDedup(Rel aggRel) {
        AggregateRel agg = aggRel.getAggregate();
        // Must be an ungrouped aggregate (a grouping key = the `count() by <dim>` group shape → leave).
        if (isUngrouped(agg) == false) return aggRel;
        if (agg.getMeasuresCount() != 1) return aggRel;
        AggregateRel.Measure measure = agg.getMeasures(0);
        AggregateFunction fn = measure.getMeasure();
        // count() emitted by isthmus takes zero arguments (count(*)). A measure WITH an argument is a
        // child metric (e.g. avg(price)) — leave it alone.
        if (fn.getArgumentsCount() != 0) return aggRel;
        if (agg.hasInput() == false) return aggRel;

        // The count's input is the parent-column PROJECTION P; its emit selects only parent/child cols and
        // DROPS __row_id__. We bypass P (count references no data column) and place the dedup group-by over
        // P's INPUT (the filter/reshape chain). threadRowId walks that chain: appends __row_id__ to the
        // deepest reshape read AND — crucially for MULTI-level — shifts every intermediate projection's emit
        // so the inserted __row_id__ doesn't displace the deeper array columns, returning row_id's final
        // index at the top of the chain. The group-by then references that exact index.
        if (agg.getInput().getRelTypeCase() != Rel.RelTypeCase.PROJECT) {
            LOGGER.info("[NESTED] parent-dedup(count): count input is not the expected parent projection, skipping");
            return aggRel;
        }
        Rel belowProject = agg.getInput().getProject().getInput();
        RowIdChain chain = threadRowId(belowProject);
        if (chain == null) {
            LOGGER.info("[NESTED] parent-dedup(count): no unnest_reshape read found beneath count, skipping");
            return aggRel;
        }

        // Insert Aggregate(group by [__row_id__@rowIdIndex]) over the row_id-carrying reshape/filter output —
        // one row per matching parent. The count() above is unchanged and now counts distinct parents.
        AggregateRel dedupAgg = AggregateRel.newBuilder()
            .setInput(chain.rel)
            .addGroupings(AggregateRel.Grouping.newBuilder().addGroupingExpressions(fieldRef(chain.rowIdIndex)).build())
            .build();
        Rel dedupAggRel = Rel.newBuilder().setAggregate(dedupAgg).build();

        AggregateRel newAgg = agg.toBuilder().setInput(dedupAggRel).build();
        LOGGER.info("[NESTED] parent-dedup(count): group-by(__row_id__ @{}) over reshape/filter (bypassing parent projection)", chain.rowIdIndex);
        return aggRel.toBuilder().setAggregate(newAgg).build();
    }

    /**
     * General parent-grain measure dedup — the fix that makes {@code sum/min/max/count/avg} of a PARENT
     * scalar column match vanilla nested (reverse_nested) semantics under a nested filter or {@code expand}.
     *
     * <p><b>The bug this fixes.</b> {@code expand} flattens a parent into one row per matching child, so a
     * parent-grain column ({@code views}) is replicated per child. A naive {@code sum(views)} then counts
     * that parent's value once per matching child (e.g. 1250 instead of 750). Vanilla nested counts each
     * matching PARENT once. {@link #tryCountDedup} already fixes the argless {@code count()} case; this
     * method generalizes the same {@code __row_id__} collapse to measures that take a parent-column argument.
     *
     * <p><b>Grain classification (research §0 amendment B, §2).</b> Each measure's argument column is
     * classified by the parent/child boundary {@code scanWidth} (= the reshape read's base_schema width):
     * an index {@code < scanWidth} is a PARENT-grain column (replicated by the fan-out → must be deduped);
     * an index {@code >= scanWidth} is a CHILD-grain column (genuinely one value per child → already correct).
     *
     * <p><b>Two-phase plan (Kimball/Looker/Gray two-level aggregation).</b> When the aggregate mixes parent-
     * and child-grain measures — or a group key — a single flat aggregate cannot be right for both grains,
     * so we build:
     * <ol>
     *   <li><b>inner</b> {@code Aggregate(group by [userGroupKeys, __row_id__])} — one row per parent per
     *       group. Parent columns are functionally determined by {@code __row_id__}, so we carry each needed
     *       parent column as an extra grouping key (the same device {@link #tryDocsDedup} uses); child-grain
     *       measures are emitted here as their normal partial (distributive {@code sum/min/max/count}).</li>
     *   <li><b>outer</b> {@code Aggregate(group by [userGroupKeys])} — parent measures now apply {@code f}
     *       over the de-replicated per-parent values; child partials are combined ({@code sum(sum)},
     *       {@code min(min)}, {@code max(max)}, {@code sum(count)}).</li>
     * </ol>
     *
     * <p><b>Scope guardrails (research §6 — bail rather than return a wrong answer).</b> This handles the
     * distributive/algebraic measures that decompose cleanly ({@code sum, min, max, count}; {@code avg} was
     * pre-reduced to {@code sum+count} by {@code AggregateReduceFunctionsRule}). It intentionally does NOT
     * rewrite when: any measure is DISTINCT or holistic ({@code count(distinct)}, {@code median},
     * {@code percentile} — no bounded partial state, §6.1/6.2); a measure's argument is not a bare column
     * ref (computed expression — grain undecidable); or the plan shape isn't the expected
     * {@code Aggregate(Project(...reshape chain))}. In every bail it returns {@code aggRel} unchanged, so
     * unsupported shapes keep today's behavior rather than silently miscomputing.
     */
    private static Rel tryParentMeasureDedup(Rel aggRel, FnCatalog fns) {
        AggregateRel agg = aggRel.getAggregate();
        if (agg.hasInput() == false || agg.getMeasuresCount() == 0) return aggRel;
        // Expected shape: Aggregate over the parent-column Project over the filter/reshape chain.
        if (agg.getInput().getRelTypeCase() != Rel.RelTypeCase.PROJECT) return aggRel;
        ProjectRel proj = agg.getInput().getProject();
        if (proj.hasInput() == false) return aggRel;

        // scanWidth = parent/child boundary in the RESHAPE output (indices < scanWidth are parent columns).
        Integer scanWidth = scanColumnCount(proj.getInput());
        if (scanWidth == null) return aggRel;

        // The Project maps the aggregate's input columns (0..projOut-1) to columns of the reshape output.
        // We need, per aggregate-referenced column, the RESHAPE-space index so we can classify its grain and
        // (for parent cols) carry it through the inner group-by. Resolve the project's output→reshape map.
        java.util.List<Integer> projToReshape = projectOutputToInputIndex(proj);
        if (projToReshape == null) return aggRel;

        // Classify every measure. Bail (return unchanged) on anything not safely decomposable.
        boolean anyParentMeasure = false;
        for (AggregateRel.Measure m : agg.getMeasuresList()) {
            AggregateFunction fn = m.getMeasure();
            String name = fns.baseName(fn.getFunctionReference());
            if (name == null) return aggRel;                              // unknown function → don't touch
            if (fn.getInvocation() == AggregateFunction.AggregationInvocation.AGGREGATION_INVOCATION_DISTINCT) {
                return aggRel;                                            // DISTINCT is holistic (§6.2) → bail
            }
            if (isDecomposable(name) == false) return aggRel;            // median/percentile/… (§6.1) → bail
            if (fn.getArgumentsCount() == 0) {
                // argless count() — parent-grain (distinct parents). Safe: handled by combine as sum(count).
                anyParentMeasure = true;
                continue;
            }
            if (fn.getArgumentsCount() != 1) return aggRel;              // multi-arg agg — out of scope
            Integer projIdx = fieldIndexOf(fn.getArguments(0).getValue());
            if (projIdx == null) return aggRel;                          // computed arg — grain undecidable
            if (projIdx >= projToReshape.size()) return aggRel;
            int reshapeIdx = projToReshape.get(projIdx);
            if (reshapeIdx < scanWidth) anyParentMeasure = true;         // parent-grain measure present
        }
        if (anyParentMeasure == false) {
            // Only child-grain measures → the flat aggregate is already correct; leave it (also avoids
            // needlessly rewriting the pure-child metric shape the old code deliberately skipped).
            return aggRel;
        }

        // Also classify the group keys' grains (a `by <dim>` on a child column is a genuine per-child group
        // and is out of scope for parent-collapse; bail to preserve behavior). Grouping expressions are
        // positional refs into the aggregate's input (the Project output).
        for (AggregateRel.Grouping g : agg.getGroupingsList()) {
            for (Expression ge : g.getGroupingExpressionsList()) {
                Integer gIdx = fieldIndexOf(ge);
                if (gIdx == null) return aggRel;                         // computed group key → bail
                if (gIdx >= projToReshape.size()) return aggRel;
                if (projToReshape.get(gIdx) >= scanWidth) return aggRel; // group by child column → bail
            }
        }

        return buildTwoPhase(aggRel, agg, proj, projToReshape, scanWidth, fns);
    }

    /**
     * Builds the two-phase (inner-dedup, outer-combine) aggregate. Preconditions verified by the caller:
     * expected shape, all measures decomposable + single-or-zero bare-column arg, group keys parent-grain.
     *
     * <p>Layout. Let the inner group key be {@code [g0..g(k-1), c0..c(j-1), __row_id__]} where {@code g*}
     * are the user group columns and {@code c*} are the DISTINCT parent columns any parent measure reads
     * (both carried as grouping keys — functionally determined by {@code __row_id__}). Child-grain measures
     * become inner partials appended after the group columns. Substrait aggregate output order is
     * {@code [grouping cols..., measures...]}, so the inner output is
     * {@code [g0.., c0.., __row_id__, childPartial0..]} — indices we reference by position in the outer phase.
     */
    private static Rel buildTwoPhase(
        Rel aggRel,
        AggregateRel agg,
        ProjectRel proj,
        java.util.List<Integer> projToReshape,
        int scanWidth,
        FnCatalog fns
    ) {
        // Thread __row_id__ through the reshape chain beneath the Project.
        RowIdChain chain = threadRowId(proj.getInput());
        if (chain == null) return aggRel;
        int rowIdIdx = chain.rowIdIndex;   // __row_id__ index in chain.rel (reshape-space, its tail)

        // Collect, in first-seen order: user group-key reshape indices, then the distinct parent-column
        // reshape indices referenced by parent measures. Each becomes an inner grouping key. A reshape index
        // >= rowIdIdx never occurs here (row_id is appended at the tail), so no shift needed for these.
        java.util.LinkedHashMap<Integer, Integer> keyReshapeToInnerSlot = new java.util.LinkedHashMap<>();
        java.util.List<Integer> groupKeyReshape = new java.util.ArrayList<>();
        for (AggregateRel.Grouping g : agg.getGroupingsList()) {
            for (Expression ge : g.getGroupingExpressionsList()) {
                int r = projToReshape.get(fieldIndexOf(ge));
                groupKeyReshape.add(r);
                keyReshapeToInnerSlot.putIfAbsent(r, keyReshapeToInnerSlot.size());
            }
        }
        // Parent-measure argument columns.
        for (AggregateRel.Measure m : agg.getMeasuresList()) {
            AggregateFunction fn = m.getMeasure();
            if (fn.getArgumentsCount() == 1) {
                int r = projToReshape.get(fieldIndexOf(fn.getArguments(0).getValue()));
                if (r < scanWidth) keyReshapeToInnerSlot.putIfAbsent(r, keyReshapeToInnerSlot.size());
            }
        }

        // ---- INNER aggregate: group by [carried cols..., __row_id__], child-grain measures as partials ----
        AggregateRel.Grouping.Builder innerGrouping = AggregateRel.Grouping.newBuilder();
        for (Integer r : keyReshapeToInnerSlot.keySet()) {
            innerGrouping.addGroupingExpressions(fieldRef(r));   // reshape-space ref (all < rowIdIdx)
        }
        int rowIdSlot = keyReshapeToInnerSlot.size();            // __row_id__ is the last grouping key
        innerGrouping.addGroupingExpressions(fieldRef(rowIdIdx));
        int innerGroupWidth = rowIdSlot + 1;

        AggregateRel.Builder inner = AggregateRel.newBuilder().setInput(chain.rel).addGroupings(innerGrouping.build());
        // Child-grain measures: re-emit the measure verbatim but repoint its argument to the reshape-space
        // index (its original arg pointed at the Project's output column). Track its inner output slot.
        java.util.List<Integer> childMeasureInnerSlot = new java.util.ArrayList<>();  // per agg measure, inner slot or -1
        int innerMeasureSlot = innerGroupWidth;
        for (AggregateRel.Measure m : agg.getMeasuresList()) {
            AggregateFunction fn = m.getMeasure();
            boolean childGrain = fn.getArgumentsCount() == 1
                && projToReshape.get(fieldIndexOf(fn.getArguments(0).getValue())) >= scanWidth;
            if (childGrain) {
                int r = projToReshape.get(fieldIndexOf(fn.getArguments(0).getValue()));
                AggregateFunction.Builder pf = fn.toBuilder().clearArguments()
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldRef(r)).build());
                inner.addMeasures(AggregateRel.Measure.newBuilder().setMeasure(pf.build()).build());
                childMeasureInnerSlot.add(innerMeasureSlot++);
            } else {
                childMeasureInnerSlot.add(-1);   // parent-grain (or argless count) → not computed in inner
            }
        }
        Rel innerRel = Rel.newBuilder().setAggregate(inner.build()).build();

        // ---- OUTER aggregate: group by the user keys (by their inner slots); combine each measure ----
        AggregateRel.Grouping.Builder outerGrouping = AggregateRel.Grouping.newBuilder();
        for (Integer r : groupKeyReshape) {
            outerGrouping.addGroupingExpressions(fieldRef(keyReshapeToInnerSlot.get(r)));
        }
        AggregateRel.Builder outer = AggregateRel.newBuilder().setInput(innerRel);
        if (agg.getGroupingsList().isEmpty() == false || outerGrouping.getGroupingExpressionsCount() > 0) {
            outer.addGroupings(outerGrouping.build());
        }
        for (int i = 0; i < agg.getMeasuresCount(); i++) {
            AggregateRel.Measure m = agg.getMeasures(i);
            AggregateFunction fn = m.getMeasure();
            String base = fns.baseName(fn.getFunctionReference());
            int childSlot = childMeasureInnerSlot.get(i);
            if (childSlot >= 0) {
                // Child-grain: COMBINE the inner partial. distributive combiner = sum for sum/count, min/max stay.
                String combiner = combinerFor(base);
                int combinerAnchor = fns.anchorFor(combiner, fn.getFunctionReference());
                if (combinerAnchor < 0) {
                    // The combiner function (e.g. sum, needed to combine child count() partials) isn't
                    // declared in this plan's extensions. Rather than emit a dangling anchor, bail — the
                    // flat aggregate keeps today's behavior for this uncommon mixed shape.
                    LOGGER.info("[NESTED] parent-dedup(measure): no '{}' combiner anchor for child {} — skipping", combiner, base);
                    return aggRel;
                }
                AggregateFunction.Builder cf = fn.toBuilder()
                    .setFunctionReference(combinerAnchor)
                    .clearArguments()
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldRef(childSlot)).build());
                outer.addMeasures(AggregateRel.Measure.newBuilder().setMeasure(cf.build()).build());
            } else if (fn.getArgumentsCount() == 0) {
                // argless count() over the de-replicated inner rows = distinct-parent count. Keep as count()
                // (its arg-less form counts inner rows, one per parent) — decomposes normally.
                outer.addMeasures(m);
            } else {
                // Parent-grain measure: apply f over the carried (de-replicated) parent column.
                int r = projToReshape.get(fieldIndexOf(fn.getArguments(0).getValue()));
                int innerSlot = keyReshapeToInnerSlot.get(r);
                AggregateFunction.Builder pf = fn.toBuilder().clearArguments()
                    .addArguments(FunctionArgument.newBuilder().setValue(fieldRef(innerSlot)).build());
                outer.addMeasures(AggregateRel.Measure.newBuilder().setMeasure(pf.build()).build());
            }
        }

        // Outer aggregate output = [outer group cols..., outer measures...]. The original aggregate's output
        // was [group cols..., measures...] in the same order, so the shapes line up 1:1 — no re-projection
        // needed. Preserve the original aggregate's RelCommon (emit/direct) so downstream indices are stable.
        if (agg.hasCommon()) outer.setCommon(agg.getCommon());
        LOGGER.info("[NESTED] parent-dedup(measure): two-phase over __row_id__@{} — {} carried key(s), {} measure(s)",
            rowIdIdx, keyReshapeToInnerSlot.size(), agg.getMeasuresCount());
        return Rel.newBuilder().setAggregate(outer.build()).build();
    }

    /**
     * Resolves a Project's output column index → its input (reshape-space) column index, for the bare
     * field-ref / emit shapes isthmus produces. Returns null if any output isn't a passthrough of an input
     * column (e.g. a computed expression), since then grain can't be decided positionally.
     *
     * <p>Two shapes: (a) an {@code emit} output_mapping selecting expressions, where each expression must be
     * a bare input field ref; (b) no emit → output = expressions in order. In both the mapped expression
     * must be a bare {@code fieldRef} into the project's input.
     */
    private static java.util.List<Integer> projectOutputToInputIndex(ProjectRel proj) {
        java.util.List<Expression> exprs = proj.getExpressionsList();
        java.util.List<Integer> out = new java.util.ArrayList<>();
        if (proj.hasCommon() && proj.getCommon().hasEmit()) {
            // With an emit, the project's logical output columns are input-columns-then-expressions, and the
            // emit selects among them by index. inputCount is unknown here, but every emitted position that
            // lands in the expression range must resolve to a bare field ref; positions in the input range
            // pass an input column through directly. We over-approximate inputCount as the max emit target
            // minus expressions; simpler and robust: require emit targets to reference expressions that are
            // bare field refs, and treat a target < (target0 of first expression) as a direct input passthrough.
            // In practice isthmus emits output_mapping into the EXPRESSION range for `fields`/`stats` inputs,
            // so map each target through the expressions list.
            int exprBase = -1;
            // Heuristic base: the smallest target that indexes an expression. Isthmus sets output_mapping to
            // [inputCount .. inputCount+exprCount). Recover inputCount = minTarget when all targets are the
            // contiguous expression range; otherwise fall back to treating targets as expression indices.
            int minTarget = Integer.MAX_VALUE, maxTarget = Integer.MIN_VALUE;
            for (int t : proj.getCommon().getEmit().getOutputMappingList()) { minTarget = Math.min(minTarget, t); maxTarget = Math.max(maxTarget, t); }
            if (exprs.isEmpty() == false && maxTarget - minTarget == exprs.size() - 1 && minTarget >= 0) {
                exprBase = minTarget;   // targets map 1:1 onto the expression list
            }
            for (int t : proj.getCommon().getEmit().getOutputMappingList()) {
                int ei = (exprBase >= 0) ? (t - exprBase) : t;
                if (ei < 0 || ei >= exprs.size()) return null;
                Integer idx = fieldIndexOf(exprs.get(ei));
                if (idx == null) return null;
                out.add(idx);
            }
            return out;
        }
        // No emit: output = expressions in order; each must be a bare field ref.
        for (Expression e : exprs) {
            Integer idx = fieldIndexOf(e);
            if (idx == null) return null;
            out.add(idx);
        }
        return out;
    }

    /** Distributive/algebraic base names this pass can decompose. avg/stddev/var are pre-reduced upstream. */
    private static boolean isDecomposable(String base) {
        switch (base) {
            case "sum":
            case "min":
            case "max":
            case "count":
                return true;
            default:
                return false;   // median, percentile, mode, count_distinct, approx_* … → holistic/unsupported
        }
    }

    /** The distributive combiner that re-aggregates a partial: sum/count → sum; min → min; max → max. */
    private static String combinerFor(String base) {
        switch (base) {
            case "count":
            case "sum":
                return "sum";
            case "min":
                return "min";
            case "max":
                return "max";
            default:
                return base;
        }
    }

    /**
     * Read-only anchor↔name catalog over {@code Plan.extensions}. Substrait function names live only at the
     * Plan level; the Rel layer holds a numeric {@code function_reference}. Names carry a {@code :sig} suffix
     * (e.g. {@code sum:i32}) — {@link #baseName} strips it.
     */
    private static final class FnCatalog {
        private final java.util.Map<Integer, String> anchorToName;

        private FnCatalog(java.util.Map<Integer, String> m) { this.anchorToName = m; }

        static FnCatalog from(Plan plan) {
            java.util.Map<Integer, String> m = new java.util.HashMap<>();
            for (io.substrait.proto.SimpleExtensionDeclaration d : plan.getExtensionsList()) {
                if (d.hasExtensionFunction()) {
                    m.put(d.getExtensionFunction().getFunctionAnchor(), d.getExtensionFunction().getName());
                }
            }
            return new FnCatalog(m);
        }

        /** Base function name (suffix stripped), or null if the anchor is unknown. */
        String baseName(int anchor) {
            String n = anchorToName.get(anchor);
            if (n == null) return null;
            int c = n.indexOf(':');
            return c < 0 ? n : n.substring(0, c);
        }

        /**
         * The anchor for a function whose base name is {@code base}. Prefer an existing declaration with a
         * matching base name (reusing isthmus's own signature); if none exists (e.g. no {@code sum} was
         * declared because the query only had {@code count}), fall back to {@code likeAnchor} — safe here
         * because the only cross-name case is count→sum, and a plan with a parent count() that needs a sum
         * combiner always also has a numeric measure whose anchor is a valid sum. If truly absent, returns
         * {@code likeAnchor} unchanged; the consumer resolves by the declared name, so a mismatch would fail
         * loudly rather than silently miscompute.
         */
        int anchorFor(String base, int likeAnchor) {
            // If the requested combiner IS the like-anchor's own base (min→min, max→max), reuse it directly.
            String likeName = anchorToName.get(likeAnchor);
            if (likeName != null) {
                int c = likeName.indexOf(':');
                if ((c < 0 ? likeName : likeName.substring(0, c)).equals(base)) return likeAnchor;
            }
            for (java.util.Map.Entry<Integer, String> e : anchorToName.entrySet()) {
                String n = e.getValue();
                int c = n.indexOf(':');
                if ((c < 0 ? n : n.substring(0, c)).equals(base)) return e.getKey();
            }
            return -1;   // not declared — caller must bail rather than emit a dangling anchor
        }
    }

    /** True if the aggregate has no grouping expression (ungrouped). */
    private static boolean isUngrouped(AggregateRel agg) {
        for (AggregateRel.Grouping g : agg.getGroupingsList()) {
            if (g.getGroupingExpressionsCount() != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * The number of PARENT (scan) columns feeding the unnest chain = the top-level field count of the
     * deepest {@code unnest_reshape} read's {@code base_schema}. Parent columns occupy indices
     * {@code [0, scanWidth)} of the reshape input; exploded child columns are appended after. Returns
     * {@code null} if no reshape read is found beneath {@code rel}.
     */
    private static Integer scanColumnCount(Rel rel) {
        switch (rel.getRelTypeCase()) {
            case FILTER:
                return rel.getFilter().hasInput() ? scanColumnCount(rel.getFilter().getInput()) : null;
            case PROJECT:
                return rel.getProject().hasInput() ? scanColumnCount(rel.getProject().getInput()) : null;
            case EXTENSION_SINGLE: {
                if (rel.getExtensionSingle().hasInput() == false) return null;
                Rel input = rel.getExtensionSingle().getInput();
                // The reshape's parent-scan width = the READ's base_schema top-level field count. The READ
                // may sit directly under the reshape, OR under a PRE-EXPAND parent Filter (e.g.
                // `where title='x' | expand comments`: Filter is between reshape and scan). Descend through
                // any Filter/Project to find that READ. If instead a deeper reshape is found, recurse into it.
                Rel scan = input;
                while (scan.getRelTypeCase() == Rel.RelTypeCase.FILTER || scan.getRelTypeCase() == Rel.RelTypeCase.PROJECT) {
                    Rel next = scan.getRelTypeCase() == Rel.RelTypeCase.FILTER
                        ? (scan.getFilter().hasInput() ? scan.getFilter().getInput() : null)
                        : (scan.getProject().hasInput() ? scan.getProject().getInput() : null);
                    if (next == null) return null;
                    scan = next;
                }
                if (scan.getRelTypeCase() == Rel.RelTypeCase.READ) {
                    return scan.getRead().hasBaseSchema() ? scan.getRead().getBaseSchema().getStruct().getTypesCount() : null;
                }
                return scanColumnCount(scan);   // deeper reshape/read below
            }
            default:
                return null;
        }
    }

    /** True if a deeper {@code unnest_reshape} ExtensionSingle exists below {@code rel} (multi-level unnest),
     *  descending through Filter/Project. Distinguishes "recurse into inner reshape" from "reached the scan". */
    private static boolean containsReshapeBelow(Rel rel) {
        switch (rel.getRelTypeCase()) {
            case EXTENSION_SINGLE:
                return rel.getExtensionSingle().getDetail().getTypeUrl().startsWith(UNNEST_RESHAPE_PREFIX)
                    || (rel.getExtensionSingle().hasInput() && containsReshapeBelow(rel.getExtensionSingle().getInput()));
            case FILTER:
                return rel.getFilter().hasInput() && containsReshapeBelow(rel.getFilter().getInput());
            case PROJECT:
                return rel.getProject().hasInput() && containsReshapeBelow(rel.getProject().getInput());
            default:
                return false;
        }
    }

    /** Descends through Filter/Project to the READ and returns its base_schema top-level width, or null. */
    private static Integer readBaseSchemaWidth(Rel rel) {
        ReadRel r = findRead(rel);
        return (r != null && r.hasBaseSchema()) ? r.getBaseSchema().getStruct().getTypesCount() : null;
    }

    private static boolean readBaseSchemaHasRowId(Rel rel) {
        ReadRel r = findRead(rel);
        return r != null && r.hasBaseSchema() && r.getBaseSchema().getNamesList().contains(ROW_ID);
    }

    /** The READ at the bottom of a Filter/Project/ExtensionSingle chain, or null. */
    private static ReadRel findRead(Rel rel) {
        switch (rel.getRelTypeCase()) {
            case READ: return rel.getRead();
            case FILTER: return rel.getFilter().hasInput() ? findRead(rel.getFilter().getInput()) : null;
            case PROJECT: return rel.getProject().hasInput() ? findRead(rel.getProject().getInput()) : null;
            case EXTENSION_SINGLE: return rel.getExtensionSingle().hasInput() ? findRead(rel.getExtensionSingle().getInput()) : null;
            default: return null;
        }
    }

    /**
     * Rebuilds {@code rel} (a READ, or a READ under a PRE-EXPAND Filter/Project) with {@code __row_id__}
     * APPENDED to the READ's base_schema. Appending at the END leaves every existing column index
     * undisturbed, so a pre-expand Filter's positional condition refs stay valid without shifting.
     * Returns null if no READ is found. (Distinct from {@link #appendRowIdToRead}, which routes through
     * {@code threadRowId} for the reshape-chain layout; this one handles the parent scan below the reshape.)
     */
    private static Rel appendRowIdThroughFilters(Rel rel) {
        switch (rel.getRelTypeCase()) {
            case READ: {
                ReadRel read = rel.getRead();
                if (read.hasBaseSchema() == false) return null;
                NamedStruct bs = read.getBaseSchema();
                if (bs.getNamesList().contains(ROW_ID)) return rel;   // idempotent
                NamedStruct newBs = bs.toBuilder()
                    .addNames(ROW_ID)
                    .setStruct(bs.getStruct().toBuilder().addTypes(i64Nullable()).build())
                    .build();
                return rel.toBuilder().setRead(read.toBuilder().setBaseSchema(newBs).build()).build();
            }
            case FILTER: {
                if (rel.getFilter().hasInput() == false) return null;
                Rel ni = appendRowIdThroughFilters(rel.getFilter().getInput());
                if (ni == null) return null;
                // __row_id__ appended at the tail → the filter's existing column refs are unchanged.
                return rel.toBuilder().setFilter(rel.getFilter().toBuilder().setInput(ni).build()).build();
            }
            case PROJECT: {
                if (rel.getProject().hasInput() == false) return null;
                Rel ni = appendRowIdThroughFilters(rel.getProject().getInput());
                if (ni == null) return null;
                return rel.toBuilder().setProject(rel.getProject().toBuilder().setInput(ni).build()).build();
            }
            default:
                return null;
        }
    }

    /** Parses the {@code |w=<N>} width suffix from an {@code unnest_reshape:<path>|w=<N>} type_url. */
    private static Integer parseWidth(String typeUrl) {
        int at = typeUrl.lastIndexOf("|w=");
        if (at < 0) return null;
        try {
            return Integer.parseInt(typeUrl.substring(at + 3));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Result of threading __row_id__ up a rel chain: the rewritten rel + the index __row_id__ occupies in its output row. */
    private static final class RowIdChain {
        final Rel rel;
        final int rowIdIndex;

        RowIdChain(Rel rel, int rowIdIndex) {
            this.rel = rel;
            this.rowIdIndex = rowIdIndex;
        }
    }

    /** Back-compat entry: returns just the rewritten rel (used where the tail index is recomputed separately). */
    private static Rel appendRowIdToRead(Rel rel) {
        RowIdChain c = threadRowId(rel);
        return c == null ? null : c.rel;
    }

    /**
     * Walks down through {@code Filter}/{@code Project}/{@code ExtensionSingle} to the deepest
     * {@code unnest_reshape} read, appends {@code __row_id__} (i64) to its {@code base_schema}, and — on
     * the way back up — keeps {@code __row_id__} addressable and every OTHER column's index stable:
     *
     * <ul>
     *   <li>Each reshape {@code ExtensionSingle} reorders {@code __row_id__} to its own tail (index = its
     *       stamped {@code |w=} width), so after this level {@code __row_id__} is at {@code Wi}.</li>
     *   <li>An intermediate {@code Project} sitting above a reshape has an isthmus emit computed for the
     *       NO-row_id layout; since row_id was inserted at {@code Wi} in that project's input, every emit
     *       entry {@code >= Wi} must shift {@code +1} to keep selecting the same logical columns, and the
     *       project must also pass {@code __row_id__} through (append {@code Wi} to its emit) so the next
     *       level still sees it. This is what makes MULTI-level dedup correct.</li>
     * </ul>
     *
     * Returns the rewritten rel plus {@code __row_id__}'s index in that rel's output, or {@code null} if no
     * reshape read is found.
     */
    private static RowIdChain threadRowId(Rel rel) {
        switch (rel.getRelTypeCase()) {
            case FILTER: {
                if (rel.getFilter().hasInput() == false) return null;
                RowIdChain in = threadRowId(rel.getFilter().getInput());
                if (in == null) return null;
                // Filter keeps its input's columns (row_id stays at in.rowIdIndex), but its CONDITION
                // references columns positionally — any ref >= the insertion index shifted right by one
                // when __row_id__ was inserted below, so shift the condition to match.
                io.substrait.proto.FilterRel.Builder fb = rel.getFilter().toBuilder().setInput(in.rel);
                if (rel.getFilter().hasCondition()) {
                    try {
                        fb.setCondition(shiftAllFieldRefs(rel.getFilter().getCondition(), in.rowIdIndex));
                    } catch (UnshiftableExpressionException ex) {
                        // Filter condition holds a ref we can't safely renumber → bail to the original plan.
                        LOGGER.debug("[NESTED] threadRowId bail: unshiftable filter condition — {}", ex.getMessage());
                        return null;
                    }
                }
                Rel newRel = rel.toBuilder().setFilter(fb.build()).build();
                return new RowIdChain(newRel, in.rowIdIndex);
            }
            case PROJECT: {
                if (rel.getProject().hasInput() == false) return null;
                RowIdChain in = threadRowId(rel.getProject().getInput());
                if (in == null) return null;
                return shiftProjectForRowId(rel, in);
            }
            case EXTENSION_SINGLE: {
                if (rel.getExtensionSingle().hasInput() == false) return null;
                Rel input = rel.getExtensionSingle().getInput();
                boolean isReshape = rel.getExtensionSingle().getDetail().getTypeUrl().startsWith(UNNEST_RESHAPE_PREFIX);
                // The reshape's input is the parent scan — either a READ directly, a READ under a PRE-EXPAND
                // parent Filter (`where title='x' | expand comments`), or a deeper reshape (multi-level).
                // If a deeper reshape is below, recurse into it. Otherwise append __row_id__ to the READ's
                // base_schema (descending through any pre-expand Filter/Project), so the reshape carries it.
                if (containsReshapeBelow(input)) {
                    RowIdChain in = threadRowId(input);
                    if (in == null) return null;
                    Rel newRel = rel.toBuilder()
                        .setExtensionSingle(rel.getExtensionSingle().toBuilder().setInput(in.rel).build())
                        .build();
                    Integer wi = parseWidth(rel.getExtensionSingle().getDetail().getTypeUrl());
                    return new RowIdChain(newRel, wi != null ? wi : in.rowIdIndex);
                }
                if (isReshape == false) return null;
                // Find the READ (through pre-expand filters), append __row_id__ there, rebuild the chain.
                Integer readWidth = readBaseSchemaWidth(input);
                if (readWidth == null) return null;
                boolean alreadyPresent = readBaseSchemaHasRowId(input);
                Rel newInput = alreadyPresent ? input : appendRowIdThroughFilters(input);
                if (newInput == null) return null;
                Rel newRel = rel.toBuilder()
                    .setExtensionSingle(rel.getExtensionSingle().toBuilder().setInput(newInput).build())
                    .build();
                // This reshape reorders __row_id__ to its tail = its stamped width Wi.
                Integer wi = parseWidth(rel.getExtensionSingle().getDetail().getTypeUrl());
                return new RowIdChain(newRel, wi != null ? wi : readWidth);
            }
            default:
                return null;
        }
    }

    /**
     * Rewrites an intermediate {@code Project} so it stays correct after {@code __row_id__} was inserted at
     * index {@code in.rowIdIndex} in its input: shift every emit entry {@code >= in.rowIdIndex} by {@code +1}
     * (those logical columns moved right by one), append {@code in.rowIdIndex} to the emit so row_id passes
     * through as this project's LAST output, and shift any expression field-references the same way. Returns
     * the rewritten project and row_id's new output index (its emit tail).
     */
    private static RowIdChain shiftProjectForRowId(Rel projectRel, RowIdChain in) {
        ProjectRel project = projectRel.getProject();
        ProjectRel.Builder pb = project.toBuilder().setInput(in.rel);

        // Shift expression field-references (>= insertion index) by +1 (recursively, incl. nested args).
        pb.clearExpressions();
        for (Expression e : project.getExpressionsList()) {
            try {
                pb.addExpressions(shiftAllFieldRefs(e, in.rowIdIndex));
            } catch (UnshiftableExpressionException ex) {
                // Project expression holds a ref we can't safely renumber → bail to the original plan.
                LOGGER.debug("[NESTED] threadRowId bail: unshiftable project expression — {}", ex.getMessage());
                return null;
            }
        }

        int newRowIdOut;
        if (project.hasCommon() && project.getCommon().hasEmit()) {
            RelCommon.Emit.Builder emit = RelCommon.Emit.newBuilder();
            for (int m : project.getCommon().getEmit().getOutputMappingList()) {
                emit.addOutputMapping(m >= in.rowIdIndex ? m + 1 : m);
            }
            // Pass __row_id__ through as the last emitted column (its input index is in.rowIdIndex).
            emit.addOutputMapping(in.rowIdIndex);
            newRowIdOut = emit.getOutputMappingCount() - 1;
            pb.setCommon(project.getCommon().toBuilder().setEmit(emit.build()).build());
        } else {
            // No emit: output = input columns then expressions. row_id (at in.rowIdIndex in the input) stays
            // at that index in the output prefix. Nothing to renumber; other columns keep their indices.
            newRowIdOut = in.rowIdIndex;
        }
        return new RowIdChain(Rel.newBuilder().setProject(pb.build()).build(), newRowIdOut);
    }

    /**
     * Signals that an expression contains a field reference at or above the shift threshold inside a variant
     * this rewriter cannot safely renumber (e.g. a window function, subquery, or lambda). Rather than emit an
     * off-by-one plan, {@link #shiftAllFieldRefs} throws this so callers can bail to the original (unrewritten)
     * plan — honoring the file-wide "bail rather than return a wrong answer" contract.
     */
    static final class UnshiftableExpressionException extends RuntimeException {
        UnshiftableExpressionException(Expression.RexTypeCase variant) {
            super("cannot shift field refs inside expression variant " + variant);
        }
    }

    /**
     * Test/utility wrapper around {@link #shiftAllFieldRefs}: returns the shifted expression, or {@code null} if
     * the expression contains an unshiftable variant (mirrors the bail-to-original behavior the callers apply).
     */
    static Expression shiftAllFieldRefsOrNull(Expression e, int threshold) {
        try {
            return shiftAllFieldRefs(e, threshold);
        } catch (UnshiftableExpressionException ex) {
            return null;
        }
    }

    /**
     * Returns {@code e} with every struct-field reference index {@code >= threshold} bumped by +1, recursively
     * through EVERY expression variant that can carry a field reference — bare refs, scalar functions, casts,
     * {@code IF/THEN} (CASE), {@code SWITCH}, and {@code IN}-list forms ({@code SingularOrList}/{@code MultiOrList}).
     * This runs after {@code __row_id__} is inserted at {@code threshold}, so every column at or past that index
     * moved right by one and every reference to it must move with it.
     *
     * <p>Literals, enums, and dynamic parameters reference no column and are returned unchanged. For opaque
     * variants that MAY reference columns but that we cannot rewrite (window functions, subqueries, lambdas), we
     * inspect their serialized form: if it contains no reference {@code >= threshold} it is returned as-is,
     * otherwise a {@link UnshiftableExpressionException} is thrown so the caller bails to the original plan
     * rather than silently miscomputing. This keeps the transform total and correctness-preserving for any input.
     */
    static Expression shiftAllFieldRefs(Expression e, int threshold) {
        // Bare field reference.
        Integer idx = fieldIndexOf(e);
        if (idx != null) {
            return idx >= threshold ? fieldRef(idx + 1) : e;
        }
        switch (e.getRexTypeCase()) {
            case SCALAR_FUNCTION: {
                // Shift each argument's value expression (covers gt($7,4), and(...), coalesce(...), etc.).
                Expression.ScalarFunction sf = e.getScalarFunction();
                Expression.ScalarFunction.Builder sb = sf.toBuilder().clearArguments();
                for (FunctionArgument arg : sf.getArgumentsList()) {
                    if (arg.hasValue()) {
                        sb.addArguments(FunctionArgument.newBuilder().setValue(shiftAllFieldRefs(arg.getValue(), threshold)).build());
                    } else {
                        sb.addArguments(arg);
                    }
                }
                return e.toBuilder().setScalarFunction(sb.build()).build();
            }
            case CAST: {
                if (e.getCast().hasInput() == false) {
                    return e;
                }
                return e.toBuilder()
                    .setCast(e.getCast().toBuilder().setInput(shiftAllFieldRefs(e.getCast().getInput(), threshold)).build())
                    .build();
            }
            case IF_THEN: {
                // CASE WHEN <cond> THEN <val> ... ELSE <val>: shift the condition and result of each clause.
                Expression.IfThen it = e.getIfThen();
                Expression.IfThen.Builder ib = it.toBuilder().clearIfs();
                for (Expression.IfThen.IfClause clause : it.getIfsList()) {
                    Expression.IfThen.IfClause.Builder cb = clause.toBuilder();
                    if (clause.hasIf()) {
                        cb.setIf(shiftAllFieldRefs(clause.getIf(), threshold));
                    }
                    if (clause.hasThen()) {
                        cb.setThen(shiftAllFieldRefs(clause.getThen(), threshold));
                    }
                    ib.addIfs(cb.build());
                }
                if (it.hasElse()) {
                    ib.setElse(shiftAllFieldRefs(it.getElse(), threshold));
                }
                return e.toBuilder().setIfThen(ib.build()).build();
            }
            case SWITCH_EXPRESSION: {
                // SWITCH <match> WHEN <literal> THEN <val> ... ELSE <val>: the WHEN keys are literals (no shift);
                // shift the match expression and each THEN result plus the ELSE.
                Expression.SwitchExpression sw = e.getSwitchExpression();
                Expression.SwitchExpression.Builder swb = sw.toBuilder().clearIfs();
                if (sw.hasMatch()) {
                    swb.setMatch(shiftAllFieldRefs(sw.getMatch(), threshold));
                }
                for (Expression.SwitchExpression.IfValue iv : sw.getIfsList()) {
                    Expression.SwitchExpression.IfValue.Builder ivb = iv.toBuilder();
                    if (iv.hasThen()) {
                        ivb.setThen(shiftAllFieldRefs(iv.getThen(), threshold));
                    }
                    swb.addIfs(ivb.build());
                }
                if (sw.hasElse()) {
                    swb.setElse(shiftAllFieldRefs(sw.getElse(), threshold));
                }
                return e.toBuilder().setSwitchExpression(swb.build()).build();
            }
            case SINGULAR_OR_LIST: {
                // <value> IN (<opt>, <opt>, ...): shift the value and every option.
                Expression.SingularOrList sol = e.getSingularOrList();
                Expression.SingularOrList.Builder solb = sol.toBuilder().clearOptions();
                if (sol.hasValue()) {
                    solb.setValue(shiftAllFieldRefs(sol.getValue(), threshold));
                }
                for (Expression opt : sol.getOptionsList()) {
                    solb.addOptions(shiftAllFieldRefs(opt, threshold));
                }
                return e.toBuilder().setSingularOrList(solb.build()).build();
            }
            case MULTI_OR_LIST: {
                // (<v1>,<v2>,...) IN ((...),(...)): shift each value expression; option records are literals.
                Expression.MultiOrList mol = e.getMultiOrList();
                Expression.MultiOrList.Builder molb = mol.toBuilder().clearValue();
                for (Expression v : mol.getValueList()) {
                    molb.addValue(shiftAllFieldRefs(v, threshold));
                }
                return e.toBuilder().setMultiOrList(molb.build()).build();
            }
            case LITERAL:
            case ENUM:
            case DYNAMIC_PARAMETER:
            case REXTYPE_NOT_SET:
                return e; // references no column — safe to leave unchanged
            default:
                // Opaque variant we don't rewrite (WINDOW_FUNCTION, SUBQUERY, LAMBDA, NESTED, ...). If it holds
                // no reference >= threshold it's unaffected by the insertion and safe to keep verbatim; if it
                // does, bail rather than emit an off-by-one plan.
                if (containsFieldRefAtOrAbove(e, threshold)) {
                    throw new UnshiftableExpressionException(e.getRexTypeCase());
                }
                return e;
        }
    }

    /**
     * Deep scan: true if {@code e} (or any descendant expression) is a bare struct-field reference whose index
     * is {@code >= threshold}. Used to decide whether an un-rewritable expression variant is actually affected
     * by the {@code __row_id__} insertion (and must trigger a bail) or is safely index-independent.
     */
    static boolean containsFieldRefAtOrAbove(Expression e, int threshold) {
        Integer idx = fieldIndexOf(e);
        if (idx != null) {
            return idx >= threshold;
        }
        for (Expression child : childExpressionsOf(e)) {
            if (containsFieldRefAtOrAbove(child, threshold)) {
                return true;
            }
        }
        return false;
    }

    /** All directly-nested child expressions of {@code e}, across every variant, for structural traversal. */
    private static java.util.List<Expression> childExpressionsOf(Expression e) {
        java.util.List<Expression> out = new java.util.ArrayList<>();
        switch (e.getRexTypeCase()) {
            case SCALAR_FUNCTION:
                for (FunctionArgument arg : e.getScalarFunction().getArgumentsList()) {
                    if (arg.hasValue()) out.add(arg.getValue());
                }
                break;
            case CAST:
                if (e.getCast().hasInput()) out.add(e.getCast().getInput());
                break;
            case IF_THEN:
                for (Expression.IfThen.IfClause c : e.getIfThen().getIfsList()) {
                    if (c.hasIf()) out.add(c.getIf());
                    if (c.hasThen()) out.add(c.getThen());
                }
                if (e.getIfThen().hasElse()) out.add(e.getIfThen().getElse());
                break;
            case SWITCH_EXPRESSION:
                if (e.getSwitchExpression().hasMatch()) out.add(e.getSwitchExpression().getMatch());
                for (Expression.SwitchExpression.IfValue iv : e.getSwitchExpression().getIfsList()) {
                    if (iv.hasThen()) out.add(iv.getThen());
                }
                if (e.getSwitchExpression().hasElse()) out.add(e.getSwitchExpression().getElse());
                break;
            case SINGULAR_OR_LIST:
                if (e.getSingularOrList().hasValue()) out.add(e.getSingularOrList().getValue());
                out.addAll(e.getSingularOrList().getOptionsList());
                break;
            case MULTI_OR_LIST:
                out.addAll(e.getMultiOrList().getValueList());
                break;
            case WINDOW_FUNCTION:
                for (FunctionArgument arg : e.getWindowFunction().getArgumentsList()) {
                    if (arg.hasValue()) out.add(arg.getValue());
                }
                out.addAll(e.getWindowFunction().getPartitionsList());
                break;
            default:
                break;
        }
        return out;
    }

    /** True if any rel in the tree is our {@code unnest_reshape:} ExtensionSingle. */
    private static boolean containsUnnestReshape(Rel rel) {
        if (rel.getRelTypeCase() == Rel.RelTypeCase.EXTENSION_SINGLE
            && rel.getExtensionSingle().getDetail().getTypeUrl().startsWith(UNNEST_RESHAPE_PREFIX)) {
            return true;
        }
        for (Rel child : inputsOf(rel)) {
            if (containsUnnestReshape(child)) return true;
        }
        return false;
    }

    private static java.util.List<Rel> inputsOf(Rel rel) {
        java.util.List<Rel> out = new java.util.ArrayList<>();
        switch (rel.getRelTypeCase()) {
            case FILTER: if (rel.getFilter().hasInput()) out.add(rel.getFilter().getInput()); break;
            case PROJECT: if (rel.getProject().hasInput()) out.add(rel.getProject().getInput()); break;
            case AGGREGATE: if (rel.getAggregate().hasInput()) out.add(rel.getAggregate().getInput()); break;
            case SORT: if (rel.getSort().hasInput()) out.add(rel.getSort().getInput()); break;
            case FETCH: if (rel.getFetch().hasInput()) out.add(rel.getFetch().getInput()); break;
            case EXTENSION_SINGLE: if (rel.getExtensionSingle().hasInput()) out.add(rel.getExtensionSingle().getInput()); break;
            default: break;
        }
        return out;
    }

    /** Extracts the index of a bare top-level struct-field reference, or {@code null} if {@code e} is not one. */
    /**
     * True if {@code e} is a parent-scalar CONSTANT: a bare {@code Literal}, or a {@code Cast} whose
     * (recursively unwrapped) input is a literal. Such an expression arises when a keyword-equality
     * predicate is constant-folded into the projection (isthmus emits {@code CAST('x')} in place of the
     * {@code field='x'} column). It references no column, so it is trivially the same for every row of a
     * parent and stays dedup-eligible; it is carried verbatim into the post-dedup re-projection.
     */
    private static boolean isConstantExpr(Expression e) {
        if (e.hasLiteral()) return true;
        if (e.hasCast() && e.getCast().hasInput()) return isConstantExpr(e.getCast().getInput());
        return false;
    }

    static Integer fieldIndexOf(Expression e) {
        if (e.hasSelection() == false) return null;
        Expression.FieldReference fr = e.getSelection();
        if (fr.hasDirectReference() == false || fr.getDirectReference().hasStructField() == false) return null;
        Expression.ReferenceSegment.StructField sf = fr.getDirectReference().getStructField();
        if (sf.hasChild()) return null; // nested reference — not a bare column ref
        return sf.getField();
    }

    /** A positional field reference expression on the input row. */
    static Expression fieldRef(int index) {
        return Expression.newBuilder()
            .setSelection(
                Expression.FieldReference.newBuilder()
                    .setDirectReference(
                        Expression.ReferenceSegment.newBuilder()
                            .setStructField(Expression.ReferenceSegment.StructField.newBuilder().setField(index).build())
                            .build()
                    )
                    .setRootReference(Expression.FieldReference.RootReference.newBuilder().build())
                    .build()
            )
            .build();
    }

    private static Type i64Nullable() {
        return Type.newBuilder()
            .setI64(Type.I64.newBuilder().setNullability(Type.Nullability.NULLABILITY_NULLABLE).build())
            .build();
    }
}

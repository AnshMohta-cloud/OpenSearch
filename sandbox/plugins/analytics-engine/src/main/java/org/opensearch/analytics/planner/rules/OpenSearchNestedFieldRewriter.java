/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.planner.rel.LogicalNestedScope;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * [NESTED] Generic Calcite rewrite that turns references to nested fields into a real UNNEST plan,
 * for ARBITRARY queries — no per-query hardcoding. This is the production direction (behind the
 * {@code nested.generic_rewrite} flag) replacing the hand-authored {@code N1Descriptor} registry.
 *
 * <p><b>What it detects.</b> A nested field reference {@code comments.author} is represented by
 * Calcite as {@code ITEM($arrayCol, 'field')} where {@code $arrayCol} is an {@code ARRAY(ROW(...))}
 * column (see {@code OpenSearchSchemaBuilder} which exposes {@code nested} mappings that way). Such
 * {@code ITEM} calls can appear inside a {@link LogicalProject}'s expressions ({@code | fields
 * comments.author}) or inside a {@link LogicalFilter}'s condition ({@code | where comments.score>4}),
 * and — since {@code | stats avg(comments.score)} is an {@code Aggregate} over a {@code Project} that
 * contains the {@code ITEM} — handling Project + Filter also covers aggregates.
 *
 * <p><b>What it does.</b> Walking the tree, at each Project/Filter whose expressions reference an
 * array column via {@code ITEM}, it injects the backend-neutral unnest operator beneath that node:
 * <pre>
 *   LogicalNestedScope(path=arrayCol)
 *     └─ &lt;original input&gt;    (all original columns, indices UNCHANGED; struct fields APPENDED)
 * </pre>
 * (see {@link org.opensearch.analytics.planner.rel.LogicalNestedScope} for the row-type contract —
 * it is Calcite-shape-equivalent to the {@code Correlate(left, Uncollect(...))} it replaces, just
 * marked/routed by a real capability lookup instead of a hardcoded backend name) and rewrites each
 * {@code ITEM($arrayCol,'f')} to a plain {@link RexInputRef} of the appended unnested column.
 * Because the scope keeps the original columns first and appends the exploded struct fields,
 * <b>every original column index is preserved</b> — so operators above the rewritten node are
 * unaffected and the transform composes cleanly across the whole tree.
 *
 * <p>For a {@link LogicalFilter}, the appended unnested columns are projected away again above the
 * filter so the row type is restored to the parent's shape (returning parent rows). NOTE: parent
 * de-duplication (a parent with two matching children currently appears twice) and multi-array /
 * same-child correlation are the remaining runtime gaps — see the package README / task list; those
 * shapes fall back to the hardcoded path when the flag is off.
 *
 * @opensearch.internal
 */
public final class OpenSearchNestedFieldRewriter {

    private static final Logger LOGGER = LogManager.getLogger(OpenSearchNestedFieldRewriter.class);

    /**
     * Synthetic scalar function: {@code nested_any_match_expr(arrayCol, '<json expr tree>') → BOOLEAN}.
     * Emitted by the filter rewrite in place of Correlate+Uncollect for ANY predicate shape on a
     * single array column — a lone equality leaf, a compound AND/OR/NOT tree, arithmetic (+,-,*,/,%),
     * or any mix thereof — e.g. {@code subs.views > 65 and subs.views % 2 = 0} (a single element must
     * satisfy the WHOLE tree carried by ONE call — matches vanilla OpenSearch's native {@code nested}
     * query + Painless script semantics for everything inside that one JSON tree: one element must
     * jointly satisfy every clause the tree contains). The second argument is a JSON string
     * describing the per-element predicate tree — see {@link ExprTreeBuilder} for the node shapes;
     * the Rust {@code nested_any_match_expr} UDF parses and evaluates it per array element,
     * short-circuiting on the first match. Row count never changes — one boolean per parent row.
     *
     * <p><b>By default, top-level AND conjuncts on the same array ARE fused into one call</b> —
     * {@code tryLambdaRewrite} combines every array-referencing conjunct into one joint tree, so
     * {@code comments.author = 'frank' AND comments.score < 50} requires a SINGLE element to satisfy
     * both, matching vanilla's strict joint-element guarantee. Behind the (removed) independent-conjunct routing option (default off), each top-level conjunct instead gets its
     * OWN call, ANDed together at the row level — trading that joint-element guarantee for letting
     * each conjunct reach its own most suitable backend independently; see that property's javadoc
     * for the accepted correctness gap. A compound condition written as a SINGLE Calcite expression
     * (an explicit OR, or an AND nested inside an OR, etc. — anything that isn't itself a top-level
     * AND operand) always becomes one call with true joint semantics, regardless of the flag.
     *
     * <p>Whether a given call is ALSO Lucene-delegable (not just DataFusion-native) is decided per
     * instance, not by a separate function: {@code CapabilityRegistry} registers this function as
     * dual-viable {@code [lucene, datafusion]} on {@code FieldType.ARRAY}, and {@code
     * OpenSearchFilterRule} additionally consults each candidate backend's {@code
     * DelegatedPredicateSerializer#canServe} to narrow that further per query — Lucene's serializer
     * inspects the JSON tree and approves only a single string-equality leaf (the shape it can
     * translate into a native {@code TermQuery}); DataFusion has no such override and always serves
     * every shape. See {@code NestedAnyMatchExprSerializer} on the Lucene side.
     */
    public static final SqlFunction NESTED_ANY_MATCH_EXPR_OP = new SqlFunction(
        "NESTED_ANY_MATCH_EXPR",
        SqlKind.OTHER_FUNCTION,
        ReturnTypes.BOOLEAN_NULLABLE,
        null,
        OperandTypes.ANY,
        SqlFunctionCategory.USER_DEFINED_FUNCTION
    );

    private OpenSearchNestedFieldRewriter() {}

    /**
     * Rewrites the tree so that every {@code ITEM}-on-array reference becomes a plain column produced
     * by an injected UNNEST. Returns the original tree unchanged if there are no nested references.
     */
    public static RelNode rewrite(RelNode root) {
        RelNode result = root.accept(new NestedShuttle());
        if (result != root) {
            LOGGER.info("[NESTED] rewrite injected UNNEST. New plan:\n{}", RelOptUtil.toString(result));
        }
        return result;
    }

    /**
     * Bottom-up shuttle: children are rewritten first (so a node always sees an already-unnested
     * input where applicable), then the node itself is rewritten if it carries {@code ITEM} refs.
     *
     * <p>{@code aggregateClaimedProjects} tracks {@link LogicalProject} instances that {@code
     * visit(LogicalAggregate)} has already routed through the unnest-injecting rewrite (because
     * their {@code ITEM} references feed a GROUP BY key or aggregate-function argument — a genuine
     * grain change, same as vanilla's {@code expand} command). {@code visit(LogicalProject)} must
     * NOT re-rewrite those as plain (first-element) projections; identity-based membership in this
     * set is the signal that a Project was already handled at the Aggregate level.
     */
    private static final class NestedShuttle extends RelShuttleImpl {
        private final java.util.Set<LogicalProject> aggregateClaimedProjects = java.util.Collections.newSetFromMap(
            new java.util.IdentityHashMap<>()
        );

        @Override
        public RelNode visit(LogicalAggregate aggregate) {
            RelNode rewrittenInput = null;
            if (aggregate.getInput() instanceof LogicalProject childProject) {
                RelNode candidate = rewriteAggregateInputProject(aggregate, childProject);
                if (candidate != childProject) {
                    aggregateClaimedProjects.add(childProject);
                    rewrittenInput = candidate.accept(this);
                }
            }
            LogicalAggregate visited = rewrittenInput != null
                ? (LogicalAggregate) aggregate.copy(aggregate.getTraitSet(), List.of(rewrittenInput))
                : (LogicalAggregate) super.visitChildren(aggregate);
            return visited;
        }

        @Override
        public RelNode visit(LogicalProject project) {
            LogicalProject visited = (LogicalProject) super.visitChildren(project);
            if (aggregateClaimedProjects.contains(project)) {
                return visited;
            }
            return rewriteProject(visited);
        }

        @Override
        public RelNode visit(LogicalFilter filter) {
            LogicalFilter visited = (LogicalFilter) super.visitChildren(filter);
            return rewriteFilter(visited);
        }
    }

    /**
     * If {@code childProject} (the Aggregate's input) references a nested array via {@code ITEM}
     * AND that reference feeds a GROUP BY key or an aggregate-function argument, injects the
     * Correlate+Uncollect unnest beneath it (the existing, unchanged logic) — this is a genuine
     * grain change (the output IS per-child), matching vanilla's requirement that {@code expand} (or
     * an explicit {@code nested()}/{@code stats ... by} group-key) is needed to see every element.
     * Returns {@code childProject} unchanged if no such reference exists (the plain-projection
     * rewrite in {@link #rewriteProject} will apply instead, once {@code visit(LogicalProject)}
     * reaches it — first-element semantics, matching vanilla's {@code parseArray} degrade behavior).
     */
    private static RelNode rewriteAggregateInputProject(LogicalAggregate aggregate, LogicalProject childProject) {
        RelNode grandchild = childProject.getInput();
        int arrayCol = firstArrayColReferenced(childProject.getProjects(), grandchild.getRowType());
        if (arrayCol < 0) {
            return childProject;
        }
        // Only claim this Project if the ITEM-bearing output column(s) are actually consumed by
        // the Aggregate — as a group key or as an aggregate call's argument. If the ITEM reference
        // feeds a column the Aggregate never touches (e.g. a passthrough SELECT column alongside an
        // unrelated aggregate), leave it for the plain-projection (first-element) rewrite.
        java.util.Set<Integer> itemBearingOutputCols = new java.util.HashSet<>();
        List<RexNode> projectExprs = childProject.getProjects();
        for (int i = 0; i < projectExprs.size(); i++) {
            if (referencesItemOnArray(projectExprs.get(i), arrayCol, grandchild.getRowType())) {
                itemBearingOutputCols.add(i);
            }
        }
        boolean consumedByAggregate = false;
        for (int groupKey : aggregate.getGroupSet()) {
            if (itemBearingOutputCols.contains(groupKey)) {
                consumedByAggregate = true;
                break;
            }
        }
        if (!consumedByAggregate) {
            for (AggregateCall call : aggregate.getAggCallList()) {
                for (int argIdx : call.getArgList()) {
                    if (itemBearingOutputCols.contains(argIdx)) {
                        consumedByAggregate = true;
                        break;
                    }
                }
            }
        }
        if (!consumedByAggregate) {
            return childProject;
        }
        return rewriteProjectViaUnnest(childProject);
    }

    /** True if {@code expr} contains {@code ITEM($arrayCol,'field')} anywhere in its tree. */
    private static boolean referencesItemOnArray(RexNode expr, int arrayCol, RelDataType inputRowType) {
        ItemFinder finder = new ItemFinder(inputRowType);
        expr.accept(finder);
        return finder.arrayCol == arrayCol;
    }

    // ---- Project: rewrite ITEM refs in the projected expressions -------------------------------

    /**
     * Plain-projection path: rewrites {@code ITEM($arrayCol,'field')} to {@code
     * ITEM(ITEM($arrayCol, 1), 'field')} — index into the array to get its first element (a ROW),
     * then extract the field from that ROW. Both are plain Calcite {@code ITEM} calls, dispatched at
     * Substrait-emission time by {@code ArrayElementAdapter} (array-index → {@code array_element},
     * struct-field → {@code get_field}) — no new operator, no row-count change.
     *
     * <p>Matches vanilla OpenSearch's own behavior for a bare dotted nested projection with no
     * inner_hits/expand request (see {@code OpenSearchExprValueFactory.parseArray}, which degrades
     * to {@code content.array().next()} — the first element — when {@code supportArrays} is false).
     */
    private static RelNode rewriteProject(LogicalProject project) {
        RelNode input = project.getInput();
        int arrayCol = firstArrayColReferenced(project.getProjects(), input.getRowType());
        if (arrayCol < 0) {
            return project;
        }
        RelOptCluster cluster = project.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();
        RelDataTypeField arrayField = input.getRowType().getFieldList().get(arrayCol);
        FirstElementRewriteShuttle shuttle = new FirstElementRewriteShuttle(arrayCol, arrayField.getType(), rexBuilder);
        List<RexNode> newExprs = new ArrayList<>(project.getProjects().size());
        for (RexNode e : project.getProjects()) {
            newExprs.add(e.accept(shuttle));
        }
        LOGGER.info(
            "[NESTED-FIRST-ELEMENT] plain projection on array col '{}' (idx {}) rewritten to ITEM(ITEM(arr,1),field) "
                + "— no unnest, first element only (matches vanilla)",
            arrayField.getName(),
            arrayCol
        );
        return LogicalProject.create(input, List.of(), newExprs, project.getRowType().getFieldNames());
    }

    /**
     * Rewrites {@code ITEM($arrayCol,'field')} references in {@code project}'s expressions to
     * columns of injected {@code LogicalNestedScope}(s) (the original, child-grain unnest path). Used
     * when the Aggregate-input guard determines a genuine grain change is required.
     *
     * <p>Delegates to {@link #stackedUnnest} so that a DEEP nested leaf — an ITEM CHAIN crossing
     * MULTIPLE {@code nested} array boundaries, e.g. {@code comments.replies.tags.name} — has EVERY
     * array level exploded, not just the outermost one. A single {@code injectUnnest} explodes only
     * the outermost array ({@code comments}) and leaves the residual inner chain
     * ({@code ITEM(ITEM($replies,'tags'),'name')}) over a STILL-array column, which mis-emits to
     * Substrait as e.g. {@code array_element(get_field(array_element(...),'tags'),'name')} and fails
     * with "Named schema must contain names for all fields". The loop keeps exploding until the leaf
     * resolves to a scalar column, so the fix is ARBITRARY-DEPTH (2, 3, 4, ... N levels) — the number
     * of passes is driven purely by the schema, never a hardcoded depth.
     */
    private static RelNode rewriteProjectViaUnnest(LogicalProject project) {
        StackedUnnestResult r = stackedUnnest(
            project.getInput(),
            project.getProjects(),
            List.of(),
            project.getCluster(),
            project.getCluster().getRexBuilder()
        );
        if (r.levels == 0) {
            return project; // nothing referenced our array
        }
        return LogicalProject.create(r.input, List.of(), r.projectExprs, project.getRowType().getFieldNames());
    }

    /** Result of a stacked unnest: the final NestedScope input, the rewritten project exprs + extra exprs, and level count. */
    private record StackedUnnestResult(RelNode input, List<RexNode> projectExprs, List<RexNode> extraExprs, int levels) {}

    /**
     * STACKED (iterative) UNNEST for ARBITRARY nesting depth. A deep path
     * {@code comments.replies.tags.name} (each level a {@code nested} ARRAY(ROW)) lowers to an ITEM
     * CHAIN {@code ITEM(ITEM(ITEM($comments,'replies'),'tags'),'name')}. One unnest explodes only the
     * OUTERMOST array; the residual chain over the still-array inner column would mis-emit as
     * {@code array_element(Struct,...)} and fail. So we iterate: each pass explodes the outermost
     * still-array level referenced by an ITEM in {@code projectExprs}, rewriting that ITEM link (in
     * BOTH {@code projectExprs} AND {@code extraExprs}) to the exploded column, until no ITEM-on-array
     * remains in {@code projectExprs} (the leaf is a scalar column). The number of passes equals the
     * schema nesting depth of the referenced leaf — a 2-level path does 2 passes, a 3-level path does
     * 3, an N-level path does N — with NO depth-specific branching and NO hardcoded constant; the loop
     * terminates purely when {@code firstArrayColReferenced} finds no more ITEM-on-array. The {@code
     * MAX_UNNEST_DEPTH} cap is a pure defensive guard against a pathological/malformed chain, never
     * reached by normal schema-driven descent.
     *
     * <p>{@code extraExprs} (e.g. a same-path filter condition) ride through the SAME per-level
     * shuttles so they resolve to the SAME exploded element columns; today's only caller passes an
     * empty list, but the parameter keeps this a faithful port of the proven logic.
     *
     * <p><b>ADAPTATION NOTE:</b> the pre-merge version iterated on a Calcite {@code
     * LogicalCorrelate} ({@code u.correlate}); HEAD's {@code injectUnnest} instead returns a
     * backend-neutral {@link LogicalNestedScope} ({@code u.nestedScope}), which is fed back as the
     * next {@code input} and whose {@code getRowType()} backs each per-level {@code ItemRewriteShuttle}.
     */
    private static StackedUnnestResult stackedUnnest(
        RelNode input,
        List<RexNode> projectExprs,
        List<RexNode> extraExprs,
        RelOptCluster cluster,
        RexBuilder rexBuilder
    ) {
        List<RexNode> exprs = new ArrayList<>(projectExprs);
        List<RexNode> extra = new ArrayList<>(extraExprs);
        int levels = 0;
        final int MAX_UNNEST_DEPTH = 64; // defensive: schema nesting is far shallower; prevents any loop
        while (true) {
            int arrayCol = firstArrayColReferenced(exprs, input.getRowType());
            if (arrayCol < 0) {
                break; // no more ITEM-on-array in the metric exprs — chain fully resolved to scalar refs
            }
            if (levels++ >= MAX_UNNEST_DEPTH) {
                LOGGER.warn("[NESTED] stacked-unnest exceeded max depth {} — leaving remaining ITEM refs", MAX_UNNEST_DEPTH);
                break;
            }
            UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
            if (u == null) {
                break; // not ARRAY(ROW) at this level — leave as-is (caller/emission handles/fails cleanly)
            }
            ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.nestedScope.getRowType());
            List<RexNode> rewritten = new ArrayList<>(exprs.size());
            for (RexNode e : exprs) {
                rewritten.add(e.accept(shuttle));
            }
            List<RexNode> rewrittenExtra = new ArrayList<>(extra.size());
            for (RexNode e : extra) {
                rewrittenExtra.add(e.accept(shuttle));
            }
            exprs = rewritten;
            extra = rewrittenExtra;
            input = u.nestedScope;
        }
        if (levels > 0) {
            LOGGER.info("[NESTED] stacked-unnest injected {} UNNEST level(s)", levels);
        }
        return new StackedUnnestResult(input, exprs, extra, levels);
    }

    /**
     * Rewrites {@code ITEM($arrayCol,'field')} to {@code ITEM(ITEM($arrayCol, 1), 'field')} in
     * place — no relational structure change, just an expression substitution. Both calls use
     * Calcite's standard {@code SqlStdOperatorTable.ITEM} operator; {@code ArrayElementAdapter}
     * (already shipped, used by PPL's {@code mvindex}/{@code spath} paths) dispatches the outer
     * array-index call to {@code array_element} and — per the new struct-input branch added
     * alongside this change — the inner struct-field call to {@code get_field}.
     */
    private static final class FirstElementRewriteShuttle extends RexShuttle {
        private final int arrayCol;
        private final RelDataType elementType;
        private final RexBuilder rexBuilder;

        FirstElementRewriteShuttle(int arrayCol, RelDataType arrayType, RexBuilder rexBuilder) {
            this.arrayCol = arrayCol;
            this.elementType = arrayType.getComponentType();
            this.rexBuilder = rexBuilder;
        }

        @Override
        public RexNode visitCall(RexCall call) {
            if ("ITEM".equals(call.getOperator().getName()) && call.getOperands().size() == 2) {
                RexNode arrayOperand = call.getOperands().get(0);
                RexNode fieldNode = call.getOperands().get(1);
                if (arrayOperand instanceof RexInputRef ref
                    && ref.getIndex() == arrayCol
                    && fieldNode instanceof RexLiteral lit
                    && lit.getTypeName() == SqlTypeName.CHAR) {
                    RexNode indexLiteral = rexBuilder.makeExactLiteral(java.math.BigDecimal.ONE);
                    RexNode firstElement = rexBuilder.makeCall(
                        elementType,
                        org.apache.calcite.sql.fun.SqlStdOperatorTable.ITEM,
                        List.of(arrayOperand, indexLiteral)
                    );
                    return rexBuilder.makeCall(
                        call.getType(),
                        org.apache.calcite.sql.fun.SqlStdOperatorTable.ITEM,
                        List.of(firstElement, fieldNode)
                    );
                }
            }
            return super.visitCall(call);
        }
    }

    // ---- Filter: rewrite ITEM-based predicates into nested_any_match scalar calls ---------------

    /**
     * Rewrites a filter containing {@code ITEM($arrayCol,'field') <op> <literal>} into a filter
     * using {@code NESTED_ANY_MATCH_EXPR($arrayCol, '<json expr tree>')}. This is the "peek
     * inside the cell" approach: the function iterates the array internally and returns TRUE/FALSE
     * per parent row — row count never changes.
     *
     * <p>Falls back to the old Correlate+Uncollect path for predicates that don't match the
     * supported shape (e.g. ITEM used in a non-comparison context, or two different arrays).
     */
    private static RelNode rewriteFilter(LogicalFilter filter) {
        RelNode input = filter.getInput();
        int arrayCol = firstArrayColReferenced(List.of(filter.getCondition()), input.getRowType());
        if (arrayCol < 0) {
            return filter;
        }
        RelOptCluster cluster = filter.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();

        // Expand any SEARCH/Sarg back into explicit comparisons BEFORE the lambda rewrite. Calcite
        // collapses same-field disjunctions and IN-lists into a single SEARCH(expr, Sarg[...]) node
        // (e.g. `depts.name='Sales' OR depts.name='Ops'` and `depts.name IN ('Sales','Ops')` both
        // become SEARCH(ITEM($0,'name'), Sarg['Ops','Sales'])). ExprTreeBuilder has no mapping for
        // SqlKind.SEARCH, so without this the rewrite reports "not applicable" and the filter falls
        // through to the unnest path — which cannot resolve the unnested column against the base scan
        // schema and fails the query outright. expandSearch is Calcite's own inverse transform, so
        // this is semantics-preserving; a condition with no Sarg is returned unchanged.
        RexNode condition = RexUtil.expandSearch(rexBuilder, null, filter.getCondition());

        // Try the lambda (nested_any_match) rewrite first — it preserves parent grain.
        RexNode lambdaCondition = tryLambdaRewrite(condition, arrayCol, input.getRowType(), rexBuilder);
        if (lambdaCondition != null) {
            LOGGER.info("[NESTED-LAMBDA] filter rewritten to nested_any_match (no unnest, row count preserved)");
            return LogicalFilter.create(input, lambdaCondition);
        }

        // Fallback: inject LogicalNestedScope (the old unnest path).
        LOGGER.info("[NESTED] filter lambda-rewrite not applicable, falling back to unnest path");
        int originalColCount = input.getRowType().getFieldCount();
        UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
        if (u == null) {
            return filter;
        }
        ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.nestedScope.getRowType());
        RexNode newCondition = condition.accept(shuttle);
        RelNode newFilter = LogicalFilter.create(u.nestedScope, newCondition);

        List<RexNode> passthrough = new ArrayList<>(originalColCount);
        List<String> names = new ArrayList<>(originalColCount);
        List<RelDataTypeField> scopeFields = u.nestedScope.getRowType().getFieldList();
        for (int i = 0; i < originalColCount; i++) {
            passthrough.add(rexBuilder.makeInputRef(scopeFields.get(i).getType(), i));
            names.add(scopeFields.get(i).getName());
        }
        return LogicalProject.create(newFilter, List.of(), passthrough, names);
    }

    /**
     * Attempts to rewrite the filter condition using {@code NESTED_ANY_MATCH_EXPR}. Splits the
     * TOP-LEVEL {@code AND} conjuncts (if any) into two groups:
     * <ul>
     *   <li>conjuncts that reference our array column — combined into the array-side condition per
     *       the (removed) independent-conjunct routing option below</li>
     *   <li>conjuncts that don't (pure parent predicates, e.g. {@code count > 0}) — passed through
     *       unchanged and ANDed back in at the row level, since parent predicates are genuinely
     *       independent per-row and don't need per-element evaluation</li>
     * </ul>
     * A non-AND condition (a single comparison, an OR, a NOT, ...) is treated as one conjunct, so
     * a single {@code comments.a = X or comments.b = Y}-style OR always becomes ONE joint call
     * regardless of the flag below.
     *
     * <p><b>Default (flag off): every array-referencing conjunct is fused into ONE joint
     * {@code NESTED_ANY_MATCH_EXPR} call</b> — a single element must satisfy the WHOLE combined
     * condition, matching vanilla's strict joint-element guarantee. This is the safe default.
     *
     * <p><b>the (removed) independent-conjunct routing option on: each array-referencing conjunct gets
     * its OWN independent {@code NESTED_ANY_MATCH_EXPR} call</b> instead, ANDed together at the row
     * level — see that property's javadoc for the accepted correctness gap this trades for
     * per-conjunct backend routing.
     *
     * <p>Returns null (triggering the Correlate+Uncollect fallback) if any array-referencing
     * conjunct's tree can't be built — e.g. it touches a DIFFERENT array column, or mixes an
     * array-of-ours reference with a parent column inside the SAME comparison (ambiguous — which
     * row's value?).
     */
    private static RexNode tryLambdaRewrite(RexNode condition, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        // A top-level OR mixing an array-referencing operand with a pure-parent operand (e.g.
        // `comments.author='alice' OR views>50`) used to always fall through to the single-joint-
        // tree path below, which fails to build (a parent column has no meaning inside a per-
        // element tree) and forces the Correlate+Uncollect fallback — which then mis-drops parent
        // rows with an absent/empty array (see BUG note on scenario "OR breaks the classification")
        // and, at 2+ levels of nesting, crashes outright (the unnest path's intermediate row
        // carries a raw multi-level Struct that DataFusion's array_element/sum can't handle). Try
        // the OR-split first; only a pure-array OR (no parent operand at all, e.g. `comments.a=X or
        // comments.b=Y`) falls through unchanged to the existing single-joint-tree treatment below.
        // NOTE on negation (investigated, NOT fixable here). SQL/PPL `where NOT (depts.name='Sales')`
        // should mean "no dept is named Sales" (¬∃e p), but a comparison on a nested field is implicitly
        // EXISTENTIALLY quantified, so folding the negation into the comparison yields ∃e ¬p — a different
        // and wrong predicate. (A doc with depts [Sales, Eng] satisfies ∃e ¬p but must not satisfy ¬∃e p.)
        //
        // We cannot correct that here: the negation is already pushed down before this rule runs. The very
        // first RelNode handed to PlannerImpl for that query is `<>(ITEM($0,'name'),'Sales')` — no NOT node
        // survives to the rewriter, and De Morgan has likewise already turned NOT(A AND B) into
        // OR(<>,<=). That lowering happens in the external PPL→Calcite frontend
        // (unified-query-ppl jar), which is not part of this repository. Fixing it requires suppressing
        // negation-pushing when the negated operand references an array/nested field, at the point where
        // the RelNode is built. A `SqlKind.NOT` branch here would be dead code.
        if (condition.getKind() == SqlKind.OR) {
            RexNode orSplit = tryOrSplitRewrite(condition, arrayCol, inputRowType, rexBuilder);
            if (orSplit != null) {
                LOGGER.info(
                    "[NESTED-LAMBDA] top-level OR split: array-side NESTED_ANY_MATCH_EXPR OR'd with pure-parent "
                        + "operand(s) at the row level (existential quantification distributes over OR — no unnest needed)"
                );
                return orSplit;
            }
        }
        List<RexNode> conjuncts = condition.getKind() == SqlKind.AND ? ((RexCall) condition).getOperands() : List.of(condition);

        ExprTreeBuilder builder = new ExprTreeBuilder(arrayCol, inputRowType);
        List<RexNode> arrayConjuncts = new ArrayList<>();
        List<RexNode> parentConjuncts = new ArrayList<>();
        for (RexNode conjunct : conjuncts) {
            if (builder.containsItemOnArray(conjunct)) {
                arrayConjuncts.add(conjunct);
            } else {
                parentConjuncts.add(conjunct);
            }
        }
        if (arrayConjuncts.isEmpty()) {
            return null; // nothing to rewrite on our array — shouldn't normally happen, fall back
        }

        RexNode combinedArrayCondition;
        {
            // Every array-referencing conjunct is combined into ONE joint per-element tree, regardless
            // of count or shape — a single equality leaf and a multi-clause compound condition both go
            // through the same NESTED_ANY_MATCH_EXPR construction below. A single element must satisfy
            // the WHOLE combined tree, matching vanilla OpenSearch's strict joint-element guarantee.
            // Whether the resulting call is ALSO Lucene-delegable (not just DataFusion-native) is decided
            // later, per instance, by OpenSearchFilterRule consulting each candidate backend's
            // DelegatedPredicateSerializer#canServe (see NESTED_ANY_MATCH_EXPR_OP's javadoc) — this
            // method never special-cases the single-leaf shape itself.
            List<Map<String, Object>> arrayTrees = new ArrayList<>();
            for (RexNode conjunct : arrayConjuncts) {
                Map<String, Object> tree = builder.build(conjunct);
                if (tree == null) {
                    return null; // unsupported shape somewhere in this conjunct — fall back entirely
                }
                arrayTrees.add(tree);
            }

            // Multiple conjuncts sharing the SAME nested-array prefix (e.g. `products.variants.specs.key=
            // "weight" AND products.variants.specs.val>50`, both under products.variants.specs) must be
            // evaluated against the SAME inner element, not independently ANDed at the top level — an
            // AND of two separately {"nested":...}-wrapped trees re-runs the inner ∃ loop once per
            // operand, so a DIFFERENT inner element could satisfy each conjunct (the "Delta" false-
            // positive bug, one level deeper). Merge any trees sharing an outer {"nested": name} wrapper
            // into ONE such wrapper around the AND of their (recursively re-merged) inner subtrees.
            arrayTrees = mergeSharedNestedPrefixes(arrayTrees);

            Map<String, Object> combinedTree = arrayTrees.size() == 1 ? arrayTrees.get(0) : Map.of("op", "AND", "args", arrayTrees);
            RexNode anyMatchCall = buildAnyMatchExprCall(combinedTree, arrayCol, inputRowType, rexBuilder);
            if (anyMatchCall == null) {
                return null; // serialization failed — fall back entirely
            }

            // ── Lucene pruning peers (superset-safe split, same idea as flat-column delegation)
            // ─────────────────────────────────────────────────────────────────────────────────
            // The fused NESTED_ANY_MATCH_EXPR above is the AUTHORITATIVE, DataFusion-evaluated,
            // element-correlated predicate — it alone is 100% correct. For each top-level-AND
            // keyword-equality conjunct on our array we ALSO emit a NESTED_ANY_MATCH_EXPR call
            // carrying just that single-equality leaf, which is dual-viable [lucene, datafusion] and
            // so gets performance-delegated to Lucene's native block-join query. That call matches
            // "parent has SOME child with field=v", a SUPERSET of the fused predicate's parent set —
            // AND-ing a superset with the authoritative predicate never changes the result, but it
            // lets Lucene's inverted index PRUNE which rows DataFusion must evaluate. Only pure
            // keyword equality qualifies. Emitted only in the compound (size>1) path — a single
            // conjunct's tree already IS the equality leaf, so the fused call above is already
            // Lucene-viable and a peer would be redundant.
            List<RexNode> lucenerPeers = new ArrayList<>();
            if (arrayConjuncts.size() > 1) {
                for (RexNode conjunct : arrayConjuncts) {
                    RexNode peer = tryDirectEqualityRewrite(conjunct, arrayCol, inputRowType, rexBuilder);
                    if (peer != null) {
                        lucenerPeers.add(peer);
                    }
                }
            }

            if (lucenerPeers.isEmpty()) {
                combinedArrayCondition = anyMatchCall;
            } else {
                // authoritative fused expr AND (Lucene-prunable keyword peers). The peers are
                // supersets, so this is semantically identical to `anyMatchCall` alone; the AND
                // exists purely so the marking layer can performance-delegate the peers to Lucene.
                List<RexNode> operands = new ArrayList<>(lucenerPeers.size() + 1);
                operands.add(anyMatchCall);
                operands.addAll(lucenerPeers);
                combinedArrayCondition = rexBuilder.makeCall(
                    rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
                    org.apache.calcite.sql.fun.SqlStdOperatorTable.AND,
                    operands
                );
                LOGGER.info(
                    "[NESTED-LAMBDA] fused NESTED_ANY_MATCH_EXPR (authoritative) + {} Lucene pruning peer(s) "
                        + "(keyword-equality conjuncts delegated to Lucene for pruning)",
                    lucenerPeers.size()
                );
            }
        }

        return combineWithParentConjuncts(combinedArrayCondition, parentConjuncts, rexBuilder);
    }

    /**
     * Splits a top-level {@code OR}'s operands into array-referencing and pure-parent groups,
     * mirroring {@link #tryLambdaRewrite}'s AND-split — but ORs the two groups back together at
     * the row level instead of ANDing them, which is valid because existential quantification
     * distributes over OR: {@code (∃e. P(e)) OR Q  ≡  ∃e. (P(e) OR Q)} for any per-row predicate
     * {@code Q} that doesn't depend on {@code e}. Concretely, {@code comments.author='alice' OR
     * views>50} becomes {@code NESTED_ANY_MATCH_EXPR(comments, {author='alice'}) OR (views>50)} —
     * each side evaluated independently (one existentially over the array, one as a plain row
     * check), then ORed — instead of forcing {@code views>50} to be expressed as a per-element
     * field (which it cannot be, since it isn't part of any array element).
     *
     * <p>Returns {@code null} (triggering the fall-through to the existing single-joint-tree
     * path, which itself falls back to Correlate+Uncollect if that also fails) in two cases: no
     * operand references our array at all (nothing to split — shouldn't normally happen since the
     * caller already confirmed {@code arrayCol} is referenced somewhere), or EVERY operand
     * references the array (a pure-array OR like {@code comments.a=X or comments.b=Y} has no
     * parent-side operand to split off, and must stay a single joint per-element tree so a
     * DIFFERENT element can't satisfy each side of the OR — same "Delta" false-positive concern
     * that already motivates fusing AND conjuncts). Each array-referencing operand is itself
     * fused into ONE joint tree via {@link ExprTreeBuilder#build} (an OR of multiple array
     * operands, e.g. {@code comments.a=X or comments.b=Y or views>50}, still needs joint-element
     * semantics across the array operands — only the parent operand(s) split cleanly away).
     */
    private static RexNode tryOrSplitRewrite(RexNode condition, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        List<RexNode> operands = ((RexCall) condition).getOperands();
        ExprTreeBuilder builder = new ExprTreeBuilder(arrayCol, inputRowType);
        List<RexNode> arrayOperands = new ArrayList<>();
        List<RexNode> parentOperands = new ArrayList<>();
        for (RexNode operand : operands) {
            if (builder.containsItemOnArray(operand)) {
                arrayOperands.add(operand);
            } else {
                parentOperands.add(operand);
            }
        }
        if (arrayOperands.isEmpty() || parentOperands.isEmpty()) {
            return null; // nothing to split (all-parent can't happen here; all-array needs the joint-tree path)
        }

        List<Map<String, Object>> arrayTrees = new ArrayList<>();
        for (RexNode operand : arrayOperands) {
            Map<String, Object> tree = builder.build(operand);
            if (tree == null) {
                return null; // unsupported shape — fall back to the joint-tree path (then unnest if that also fails)
            }
            arrayTrees.add(tree);
        }
        arrayTrees = mergeSharedNestedPrefixes(arrayTrees);
        Map<String, Object> combinedArrayTree = arrayTrees.size() == 1 ? arrayTrees.get(0) : Map.of("op", "OR", "args", arrayTrees);
        RexNode anyMatchCall = buildAnyMatchExprCall(combinedArrayTree, arrayCol, inputRowType, rexBuilder);
        if (anyMatchCall == null) {
            return null; // serialization failed — fall back
        }

        List<RexNode> allOperands = new ArrayList<>(parentOperands.size() + 1);
        allOperands.add(anyMatchCall);
        allOperands.addAll(parentOperands);
        return rexBuilder.makeCall(rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN), org.apache.calcite.sql.fun.SqlStdOperatorTable.OR, allOperands);
    }

    /** Builds a single {@code NESTED_ANY_MATCH_EXPR(arrayCol, jsonTree)} call, or {@code null} if
     *  the tree can't be serialized. */
    private static RexNode buildAnyMatchExprCall(Map<String, Object> tree, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        String json;
        try {
            json = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().writeValueAsString(tree);
        } catch (Exception e) {
            LOGGER.warn("[NESTED-LAMBDA] failed to serialize expr tree, falling back to unnest", e);
            return null;
        }
        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        RexNode exprLit = rexBuilder.makeLiteral(json);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_EXPR_OP,
            List.of(arrayRef, exprLit)
        );
    }

    /**
     * Fast-path / superset-pruning-peer check: if {@code conjunct} is exactly
     * {@code ITEM($arrayCol,'field') = 'literal'} (either operand order) with a STRING literal
     * value, emits a {@code NESTED_ANY_MATCH_EXPR(arrayCol, '{"op":"=","args":[{"field":F},{"lit":V}]}')}
     * call carrying just that single-equality-leaf tree. {@code NestedAnyMatchExprSerializer#canServe}
     * (in the Lucene backend) recognizes exactly this single-leaf shape, so this call is registered as
     * a dual-viable [lucene, datafusion] filter capability, enabling performance-delegation to Lucene's
     * native nested block-join query — unlike a compound tree, which stays DataFusion-only.
     *
     * <p>Deliberately restricted to a STRING-literal comparison value: only keyword-typed nested
     * leaves make sense as a Lucene term lookup in this composite (parquet+lucene) setup, and no
     * leaf-level field-type info is available at this point — nested leaf fields have no entry in
     * {@code FieldStorageResolver} (it explicitly skips {@code "nested"}-typed fields). Requiring a
     * string literal is the same conservative heuristic this class already uses elsewhere ({@code
     * ItemFinder}/{@code ExprTreeBuilder}) to infer "this looks like a keyword comparison" without
     * real type resolution. A numeric/boolean-literal comparison falls through to the generic
     * {@code NESTED_ANY_MATCH_EXPR} path unchanged, staying DataFusion-only rather than risk
     * mis-registering a Lucene capability for a field Lucene doesn't actually index in this format.
     *
     * <p>Only {@code EQUALS} is handled (not {@code NOT_EQUALS}) — a nested "field != value"
     * existence check has no Lucene query primitive as simple as a single {@code TermQuery} and
     * isn't needed for the common case this fast path targets. Returns {@code null} for anything
     * else (including {@code NOT_EQUALS}, non-comparison kinds, or a non-string literal), which
     * triggers the generic path in the caller.
     */
    private static RexNode tryDirectEqualityRewrite(RexNode conjunct, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        if (conjunct.getKind() != SqlKind.EQUALS || !(conjunct instanceof RexCall call) || call.getOperands().size() != 2) {
            return null;
        }
        RexNode left = call.getOperands().get(0);
        RexNode right = call.getOperands().get(1);

        RexCall itemCall;
        RexLiteral valueLit;
        if (isItemOnArray(left, arrayCol) && right instanceof RexLiteral lit) {
            itemCall = (RexCall) left;
            valueLit = lit;
        } else if (isItemOnArray(right, arrayCol) && left instanceof RexLiteral lit) {
            itemCall = (RexCall) right;
            valueLit = lit;
        } else {
            return null;
        }
        if (valueLit.getTypeName() != SqlTypeName.CHAR && valueLit.getTypeName() != SqlTypeName.VARCHAR) {
            return null; // not a string comparison — leave for the generic path
        }
        RexNode fieldNameNode = itemCall.getOperands().get(1);
        if (!(fieldNameNode instanceof RexLiteral fieldLit) || fieldLit.getTypeName() != SqlTypeName.CHAR) {
            return null;
        }
        String fieldName = fieldLit.getValueAs(String.class);
        String value = valueLit.getValueAs(String.class);

        Map<String, Object> equalityLeafTree = Map.of(
            "op",
            "=",
            "args",
            List.of(Map.of("field", fieldName), Map.of("lit", value))
        );
        return buildAnyMatchExprCall(equalityLeafTree, arrayCol, inputRowType, rexBuilder);
    }

    /** True if {@code node} is {@code ITEM($arrayCol, 'field')} — a direct nested-leaf reference on our array. */
    private static boolean isItemOnArray(RexNode node, int arrayCol) {
        if (!(node instanceof RexCall call) || !"ITEM".equals(call.getOperator().getName()) || call.getOperands().size() != 2) {
            return false;
        }
        return call.getOperands().get(0) instanceof RexInputRef ref && ref.getIndex() == arrayCol;
    }

    /**
     * Groups {@code trees} by their outermost {@code {"nested": name, "inner": ...}} wrapper (trees
     * with no such wrapper — i.e. today's single-level shape — are left alone, in original relative
     * order), and for any group of 2+ trees sharing the same {@code name}, replaces them with ONE
     * {@code {"nested": name, "inner": <recursively re-merged AND of their inner subtrees>}} —
     * required so multiple conjuncts on the same nested-array path are evaluated against the SAME
     * inner element rather than independently ANDed (see the call site's javadoc for why an
     * independent AND is a correctness bug, not just a missed optimization).
     */
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> mergeSharedNestedPrefixes(List<Map<String, Object>> trees) {
        LinkedHashMap<String, List<Map<String, Object>>> byNestedName = new LinkedHashMap<>();
        List<Map<String, Object>> unwrapped = new ArrayList<>();
        for (Map<String, Object> tree : trees) {
            Object nestedName = tree.get("nested");
            if (nestedName instanceof String name) {
                byNestedName.computeIfAbsent(name, k -> new ArrayList<>()).add(tree);
            } else {
                unwrapped.add(tree);
            }
        }
        List<Map<String, Object>> merged = new ArrayList<>(unwrapped);
        for (Map.Entry<String, List<Map<String, Object>>> entry : byNestedName.entrySet()) {
            List<Map<String, Object>> group = entry.getValue();
            if (group.size() == 1) {
                merged.add(group.get(0));
                continue;
            }
            List<Map<String, Object>> innerTrees = new ArrayList<>(group.size());
            for (Map<String, Object> tree : group) {
                innerTrees.add((Map<String, Object>) tree.get("inner"));
            }
            innerTrees = mergeSharedNestedPrefixes(innerTrees); // catch a further shared prefix one level deeper
            Map<String, Object> mergedInner = innerTrees.size() == 1 ? innerTrees.get(0) : Map.of("op", "AND", "args", innerTrees);
            Map<String, Object> mergedWrapper = new LinkedHashMap<>();
            mergedWrapper.put("nested", entry.getKey());
            mergedWrapper.put("inner", mergedInner);
            merged.add(mergedWrapper);
        }
        return merged;
    }

    /** ANDs {@code arrayCall} together with any parent-only conjuncts (passed through unchanged,
     *  since they're independent per-row and don't need per-element evaluation); returns {@code
     *  arrayCall} directly when there are none. */
    private static RexNode combineWithParentConjuncts(RexNode arrayCall, List<RexNode> parentConjuncts, RexBuilder rexBuilder) {
        if (parentConjuncts.isEmpty()) {
            return arrayCall;
        }
        List<RexNode> allOperands = new ArrayList<>(parentConjuncts.size() + 1);
        allOperands.add(arrayCall);
        allOperands.addAll(parentConjuncts);
        return rexBuilder.makeCall(rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN), org.apache.calcite.sql.fun.SqlStdOperatorTable.AND, allOperands);
    }

    /**
     * Walks a Calcite expression tree and builds an equivalent JSON-serializable tree describing the
     * per-element predicate, for the {@code NESTED_ANY_MATCH_EXPR} wire format. Node shapes:
     * <ul>
     *   <li>{@code {"op":"AND"|"OR","args":[...]}} — boolean connective</li>
     *   <li>{@code {"op":"NOT","args":[...]}} — negation</li>
     *   <li>{@code {"op":">"|">="|"<"|"<="|"="|"!=","args":[...]}} — comparison (exactly 2 args)</li>
     *   <li>{@code {"op":"+"|"-"|"*"|"/"|"%","args":[...]}} — arithmetic (exactly 2 args)</li>
     *   <li>{@code {"field":"fieldName"}} — read a field off the CURRENT array element</li>
     *   <li>{@code {"lit":value}} — a literal number/string/boolean</li>
     * </ul>
     * Top-level entry point is {@link #build}, which returns {@code null} if the condition contains
     * a reference to a DIFFERENT array column (unsupported — multi-array predicates fall back to
     * unnest) or an operator this builder doesn't know how to translate.
     */
    private static final class ExprTreeBuilder {
        private final int arrayCol;
        private final RelDataType inputRowType;

        ExprTreeBuilder(int arrayCol, RelDataType inputRowType) {
            this.arrayCol = arrayCol;
            this.inputRowType = inputRowType;
        }

        /**
         * Splits a chain {@code ITEM(ITEM(...ITEM($arrayCol,'f1')...,'f2'),'f3')} into its ordered
         * list of field-name hops, {@code ["f1","f2","f3"]} — one entry per nested-array boundary
         * plus the final leaf field. Returns {@code null} if {@code node} is not an ITEM chain
         * rooted at {@code arrayCol} at all (e.g. a literal, a different column, or an unsupported
         * shape) — the caller falls back accordingly.
         */
        private List<String> chainHops(RexNode node) {
            List<String> hops = new ArrayList<>();
            RexNode cur = node;
            while (cur instanceof RexCall call && "ITEM".equals(call.getOperator().getName()) && call.getOperands().size() == 2) {
                RexNode fieldNode = call.getOperands().get(1);
                if (!(fieldNode instanceof RexLiteral lit) || lit.getTypeName() != SqlTypeName.CHAR) {
                    return null;
                }
                hops.add(0, lit.getValueAs(String.class));
                cur = call.getOperands().get(0);
            }
            if (hops.isEmpty() || !(cur instanceof RexInputRef ref) || ref.getIndex() != arrayCol) {
                return null;
            }
            return hops;
        }

        /**
         * Given the field-name hops of a chain rooted at {@code arrayCol}, walks each intermediate
         * hop's type (starting from {@code arrayCol}'s own component/element ROW type) to find how
         * many of the LEADING hops are themselves nested-array (further {@code ARRAY(ROW(...))})
         * boundaries, as opposed to the final leaf. E.g. for {@code products.variants.color} with
         * {@code hops = ["variants","color"]}: {@code variants} is itself an array (1 boundary),
         * {@code color} is the terminal leaf field — returns 1. For the single-level {@code
         * hops = ["score"]}: {@code score} is the terminal leaf directly — returns 0 (no further
         * nesting to descend through; this is the pre-existing single-level shape).
         */
        private int countNestedBoundaries(List<String> hops) {
            RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
            int boundaries = 0;
            for (int i = 0; i < hops.size() - 1; i++) {
                if (elementType == null || !elementType.isStruct()) {
                    break;
                }
                RelDataTypeField field = elementType.getField(hops.get(i), true, false);
                if (field == null || field.getType().getSqlTypeName() != SqlTypeName.ARRAY) {
                    break;
                }
                boundaries++;
                elementType = field.getType().getComponentType();
            }
            return boundaries;
        }

        /**
         * Returns null if the tree can't be expressed (unsupported operator, or ITEM on the wrong
         * array). A leaf {@code ITEM($arrayCol,'field')} — single-hop, no further nested-array
         * boundary — becomes {@code {"field":"field"}} directly. A multi-hop chain crossing one or
         * more further nested-array boundaries (e.g. {@code products.variants.color}) can only be
         * expressed by wrapping a whole COMPARISON in {@code {"nested":...,"inner":...}} (the Rust
         * UDF dispatches {@code "nested"} from its boolean evaluator, {@code eval_bool}, never from
         * its value evaluator, {@code eval_value} — see {@code nested_any_match_expr.rs}), so that
         * descent happens in the comparison branch below, not here. A multi-hop chain reaching this
         * leaf branch directly (with no enclosing comparison for the comparison branch to have
         * already descended through) has no valid representation and falls back.
         */
        Map<String, Object> build(RexNode node) {
            if (node instanceof RexCall itemCall && "ITEM".equals(itemCall.getOperator().getName()) && itemCall.getOperands().size() == 2) {
                RexNode arrayOperand = itemCall.getOperands().get(0);
                RexNode fieldNode = itemCall.getOperands().get(1);
                if (arrayOperand instanceof RexInputRef ref && fieldNode instanceof RexLiteral lit && lit.getTypeName() == SqlTypeName.CHAR) {
                    if (ref.getIndex() != arrayCol) {
                        return null; // ITEM on a DIFFERENT array — unsupported, fall back
                    }
                    return Map.of("field", lit.getValueAs(String.class));
                }
                return null; // multi-hop chain with no enclosing comparison — unsupported here, fall back
            }

            if (node instanceof RexLiteral lit) {
                // String/char literals come back from getValueAs(Comparable.class) as Calcite's
                // internal NlsString (carrying charset/collation) — JSON-serializing that produces
                // a nested object, not a plain string, which the Rust-side parser can't read as a
                // string value. getValueAs(String.class) unwraps NlsString to a plain Java String;
                // for non-string types (numbers, booleans) fall back to the generic Comparable path.
                Object value;
                if (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR) {
                    value = lit.getValueAs(String.class);
                } else {
                    value = lit.getValueAs(Comparable.class);
                }
                return Map.of("lit", value == null ? "null" : value);
            }

            if (node instanceof RexCall call) {
                // A CAST wrapping any of the above is transparent for this tree (the Rust side
                // compares numerically regardless of source width).
                if (call.getKind() == SqlKind.CAST) {
                    return build(call.getOperands().get(0));
                }
                String opSymbol = opSymbolFor(call);
                if (opSymbol == null) {
                    // Unknown operator. If it references our array at all, we can't safely pass it
                    // through as a pure-parent predicate (it's ambiguous), so fail closed.
                    return containsItemOnArray(call) ? null : passthroughAsLiteralRef(call);
                }
                // COMPARISON crossing a nested-array boundary (e.g. products.variants.color = 'red',
                // where 'variants' is itself nested one level below arrayCol): exactly one operand
                // contains a chain descending through 1+ further nested-array boundaries, and no
                // OTHER operand touches our array at all (see this method's javadoc — comparing two
                // DIFFERENT nested chains to each other is not handled and falls back). Wrap the
                // WHOLE comparison in {"nested": hop, "inner": ...} once per boundary crossed —
                // required because the Rust UDF dispatches "nested" only from its boolean evaluator,
                // never from the value evaluator that runs a comparison's individual operands.
                if (isComparisonOp(opSymbol)) {
                    Map<String, Object> descended = tryBuildComparisonWithDescent(call, opSymbol);
                    if (descended != null) {
                        return descended;
                    }
                }
                List<Object> args = new ArrayList<>(call.getOperands().size());
                for (RexNode operand : call.getOperands()) {
                    Map<String, Object> argTree = build(operand);
                    if (argTree == null) {
                        return null;
                    }
                    args.add(argTree);
                }
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("op", opSymbol);
                result.put("args", args);
                return result;
            }

            // A plain column reference NOT on our array (e.g. a parent-row column mixed into the
            // expression) — not representable inside a per-element tree; fail closed rather than
            // silently dropping it.
            return null;
        }

        /**
         * A sub-expression with no ITEM-on-our-array reference at all is a pure parent-row value
         * (e.g. a literal, or a reference to a different, non-array column) — not evaluable per
         * array element. Rather than guess, we fail closed: the caller (rewriteFilter) then falls
         * back to the Correlate+Uncollect path, which resolves parent columns correctly by carrying
         * them through the join unchanged.
         */
        private Map<String, Object> passthroughAsLiteralRef(RexNode node) {
            return null;
        }

        /**
         * True if {@code node} contains an {@code ITEM} anywhere in its tree that is rooted (after
         * walking down through any further chained {@code ITEM} calls — see {@link
         * OpenSearchNestedFieldRewriter#itemArrayCol}) at {@code arrayCol}. A multi-level dotted path
         * like {@code products.variants.color} must be recognized here — otherwise the conjunct gets
         * silently classified as a PARENT predicate (see {@code tryLambdaRewrite}'s split into {@code
         * arrayConjuncts}/{@code parentConjuncts}) and passed through unfiltered, which is a
         * wrong-results bug, not a crash or a safe fallback (see FR-3 in the project plan).
         */
        private boolean containsItemOnArray(RexNode node) {
            if (node instanceof RexCall call) {
                if ("ITEM".equals(call.getOperator().getName())
                    && call.getOperands().size() == 2
                    && itemArrayCol(call, inputRowType) == arrayCol) {
                    return true;
                }
                for (RexNode op : call.getOperands()) {
                    if (containsItemOnArray(op)) return true;
                }
            }
            return false;
        }

        private static boolean isComparisonOp(String opSymbol) {
            return switch (opSymbol) {
                case ">", ">=", "<", "<=", "=", "!=", "EXISTS", "NOT_EXISTS" -> true;
                default -> false;
            };
        }

        /**
         * Attempts to build {@code call} (a comparison with operator symbol {@code opSymbol} —
         * including the 1-operand {@code EXISTS}/{@code NOT_EXISTS}, i.e. {@code IS NOT NULL}/{@code
         * IS NULL}) as a descended, {@code {"nested":...,"inner":...}}-wrapped tree, for the case
         * where exactly one operand is an ITEM chain crossing 1+ further nested-array boundaries
         * beyond {@code arrayCol} itself (e.g. {@code products.variants.color = 'red'}, or {@code
         * products.variants.color IS NOT NULL}). Returns {@code null} — NOT a fall-back-worthy
         * failure, just "this comparison doesn't need descent" — when no operand crosses a boundary
         * (the ordinary single-level path in {@link #build} handles it), and returns {@code null}
         * (this time meaning "unsupported, fall back entirely") when MORE THAN ONE operand touches
         * our array — comparing two different nested chains to each other, or a chain against
         * another reference into the same array at a different depth, has no defined per-element
         * correlation semantics here and is intentionally not guessed at.
         */
        private Map<String, Object> tryBuildComparisonWithDescent(RexCall call, String opSymbol) {
            List<RexNode> operands = call.getOperands();
            if (operands.size() != 1 && operands.size() != 2) {
                return null;
            }
            int chainOperandIdx = -1;
            List<String> hops = null;
            int boundaries = 0;
            for (int i = 0; i < operands.size(); i++) {
                RexNode operand = operands.get(i);
                if (!containsItemOnArray(operand)) {
                    continue;
                }
                List<String> operandHops = chainHops(operand);
                if (operandHops == null) {
                    return null; // touches our array but isn't a plain chain (e.g. arithmetic on it) — unsupported here
                }
                int operandBoundaries = countNestedBoundaries(operandHops);
                if (operandBoundaries == 0) {
                    continue; // single-level reference — no descent needed for THIS operand
                }
                if (chainOperandIdx != -1) {
                    return null; // more than one operand crosses a boundary — unsupported, fall back
                }
                chainOperandIdx = i;
                hops = operandHops;
                boundaries = operandBoundaries;
            }
            if (chainOperandIdx == -1) {
                return null; // no operand needed descent — let the caller build the ordinary shape
            }
            // Build the innermost comparison directly from the LEAF field name (the final hop),
            // bypassing the ordinary ITEM-based leaf branch in build() entirely — after descent, the
            // comparison operates on the innermost array's element, not on arrayCol's own element, so
            // there is no ITEM RexNode left to re-walk; {"field": leaf} is exactly what eval_value
            // expects once inside that many "nested" wrappers.
            String leafField = hops.get(hops.size() - 1);
            Map<String, Object> fieldNode = Map.of("field", leafField);
            List<Object> comparisonArgs;
            if (operands.size() == 1) {
                comparisonArgs = List.of(fieldNode); // EXISTS/NOT_EXISTS: the chain IS the sole operand
            } else {
                RexNode otherOperand = operands.get(1 - chainOperandIdx);
                if (containsItemOnArray(otherOperand)) {
                    return null; // the non-chain operand ALSO touches our array — ambiguous, fall back
                }
                Map<String, Object> otherTree = build(otherOperand);
                if (otherTree == null) {
                    return null;
                }
                comparisonArgs = chainOperandIdx == 0 ? List.of(fieldNode, otherTree) : List.of(otherTree, fieldNode);
            }
            Map<String, Object> innerComparison = new LinkedHashMap<>();
            innerComparison.put("op", opSymbol);
            innerComparison.put("args", comparisonArgs);

            Map<String, Object> wrapped = innerComparison;
            for (int level = boundaries - 1; level >= 0; level--) {
                Map<String, Object> nestedWrapper = new LinkedHashMap<>();
                nestedWrapper.put("nested", hops.get(level));
                nestedWrapper.put("inner", wrapped);
                wrapped = nestedWrapper;
            }
            return wrapped;
        }

        /**
         * Maps a RexCall to its JSON-tree operator symbol. Most operators are recognized by
         * Calcite's own {@code SqlKind} (Calcite's built-in comparison/arithmetic operators).
         * PPL's own custom operators (registered as {@link org.apache.calcite.sql.SqlFunction}
         * UDFs, e.g. {@code PPLBuiltinOperators.MOD} — {@code new ModFunction().toUDF("MOD")} in
         * the sql-plugin) carry {@code SqlKind.OTHER_FUNCTION} regardless of what they compute, so
         * for that catch-all kind we fall back to matching the operator's NAME instead — the same
         * by-name pattern already used for {@code ITEM} elsewhere in this class.
         */
        private static String opSymbolFor(RexCall call) {
            SqlKind kind = call.getKind();
            String byKind = switch (kind) {
                case AND -> "AND";
                case OR -> "OR";
                case NOT -> "NOT";
                case GREATER_THAN -> ">";
                case GREATER_THAN_OR_EQUAL -> ">=";
                case LESS_THAN -> "<";
                case LESS_THAN_OR_EQUAL -> "<=";
                case EQUALS -> "=";
                case NOT_EQUALS -> "!=";
                case IS_NOT_NULL -> "EXISTS";
                case IS_NULL -> "NOT_EXISTS";
                case PLUS -> "+";
                case MINUS -> "-";
                case TIMES -> "*";
                case DIVIDE -> "/";
                case MOD -> "%";
                default -> null;
            };
            if (byKind != null) {
                return byKind;
            }
            if (kind == SqlKind.OTHER_FUNCTION) {
                return switch (call.getOperator().getName().toUpperCase(java.util.Locale.ROOT)) {
                    case "MOD", "MODULUS", "MODULUSFUNCTION" -> "%";
                    default -> null;
                };
            }
            return null;
        }
    }

    // ---- Shared: build LogicalNestedScope(input, arrayCol) appending the struct fields ----------

    /** Result of injecting an unnest: the new NestedScope rel + the index where unnested fields begin. */
    private record UnnestResult(LogicalNestedScope nestedScope, int unnestedFieldIndex, Map<String, Integer> fieldToIndex) {}

    /**
     * Injects {@code LogicalNestedScope(input, arrayCol)} — the backend-neutral "expand this array,
     * keep parent identity" operator (see that class's javadoc); marked into {@code
     * OpenSearchNestedScope} later by {@code OpenSearchNestedScopeRule} during the marking phase, with
     * viable backends computed from a real capability lookup rather than hardcoded. Output is
     * {@code [original cols..., unnested struct fields...]} — original indices preserved, struct
     * fields appended starting at {@code input.fieldCount}.
     */
    private static UnnestResult injectUnnest(RelNode input, int arrayCol, RelOptCluster cluster, RexBuilder rexBuilder) {
        RelDataType inputRowType = input.getRowType();
        RelDataTypeField arrayField = inputRowType.getFieldList().get(arrayCol);
        RelDataType elementType = arrayField.getType().getComponentType();
        if (elementType == null || !elementType.isStruct()) {
            LOGGER.warn("[NESTED] array column '{}' is not ARRAY(ROW) — skipping unnest", arrayField.getName());
            return null;
        }

        LogicalNestedScope nestedScope = LogicalNestedScope.create(input, arrayCol);

        int originalColCount = inputRowType.getFieldCount();
        Map<String, Integer> fieldToIndex = new LinkedHashMap<>();
        List<RelDataTypeField> scopeFields = nestedScope.getRowType().getFieldList();
        for (int i = originalColCount; i < scopeFields.size(); i++) {
            fieldToIndex.put(scopeFields.get(i).getName(), i);
        }
        LOGGER.info(
            "[NESTED] injected NestedScope UNNEST on array col '{}' (idx {}); unnested fields {} at indices {}..{}",
            arrayField.getName(),
            arrayCol,
            fieldToIndex.keySet(),
            originalColCount,
            scopeFields.size() - 1
        );
        return new UnnestResult(nestedScope, originalColCount, fieldToIndex);
    }

    // ---- ITEM detection + rewriting ------------------------------------------------------------

    /**
     * Finds the first array-column index referenced by an {@code ITEM($arrayCol,'field')} anywhere
     * within the given expressions, or -1 if none. (Single-array per rewrite step for now; multiple
     * distinct arrays in one node is a follow-up — see class javadoc.)
     */
    private static int firstArrayColReferenced(List<RexNode> exprs, RelDataType inputRowType) {
        ItemFinder finder = new ItemFinder(inputRowType);
        for (RexNode e : exprs) {
            e.accept(finder);
        }
        return finder.arrayCol;
    }

    /** Walks an expression tree recording the array-column index of the first {@code ITEM}-on-array. */
    private static final class ItemFinder extends RexShuttle {
        private final RelDataType inputRowType;
        private int arrayCol = -1;

        ItemFinder(RelDataType inputRowType) {
            this.inputRowType = inputRowType;
        }

        @Override
        public RexNode visitCall(RexCall call) {
            if (arrayCol < 0) {
                int c = itemArrayCol(call, inputRowType);
                if (c >= 0) {
                    arrayCol = c;
                }
            }
            return super.visitCall(call);
        }
    }

    /**
     * Replaces every {@code ITEM($arrayCol,'field')} (for the target array column) with a plain
     * {@link RexInputRef} to the appended unnested column of that field.
     */
    private static final class ItemRewriteShuttle extends RexShuttle {
        private final int arrayCol;
        private final Map<String, Integer> fieldToIndex;
        private final RexBuilder rexBuilder;
        private final RelDataType correlateRowType;

        ItemRewriteShuttle(int arrayCol, int unnestedStartIdx, RexBuilder rexBuilder, RelDataType correlateRowType) {
            this.arrayCol = arrayCol;
            this.rexBuilder = rexBuilder;
            this.correlateRowType = correlateRowType;
            this.fieldToIndex = new LinkedHashMap<>();
            for (int i = unnestedStartIdx; i < correlateRowType.getFieldCount(); i++) {
                String colName = correlateRowType.getFieldList().get(i).getName();
                fieldToIndex.put(colName, i);
                // Calcite deduplicates field names by appending a numeric suffix (e.g. "name" → "name0")
                // when the parent already has a field with the same name. Map the original (unsuffixed)
                // name too so ITEM($arrayCol, 'name') resolves to the correct unnested column.
                String stripped = colName.replaceAll("\\d+$", "");
                if (!stripped.equals(colName) && !fieldToIndex.containsKey(stripped)) {
                    fieldToIndex.put(stripped, i);
                }
            }
        }

        @Override
        public RexNode visitCall(RexCall call) {
            if ("ITEM".equals(call.getOperator().getName()) && call.getOperands().size() == 2) {
                RexNode arrayRef = call.getOperands().get(0);
                RexNode fieldNode = call.getOperands().get(1);
                if (arrayRef instanceof RexInputRef ref
                    && ref.getIndex() == arrayCol
                    && fieldNode instanceof RexLiteral lit
                    && lit.getTypeName() == SqlTypeName.CHAR) {
                    String field = lit.getValueAs(String.class);
                    Integer idx = fieldToIndex.get(field);
                    if (idx != null) {
                        return rexBuilder.makeInputRef(correlateRowType.getFieldList().get(idx).getType(), idx);
                    }
                }
            }
            return super.visitCall(call);
        }
    }

    /** If {@code call} is {@code ITEM($N,'field')} with {@code $N} an ARRAY column, returns N; else -1. */
    private static int itemArrayCol(RexCall call, RelDataType inputRowType) {
        if (!"ITEM".equals(call.getOperator().getName()) || call.getOperands().size() != 2) {
            return -1;
        }
        RexNode fieldNode = call.getOperands().get(1);
        if (!(fieldNode instanceof RexLiteral lit) || lit.getTypeName() != SqlTypeName.CHAR) {
            return -1;
        }
        RexNode arrayRef = call.getOperands().get(0);
        while (arrayRef instanceof RexCall innerCall
            && "ITEM".equals(innerCall.getOperator().getName())
            && innerCall.getOperands().size() == 2
            && innerCall.getOperands().get(1) instanceof RexLiteral innerLit
            && innerLit.getTypeName() == SqlTypeName.CHAR) {
            arrayRef = innerCall.getOperands().get(0);
        }
        if (!(arrayRef instanceof RexInputRef ref)) {
            return -1;
        }
        int colIndex = ref.getIndex();
        if (colIndex >= inputRowType.getFieldCount()) {
            return -1;
        }
        RelDataType colType = inputRowType.getFieldList().get(colIndex).getType();
        return colType.getSqlTypeName() == SqlTypeName.ARRAY ? colIndex : -1;
    }
}

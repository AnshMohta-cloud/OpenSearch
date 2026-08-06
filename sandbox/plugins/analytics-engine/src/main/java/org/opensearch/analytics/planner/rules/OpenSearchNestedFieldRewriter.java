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
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Uncollect;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalCorrelate;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalValues;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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
 * array column via {@code ITEM}, it injects the textbook Calcite UNNEST shape beneath that node:
 * <pre>
 *   LogicalCorrelate(INNER, requiredColumns={arrayCol})
 *     ├─ &lt;original input&gt;                              (all original columns, indices UNCHANGED)
 *     └─ Uncollect( Project($cor0.arrayCol, LogicalValues.oneRow) )   (struct fields, APPENDED)
 * </pre>
 * and rewrites each {@code ITEM($arrayCol,'f')} to a plain {@link RexInputRef} of the appended
 * unnested column. Because the correlate keeps the left (original) columns first and appends the
 * exploded struct fields, <b>every original column index is preserved</b> — so operators above the
 * rewritten node are unaffected and the transform composes cleanly across the whole tree.
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
     * Kill-switch for independent per-conjunct backend routing on a multi-conjunct same-array nested
     * filter (e.g. {@code comments.author='frank' AND comments.score<50}). Default {@code false}.
     *
     * <p><b>This is a deliberate, accepted correctness gap, not a safe default.</b> When enabled, each
     * array-referencing conjunct is rewritten and marked independently — a keyword-equality conjunct
     * becomes its own dual-viable {@code NESTED_ANY_MATCH} leaf (Lucene-delegable), any other conjunct
     * becomes its own single-conjunct {@code NESTED_ANY_MATCH_EXPR} leaf (DataFusion-only) — exactly
     * mirroring how independent flat-column conjuncts (e.g. {@code title=/views>}) are already marked
     * and routed. This intentionally DROPS the joint-per-element guarantee this class otherwise
     * enforces everywhere else: vanilla nested semantics require ONE array element to satisfy every
     * conjunct together, but independently-annotated leaves get independently evaluated and ANDed at
     * the row level, so a row where DIFFERENT elements each satisfy a different conjunct is wrongly
     * included (e.g. {@code comments=[{frank,90},{carol,5}]} wrongly matches {@code author='frank' AND
     * score<50} — carol's low score, not frank's, satisfies the second conjunct). Requested explicitly
     * to unblock incremental "does this reach the right backend" plumbing work ahead of a real fix
     * (fusing same-array conjuncts into one combined Lucene query, tracked separately) — do not enable
     * by default and do not remove this javadoc's warning when touching this flag.
     *
     * <p>Read fresh each call (not cached) so it can be toggled per-run without rebuilding.
     */
    public static final String INDEPENDENT_CONJUNCT_ROUTING_PROPERTY = "opensearch.analytics.nested.independent_conjunct_routing";

    private static boolean independentConjunctRoutingEnabled() {
        return Boolean.parseBoolean(System.getProperty(INDEPENDENT_CONJUNCT_ROUTING_PROPERTY, "false"));
    }

    /**
     * Opt-in CHILD-GRAIN nested split (default off). When enabled, a fused nested predicate whose
     * keyword-equality conjunct(s) are Lucene-eligible is emitted so the keyword clause is evaluated by
     * Lucene at CHILD-doc grain and its per-element verdict is intersected with the DataFusion range/other
     * clauses AT THE SAME ELEMENT (before the ∃ roll-up) — the maximally-selective, element-exact split.
     * The keyword conjunct in the {@code NESTED_ANY_MATCH_EXPR} JSON tree is replaced by a
     * {@code {"lucene": <clauseIdx>}} node, and the paired {@code NESTED_ANY_MATCH} peer is tagged
     * child-grain (shipped as a child-scoped query, not a block-join) so the executor consumes it per element.
     *
     * <p>When OFF (default), the safe SUPERSET pruning peer is emitted instead (keyword clause still runs on
     * Lucene, but as a parent-grain superset prune AND-ed with the authoritative DataFusion predicate). Both
     * are correct; child-grain is strictly tighter pruning. Gated so the proven superset path stays the
     * default while the child-grain path is validated. Read fresh each call.
     */
    public static final String CHILD_GRAIN_SPLIT_PROPERTY = "opensearch.analytics.nested.child_grain_split";

    private static boolean childGrainSplitEnabled() {
        return Boolean.parseBoolean(System.getProperty(CHILD_GRAIN_SPLIT_PROPERTY, "false"));
    }

    /**
     * Synthetic scalar function: {@code nested_any_match(arrayCol, 'fieldName', 'op', literal) → BOOLEAN}.
     * Emitted by the filter rewrite in place of Correlate+Uncollect. The Substrait emission maps this
     * to a scalar function call "nested_any_match"; on the Rust side a UDF of the same name builds the
     * equivalent {@code array_any_match(col, s -> get_field(s, field) op value)} lambda expression and
     * evaluates it. Row count is never changed — one boolean per parent row.
     */
    public static final SqlFunction NESTED_ANY_MATCH_OP = new SqlFunction(
        "NESTED_ANY_MATCH",
        SqlKind.OTHER_FUNCTION,
        ReturnTypes.BOOLEAN_NULLABLE,
        null,
        OperandTypes.ANY,
        SqlFunctionCategory.USER_DEFINED_FUNCTION
    );

    /**
     * Synthetic scalar function {@code nested_any_match_child(arrayCol, 'field', 'EQUALS', literal, clauseIdx)}
     * — the CHILD-GRAIN sibling of {@link #NESTED_ANY_MATCH_OP} for the opt-in child-grain split. Same
     * keyword-equality meaning, but its Lucene serializer ships a CHILD-scoped query (a term on
     * {@code path.field} restricted to that path's child docs, NOT wrapped in a block-join), so the delegated
     * scorer yields CHILD docIds. The executor collects those at child-element grain and feeds the per-element
     * verdict into the paired {@code NESTED_ANY_MATCH_EXPR} residual's {@code {"lucene": clauseIdx}} node —
     * intersecting keyword (Lucene) and range/other (DataFusion) clauses AT THE SAME ELEMENT before the ∃
     * roll-up. The trailing {@code clauseIdx} (Int) pairs this peer with its JSON node.
     */
    public static final SqlFunction NESTED_ANY_MATCH_CHILD_OP = new SqlFunction(
        "NESTED_ANY_MATCH_CHILD",
        SqlKind.OTHER_FUNCTION,
        ReturnTypes.BOOLEAN_NULLABLE,
        null,
        OperandTypes.ANY,
        SqlFunctionCategory.USER_DEFINED_FUNCTION
    );

    /**
     * Synthetic scalar function: {@code nested_any_match_expr(arrayCol, '<json expr tree>') → BOOLEAN}.
     * Generalization of {@link #NESTED_ANY_MATCH_OP} for compound (AND/OR/NOT), arithmetic
     * (+,-,*,/,%), or otherwise-shaped predicates on ONE array column that {@code NESTED_ANY_MATCH}'s
     * flat (field, op, value) triple can't express — e.g. {@code subs.views > 65 and subs.views % 2 =
     * 0} (a single element must satisfy the WHOLE compound condition — matches vanilla OpenSearch's
     * native {@code nested} query + Painless script semantics, confirmed by direct comparison: vanilla
     * requires ONE element to jointly satisfy every clause, never independent per-clause existence
     * checks). The second argument is a JSON string describing the per-element predicate tree — see
     * {@link ExprTreeBuilder} for the node shapes; the Rust {@code nested_any_match_expr} UDF parses
     * and evaluates it per array element, short-circuiting on the first match. Row count never changes.
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
     * columns of an injected Correlate+Uncollect (the original, child-grain unnest path). Used when
     * the Aggregate-input guard determines a genuine grain change is required.
     */
    private static RelNode rewriteProjectViaUnnest(LogicalProject project) {
        RelNode input = project.getInput();
        int arrayCol = firstArrayColReferenced(project.getProjects(), input.getRowType());
        if (arrayCol < 0) {
            return project;
        }
        RelOptCluster cluster = project.getCluster();
        RexBuilder rexBuilder = cluster.getRexBuilder();
        UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
        if (u == null) {
            return project;
        }
        ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.correlate.getRowType());
        List<RexNode> newExprs = new ArrayList<>(project.getProjects().size());
        for (RexNode e : project.getProjects()) {
            newExprs.add(e.accept(shuttle));
        }
        return LogicalProject.create(u.correlate, List.of(), newExprs, project.getRowType().getFieldNames());
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
     * using {@code NESTED_ANY_MATCH($arrayCol, 'field', '<op>', <literal>)}. This is the "peek
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

        // Try the lambda (nested_any_match) rewrite first — it preserves parent grain.
        RexNode lambdaCondition = tryLambdaRewrite(filter.getCondition(), arrayCol, input.getRowType(), rexBuilder);
        if (lambdaCondition != null) {
            LOGGER.info("[NESTED-LAMBDA] filter rewritten to nested_any_match (no unnest, row count preserved)");
            return LogicalFilter.create(input, lambdaCondition);
        }

        // Fallback: inject Correlate+Uncollect (the old unnest path).
        LOGGER.info("[NESTED] filter lambda-rewrite not applicable, falling back to unnest path");
        int originalColCount = input.getRowType().getFieldCount();
        UnnestResult u = injectUnnest(input, arrayCol, cluster, rexBuilder);
        if (u == null) {
            return filter;
        }
        ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.correlate.getRowType());
        RexNode newCondition = filter.getCondition().accept(shuttle);
        RelNode newFilter = LogicalFilter.create(u.correlate, newCondition);

        List<RexNode> passthrough = new ArrayList<>(originalColCount);
        List<String> names = new ArrayList<>(originalColCount);
        List<RelDataTypeField> corrFields = u.correlate.getRowType().getFieldList();
        for (int i = 0; i < originalColCount; i++) {
            passthrough.add(rexBuilder.makeInputRef(corrFields.get(i).getType(), i));
            names.add(corrFields.get(i).getName());
        }
        return LogicalProject.create(newFilter, List.of(), passthrough, names);
    }

    /**
     * Attempts to rewrite the filter condition using {@code NESTED_ANY_MATCH_EXPR}. Splits the
     * TOP-LEVEL {@code AND} conjuncts (if any) into two groups:
     * <ul>
     *   <li>conjuncts that reference our array column — these are combined into ONE joint
     *       per-element expression tree (a single element must satisfy ALL of them together,
     *       matching vanilla's semantics), wrapped in one {@code NESTED_ANY_MATCH_EXPR} call</li>
     *   <li>conjuncts that don't (pure parent predicates, e.g. {@code count > 0}) — passed through
     *       unchanged and ANDed back in at the row level, since parent predicates are genuinely
     *       independent per-row and don't need per-element evaluation</li>
     * </ul>
     * A non-AND condition (a single comparison, an OR, a NOT, ...) is treated as one conjunct.
     * Returns null (triggering the Correlate+Uncollect fallback) if any array-referencing conjunct's
     * tree can't be built — e.g. it touches a DIFFERENT array column, or mixes an array-of-ours
     * reference with a parent column inside the SAME comparison (ambiguous — which row's value?).
     */
    private static RexNode tryLambdaRewrite(RexNode condition, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
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

        // Fast path: a single, standalone keyword-equality conjunct on our array (e.g.
        // `comments.author = "alice"`, alone or ANDed only with parent-only conjuncts) becomes
        // NESTED_ANY_MATCH — the flat (field, op, value) predecessor of NESTED_ANY_MATCH_EXPR.
        // Unlike NESTED_ANY_MATCH_EXPR (DataFusion-only), NESTED_ANY_MATCH is registered as a
        // dual-viable [lucene, datafusion] filter capability (see LuceneAnalyticsBackendPlugin),
        // so it can be performance-delegated to Lucene's native nested block-join query exactly
        // like a flat EQUALS predicate is today — giving nested equality the same
        // Lucene-delegation behavior as flat fields. Only fires for a SINGLE array conjunct: a
        // compound joint condition (e.g. `author='alice' AND score>10`) must stay on the generic
        // path below, since a partial match on just the equality clause would be wrong (see class
        // javadoc on joint semantics).
        if (arrayConjuncts.size() == 1) {
            RexNode directMatch = tryDirectEqualityRewrite(arrayConjuncts.get(0), arrayCol, inputRowType, rexBuilder);
            if (directMatch != null) {
                LOGGER.info("[NESTED-LAMBDA] filter rewritten to nested_any_match (flat, Lucene-delegable)");
                return combineWithParentConjuncts(directMatch, parentConjuncts, rexBuilder);
            }
        } else if (independentConjunctRoutingEnabled()) {
            // See INDEPENDENT_CONJUNCT_ROUTING_PROPERTY javadoc: deliberately unsafe, requested
            // explicitly to unblock "does each conjunct reach its appropriate backend" plumbing work.
            // Rewrite each array conjunct to its OWN leaf (independently marked/routed, exactly like
            // independent flat-column conjuncts) instead of fusing them into one joint tree.
            RexNode independentAnd = tryIndependentConjunctRewrite(arrayConjuncts, arrayCol, inputRowType, rexBuilder);
            if (independentAnd != null) {
                LOGGER.warn(
                    "[NESTED-LAMBDA] filter rewritten to INDEPENDENT per-conjunct nested leaves "
                        + "(joint per-element semantics NOT enforced — see INDEPENDENT_CONJUNCT_ROUTING_PROPERTY)"
                );
                return combineWithParentConjuncts(independentAnd, parentConjuncts, rexBuilder);
            }
        }

        List<Map<String, Object>> arrayTrees = new ArrayList<>();
        for (RexNode conjunct : arrayConjuncts) {
            Map<String, Object> tree = builder.build(conjunct);
            if (tree == null) {
                return null; // unsupported shape somewhere in this conjunct — fall back entirely
            }
            arrayTrees.add(tree);
        }

        // ── CHILD-GRAIN split (opt-in) ────────────────────────────────────────────────────────────
        // When enabled, each keyword-equality conjunct that Lucene can evaluate is REPLACED in the JSON
        // tree by a {"lucene": <clauseIdx>} node and paired with a NESTED_ANY_MATCH_CHILD peer. The
        // executor evaluates the residual (with the range/other clauses on the decoded array) and, at the
        // {"lucene"} node, consumes the Lucene peer's per-element verdict — so keyword (Lucene) and range
        // (DataFusion) intersect at the SAME element before the ∃ roll-up. childPeers[i] pairs with
        // clauseIdx i. Only fires in the compound (>1) path; single keyword-equality already uses the
        // whole-node block-join fast path (tryDirectEqualityRewrite) above.
        List<RexNode> childPeers = new ArrayList<>();
        if (childGrainSplitEnabled() && arrayConjuncts.size() > 1) {
            for (int i = 0; i < arrayConjuncts.size(); i++) {
                RexNode peer = tryDirectEqualityChildRewrite(arrayConjuncts.get(i), arrayCol, inputRowType, rexBuilder, childPeers.size());
                if (peer != null) {
                    // Route this conjunct's keyword-equality to Lucene at element grain: mark it
                    // {"lucene": clauseIdx} and KEEP the original subtree under "fallback". The UDF uses
                    // Lucene's per-element verdict when the child-grain executor supplies it, and evaluates
                    // the fallback natively otherwise (the plain UDF path, or a Tree/OR-NOT plan where the
                    // peer was demoted to native) — so NESTED_ANY_MATCH_EXPR is correct on EVERY path and
                    // Lucene is a pure accelerant, never a correctness dependency.
                    Map<String, Object> luceneNode = Map.of("lucene", childPeers.size(), "fallback", arrayTrees.get(i));
                    arrayTrees.set(i, luceneNode);
                    childPeers.add(peer);
                }
            }
        }

        Map<String, Object> combinedTree = arrayTrees.size() == 1
            ? arrayTrees.get(0)
            : Map.of("op", "AND", "args", arrayTrees);

        String json;
        try {
            json = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().writeValueAsString(combinedTree);
        } catch (Exception e) {
            LOGGER.warn("[NESTED-LAMBDA] failed to serialize expr tree, falling back to unnest", e);
            return null;
        }
        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        RexNode exprLit = rexBuilder.makeLiteral(json);
        RexNode anyMatchCall = rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_EXPR_OP,
            List.of(arrayRef, exprLit)
        );

        // ── Lucene pruning peers (the child-predicate split, done the superset-safe way non-flat
        // delegation already works) ────────────────────────────────────────────────────────────
        // The fused NESTED_ANY_MATCH_EXPR above is the AUTHORITATIVE, DataFusion-evaluated,
        // element-correlated predicate (one element must satisfy the WHOLE compound jointly) — it
        // alone is 100% correct. For each top-level-AND keyword-equality conjunct on our array we
        // ALSO emit a NESTED_ANY_MATCH call, which is dual-viable [lucene, datafusion] and so gets
        // performance-delegated to Lucene's native ToParentBlockJoinQuery. NESTED_ANY_MATCH(field=v)
        // matches "parent has SOME child with field=v", which is a SUPERSET of the fused predicate's
        // parent set (any parent satisfying `field=v AND <rest>` on a single element necessarily has
        // SOME child with field=v). AND-ing a superset with the authoritative predicate never changes
        // the result — but it lets Lucene's inverted index PRUNE which parquet rows DataFusion must
        // read. This is exactly how flat-field performance delegation works (peer bitset intersected
        // with the driving backend's native eval), extended to nested keyword children. Only pure
        // keyword equality qualifies (Lucene indexes keyword/text child leaves; numeric/range/other
        // stay on DataFusion). Emitted only in the compound (size>1) path; the single-conjunct case is
        // already handled by tryDirectEqualityRewrite above.
        // Child-grain peers (opt-in, built above): each pairs with a {"lucene": i} node in the JSON. When
        // child-grain is active we do NOT also emit superset peers — the child peers ARE the keyword
        // delegation, at element grain.
        List<RexNode> lucenerPeers = new ArrayList<>(childPeers);
        if (childPeers.isEmpty()) {
            // Superset pruning peers (default path): keyword conjunct stays in the JSON (authoritative on
            // DataFusion) and also runs on Lucene as a parent-grain superset prune.
            for (RexNode conjunct : arrayConjuncts) {
                RexNode peer = tryDirectEqualityRewrite(conjunct, arrayCol, inputRowType, rexBuilder);
                if (peer != null) {
                    lucenerPeers.add(peer);
                }
            }
        }

        RexNode nestedPredicate;
        if (lucenerPeers.isEmpty()) {
            nestedPredicate = anyMatchCall;
        } else {
            // authoritative fused expr AND (Lucene-prunable keyword peers). The peers are supersets, so
            // this is semantically identical to `anyMatchCall` alone; the AND exists purely so the
            // marking layer can performance-delegate the peers to Lucene for pruning.
            List<RexNode> operands = new ArrayList<>(lucenerPeers.size() + 1);
            operands.add(anyMatchCall);
            operands.addAll(lucenerPeers);
            nestedPredicate = rexBuilder.makeCall(
                rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
                org.apache.calcite.sql.fun.SqlStdOperatorTable.AND,
                operands
            );
            LOGGER.info(
                "[NESTED-LAMBDA] fused NESTED_ANY_MATCH_EXPR (authoritative) + {} Lucene pruning peer(s) "
                    + "(keyword-equality conjuncts delegated to Lucene block-join for pruning)",
                lucenerPeers.size()
            );
        }

        return combineWithParentConjuncts(nestedPredicate, parentConjuncts, rexBuilder);
    }

    /**
     * See {@link #INDEPENDENT_CONJUNCT_ROUTING_PROPERTY}: rewrites each array-referencing conjunct to
     * its OWN independent leaf instead of fusing them into one joint tree. A keyword-equality conjunct
     * becomes {@code NESTED_ANY_MATCH} (dual-viable, Lucene-delegable); anything else becomes its own
     * single-conjunct {@code NESTED_ANY_MATCH_EXPR} (DataFusion-only). ANDs all the resulting leaves
     * together. Returns {@code null} (triggering the generic joint-tree path) if any conjunct's tree
     * can't be built at all — same fallback contract as {@link #tryLambdaRewrite}.
     *
     * <p>Each leaf is independently viable/annotated by the marking rules downstream exactly like
     * independent flat-column conjuncts are — this is what makes each conjunct reach its own
     * appropriate backend, at the deliberate cost of joint per-element correctness (see the flag's
     * javadoc).
     */
    private static RexNode tryIndependentConjunctRewrite(
        List<RexNode> arrayConjuncts,
        int arrayCol,
        RelDataType inputRowType,
        RexBuilder rexBuilder
    ) {
        ExprTreeBuilder builder = new ExprTreeBuilder(arrayCol, inputRowType);
        List<RexNode> leaves = new ArrayList<>(arrayConjuncts.size());
        for (RexNode conjunct : arrayConjuncts) {
            RexNode directMatch = tryDirectEqualityRewrite(conjunct, arrayCol, inputRowType, rexBuilder);
            if (directMatch != null) {
                leaves.add(directMatch);
                continue;
            }
            Map<String, Object> tree = builder.build(conjunct);
            if (tree == null) {
                return null; // unsupported shape somewhere in this conjunct — fall back entirely
            }
            RexNode leaf = singleConjunctAnyMatchExpr(tree, arrayCol, inputRowType, rexBuilder);
            if (leaf == null) {
                return null; // serialization failed — fall back entirely
            }
            leaves.add(leaf);
        }
        if (leaves.size() == 1) {
            return leaves.get(0);
        }
        return rexBuilder.makeCall(rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN), org.apache.calcite.sql.fun.SqlStdOperatorTable.AND, leaves);
    }

    /** Builds a single-conjunct {@code NESTED_ANY_MATCH_EXPR(arrayCol, jsonTree)} call, or {@code null}
     *  if the tree can't be serialized. */
    private static RexNode singleConjunctAnyMatchExpr(Map<String, Object> tree, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        String json;
        try {
            json = com.fasterxml.jackson.databind.json.JsonMapper.builder().build().writeValueAsString(tree);
        } catch (Exception e) {
            LOGGER.warn("[NESTED-LAMBDA] failed to serialize independent-conjunct expr tree", e);
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
     * Fast-path check: if {@code conjunct} is exactly {@code ITEM($arrayCol,'field') = 'literal'}
     * (either operand order) with a STRING literal value, emits {@code NESTED_ANY_MATCH(arrayCol,
     * 'field', 'EQUALS', 'literal')} — the flat 4-arg predecessor of {@code NESTED_ANY_MATCH_EXPR}
     * (see {@link #NESTED_ANY_MATCH_OP}'s javadoc). This shape is registered as a dual-viable
     * [lucene, datafusion] filter capability, enabling performance-delegation to Lucene's native
     * nested block-join query, unlike the always-DataFusion-only {@code NESTED_ANY_MATCH_EXPR}.
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

        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_OP,
            List.of(arrayRef, rexBuilder.makeLiteral(fieldName), rexBuilder.makeLiteral("EQUALS"), rexBuilder.makeLiteral(value))
        );
    }

    /**
     * CHILD-GRAIN sibling of {@link #tryDirectEqualityRewrite} for the opt-in child-grain split: for a
     * keyword-equality conjunct {@code ITEM($arrayCol,'field') = 'value'}, emits
     * {@code NESTED_ANY_MATCH_CHILD(arrayRef, 'field', 'EQUALS', 'value', clauseIdx)} — the child-scoped
     * Lucene peer paired with the {@code {"lucene": clauseIdx}} node in the residual JSON. Returns
     * {@code null} for any non-(keyword-equality) shape, so the caller leaves that conjunct in the JSON tree
     * for DataFusion. Same operand parse as the parent-grain variant plus the trailing {@code clauseIdx}.
     */
    private static RexNode tryDirectEqualityChildRewrite(
        RexNode conjunct,
        int arrayCol,
        RelDataType inputRowType,
        RexBuilder rexBuilder,
        int clauseIdx
    ) {
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
            return null;
        }
        RexNode fieldNameNode = itemCall.getOperands().get(1);
        if (!(fieldNameNode instanceof RexLiteral fieldLit) || fieldLit.getTypeName() != SqlTypeName.CHAR) {
            return null;
        }
        String fieldName = fieldLit.getValueAs(String.class);
        String value = valueLit.getValueAs(String.class);
        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_CHILD_OP,
            List.of(
                arrayRef,
                rexBuilder.makeLiteral(fieldName),
                rexBuilder.makeLiteral("EQUALS"),
                rexBuilder.makeLiteral(value),
                rexBuilder.makeLiteral(
                    java.math.BigDecimal.valueOf(clauseIdx),
                    rexBuilder.getTypeFactory().createSqlType(SqlTypeName.INTEGER),
                    false
                )
            )
        );
    }

    /** True if {@code node} is exactly {@code ITEM($arrayCol, <anything>)}. */
    private static boolean isItemOnArray(RexNode node, int arrayCol) {
        if (!(node instanceof RexCall call) || !"ITEM".equals(call.getOperator().getName()) || call.getOperands().size() != 2) {
            return false;
        }
        return call.getOperands().get(0) instanceof RexInputRef ref && ref.getIndex() == arrayCol;
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

        /** Returns null if the tree can't be expressed (unsupported operator, or ITEM on the wrong array). */
        Map<String, Object> build(RexNode node) {
            // ITEM($arrayCol, 'field') -> {"field": "field"}
            if (node instanceof RexCall itemCall
                && "ITEM".equals(itemCall.getOperator().getName())
                && itemCall.getOperands().size() == 2) {
                RexNode arrayOperand = itemCall.getOperands().get(0);
                RexNode fieldNode = itemCall.getOperands().get(1);
                if (arrayOperand instanceof RexInputRef ref && fieldNode instanceof RexLiteral lit && lit.getTypeName() == SqlTypeName.CHAR) {
                    if (ref.getIndex() != arrayCol) {
                        return null; // ITEM on a DIFFERENT array — unsupported, fall back
                    }
                    return Map.of("field", lit.getValueAs(String.class));
                }
                return null;
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

        private boolean containsItemOnArray(RexNode node) {
            if (node instanceof RexCall call) {
                if ("ITEM".equals(call.getOperator().getName()) && call.getOperands().size() == 2) {
                    RexNode ref = call.getOperands().get(0);
                    if (ref instanceof RexInputRef r && r.getIndex() == arrayCol) {
                        return true;
                    }
                }
                for (RexNode op : call.getOperands()) {
                    if (containsItemOnArray(op)) return true;
                }
            }
            return false;
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

    // ---- Shared: build Correlate(input, Uncollect(array)) appending the struct fields ----------

    /** Result of injecting an unnest: the new Correlate rel + the index where unnested fields begin. */
    private record UnnestResult(LogicalCorrelate correlate, int unnestedFieldIndex, Map<String, Integer> fieldToIndex) {}

    /**
     * Injects {@code Correlate(input, Uncollect(Project($cor0.arrayCol, oneRow)))}. The correlate's
     * output is {@code [original cols..., unnested struct fields...]} — original indices preserved,
     * struct fields appended starting at {@code input.fieldCount}.
     */
    private static UnnestResult injectUnnest(RelNode input, int arrayCol, RelOptCluster cluster, RexBuilder rexBuilder) {
        RelDataType inputRowType = input.getRowType();
        RelDataTypeField arrayField = inputRowType.getFieldList().get(arrayCol);
        RelDataType elementType = arrayField.getType().getComponentType();
        if (elementType == null || !elementType.isStruct()) {
            LOGGER.warn("[NESTED] array column '{}' is not ARRAY(ROW) — skipping unnest", arrayField.getName());
            return null;
        }

        CorrelationId correlId = cluster.createCorrel();
        RexNode correlVar = rexBuilder.makeCorrel(inputRowType, correlId);
        RexNode correlArrayAccess = rexBuilder.makeFieldAccess(correlVar, arrayCol);

        RelNode oneRow = LogicalValues.createOneRow(cluster);
        RelNode rightProject = LogicalProject.create(oneRow, List.of(), List.of(correlArrayAccess), List.of(arrayField.getName()));
        RelNode uncollect = Uncollect.create(rightProject.getTraitSet(), rightProject, false, List.of());

        LogicalCorrelate correlate = LogicalCorrelate.create(
            input,
            uncollect,
            List.of(),
            correlId,
            ImmutableBitSet.of(arrayCol),
            JoinRelType.INNER
        );

        int originalColCount = inputRowType.getFieldCount();
        Map<String, Integer> fieldToIndex = new LinkedHashMap<>();
        List<RelDataTypeField> corrFields = correlate.getRowType().getFieldList();
        for (int i = originalColCount; i < corrFields.size(); i++) {
            fieldToIndex.put(corrFields.get(i).getName(), i);
        }
        LOGGER.info(
            "[NESTED] injected UNNEST on array col '{}' (idx {}); unnested fields {} at indices {}..{}",
            arrayField.getName(),
            arrayCol,
            fieldToIndex.keySet(),
            originalColCount,
            corrFields.size() - 1
        );
        return new UnnestResult(correlate, originalColCount, fieldToIndex);
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
        RexNode arrayRef = call.getOperands().get(0);
        RexNode fieldNode = call.getOperands().get(1);
        if (!(arrayRef instanceof RexInputRef ref)) {
            return -1;
        }
        if (!(fieldNode instanceof RexLiteral lit) || lit.getTypeName() != SqlTypeName.CHAR) {
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

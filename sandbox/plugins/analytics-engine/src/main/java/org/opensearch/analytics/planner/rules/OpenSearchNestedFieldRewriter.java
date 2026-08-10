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
     * Test-only: resolve a (possibly compound-dotted) nested field {@code key} against an array
     * element ROW {@code elementType} to its {@code NESTED_ANY_MATCH_EXPR} JSON node
     * ({@code {"field"}} for a leaf, or a {@code {"nested","inner"}} descent chain for a deep
     * nested-of-nested path), or {@code null} if unrepresentable. Delegates to the package-private
     * {@code ExprTreeBuilder} so the deep-descent emission can be unit-tested without a full plan.
     */
    static Map<String, Object> resolveNestedKeyForTest(String key, RelDataType elementType) {
        return ExprTreeBuilder.resolveKeyForTest(key, elementType);
    }

    /**
     * Test-only: resolve a deep path {@code key} to a descent whose LEAF is the pushed-down comparison
     * {@code {"op":op,"args":[{"field":leaf},{"lit":value}]}} — the shape {@code tryDeepComparison}
     * produces for {@code deep.path.leaf <op> value}.
     */
    static Map<String, Object> resolveDeepComparisonForTest(String key, RelDataType elementType, String op, Object value) {
        return ExprTreeBuilder.resolveDeepComparisonForTest(key, elementType, op, value);
    }

    /** Test-only: expose {@link #fuseByNestedPrefix} for verifying same-prefix conjunct fusion. */
    static Map<String, Object> fuseByNestedPrefixForTest(List<Map<String, Object>> trees) {
        return fuseByNestedPrefix(trees);
    }

    /** Test-only: expose {@link #tryDeepEqualityPrunePeer} for verifying deep prune-peer emission (Phase B). */
    static RexNode tryDeepEqualityPrunePeerForTest(RexNode conjunct, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        return tryDeepEqualityPrunePeer(conjunct, arrayCol, inputRowType, rexBuilder);
    }

    /** Test-only: expose {@link #tryDirectEqualityChildRewrite} for verifying deep child-peer emission (Phase C). */
    static RexNode tryDirectEqualityChildRewriteForTest(
        RexNode conjunct,
        int arrayCol,
        RelDataType inputRowType,
        RexBuilder rexBuilder,
        int clauseIdx
    ) {
        return tryDirectEqualityChildRewrite(conjunct, arrayCol, inputRowType, rexBuilder, clauseIdx);
    }

    /** Test-only: expose {@link #wrapLeafWithLuceneNode} for verifying the {@code {"lucene"}} node lands at the
     *  deepest inner position of a descent (Phase C). */
    static Map<String, Object> wrapLeafWithLuceneNodeForTest(Map<String, Object> tree, int clauseIdx) {
        return wrapLeafWithLuceneNode(tree, clauseIdx);
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
            if (aggregateClaimedProjects.contains(project)) {
                return (LogicalProject) super.visitChildren(project);
            }
            // If this Project references a nested leaf AND its input is a same-path Filter, we must
            // correlate the filter at ELEMENT grain — but that needs the ORIGINAL Filter (an ITEM
            // condition), NOT the parent-grain NESTED_ANY_MATCH_EXPR that visiting the child first would
            // produce. So handle it on the pre-descent tree here, before super.visitChildren rewrites the
            // Filter. `rewriteProject` returns the input Project unchanged when it's not this shape, in
            // which case we fall through to the normal bottom-up path.
            int arrayCol = firstArrayColReferenced(project.getProjects(), project.getInput().getRowType());
            if (arrayCol >= 0 && project.getInput() instanceof LogicalFilter) {
                RelNode correlated = rewriteProject(project);
                if (correlated != project) {
                    return correlated; // same-path correlation (or fan-out) claimed it; do not re-descend
                }
            }
            LogicalProject visited = (LogicalProject) super.visitChildren(project);
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
        return correlatedOrPlainUnnest(childProject, arrayCol);
    }

    /**
     * Unnest a Project that references a nested array, correlating a same-path Filter at ELEMENT grain when
     * present. If {@code project.getInput()} is a Filter whose condition references the SAME nested array as
     * the projected leaf (e.g. {@code where products.variants.color='red'} feeding {@code max(products.
     * variants.qty)} OR {@code fields ..., products.variants.qty}), the filter must apply per-ELEMENT — only
     * matching elements may reach the metric/projection. Leaving the Filter below the unnest makes it a
     * parent-grain ∃ ("has SOME red variant"), then ALL variants of matching parents explode (wrong). Fix:
     * strip the Filter, explode the metric path, thread the filter CONDITION through the SAME per-level
     * shuttles (so {@code color} resolves to the SAME exploded element column as {@code qty}), and re-emit
     * the filter ABOVE the unnest at element grain. Shared by the aggregate-input and bare-projection paths.
     *
     * <p>DIVERGENCE GATE: explosion is driven by the PROJECT exprs only. The filter is correlated ONLY if
     * its condition fully resolves to element columns within those levels (no residual ITEM-on-array). A
     * divergent path (filter on {@code products.tags.*}, metric on {@code products.variants.*}) leaves an
     * unresolved ITEM → gate fails → fall back to {@link #rewriteProjectViaUnnest} (prior behavior).
     */
    private static RelNode correlatedOrPlainUnnest(LogicalProject project, int arrayCol) {
        if (project.getInput() instanceof LogicalFilter filter) {
            RelOptCluster cluster = project.getCluster();
            RexBuilder rexBuilder = cluster.getRexBuilder();
            int filterArrayCol = firstArrayColReferenced(List.of(filter.getCondition()), filter.getInput().getRowType());
            if (filterArrayCol == arrayCol) {
                StackedUnnestResult r = stackedUnnest(
                    filter.getInput(), project.getProjects(), List.of(filter.getCondition()), cluster, rexBuilder);
                if (r.levels > 0
                    && firstArrayColReferenced(List.of(r.extraExprs.get(0)), r.input.getRowType()) < 0) {
                    LOGGER.info("[NESTED] same-path filter correlated at element grain ({} unnest level(s))", r.levels);
                    RelNode elementFilter = LogicalFilter.create(r.input, r.extraExprs.get(0));
                    return LogicalProject.create(
                        elementFilter, List.of(), r.projectExprs, project.getRowType().getFieldNames());
                }
            }
        }
        return rewriteProjectViaUnnest(project);
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
            // No ITEM-on-array leaf reference. Either no nested ref at all, or a WHOLE-ARRAY projection
            // (`fields title, comments` — a bare $comments column, kept whole, one row per parent).
            // Leave unchanged — passthrough is correct (vanilla returns the array as a single cell).
            return project;
        }
        // A nested LEAF is projected (`fields title, comments.author` → ITEM($comments,'author')). Vanilla
        // FANS OUT to one row per child element (row-per-comment), NOT the first element. Route to the
        // stacked-UNNEST path — which explodes each array level so a deep leaf (comments.replies.txt) also
        // fans out correctly — instead of the (incorrect) first-element rewrite. This matches vanilla's
        // dotted-nested-leaf projection semantics (each nested object is a separate hit). If a same-path
        // WHERE feeds this projection (`where comments.score>4 | fields title, comments.author`), the
        // filter is correlated at element grain so only matching child rows survive (shared with the
        // aggregate path) — otherwise a plain per-element fan-out.
        LOGGER.info(
            "[NESTED-PROJECT-FANOUT] plain projection references nested leaf on array col idx {} — "
                + "fanning out via UNNEST (row per child element)",
            arrayCol
        );
        return correlatedOrPlainUnnest(project, arrayCol);
    }

    /**
     * Rewrites {@code ITEM($arrayCol,'field')} references in {@code project}'s expressions to
     * columns of an injected Correlate+Uncollect (the original, child-grain unnest path). Used when
     * the Aggregate-input guard determines a genuine grain change is required.
     */
    private static RelNode rewriteProjectViaUnnest(LogicalProject project) {
        StackedUnnestResult r = stackedUnnest(
            project.getInput(), project.getProjects(), List.of(), project.getCluster(), project.getCluster().getRexBuilder());
        if (r.levels == 0) {
            return project; // nothing referenced our array
        }
        return LogicalProject.create(r.input, List.of(), r.projectExprs, project.getRowType().getFieldNames());
    }

    /** Result of a stacked unnest: the final Correlate input, the rewritten project exprs + extra exprs, and level count. */
    private record StackedUnnestResult(RelNode input, List<RexNode> projectExprs, List<RexNode> extraExprs, int levels) {}

    /**
     * STACKED (recursive) UNNEST for arbitrary nesting depth. A deep path
     * {@code regions.divisions...tags.priority} (each level a {@code nested} ARRAY(ROW)) lowers to an
     * ITEM CHAIN {@code ITEM(ITEM(...($regions,'divisions')...),'priority')}. One unnest explodes only the
     * OUTERMOST array; the residual chain over the still-array inner column would mis-emit as
     * {@code array_element(Struct,...)} and 500. So we iterate: each pass explodes the outermost still-array
     * level referenced by an ITEM in {@code projectExprs}, rewriting that ITEM link (in BOTH
     * {@code projectExprs} AND {@code extraExprs}) to the exploded column, until no ITEM-on-array remains in
     * {@code projectExprs} (leaf is a scalar column). {@code extraExprs} (e.g. a same-path filter condition)
     * ride through the SAME per-level shuttles so they resolve to the SAME exploded element columns — this is
     * what makes a filter element-correlated with the metric. Explosion depth is driven by {@code projectExprs}
     * ONLY (the metric grain), so the filter never over-explodes.
     */
    private static StackedUnnestResult stackedUnnest(
        RelNode input, List<RexNode> projectExprs, List<RexNode> extraExprs, RelOptCluster cluster, RexBuilder rexBuilder) {
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
            ItemRewriteShuttle shuttle = new ItemRewriteShuttle(arrayCol, u.unnestedFieldIndex, rexBuilder, u.correlate.getRowType());
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
            input = u.correlate;
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

        ExprTreeBuilder builder = new ExprTreeBuilder(arrayCol, inputRowType, rexBuilder);
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

        // ── CHILD-GRAIN split (opt-in), at ANY depth ────────────────────────────────────────────────
        // When enabled, each keyword-equality conjunct that Lucene can evaluate — at the outer element OR
        // any nested level down (Phase C) — has its LEAF comparison in the JSON tree REPLACED by a
        // {"lucene": <clauseIdx>} node and paired with a NESTED_ANY_MATCH_CHILD peer carrying the full deep
        // path. The executor evaluates the residual (with the range/other clauses on the decoded array) and,
        // at the {"lucene"} node, consumes the Lucene peer's per-element verdict — so keyword (Lucene) and
        // range (DataFusion) intersect at the SAME element before the ∃ roll-up. childPeers[i] pairs with
        // clauseIdx i. Only fires in the compound (>1) path; single keyword-equality already uses the
        // whole-node block-join fast path (tryDirectEqualityRewrite) above.
        //
        // CRUCIAL for deep clauses: arrayTrees.get(i) may be a {"nested":...,"inner":...} descent, so we must
        // NOT wrap the WHOLE conjunct at top level (that would strand the {"lucene"} node above the descent,
        // where the Rust json_lucene_paths would recover an EMPTY path). Instead wrapLeafWithLuceneNode
        // descends through the {"nested"} heads and replaces ONLY the innermost leaf comparison, so the
        // {"lucene"} node lands at the DEEPEST inner position, wrapped by its {"nested"} heads — exactly the
        // shape json_lucene_paths walks to recover ["divisions","teams","members"]. A single-level conjunct
        // (leaf at top, no {"nested"}) degenerates to {"lucene": idx, "fallback": <leaf>} at the top, its
        // recovered path empty — the original single-level bridge behavior, unchanged.
        List<RexNode> childPeers = new ArrayList<>();
        if (childGrainSplitEnabled() && arrayConjuncts.size() > 1) {
            for (int i = 0; i < arrayConjuncts.size(); i++) {
                int clauseIdx = childPeers.size();
                RexNode peer = tryDirectEqualityChildRewrite(arrayConjuncts.get(i), arrayCol, inputRowType, rexBuilder, clauseIdx);
                if (peer != null) {
                    // Route this conjunct's keyword-equality to Lucene at element grain: mark its LEAF
                    // {"lucene": clauseIdx} and KEEP the original leaf comparison under "fallback" (preserving
                    // any enclosing {"nested"} descent heads). The UDF uses Lucene's per-element verdict when
                    // the child-grain executor supplies it, and evaluates the fallback natively otherwise (the
                    // plain UDF path, or a Tree/OR-NOT plan where the peer was demoted to native) — so
                    // NESTED_ANY_MATCH_EXPR is correct on EVERY path and Lucene is a pure accelerant, never a
                    // correctness dependency.
                    arrayTrees.set(i, wrapLeafWithLuceneNode(arrayTrees.get(i), clauseIdx));
                    childPeers.add(peer);
                }
            }
        }

        // Combine the per-conjunct trees. A naive top-level AND[t0,t1] is WRONG when two conjuncts
        // descend into the SAME inner nested array (e.g. `tags.label='urgent' AND tags.priority=2`):
        // AND-ing two independent {"nested":"...tags",...} descents means "∃ tag urgent" AND "∃ tag
        // prio=2" — satisfiable by DIFFERENT tag elements (the Delta bug, one level deeper). Vanilla
        // correlates both on the SAME tag. So we FUSE conjuncts by their shared nested-descent prefix,
        // pushing the AND to the deepest common level: nested(...tags, AND[label=urgent, priority=2]).
        //
        // We ALWAYS fuse, even when a top-level child-grain {"lucene"} peer is present. A top-level split
        // node ({"lucene":i}, replacing a keyword conjunct on a DIRECT leaf of the array) has no {"nested"}
        // head, so fuseByNestedPrefix leaves it as a flat top-level sibling — exactly where the executor
        // resolves it at the outer element grain — while DEEP descents that share a nested prefix are still
        // correlated on the SAME innermost element. A prior version emitted a flat AND[t0,t1,...] whenever
        // any top-level split fired, which WRONGLY un-fused deep same-path conjuncts (e.g.
        // members.mname='al' AND members.age>35, satisfiable by DIFFERENT members) → false positive
        // (megadeep TRAP E). Fusing is safe because {"lucene"} nodes never carry a {"nested"} head and so
        // are never buried below a descent; and even if one were, the UDF descends with lucene=None and
        // falls back to the node's own subtree — correct, just un-accelerated.
        Map<String, Object> combinedTree = arrayTrees.size() == 1 ? arrayTrees.get(0) : fuseByNestedPrefix(arrayTrees);

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
        // Lucene peers, emitted as FLAT AND siblings of the authoritative NESTED_ANY_MATCH_EXPR:
        //   - child-grain peers (NESTED_ANY_MATCH_CHILD, opt-in): element-grain keyword intersection,
        //     each paired with a {"lucene": i} node in the JSON.
        //   - parent-grain superset PRUNE peers (NESTED_ANY_MATCH): "parent has SOME child with field=v",
        //     a SUPERSET of the fused set (AND-ing never changes results) whose Lucene block-join yields
        //     ROOT docs = a row bitset, so it drives parquet row-group pruning exactly like flat-field
        //     delegation. ALWAYS emitted (both child-split ON and OFF) so pruning is never lost.
        // Correctness of emitting BOTH grains: they are DISJOINT doc-id spaces (child docs vs root docs),
        // so the DelegatedPredicateCombiner must NOT fuse them into one Lucene BoolQuery — it detects this
        // mixed-grain case and materializes each as its OWN delegation_possible leaf (see
        // DelegatedPredicateCombiner.hasChildScopedPeer/hasParentGrainPeer). The child peer is then consumed
        // at child grain by the executor; the parent peer lands in the parent-grain performance locks and
        // prunes row-groups — independently, no cardinality-0 fusion.
        List<RexNode> lucenerPeers = new ArrayList<>(childPeers);
        for (RexNode conjunct : arrayConjuncts) {
            RexNode peer = tryDirectEqualityRewrite(conjunct, arrayCol, inputRowType, rexBuilder);
            if (peer == null) {
                // Phase B: a DEEP keyword conjunct (leaf ≥1 nested level down) gets a parent-grain prune
                // peer on its full dotted path — a superset block-join that yields ROOT docs, pruning
                // parquet row-groups at any depth (result-neutral; see tryDeepEqualityPrunePeer).
                peer = tryDeepEqualityPrunePeer(conjunct, arrayCol, inputRowType, rexBuilder);
            }
            if (peer != null) {
                lucenerPeers.add(peer);
            }
        }

        RexNode nestedPredicate;
        if (lucenerPeers.isEmpty()) {
            nestedPredicate = anyMatchCall;
        } else {
            List<RexNode> operands = new ArrayList<>(lucenerPeers.size() + 1);
            operands.add(anyMatchCall);
            operands.addAll(lucenerPeers);
            nestedPredicate = rexBuilder.makeCall(
                rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
                org.apache.calcite.sql.fun.SqlStdOperatorTable.AND,
                operands
            );
            LOGGER.info(
                "[NESTED-LAMBDA] fused NESTED_ANY_MATCH_EXPR (authoritative) + {} child-grain + "
                    + "{} parent-grain prune peer(s)",
                childPeers.size(),
                lucenerPeers.size() - childPeers.size()
            );
        }

        return combineWithParentConjuncts(nestedPredicate, parentConjuncts, rexBuilder);
    }

    /**
     * Fuses a list of per-conjunct predicate trees so that conjuncts descending into the SAME inner
     * nested array are correlated on the SAME element (AND pushed INSIDE the shared ∃), rather than
     * AND-ed as independent existentials at parent grain (which would match different elements — the
     * "Delta" bug, one nesting level down). Trees sharing the same {@code {"nested":X,...}} head are
     * grouped and their {@code inner}s recursively fused under that head; leaf/other trees and distinct
     * heads are AND-ed at the current level. Preserves first-seen order for stable output.
     *
     * <p>Example: {@code [ nested(tags,label=urgent), nested(tags,priority=2) ]}
     * → {@code nested(tags, AND[label=urgent, priority=2])} — one tag must satisfy both, matching the
     * vanilla single-{@code nested}-block (both {@code must} terms) semantics.
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> fuseByNestedPrefix(List<Map<String, Object>> trees) {
        // Partition into: nested-descent trees grouped by head field (order-preserving), and others.
        LinkedHashMap<String, List<Map<String, Object>>> byHead = new LinkedHashMap<>();
        List<Map<String, Object>> others = new ArrayList<>();
        for (Map<String, Object> t : trees) {
            Object head = t.get("nested");
            if (head instanceof String h && t.get("inner") instanceof Map) {
                byHead.computeIfAbsent(h, k -> new ArrayList<>()).add(t);
            } else {
                others.add(t);
            }
        }

        List<Map<String, Object>> combined = new ArrayList<>();
        // Emit in first-seen order: walk original list, emitting each head-group once at its first hit.
        java.util.Set<String> emitted = new java.util.HashSet<>();
        for (Map<String, Object> t : trees) {
            Object head = t.get("nested");
            if (head instanceof String h && byHead.containsKey(h)) {
                if (emitted.add(h)) {
                    List<Map<String, Object>> group = byHead.get(h);
                    if (group.size() == 1) {
                        combined.add(group.get(0));
                    } else {
                        // Recursively fuse the inners under this shared head.
                        List<Map<String, Object>> inners = new ArrayList<>(group.size());
                        for (Map<String, Object> g : group) {
                            inners.add((Map<String, Object>) g.get("inner"));
                        }
                        Map<String, Object> fused = new LinkedHashMap<>();
                        fused.put("nested", h);
                        fused.put("inner", fuseByNestedPrefix(inners));
                        combined.add(fused);
                    }
                }
            } else {
                combined.add(t); // a non-nested leaf/other tree, in place
            }
        }

        return combined.size() == 1 ? combined.get(0) : Map.of("op", "AND", "args", combined);
    }

    /**
     * Places a {@code {"lucene": clauseIdx, "fallback": <leaf>}} node at the DEEPEST inner position of a
     * (possibly {@code {"nested"}}-wrapped) conjunct tree, so the Rust {@code json_lucene_paths} recovers the
     * clause's full descent path by walking the {@code {"nested"}} heads ABOVE it. Descends through the
     * {@code {"nested":X,"inner":...}} descent chain (rebuilding each head so the input tree is not mutated),
     * and at the innermost non-descent node (the leaf comparison, e.g. {@code {"op":"=","args":[...]}})
     * replaces it with the {@code {"lucene"}} node keeping that leaf as its {@code fallback}.
     *
     * <p>For a SINGLE-level conjunct (a bare leaf comparison with no {@code {"nested"}} head) this degenerates
     * to {@code {"lucene": clauseIdx, "fallback": <leaf>}} at the top — the recovered path is empty (the
     * original single-level bridge behavior). For a DEEP conjunct
     * {@code {"nested":"divisions","inner":{"nested":"teams","inner":{"nested":"members","inner":<leaf>}}}}
     * the {@code {"lucene"}} node replaces {@code <leaf>} at the deepest position, so the recovered path is
     * {@code ["divisions","teams","members"]} — matching the peer's dotted {@code field} minus its leaf.
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> wrapLeafWithLuceneNode(Map<String, Object> tree, int clauseIdx) {
        if (tree.get("nested") instanceof String head && tree.get("inner") instanceof Map<?, ?> inner) {
            Map<String, Object> rebuilt = new LinkedHashMap<>();
            rebuilt.put("nested", head);
            rebuilt.put("inner", wrapLeafWithLuceneNode((Map<String, Object>) inner, clauseIdx));
            return rebuilt;
        }
        // Innermost non-descent node = the leaf comparison. Wrap it, keeping it as the fallback subtree.
        return Map.of("lucene", clauseIdx, "fallback", tree);
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

        // Single-level ONLY. A deep compound-dotted key (e.g. "replies.txt", where `replies` is an
        // inner `nested` array) is NOT a flat keyword leaf — the NESTED_ANY_MATCH serializer would
        // build a single-level Lucene term on `comments.replies.txt`, which is wrong (that leaf lives
        // one array level down, ∃-over-∃). Decline so the caller uses the generic recursive
        // NESTED_ANY_MATCH_EXPR tree, which emits a {"nested":"replies","inner":{"field":"txt"}} descent.
        if (fieldName.indexOf('.') >= 0) {
            return null;
        }

        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_OP,
            List.of(arrayRef, rexBuilder.makeLiteral(fieldName), rexBuilder.makeLiteral("EQUALS"), rexBuilder.makeLiteral(value))
        );
    }

    /**
     * DEEP (nested-of-nested) sibling of {@link #tryDirectEqualityRewrite}: for a keyword-equality conjunct
     * whose leaf lives one or more nested levels DOWN (e.g. {@code orgs.divisions.teams.members.mname='al'},
     * an ITEM chain rooted at {@code arrayCol}), emits a PARENT-GRAIN prune peer
     * {@code NESTED_ANY_MATCH(arrayRef, 'divisions.teams.members.mname', 'EQUALS', 'al')} carrying the FULL
     * descent-plus-leaf path as the {@code field} operand. {@code NestedAnyMatchSerializer} wraps it in a
     * {@code NestedQueryBuilder} on the deepest nested path (the leaf minus its last segment), which Lucene
     * runs as a block-join yielding ROOT docs — "root has SOME element at that deep path with leaf=value".
     *
     * <p><b>Why this is always safe.</b> "root has SOME deep-leaf=value" is a SUPERSET of any correlated
     * compound predicate on that path (a root satisfying the authoritative {@code NESTED_ANY_MATCH_EXPR}
     * necessarily has such an element), so AND-ing it as a prune peer never changes results — it only lets
     * Lucene's inverted index prune parquet row-groups (identical contract to the single-level parent peer,
     * generalized to any depth). For a STANDALONE deep keyword it is additionally the EXACT answer. Returns
     * {@code null} unless the conjunct is a keyword-equality on an ALL-NESTED ITEM chain rooted at
     * {@code arrayCol} whose leaf is a scalar (validated against the element ROW type); single-level paths
     * return {@code null} here (handled by {@link #tryDirectEqualityRewrite}).
     */
    private static RexNode tryDeepEqualityPrunePeer(RexNode conjunct, int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
        if (conjunct.getKind() != SqlKind.EQUALS || !(conjunct instanceof RexCall call) || call.getOperands().size() != 2) {
            return null;
        }
        RexNode left = call.getOperands().get(0);
        RexNode right = call.getOperands().get(1);
        RexNode arraySide;
        RexLiteral valueLit;
        if (right instanceof RexLiteral lit && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
            arraySide = left;
            valueLit = lit;
        } else if (left instanceof RexLiteral lit && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
            arraySide = right;
            valueLit = lit;
        } else {
            return null; // not a string-equality conjunct
        }

        List<String> raw = new ArrayList<>();
        RexNode root = peelItemChainForPeer(arraySide, raw);
        if (!(root instanceof RexInputRef ref) || ref.getIndex() != arrayCol || raw.isEmpty()) {
            return null; // not an ITEM chain rooted at our array column
        }
        // Normalize both lowerings (ITEM chain vs single compound-dotted ITEM) to per-level segments.
        List<String> segments = new ArrayList<>();
        for (String s : raw) {
            for (String part : s.split("\\.")) {
                if (part.isEmpty() == false) {
                    segments.add(part);
                }
            }
        }
        if (segments.size() < 2) {
            return null; // single-level — tryDirectEqualityRewrite handles it
        }
        RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
        if (isAllNestedScalarLeafPath(segments, elementType) == false) {
            return null; // some intermediate segment isn't a nested ARRAY(ROW), or the leaf isn't scalar
        }

        String value = valueLit.getValueAs(String.class);
        String dottedField = String.join(".", segments);
        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_OP,
            List.of(arrayRef, rexBuilder.makeLiteral(dottedField), rexBuilder.makeLiteral("EQUALS"), rexBuilder.makeLiteral(value))
        );
    }

    /** Static twin of {@code ExprTreeBuilder.peelItemChain} for the deep prune-peer path (left-to-right segments). */
    private static RexNode peelItemChainForPeer(RexNode node, List<String> segmentsOut) {
        if (node instanceof RexCall call
            && "ITEM".equals(call.getOperator().getName())
            && call.getOperands().size() == 2
            && call.getOperands().get(1) instanceof RexLiteral lit
            && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
            RexNode root = peelItemChainForPeer(call.getOperands().get(0), segmentsOut);
            segmentsOut.add(lit.getValueAs(String.class));
            return root;
        }
        return node;
    }

    /**
     * True iff {@code segments[0..n-2]} each name a nested {@code ARRAY(ROW)} field walked from
     * {@code elementType}, and {@code segments[n-1]} names a scalar leaf. Mirrors the descent validation in
     * {@code buildSegmentDescent} so a deep prune peer is only emitted for a genuine all-nested keyword path.
     */
    private static boolean isAllNestedScalarLeafPath(List<String> segments, RelDataType elementType) {
        RelDataType cur = elementType;
        for (int i = 0; i < segments.size(); i++) {
            if (cur == null || cur.isStruct() == false) {
                return false;
            }
            RelDataTypeField field = cur.getField(segments.get(i), /*caseSensitive*/ true, /*elideRecord*/ false);
            if (field == null) {
                return false;
            }
            RelDataType fieldType = field.getType();
            RelDataType innerElement = fieldType.getComponentType(); // non-null iff ARRAY
            boolean lastSeg = i == segments.size() - 1;
            if (lastSeg) {
                return innerElement == null && fieldType.isStruct() == false; // scalar leaf
            }
            if (innerElement == null || innerElement.isStruct() == false) {
                return false; // intermediate must be a nested ARRAY(ROW)
            }
            cur = innerElement;
        }
        return false;
    }

    /**
     * CHILD-GRAIN sibling of {@link #tryDirectEqualityRewrite} for the opt-in child-grain split, generalized
     * to ANY depth (Phase C). For a keyword-equality conjunct whose leaf is reached by an ITEM chain rooted at
     * {@code arrayCol} — single-level {@code ITEM($arrayCol,'field') = 'value'} OR a deep nested-of-nested
     * chain {@code ITEM(ITEM(...($arrayCol,'divisions')...),'mname') = 'value'} — emits
     * {@code NESTED_ANY_MATCH_CHILD(arrayRef, '<path>', 'EQUALS', 'value', clauseIdx)}. The {@code field}
     * operand carries the FULL dotted descent-plus-leaf path (e.g. {@code 'divisions.teams.members.mname'};
     * leaf-only for a single level) — the same convention {@link #tryDeepEqualityPrunePeer} uses — so the
     * child serializer can scope the Lucene term query to the DEEPEST {@code _nested_path}. It is the
     * child-scoped Lucene peer paired with the {@code {"lucene": clauseIdx}} node placed at the DEEPEST inner
     * position of the residual JSON descent (see {@link #wrapLeafWithLuceneNode}).
     *
     * <p>Uses the same peel + all-nested-scalar-leaf validation as {@link #tryDeepEqualityPrunePeer}, but
     * (a) does NOT reject a single-level path (both grains are accelerated), and (b) emits the 5-operand CHILD
     * function with the trailing {@code clauseIdx}. Returns {@code null} for any non-(keyword-equality) shape,
     * or a path that isn't an all-nested-ARRAY(ROW) chain with a scalar leaf, so the caller leaves that
     * conjunct in the JSON tree for DataFusion.
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
        RexNode arraySide;
        RexLiteral valueLit;
        if (right instanceof RexLiteral lit && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
            arraySide = left;
            valueLit = lit;
        } else if (left instanceof RexLiteral lit && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
            arraySide = right;
            valueLit = lit;
        } else {
            return null; // not a string-equality conjunct
        }

        // Peel the ITEM chain (one link for a single-level leaf, N links for a deep nested-of-nested path)
        // and normalize both lowerings (ITEM chain vs one compound-dotted ITEM) to per-level segments —
        // identical to tryDeepEqualityPrunePeer, so the child peer and the parent-grain prune peer always
        // agree on the path.
        List<String> raw = new ArrayList<>();
        RexNode root = peelItemChainForPeer(arraySide, raw);
        if (!(root instanceof RexInputRef ref) || ref.getIndex() != arrayCol || raw.isEmpty()) {
            return null; // not an ITEM chain rooted at our array column
        }
        List<String> segments = new ArrayList<>();
        for (String s : raw) {
            for (String part : s.split("\\.")) {
                if (part.isEmpty() == false) {
                    segments.add(part);
                }
            }
        }
        RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
        if (isAllNestedScalarLeafPath(segments, elementType) == false) {
            return null; // some intermediate segment isn't a nested ARRAY(ROW), or the leaf isn't scalar
        }

        String value = valueLit.getValueAs(String.class);
        // Leaf-only for single-level ("author"); full dotted descent-plus-leaf path for deep
        // ("divisions.teams.members.mname"). The child serializer splits it to scope the term to the
        // deepest _nested_path — same field convention as NESTED_ANY_MATCH's deep prune peer.
        String dottedField = String.join(".", segments);
        RexNode arrayRef = rexBuilder.makeInputRef(inputRowType.getFieldList().get(arrayCol).getType(), arrayCol);
        return rexBuilder.makeCall(
            rexBuilder.getTypeFactory().createSqlType(SqlTypeName.BOOLEAN),
            NESTED_ANY_MATCH_CHILD_OP,
            List.of(
                arrayRef,
                rexBuilder.makeLiteral(dottedField),
                rexBuilder.makeLiteral("EQUALS"),
                rexBuilder.makeLiteral(value),
                // clauseIdx as a STRING literal, not INTEGER: substrait/isthmus requires all operands at a
                // variadic position to share a type, and the preceding operands are CHAR. A trailing INTEGER
                // trips "Unable to convert call NESTED_ANY_MATCH_CHILD(list, char, char, char, i32)". Keeping
                // every literal CHAR matches the proven NESTED_ANY_MATCH (4-arg) conversion path; the Rust
                // classifier parses the string back to an index.
                rexBuilder.makeLiteral(Integer.toString(clauseIdx))
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
     *   <li>{@code {"nested":"field","inner":{...}}} — descend into an inner nested {@code ARRAY(ROW)}
     *       (nested-of-nested); the Rust UDF opens a fresh ∃-loop per array level. Emitted for a deep
     *       dotted path like {@code comments.replies.txt} where {@code replies} is itself {@code nested}</li>
     *   <li>{@code {"lit":value}} — a literal number/string/boolean</li>
     * </ul>
     * Top-level entry point is {@link #build}, which returns {@code null} if the condition contains
     * a reference to a DIFFERENT array column (unsupported — multi-array predicates fall back to
     * unnest) or an operator this builder doesn't know how to translate.
     */
    private static final class ExprTreeBuilder {
        private final int arrayCol;
        private final RelDataType inputRowType;
        private final RexBuilder rexBuilder; // for expanding SEARCH/Sarg back to OR-of-comparisons; nullable in tests

        ExprTreeBuilder(int arrayCol, RelDataType inputRowType) {
            this(arrayCol, inputRowType, null);
        }

        ExprTreeBuilder(int arrayCol, RelDataType inputRowType, RexBuilder rexBuilder) {
            this.arrayCol = arrayCol;
            this.inputRowType = inputRowType;
            this.rexBuilder = rexBuilder;
        }

        /**
         * Test-only hook: resolve a (possibly compound-dotted) field {@code key} against an array
         * element ROW type to its {@code {"field"}} / {@code {"nested","inner"}} JSON node, without
         * building a full RexNode/plan. Exercises {@link #buildSegmentDescent} directly.
         */
        static Map<String, Object> resolveKeyForTest(String key, RelDataType elementType) {
            // Accepts either a dotted compound key ("replies.txt") or a single segment; buildSegmentDescent
            // splits dotted segments internally, so passing the whole key as one segment exercises both.
            return new ExprTreeBuilder(0, null).buildSegmentDescent(java.util.List.of(key), 0, elementType);
        }

        /**
         * Test-only hook: resolve a deep path {@code key} to a descent whose LEAF is the comparison
         * {@code {"op": op, "args": [{"field": leaf}, {"lit": value}]}} — i.e. the pushed-down deep
         * comparison shape. Exercises the leafFn overload of {@link #buildSegmentDescent} that
         * {@link #tryDeepComparison} uses, without RexNode plumbing.
         */
        static Map<String, Object> resolveDeepComparisonForTest(String key, RelDataType elementType, String op, Object value) {
            return new ExprTreeBuilder(0, null).buildSegmentDescent(
                java.util.List.of(key),
                0,
                elementType,
                (leaf, leafRow) -> {
                    Map<String, Object> cmp = new LinkedHashMap<>();
                    cmp.put("op", op);
                    cmp.put("args", java.util.List.of(Map.of("field", leaf), Map.of("lit", value)));
                    return cmp;
                }
            );
        }

        /** Returns null if the tree can't be expressed (unsupported operator, or ITEM on the wrong array). */
        Map<String, Object> build(RexNode node) {
            // Calcite folds `expr = 'a' OR expr = 'b'` (same expr, multiple values) into
            // SEARCH(expr, Sarg['a','b']). We don't translate SEARCH directly; expand it back to the
            // equivalent OR-of-comparisons (or range ANDs) via the standard RexUtil.expandSearch, then
            // build the expansion — so `deep.path.leaf IN (a,b)` / `... = a OR ... = b` flow through the
            // normal deep-comparison path. Only when we have a RexBuilder (production path; null in unit
            // tests that construct nodes directly).
            if (node.getKind() == SqlKind.SEARCH && rexBuilder != null) {
                RexNode expanded = org.apache.calcite.rex.RexUtil.expandSearch(rexBuilder, null, node);
                if (expanded.getKind() != SqlKind.SEARCH) { // guard against no-op (unexpandable Sarg)
                    return build(expanded);
                }
                return null;
            }
            // A reference into our nested array column. A deep path `comments.replies.txt` (where both
            // `comments` and `replies` are `nested`) lowers — via the sql-plugin QualifiedNameResolver —
            // to a CHAIN of ITEM calls, one per path segment:
            //     ITEM(ITEM($comments,'replies'),'txt')
            // (rooted at a RexInputRef of our array column). A single-level `comments.author` is the
            // one-link chain ITEM($comments,'author'). We PEEL the chain outside-in into an ordered
            // segment list [replies, txt], then walk the element ROW type: each segment that is itself
            // an ARRAY(ROW) (an inner `nested`) becomes a {"nested": seg, "inner": ...} descent node the
            // Rust UDF recurses into (opening a fresh ∃-loop per array level); the final leaf segment
            // becomes {"field": leaf}. (A single ITEM carrying a compound-dotted key like 'replies.txt'
            // is ALSO handled: buildSegmentDescent splits any dotted segment, so both lowering shapes
            // resolve identically.)
            if (node instanceof RexCall itemCall
                && "ITEM".equals(itemCall.getOperator().getName())
                && itemCall.getOperands().size() == 2) {
                List<String> segments = new ArrayList<>();
                RexNode root = peelItemChain(itemCall, segments);
                if (root instanceof RexInputRef ref && ref.getIndex() == arrayCol && !segments.isEmpty()) {
                    // Element ROW type of our array column (ARRAY(ROW(...))).
                    RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
                    return buildSegmentDescent(segments, 0, elementType);
                }
                return null; // ITEM on a DIFFERENT array / non-literal key / not rooted at our col — fall back
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
                // LIKE / ILIKE: Calcite lowers `col LIKE 'p'` to LIKE/ILIKE(value, pattern, escape) —
                // a 3-operand call. We emit {"op":"LIKE"|"ILIKE","args":[<value-tree>, <pattern-lit>]},
                // DROPPING the escape operand (this path's patterns don't use a custom escape). Deep
                // paths hoist the ∃-descent out (∃ element: element.leaf LIKE p), same as comparisons.
                String likeOp = likeOpSymbol(call);
                if (likeOp != null && call.getOperands().size() >= 2) {
                    Map<String, Object> deep = tryDeepLikeOrComparison(call, likeOp);
                    if (deep != null) {
                        return deep;
                    }
                    Map<String, Object> valTree = build(call.getOperands().get(0));
                    Map<String, Object> patTree = build(call.getOperands().get(1));
                    if (valTree == null || patTree == null || containsNestedDescent(valTree) || containsNestedDescent(patTree)) {
                        return null;
                    }
                    Map<String, Object> r = new LinkedHashMap<>();
                    r.put("op", likeOp);
                    r.put("args", List.of(valTree, patTree));
                    return r;
                }
                String opSymbol = opSymbolFor(call);
                if (opSymbol == null) {
                    // Unknown operator. If it references our array at all, we can't safely pass it
                    // through as a pure-parent predicate (it's ambiguous), so fail closed.
                    return containsItemOnArray(call) ? null : passthroughAsLiteralRef(call);
                }
                // Deep comparison: `deep.path.leaf OP value` where the path crosses an inner nested
                // array. The ∃-descent must wrap the comparison (∃ element: element.leaf OP value), NOT
                // sit as a value operand of it. Hoist it out and push the comparison to the leaf level.
                if (isComparisonOp(opSymbol)) {
                    Map<String, Object> deep = tryDeepComparison(call, opSymbol);
                    if (deep != null) {
                        return deep;
                    }
                }
                // A {"nested"} ∃-descent is a BOOLEAN sub-expression. It is VALID as a direct child of a
                // boolean connective (AND/OR/NOT) — e.g. `deep.a=x OR deep.b=y` → OR(∃-descent, ∃-descent)
                // — but must NEVER be an operand of a comparison/arithmetic op (the UDF's value path
                // rejects it; the descent+comparison must instead be produced by tryDeepComparison, which
                // pushes the op to the leaf). So we allow nested-descent children only under AND/OR/NOT.
                boolean booleanConnective = "AND".equals(opSymbol) || "OR".equals(opSymbol) || "NOT".equals(opSymbol);
                List<Object> args = new ArrayList<>(call.getOperands().size());
                for (RexNode operand : call.getOperands()) {
                    Map<String, Object> argTree = build(operand);
                    if (argTree == null) {
                        return null;
                    }
                    // Fail closed if a ∃-descent lands in a value position (under a comparison/arithmetic
                    // op). Under a boolean connective it is legitimate, so permit it there.
                    if (!booleanConnective && containsNestedDescent(argTree)) {
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
         * Peels an {@code ITEM(ITEM(...($col,'a'),'b'),'c')} chain OUTSIDE-IN into the ordered field
         * segment list {@code [a, b, c]}, returning the innermost non-ITEM operand (the array root,
         * normally a {@code RexInputRef}). A non-ITEM node returns itself with no segments appended; a
         * malformed link (non-2-arg ITEM, or non-CHAR/VARCHAR key) stops peeling and returns that node
         * so the caller's root check fails and the whole predicate falls back.
         */
        private static RexNode peelItemChain(RexNode node, List<String> segmentsOut) {
            if (node instanceof RexCall call
                && "ITEM".equals(call.getOperator().getName())
                && call.getOperands().size() == 2
                && call.getOperands().get(1) instanceof RexLiteral lit
                && (lit.getTypeName() == SqlTypeName.CHAR || lit.getTypeName() == SqlTypeName.VARCHAR)) {
                RexNode root = peelItemChain(call.getOperands().get(0), segmentsOut);
                segmentsOut.add(lit.getValueAs(String.class)); // outer-most appended last → left-to-right order
                return root;
            }
            return node;
        }

        /**
         * Walks {@code segments[from..]} against {@code elementType} (the current array element ROW),
         * emitting a leaf {@code {"field": name}} or a chain of {@code {"nested": seg, "inner": ...}}
         * descent nodes for a deep nested-of-nested path. A segment whose field type is itself an
         * {@code ARRAY(ROW)} (an inner {@code nested}) is a descent boundary; remaining segments resolve
         * against the INNER element ROW. The final segment must be a scalar leaf. Returns {@code null}
         * (→ caller falls back) if any segment is missing, or the shape is unrepresentable. Any segment
         * that itself carries dots (the compound-key lowering shape) is split here too, so both the
         * ITEM-chain and single-compound-ITEM lowerings resolve identically.
         *
         * <p>Examples (element ROW = {@code ROW(author, replies: ARRAY(ROW(txt, likes)))}):
         * <ul>
         *   <li>{@code [author]}      → {@code {"field":"author"}}</li>
         *   <li>{@code [replies, txt]} or {@code [replies.txt]} → {@code {"nested":"replies","inner":{"field":"txt"}}}</li>
         * </ul>
         */
        private Map<String, Object> buildSegmentDescent(List<String> segments, int from, RelDataType elementType) {
            // Default leaf: a bare field read {"field": leaf}. Deep comparisons use the leafFn overload
            // to instead place the pushed-down comparison at the leaf (inside all the ∃ loops).
            return buildSegmentDescent(segments, from, elementType, (leaf, leafRow) -> Map.of("field", leaf));
        }

        /**
         * Walks {@code segments[from..]} against {@code elementType}, emitting {@code {"nested",...}}
         * descent nodes for each intermediate array level and, at the final (leaf) segment, whatever
         * {@code leafFn(leafField, leafElementRow)} returns. This is the single walk shared by a bare
         * field reference (leaf → {@code {"field": leaf}}) and a deep comparison (leaf → the pushed-down
         * {@code {"op":..,"args":[..]}}). Returns {@code null} if any segment is missing / the shape is
         * unrepresentable / leafFn declines (returns null).
         */
        private Map<String, Object> buildSegmentDescent(
            List<String> segments,
            int from,
            RelDataType elementType,
            java.util.function.BiFunction<String, RelDataType, Map<String, Object>> leafFn
        ) {
            if (elementType == null || !elementType.isStruct() || from >= segments.size()) {
                return null;
            }
            String seg = segments.get(from);
            // A segment may itself be compound-dotted (single-ITEM lowering): split and recurse on the
            // remainder within the SAME element ROW before consuming the next chain link.
            int dot = seg.indexOf('.');
            if (dot >= 0) {
                List<String> expanded = new ArrayList<>();
                expanded.add(seg.substring(0, dot));
                expanded.add(seg.substring(dot + 1));
                for (int i = from + 1; i < segments.size(); i++) {
                    expanded.add(segments.get(i));
                }
                return buildSegmentDescent(expanded, 0, elementType, leafFn);
            }

            RelDataTypeField field = elementType.getField(seg, /*caseSensitive*/ true, /*elideRecord*/ false);
            if (field == null) {
                return null; // segment not in this ROW — unrepresentable, fall back
            }
            RelDataType fieldType = field.getType();
            RelDataType innerElement = fieldType.getComponentType(); // non-null iff ARRAY
            boolean lastSeg = from == segments.size() - 1;

            if (lastSeg) {
                // Final segment must be a scalar leaf, NOT an unterminated array/struct: comparing a
                // whole nested array/struct (e.g. `comments.replies = X`) is not a per-element scalar
                // comparison and must fall back rather than silently mis-evaluate.
                if (innerElement != null || fieldType.isStruct()) {
                    return null;
                }
                return leafFn.apply(seg, elementType);
            }

            // More segments remain — this one MUST be an inner nested ARRAY(ROW) to descend into.
            if (innerElement == null || !innerElement.isStruct()) {
                return null; // dotted into a non-nested (or plain-struct) field — unsupported here
            }
            Map<String, Object> inner = buildSegmentDescent(segments, from + 1, innerElement, leafFn);
            if (inner == null) {
                return null;
            }
            Map<String, Object> descent = new LinkedHashMap<>();
            descent.put("nested", seg);
            descent.put("inner", inner);
            return descent;
        }

        /**
         * If {@code call} is a comparison ({@code = != &lt; &lt;= &gt; &gt;=}) with exactly one operand a
         * reference into our nested array (an ITEM chain rooted at {@code arrayCol}) and the other a plain
         * value (literal / parent expr — anything build() can represent that is NOT itself on our array),
         * HOIST the ∃-descent OUTSIDE the comparison: descend to the leaf's element level and push the
         * comparison down to that level, so it reads
         * {@code nested(a, nested(b, ... {"op":OP,"args":[{"field":leaf}, other]} ...))}. This is the
         * correct existential semantics — {@code ∃ element (element.leaf OP value)} — and keeps the
         * {@code {"nested"}} node in BOOLEAN position (never as a value operand, which the UDF rejects).
         * Returns {@code null} to defer to the generic path (single-level, both-sides-array, or neither).
         */
        private Map<String, Object> tryDeepComparison(RexCall call, String opSymbol) {
            if (call.getOperands().size() != 2) {
                return null;
            }
            RexNode left = call.getOperands().get(0);
            RexNode right = call.getOperands().get(1);

            List<String> leftSeg = new ArrayList<>();
            boolean leftOnArray = peelItemChain(left, leftSeg) instanceof RexInputRef lr && lr.getIndex() == arrayCol && !leftSeg.isEmpty();
            List<String> rightSeg = new ArrayList<>();
            boolean rightOnArray = peelItemChain(right, rightSeg) instanceof RexInputRef rr && rr.getIndex() == arrayCol && !rightSeg.isEmpty();

            // Exactly one side must be a path on our array; the other must NOT be (both-array same-level
            // correlation and cross-array comparisons are out of scope → generic/fail-closed).
            final List<String> pathSeg;
            final RexNode otherNode;
            final boolean pathOnLeft;
            if (leftOnArray && !rightOnArray) {
                pathSeg = leftSeg;
                otherNode = right;
                pathOnLeft = true;
            } else if (rightOnArray && !leftOnArray) {
                pathSeg = rightSeg;
                otherNode = left;
                pathOnLeft = false;
            } else {
                return null;
            }

            // The other side must be representable and must NOT contain a reference to our array.
            Map<String, Object> otherTree = build(otherNode);
            if (otherTree == null || containsNestedDescent(otherTree)) {
                return null;
            }

            RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
            return buildSegmentDescent(pathSeg, 0, elementType, (leaf, leafRow) -> {
                Map<String, Object> fieldNode = Map.of("field", leaf);
                Map<String, Object> cmp = new LinkedHashMap<>();
                cmp.put("op", opSymbol);
                cmp.put("args", pathOnLeft ? List.of(fieldNode, otherTree) : List.of(otherTree, fieldNode));
                return cmp;
            });
        }

        /** True if {@code tree} is (or contains at any depth) a {@code {"nested"}} descent node. */
        private static boolean containsNestedDescent(Object tree) {
            if (tree instanceof Map<?, ?> m) {
                if (m.containsKey("nested")) {
                    return true;
                }
                for (Object v : m.values()) {
                    if (containsNestedDescent(v)) {
                        return true;
                    }
                }
            } else if (tree instanceof List<?> l) {
                for (Object v : l) {
                    if (containsNestedDescent(v)) {
                        return true;
                    }
                }
            }
            return false;
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
        private static boolean isComparisonOp(String op) {
            return switch (op) {
                case "=", "!=", "<", "<=", ">", ">=" -> true;
                default -> false;
            };
        }

        /** Returns "LIKE"/"ILIKE" if the call is a SQL (case-insensitive) LIKE, else null. */
        private static String likeOpSymbol(RexCall call) {
            SqlKind kind = call.getKind();
            if (kind == SqlKind.LIKE) {
                // Calcite folds PPL `like` to ILIKE (case-insensitive) by name; distinguish by operator name.
                String name = call.getOperator().getName().toUpperCase(java.util.Locale.ROOT);
                return name.contains("ILIKE") ? "ILIKE" : "LIKE";
            }
            return null;
        }

        /**
         * Deep-path hoist for LIKE (or any 2-effective-arg leaf op): if operand 0 is a nested-array path
         * (ITEM chain rooted at arrayCol) and operand 1 is a plain pattern literal, descend to the leaf
         * and push {@code {"op":likeOp,"args":[{"field":leaf},<pattern>]}} inside the ∃ loops. Mirrors
         * {@link #tryDeepComparison}. Returns null to defer to the flat 2-arg emission.
         */
        private Map<String, Object> tryDeepLikeOrComparison(RexCall call, String likeOp) {
            List<String> pathSeg = new ArrayList<>();
            boolean leftOnArray = peelItemChain(call.getOperands().get(0), pathSeg) instanceof RexInputRef r
                && r.getIndex() == arrayCol && !pathSeg.isEmpty();
            // require a MULTI-level path (deep); single-level flat LIKE goes through the plain emission
            // (which yields {"op":LIKE,"args":[{"field":..},..]} — already element-correct at one level).
            if (!leftOnArray || pathSeg.size() < 2) {
                return null;
            }
            Map<String, Object> patTree = build(call.getOperands().get(1));
            if (patTree == null || containsNestedDescent(patTree)) {
                return null;
            }
            RelDataType elementType = inputRowType.getFieldList().get(arrayCol).getType().getComponentType();
            return buildSegmentDescent(pathSeg, 0, elementType, (leaf, leafRow) -> {
                Map<String, Object> cmp = new LinkedHashMap<>();
                cmp.put("op", likeOp);
                cmp.put("args", List.of(Map.of("field", leaf), patTree));
                return cmp;
            });
        }

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

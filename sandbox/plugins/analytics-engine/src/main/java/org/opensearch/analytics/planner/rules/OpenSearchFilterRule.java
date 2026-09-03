/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.planner.CapabilityRegistry;
import org.opensearch.analytics.planner.PlannerContext;
import org.opensearch.analytics.planner.RelNodeUtils;
import org.opensearch.analytics.planner.rel.AnnotatedPredicate;
import org.opensearch.analytics.planner.rel.OpenSearchFilter;
import org.opensearch.analytics.planner.rel.OpenSearchNestedScope;
import org.opensearch.analytics.planner.rel.OpenSearchRelNode;
import org.opensearch.analytics.settings.DelegationBlockList;
import org.opensearch.analytics.spi.DelegatedPredicateSerializer;
import org.opensearch.analytics.spi.DelegationType;
import org.opensearch.analytics.spi.FieldReferences;
import org.opensearch.analytics.spi.FieldStorageInfo;
import org.opensearch.analytics.spi.FieldType;
import org.opensearch.analytics.spi.ScalarFunction;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Converts {@link Filter} → {@link OpenSearchFilter}.
 *
 * <p>Annotates each leaf predicate with viable backends by checking field storage
 * in the child's {@link FieldStorageInfo} and operator support in backend capabilities.
 * Wraps each leaf in an {@link AnnotatedPredicate} RexNode. Computes
 * operator-level viable backends from per-predicate annotations and
 * delegation capabilities.
 *
 * <p>A separate pass after CBO reads these annotations to generate alternative
 * StagePlans with different delegation strategies.
 *
 * @opensearch.internal
 */
public class OpenSearchFilterRule extends RelOptRule {

    private static final Logger LOGGER = LogManager.getLogger(OpenSearchFilterRule.class);

    private final PlannerContext context;

    public OpenSearchFilterRule(PlannerContext context) {
        super(operand(Filter.class, operand(RelNode.class, any())), "OpenSearchFilterRule");
        this.context = context;
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        Filter filter = call.rel(0);
        RelNode child = call.rel(1);

        if (filter instanceof OpenSearchFilter) {
            return;
        }

        if (!(child instanceof OpenSearchRelNode openSearchInput)) {
            throw new IllegalStateException(
                "Filter rule encountered unmarked child ["
                    + child.getClass().getSimpleName()
                    + "]. Ensure all child operators are marked before filter."
            );
        }

        List<String> childViableBackends = openSearchInput.getViableBackends();
        List<FieldStorageInfo> childFieldStorage = openSearchInput.getOutputFieldStorage();

        // Annotate every leaf predicate with viable backends.
        RexNode annotatedCondition = annotateCondition(filter.getCondition(), childFieldStorage, childViableBackends, child);

        // Compute operator-level viable backends: must be viable for child AND handle predicates
        List<String> viableBackends = computeFilterViableBackends(annotatedCondition, childViableBackends);

        if (viableBackends.isEmpty()) {
            throw new IllegalStateException(
                "No backend can execute filter: no viable backend among "
                    + childViableBackends
                    + " can evaluate all predicates and no delegation path exists"
            );
        }

        LOGGER.debug("Filter viable backends: {} (child viable: {})", viableBackends, childViableBackends);
        LOGGER.info(
            "[TRACE-STEP] mark(OpenSearchFilterRule): operator-level viableBackends={} (childViableBackends={}), annotatedCondition=`{}`",
            viableBackends,
            childViableBackends,
            annotatedCondition
        );

        call.transformTo(
            new OpenSearchFilter(
                filter.getCluster(),
                child.getTraitSet(),
                RelNodeUtils.unwrapHep(filter.getInput()),
                annotatedCondition,
                viableBackends
            )
        );
    }

    // ---- Predicate annotation ----

    /**
     * Recursively walks the condition tree. Boolean connectives (AND, OR, NOT) are
     * preserved — we recurse into their children. Leaf predicates are wrapped in
     * {@link AnnotatedPredicate} with viable backends resolved from child's field storage.
     */
    private RexNode annotateCondition(
        RexNode condition,
        List<FieldStorageInfo> fieldStorageInfos,
        List<String> childViableBackends,
        RelNode child
    ) {
        // Whether this is a NESTED query decides delegation policy for the whole condition, not just
        // for its nested leaves — see the exclusion in resolveViableBackends. Computed once here,
        // before annotation.
        //
        // Detected from the PLAN, not from the predicate, because the nested rewriter has two output
        // shapes and only one of them leaves a marker in the predicate:
        // * lambda shape — the leaf becomes NESTED_ANY_MATCH_EXPR(arrayCol, jsonTree);
        // * unnest shape — the leaf becomes an ordinary comparison on a post-unnest column, e.g.
        // `=($16,'retry')` / `=(ITEM($15,'retry_count'),'3')`, with no marker at all, and those
        // columns are derivedColumn placeholders so FieldType.NESTED is gone too.
        // An OpenSearchNestedScope is introduced under the filter in BOTH shapes, so the plan is the
        // only representation-independent signal. The predicate check is kept OR'd in as cheap
        // insurance for a nested leaf arriving without a scope operator above the scan.
        boolean filterHasNested = subtreeHasNestedScope(child) || containsNestedPredicate(condition);
        return annotateCondition(condition, fieldStorageInfos, childViableBackends, false, filterHasNested);
    }

    /** Whether {@code n}'s subtree contains an {@link OpenSearchNestedScope} — i.e. this plan reads a nested path. */
    private static boolean subtreeHasNestedScope(RelNode n) {
        RelNode u = RelNodeUtils.unwrapHep(n);
        if (u instanceof OpenSearchNestedScope) {
            return true;
        }
        for (RelNode input : u.getInputs()) {
            if (subtreeHasNestedScope(input)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param underOrNot      true when any ancestor of this node is an OR/NOT.
     * @param filterHasNested true when this is a nested query — an {@link OpenSearchNestedScope} under
     *                        the filter, or a {@code NESTED_ANY_MATCH_EXPR} in the condition. Both are
     *                        threaded so a leaf's viability can depend on its boolean position and on
     *                        whether the query is nested — see the exclusion in
     *                        {@link #resolveViableBackends}.
     */
    private RexNode annotateCondition(
        RexNode condition,
        List<FieldStorageInfo> fieldStorageInfos,
        List<String> childViableBackends,
        boolean underOrNot,
        boolean filterHasNested
    ) {
        if (!(condition instanceof RexCall rexCall)) {
            return condition;
        }
        if (rexCall.getKind() == SqlKind.AND || rexCall.getKind() == SqlKind.OR || rexCall.getKind() == SqlKind.NOT) {
            // Monotonic: once under an OR/NOT every descendant is too, at any depth of nested AND.
            boolean childUnderOrNot = underOrNot || rexCall.getKind() == SqlKind.OR || rexCall.getKind() == SqlKind.NOT;
            List<RexNode> annotatedOperands = new ArrayList<>();
            for (RexNode operand : rexCall.getOperands()) {
                annotatedOperands.add(annotateCondition(operand, fieldStorageInfos, childViableBackends, childUnderOrNot, filterHasNested));
            }
            return rexCall.clone(rexCall.getType(), annotatedOperands);
        }
        List<String> viableBackends = resolveViableBackends(rexCall, fieldStorageInfos, childViableBackends, underOrNot, filterHasNested);
        // TODO: viableBackends here is computed from each backend's declared FilterCapability
        // (see resolveViableBackends below). Today a backend can advertise a function as
        // filter-capable without actually shipping a DelegatedPredicateSerializer for it; the
        // mismatch only surfaces when FragmentConversion tries to delegate (correctness) or
        // wrap as performance-delegation. CapabilityRegistry should validate at startup that
        // every declared FilterCapability has a matching serializer registered, and reject
        // the plugin otherwise — fail-fast at boot rather than at first dual-viable query.
        // Needs revisiting.
        int leafAnnotationId = context.nextAnnotationId();
        LOGGER.info(
            "[TRACE-STEP] mark(OpenSearchFilterRule): leaf predicate id={} `{}` -> viableBackends={}",
            leafAnnotationId,
            rexCall,
            viableBackends
        );
        return new AnnotatedPredicate(rexCall.getType(), rexCall, viableBackends, leafAnnotationId);
    }

    /**
     * Determines which backends can evaluate this leaf predicate.
     * Extracts all field references, looks up their {@link FieldStorageInfo} from the child,
     * checks backend format support, operator capability, and operator+fieldType support.
     * Intersects across all referenced fields.
     */
    private List<String> resolveViableBackends(
        RexCall predicate,
        List<FieldStorageInfo> fieldStorageInfos,
        List<String> childViableBackends
    ) {
        return resolveViableBackends(predicate, fieldStorageInfos, childViableBackends, false, false);
    }

    /**
     * Whether {@code node}'s subtree mentions a {@code NESTED_ANY_MATCH_EXPR} call — i.e. whether this
     * is a "nested query". Descends through {@link AnnotatedPredicate} wrappers, which extend
     * {@link RexCall}, so the annotation must be unwrapped before its operator is inspected.
     */
    private static boolean containsNestedPredicate(RexNode node) {
        if (node instanceof AnnotatedPredicate ap) {
            return containsNestedPredicate(ap.unwrap());
        }
        if (node instanceof RexCall call) {
            if (ScalarFunction.fromSqlOperatorWithFallback(call.getOperator()) == ScalarFunction.NESTED_ANY_MATCH_EXPR) {
                return true;
            }
            for (RexNode operand : call.getOperands()) {
                if (containsNestedPredicate(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    private List<String> resolveViableBackends(
        RexCall predicate,
        List<FieldStorageInfo> fieldStorageInfos,
        List<String> childViableBackends,
        boolean underOrNot,
        boolean filterHasNested
    ) {
        PredicateContents contents = new PredicateContents(new HashSet<>(), new ArrayList<>());
        for (RexNode operand : predicate.getOperands()) {
            collect(operand, contents);
        }
        Set<Integer> fieldIndices = contents.fieldIndices();

        CapabilityRegistry registry = context.getCapabilityRegistry();

        ScalarFunction function = ScalarFunction.fromSqlOperatorWithFallback(predicate.getOperator());
        if (function == null) {
            throw new IllegalStateException(
                "Unrecognized filter operator [" + predicate.getOperator().getName() + " / " + predicate.getKind() + "]"
            );
        }

        if (fieldIndices.isEmpty()) {
            // Multi-field full-text functions (multi_match, query_string, simple_query_string)
            // encode field names as string literals in nested MAPs rather than RexInputRef.
            // Extract the literal field names and resolve viability per-field using the actual
            // FieldStorageInfo lookup — same code path as RexInputRef-based fields. This ensures
            // that, e.g., query_string(['severityNumber'], ...) on an INTEGER field doesn't get
            // routed to a backend that only declared (QUERY_STRING, TEXT) capability.
            if (function.getCategory() == ScalarFunction.Category.FULL_TEXT) {
                if (TextRelevanceFieldValidator.usesLiteralFieldEncoding(function)) {
                    // A backend that declares full-text capability for these multi-field functions
                    // MUST register a DelegatedPredicateSerializer whose referencedFields() surfaces
                    // the fields named inside the query string (not just the `fields` MAP). A missing
                    // serializer (or one that does not implement referencedFields) is a wiring error,
                    // not a query error — fail explicitly rather than under-validating.
                    DelegatedPredicateSerializer serializer = registry.predicateSerializer(function);
                    FieldReferences refs = serializer == null ? null : serializer.referencedFields(predicate, fieldStorageInfos);
                    if (refs == null) {
                        throw new IllegalStateException(
                            "No field-reference extraction available for full-text function ["
                                + predicate.getOperator().getName()
                                + "]. A backend declaring this function's filter capability must provide a"
                                + " DelegatedPredicateSerializer that implements referencedFields()."
                        );
                    }
                    List<String> literalFieldNames = refs.literalFields();
                    boolean lenient = refs.lenient();
                    if (literalFieldNames.isEmpty()) {
                        // No explicit literal fields to type-check: only patterns and/or default-field
                        // fan-out, which OpenSearch resolves best-effort at execution. Fall back to the
                        // TEXT-type assumption and let the full-text-capable backend handle it.
                        return new ArrayList<>(registry.filterBackendsAnyFormat(function, FieldType.TEXT));
                    }
                    // Eagerly reject text-relevance functions on non-text/keyword fields, unless the
                    // caller explicitly set lenient=true
                    if (lenient == false) {
                        TextRelevanceFieldValidator.rejectNonTextFieldsForTextFunction(
                            predicate.getOperator().getName(),
                            literalFieldNames,
                            fieldStorageInfos
                        );
                    }
                    Set<String> viableSet = new HashSet<>(registry.filterCapableBackends());
                    for (String fieldName : literalFieldNames) {
                        FieldStorageInfo storageInfo = null;
                        for (FieldStorageInfo info : fieldStorageInfos) {
                            if (fieldName.equals(info.getFieldName())) {
                                storageInfo = info;
                                break;
                            }
                        }
                        if (storageInfo == null) {
                            // An explicitly-named literal field absent from the scan's schema is an
                            // unknown field. (Wildcard/regex field tokens never reach here: they are
                            // classified as patterns and handled by the empty-literals branch above.)
                            throw new IllegalArgumentException("Field [" + fieldName + "] not found.");
                        }
                        viableSet.retainAll(registry.filterBackendsForField(function, storageInfo));
                    }
                    if (viableSet.isEmpty()) {
                        // No viable alternatives is a client error (IllegalArgumentException -> HTTP 400) so the actionable
                        // message is surfaced to the caller instead of being redacted as a 500.
                        throw new IllegalArgumentException(
                            "No backend can evaluate filter predicate ["
                                + predicate.getKind()
                                + "] on literal-named fields "
                                + literalFieldNames
                        );
                    }
                    return new ArrayList<>(viableSet);
                }
                // FULL_TEXT but not a literal-field-encoding function (e.g. QUERY no-field variant,
                // MATCHALL): no explicit field list to validate — fall back to TEXT type assumption.
                return new ArrayList<>(registry.filterBackendsAnyFormat(function, FieldType.TEXT));
            }
            // No field reference (non-deterministic, or an unfoldable constant like
            // mktime('...') > N): let any child-viable backend evaluate it.
            return new ArrayList<>(childViableBackends);
        }

        Set<String> viableSet = new HashSet<>(registry.filterCapableBackends());

        for (int fieldIndex : fieldIndices) {
            FieldStorageInfo storageInfo = FieldStorageInfo.resolve(fieldStorageInfos, fieldIndex);

            Set<String> fieldViable;
            if (storageInfo.isDerived()) {
                // Derived columns (post-Aggregate, post-Join, post-Union, post-Project) are
                // computed in memory by the producer. The filter can only run on a backend
                // the producer is also viable for (its child's viableBackends), and further
                // only on backends that support this function on the field's logical type —
                // delegation isn't applicable because there's no physical storage to delegate
                // a scan against. Surfaced by testHavingFilterAfterJoin_multiShard etc., where
                // a HAVING clause filters on a stats-derived column.
                fieldViable = new HashSet<>(childViableBackends);
                fieldViable.retainAll(registry.filterBackendsAnyFormat(function, storageInfo.getFieldType()));
            } else {
                // Format-aware: backends that can access this field's storage (doc values + index).
                // A backend is viable only if it has the field in its own storage formats — ensuring
                // delegation targets are also field-storage-aware (e.g. Lucene is viable for a keyword
                // field only when the field has indexFormats=[lucene] set in the mapping).
                // TODO: for FULL_TEXT operators, extract required params from RexCall
                fieldViable = new HashSet<>(registry.filterBackendsForField(function, storageInfo));
            }

            if (function == ScalarFunction.NESTED_ANY_MATCH_EXPR && LOGGER.isDebugEnabled()) {
                // Nested-predicate viability is easy to get wrong and hard to see: a missing/derived
                // FieldStorageInfo silently drops Lucene before canServe is consulted. Log the inputs
                // to that decision so a non-delegating nested query can be diagnosed from one line.
                LOGGER.debug(
                    "[NAM-VIAB] field=[{}] mappingType={} fieldType={} derived={} docValueFormats={} "
                        + "indexFormats={} childViable={} filterCapable={} fieldViable={} viableSoFar={}",
                    storageInfo.getFieldName(),
                    storageInfo.getMappingType(),
                    storageInfo.getFieldType(),
                    storageInfo.isDerived(),
                    storageInfo.getDocValueFormats(),
                    storageInfo.getIndexFormats(),
                    childViableBackends,
                    registry.filterCapableBackends(),
                    fieldViable,
                    viableSet
                );
            }

            viableSet.retainAll(fieldViable);
        }

        // Static (function, fieldType) registration alone can be too coarse to decide viability for
        // some functions (e.g. NESTED_ANY_MATCH_EXPR, whose RexCall carries an arbitrary JSON-tree
        // operand) — give each still-viable backend's serializer a chance to inspect the actual call
        // and narrow further. Backends without a serializer, or whose serializer doesn't override
        // canServe, are unaffected (default true). This can only NARROW viableSet, never grant
        // viability the static check didn't already establish.
        viableSet.removeIf(candidateName -> {
            DelegatedPredicateSerializer serializer = registry.getBackend(candidateName).delegatedPredicateSerializers().get(function);
            if (serializer == null) {
                // No serializer for this function on this backend means the backend evaluates it
                // NATIVELY rather than by receiving a delegated predicate — nothing to narrow.
                return false;
            }
            if (!serializer.canServe(predicate, fieldStorageInfos)) {
                return true;
            }
            // In a NESTED query — one whose filter mentions NESTED_ANY_MATCH_EXPR anywhere — no leaf
            // under an OR/NOT may be delegated. combine() would demote such a leaf from performance to
            // CORRECTNESS delegation, which REPLACES the original and makes the peer's bitset the
            // answer for it. For a nested predicate that is unsound: the peer evaluates
            // NESTED_ANY_MATCH_EXPR at ROW grain, so a conjunction of nested leaves becomes a row-level
            // must-clause set answering (∃e A) ∧ (∃e B) — a strict SUPERSET of ∃e(A ∧ B), matching rows
            // whose elements satisfy the clauses separately. Fine as a pruning bound, wrong as an
            // answer. The rule is applied to the whole condition rather than only to nested leaves so
            // that the driving backend owns the ENTIRE result of a nested query: the combiner fuses
            // same-backend siblings without regard to scalar-vs-nested, so a scalar leaf sharing a
            // boolean group with a nested one would otherwise still answer from the peer.
            //
            // A PURE SCALAR filter is untouched (filterHasNested == false): a flat term query is exact,
            // so demotion stays sound there and the scalar path keeps its existing pruning.
            //
            // Excluding HERE rather than at classify/strip time is deliberate: viability is the single
            // source of truth that operator-level viability, delegation classification and strip all
            // read. Dropping the leaf later leaves the annotation still advertising the peer, and those
            // two disagreeing produced wrong results. With the peer removed here, narrowTo() derives no
            // performance peers either, so no marker is emitted and the driving backend evaluates the
            // predicate natively.
            // Strict prune-only makes this exclusion both unnecessary and harmful. Unnecessary
            // because the peer can no longer own any part of the answer — no delegated_predicate is
            // emitted and the driving backend always retains the full predicate — so the wrong-result
            // shapes this guards against are structurally impossible. Harmful because removing the
            // peer here makes the leaf look unservable to PruneOnlyRelaxer, which then drops it from
            // the relaxation and loses pruning that is exactly sound: every Lucene-servable nested
            // predicate is a row-grain superset (NestedAnyMatchExprSerializer refuses AND-bearing
            // trees outright, and existential quantification distributes over OR exactly).
            if (context.lucenePruneOnly()) {
                return false;
            }
            return underOrNot && filterHasNested;
        });

        // Every nested scalar function in the predicate must also be evaluable by a candidate backend
        for (RexCall scalarFunctionCall : contents.scalarFunctionCalls()) {
            // Calcite-internal value constructors (named-parameter MAP/ARRAY/ROW used by full-text
            // operators like match() to pass `field`, `query`, etc.) aren't real scalar functions
            // they're parameter-passing scaffolding. Skip them
            SqlKind kind = scalarFunctionCall.getKind();
            if (kind == SqlKind.MAP_VALUE_CONSTRUCTOR || kind == SqlKind.ARRAY_VALUE_CONSTRUCTOR || kind == SqlKind.ROW) {
                continue;
            }
            ScalarFunction scalarFunc = ScalarFunction.fromSqlOperatorWithFallback(scalarFunctionCall.getOperator());
            if (scalarFunc == null) {
                throw new IllegalStateException(
                    "Unrecognized scalar function ["
                        + scalarFunctionCall.getOperator().getName()
                        + "] in call ["
                        + scalarFunctionCall
                        + "] within filter predicate ["
                        + predicate
                        + "]"
                );
            }
            FieldType returnType = FieldType.fromSqlTypeName(scalarFunctionCall.getType().getSqlTypeName());
            // Polymorphic UDF fallback (e.g. SCALAR_MAX/MIN return SqlTypeName.ANY): infer
            // FieldType from the first concrete operand. Backend capabilities for these UDFs
            // are declared over operand types, so this preserves correct dispatch — see
            // OpenSearchProjectRule.resolveScalarViableBackends for the parallel fallback.
            if (returnType == null) {
                for (RexNode operand : scalarFunctionCall.getOperands()) {
                    FieldType operandType = FieldType.fromSqlTypeName(operand.getType().getSqlTypeName());
                    if (operandType != null) {
                        returnType = operandType;
                        break;
                    }
                }
                if (returnType == null) {
                    throw new IllegalStateException(
                        "Unmapped return type ["
                            + scalarFunctionCall.getType().getSqlTypeName()
                            + "] for scalar function ["
                            + scalarFunc
                            + "] in call ["
                            + scalarFunctionCall
                            + "] within filter predicate ["
                            + predicate
                            + "]"
                    );
                }
            }
            viableSet.retainAll(registry.scalarBackendsAnyFormat(scalarFunc, returnType));
        }

        // Per-backend delegation block-list. Drop backends that block this predicate so it is never
        // marked viable for them — but only when a non-blocked backend survives: blocking is a
        // delegation knob and must never make a predicate unexecutable.
        DelegationBlockList blockList = context.getDelegationBlockList();
        if (!blockList.isEmpty()) {
            boolean someBackendSurvives = viableSet.stream().anyMatch(backend -> !blockList.isBlocked(backend, function));
            if (someBackendSurvives) {
                viableSet.removeIf(backend -> blockList.isBlocked(backend, function));
            }
        }

        if (viableSet.isEmpty()) {
            throw new IllegalStateException(
                "No backend can evaluate filter predicate ["
                    + predicate.getKind()
                    + "] on fields "
                    + fieldIndices.stream()
                        .filter(i -> i < fieldStorageInfos.size())
                        .map(i -> fieldStorageInfos.get(i).getFieldName() + ":" + fieldStorageInfos.get(i).getMappingType())
                        .toList()
            );
        }
        return new ArrayList<>(viableSet);
    }

    /**
     * Result of a single walk over a predicate's operand subtree.
     *
     * <p>{@code fieldIndices} — RexInputRef indices feeding the field-storage intersection.
     * <p>{@code scalarFunctionCalls} — nested RexCalls feeding the scalar-function capability intersection.
     *
     * <p>TODO: ensure that the code for tagging and checking the scalar function of a predicate
     * remains the same as the code for tagging and checking its nested inner expressions as much
     * as possible.
     */
    private record PredicateContents(Set<Integer> fieldIndices, List<RexCall> scalarFunctionCalls) {
    }

    /** Recurses the operand subtree, populating {@code contents} in-place. */
    private void collect(RexNode node, PredicateContents contents) {
        if (node instanceof RexInputRef inputRef) {
            contents.fieldIndices().add(inputRef.getIndex());
        } else if (node instanceof RexCall rexCall) {
            contents.scalarFunctionCalls().add(rexCall);
            for (RexNode operand : rexCall.getOperands()) {
                collect(operand, contents);
            }
        }
    }

    // ---- Operator-level viable backends ----

    /**
     * Computes which backends can be the primary executor of this filter.
     * A backend is viable if:
     * 1. It is the child backend (data flows from child), AND for every predicate
     *    it can evaluate natively or delegate to another backend.
     * 2. OR it is a different backend that the child can delegate the entire filter to
     *    (child supports FILTER delegation, this backend accepts it, and this backend
     *    can handle all predicates natively).
     */
    private List<String> computeFilterViableBackends(RexNode annotatedCondition, List<String> childViableBackends) {
        List<AnnotatedPredicate> predicates = new ArrayList<>();
        collectAnnotatedPredicates(annotatedCondition, predicates);

        if (predicates.isEmpty()) {
            return new ArrayList<>(childViableBackends);
        }

        List<String> viable = new ArrayList<>();
        CapabilityRegistry registry = context.getCapabilityRegistry();

        for (String candidateName : childViableBackends) {
            if (!registry.filterCapableBackends().contains(candidateName)) {
                continue;
            }

            boolean canHandleAll = true;
            for (AnnotatedPredicate predicate : predicates) {
                if (!registry.canHandle(candidateName, predicate.getViableBackends(), DelegationType.FILTER)) {
                    canHandleAll = false;
                    break;
                }
            }
            if (canHandleAll) {
                viable.add(candidateName);
            }
        }
        return viable;
    }

    // ---- Strategy determination ----

    /** Recursively collects all {@link AnnotatedPredicate} leaves from the condition tree. */
    private void collectAnnotatedPredicates(RexNode node, List<AnnotatedPredicate> result) {
        if (node instanceof AnnotatedPredicate annotatedPredicate) {
            result.add(annotatedPredicate);
        } else if (node instanceof RexCall rexCall) {
            for (RexNode operand : rexCall.getOperands()) {
                collectAnnotatedPredicates(operand, result);
            }
        }
    }
}

/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.common.lucene.search;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.opensearch.OpenSearchException;
import org.opensearch.common.Nullable;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.SeqNoFieldMapper;
import org.opensearch.lucene.queries.ExtendedCommonTermsQuery;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Lucene queries class
 *
 * @opensearch.internal
 */
public class Queries {

    public static Query newMatchAllQuery() {
        return new MatchAllDocsQuery();
    }

    /** Return a query that matches no document. */
    public static Query newMatchNoDocsQuery(String reason) {
        return new MatchNoDocsQuery(reason);
    }

    public static Query newUnmappedFieldQuery(String field) {
        return newUnmappedFieldsQuery(Collections.singletonList(field));
    }

    public static Query newUnmappedFieldsQuery(Collection<String> fields) {
        return Queries.newMatchNoDocsQuery("unmapped fields " + fields);
    }

    public static Query newLenientFieldQuery(String field, RuntimeException e) {
        String message = OpenSearchException.getExceptionName(e) + ":[" + e.getMessage() + "]";
        return Queries.newMatchNoDocsQuery("failed [" + field + "] query, caused by " + message);
    }

    /**
     * Creates a new non-nested docs query — a filter matching root (parent) documents, used to exclude
     * hidden nested child docs from root searches and to build the parent {@code BitSet} for block joins.
     *
     * <p>Classic Lucene indexes stamp {@code _primary_term} on every root doc, so a {@code FieldExistsQuery}
     * on it selects roots. Composite (pluggable data format) indexes store {@code _primary_term} columnar
     * (absent from the Lucene secondary) and instead mark the block join with a Lucene parent field
     * ({@link Lucene#PARENT_FIELD}, set via {@code IndexWriterConfig.setParentField}); a
     * {@code FieldExistsQuery} on that field selects roots there. Matching EITHER field makes this filter
     * correct for both index types. The two field-exists clauses always mark the same set of root docs: where
     * both fields are present (e.g. a classic index that also sets a Lucene parent field when index-sorted),
     * the disjunction still selects exactly the roots; where only one is present (classic without a parent
     * field → only {@code _primary_term}; composite → only {@link Lucene#PARENT_FIELD}), the other clause
     * simply matches nothing. A doc is a root iff it has at least one of the two.
     */
    public static Query newNonNestedFilter() {
        return new BooleanQuery.Builder().add(new FieldExistsQuery(SeqNoFieldMapper.PRIMARY_TERM_NAME), Occur.SHOULD)
            .add(new FieldExistsQuery(Lucene.PARENT_FIELD), Occur.SHOULD)
            .setMinimumNumberShouldMatch(1)
            .build();
    }

    public static BooleanQuery filtered(@Nullable Query query, @Nullable Query filter) {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        if (query != null) {
            builder.add(new BooleanClause(query, Occur.MUST));
        }
        if (filter != null) {
            builder.add(new BooleanClause(filter, Occur.FILTER));
        }
        return builder.build();
    }

    /** Return a query that matches all documents but those that match the given query. */
    public static Query not(Query q) {
        return new BooleanQuery.Builder().add(new MatchAllDocsQuery(), Occur.MUST).add(q, Occur.MUST_NOT).build();
    }

    static boolean isNegativeQuery(Query q) {
        if (!(q instanceof BooleanQuery)) {
            return false;
        }
        List<BooleanClause> clauses = ((BooleanQuery) q).clauses();
        return clauses.isEmpty() == false && clauses.stream().allMatch(BooleanClause::isProhibited);
    }

    public static Query fixNegativeQueryIfNeeded(Query q) {
        if (isNegativeQuery(q)) {
            BooleanQuery bq = (BooleanQuery) q;
            BooleanQuery.Builder builder = new BooleanQuery.Builder();
            for (BooleanClause clause : bq) {
                builder.add(clause);
            }
            builder.add(newMatchAllQuery(), BooleanClause.Occur.FILTER);
            return builder.build();
        }
        return q;
    }

    public static Query applyMinimumShouldMatch(BooleanQuery query, @Nullable String minimumShouldMatch) {
        if (minimumShouldMatch == null) {
            return query;
        }
        int optionalClauses = 0;
        for (BooleanClause c : query.clauses()) {
            if (c.occur() == BooleanClause.Occur.SHOULD) {
                optionalClauses++;
            }
        }

        int msm = calculateMinShouldMatch(optionalClauses, minimumShouldMatch);
        if (0 < msm) {
            BooleanQuery.Builder builder = new BooleanQuery.Builder();
            for (BooleanClause clause : query) {
                builder.add(clause);
            }
            builder.setMinimumNumberShouldMatch(msm);
            return builder.build();
        } else {
            return query;
        }
    }

    /**
     * Potentially apply minimum should match value if we have a query that it can be applied to,
     * otherwise return the original query.
     */
    public static Query maybeApplyMinimumShouldMatch(Query query, @Nullable String minimumShouldMatch) {
        if (query instanceof BooleanQuery) {
            return applyMinimumShouldMatch((BooleanQuery) query, minimumShouldMatch);
        } else if (query instanceof ExtendedCommonTermsQuery) {
            ((ExtendedCommonTermsQuery) query).setLowFreqMinimumNumberShouldMatch(minimumShouldMatch);
        }
        return query;
    }

    private static Pattern spaceAroundLessThanPattern = Pattern.compile("(\\s+<\\s*)|(\\s*<\\s+)");
    private static Pattern spacePattern = Pattern.compile(" ");
    private static Pattern lessThanPattern = Pattern.compile("<");

    public static int calculateMinShouldMatch(int optionalClauseCount, String spec) {
        int result = optionalClauseCount;
        spec = spec.trim();

        if (-1 < spec.indexOf("<")) {
            /* we have conditional spec(s) */
            spec = spaceAroundLessThanPattern.matcher(spec).replaceAll("<");
            for (String s : spacePattern.split(spec)) {
                String[] parts = lessThanPattern.split(s, 0);
                int upperBound = Integer.parseInt(parts[0]);
                if (optionalClauseCount <= upperBound) {
                    return result;
                } else {
                    result = calculateMinShouldMatch(optionalClauseCount, parts[1]);
                }
            }
            return result;
        }

        /* otherwise, simple expression */

        if (-1 < spec.indexOf('%')) {
            /* percentage - assume the % was the last char.  If not, let Integer.parseInt fail. */
            spec = spec.substring(0, spec.length() - 1);
            int percent = Integer.parseInt(spec);
            float calc = (result * percent) * (1 / 100f);
            result = calc < 0 ? result + (int) calc : (int) calc;
        } else {
            int calc = Integer.parseInt(spec);
            result = calc < 0 ? result + calc : calc;
        }

        return result < 0 ? 0 : result;
    }

    public static Query newMatchNoDocsQueryWithoutRewrite(String reason) {
        return new MatchNoDocsWithoutRewriteQuery(reason);
    }

    /**
     * Matches no docs w/o rewriting the query
     *
     * @opensearch.internal
     */
    static class MatchNoDocsWithoutRewriteQuery extends Query {
        private final String reason;

        public MatchNoDocsWithoutRewriteQuery(String reason) {
            this.reason = reason;
        }

        @Override
        public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
            return new Weight(this) {
                @Override
                public Explanation explain(LeafReaderContext context, int doc) {
                    return Explanation.noMatch(reason);
                }

                @Override
                public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                    return null;
                }

                @Override
                public boolean isCacheable(LeafReaderContext ctx) {
                    return true;
                }
            };
        }

        @Override
        public String toString(String field) {
            return "MatchNoDocsWithoutRewriteQuery(" + reason + ")";
        }

        @Override
        public void visit(QueryVisitor visitor) {
            // noop
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof MatchNoDocsWithoutRewriteQuery && Objects.equals(this.reason, ((MatchNoDocsWithoutRewriteQuery) o).reason);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(reason);
        }
    }

}

"""v2 hybrid retrieval: dense + BM25 merged with Reciprocal Rank Fusion.

Dense retrieval finds semantically similar passages but is blind to exact
tokens; BM25 matches tokens exactly but is blind to paraphrase. RRF merges the
two ranked lists on rank alone, so neither score scale dominates the other.

The returned SearchResult.score is the RRF score (roughly 0.01-0.03), NOT a
cosine similarity -- it is not comparable to a v1 score. The underlying dense
and BM25 scores and ranks are preserved in metadata under "fusion".
"""

import logging
import time

import config
from retrieve.bm25 import get_index
from retrieve.fusion import reciprocal_rank_fusion
from retrieve.search import describe_filters, dense_search
from retrieve.tracing import get_tracer, record_results

log = logging.getLogger(__name__)


def search_hybrid(query, top_k=config.DEFAULT_TOP_K, ticker=None, year=None,
                  section=None, exclude_sparse=True, exclude_by_ref=True,
                  dense_candidates=None, bm25_candidates=None, rrf_k=None):
    """Return top_k chunks by fused dense+BM25 rank."""
    dense_n = dense_candidates or config.DENSE_CANDIDATES
    bm25_n = bm25_candidates or config.BM25_CANDIDATES

    tracer = get_tracer()
    filters = describe_filters(ticker, year, section, exclude_sparse, exclude_by_ref)
    started = time.perf_counter()

    with tracer.start_as_current_span("retrieve.search") as span:
        _set(span, "openinference.span.kind", "RETRIEVER")
        _set(span, "input.value", query)
        _set(span, "retrieval.version", "v2")
        _set(span, "retrieval.strategy", "hybrid dense+bm25 rrf")
        _set(span, "retrieval.top_k", top_k)
        _set(span, "retrieval.dense_candidates", dense_n)
        _set(span, "retrieval.bm25_candidates", bm25_n)
        _set(span, "retrieval.rrf_k", rrf_k or config.RRF_K)
        _set(span, "retrieval.collection", config.COLLECTION_NAME)
        _set(span, "retrieval.embedding_model", config.EMBEDDING_MODEL)
        for key, value in filters.items():
            _set(span, f"retrieval.filter.{key}", "None" if value is None else value)

        try:
            dense_results, embed_ms, query_ms = dense_search(
                query, dense_n, ticker, year, section,
                exclude_sparse, exclude_by_ref, tracer=tracer)

            with tracer.start_as_current_span("bm25.search") as bm_span:
                _set(bm_span, "openinference.span.kind", "RETRIEVER")
                t0 = time.perf_counter()
                bm25_pairs = get_index().search(
                    query, top_n=bm25_n, ticker=ticker, year=year, section=section,
                    exclude_sparse=exclude_sparse, exclude_by_ref=exclude_by_ref)
                bm25_ms = (time.perf_counter() - t0) * 1000
                _set(bm_span, "retrieval.latency_ms", round(bm25_ms, 2))
                _set(bm_span, "retrieval.hits", len(bm25_pairs))
                if bm25_pairs:
                    _set(bm_span, "retrieval.top_score", bm25_pairs[0][1])

            bm25_results = [result for result, _score in bm25_pairs]

            with tracer.start_as_current_span("fusion.rrf") as fuse_span:
                bm25_w = config.BM25_WEIGHT
                fused = reciprocal_rank_fusion(
                    {"dense": dense_results, "bm25": bm25_results},
                    top_k=top_k, k=rrf_k,
                    weights={"bm25": bm25_w})
                _set(fuse_span, "fusion.method", "reciprocal_rank_fusion")
                _set(fuse_span, "fusion.k", rrf_k or config.RRF_K)
                _set(fuse_span, "fusion.bm25_weight", bm25_w)
                _set(fuse_span, "fusion.dense_in", len(dense_results))
                _set(fuse_span, "fusion.bm25_in", len(bm25_results))
                _set(fuse_span, "fusion.out", len(fused))
        except Exception as exc:
            _record_exception(span, exc)
            raise

        results = []
        both = 0
        for result, rrf_score, contributions in fused:
            result.score = rrf_score
            result.metadata = dict(result.metadata)
            result.metadata["fusion"] = {
                "rrf_score": rrf_score,
                "dense_rank": contributions.get("dense", (None, None))[0],
                "dense_score": contributions.get("dense", (None, None))[1],
                "bm25_rank": contributions.get("bm25", (None, None))[0],
                "bm25_score": contributions.get("bm25", (None, None))[1],
                "retrievers": sorted(contributions),
            }
            if len(contributions) > 1:
                both += 1
            results.append(result)

        total_ms = (time.perf_counter() - started) * 1000
        _set(span, "retrieval.embed_latency_ms", round(embed_ms, 2))
        _set(span, "retrieval.query_latency_ms", round(query_ms, 2))
        _set(span, "retrieval.bm25_latency_ms", round(bm25_ms, 2))
        _set(span, "retrieval.total_latency_ms", round(total_ms, 2))
        _set(span, "retrieval.result_count", len(results))
        # How much the two retrievers actually agreed -- the useful diagnostic
        # for whether fusion is doing anything.
        _set(span, "retrieval.agreement", both)
        if results:
            _set(span, "retrieval.top_score", float(results[0].score))
        record_results(span, results)

    log.debug("search_hybrid(%r) -> %d hits in %.1fms", query, len(results), total_ms)
    return results


def _set(span, key, value):
    if span is not None and hasattr(span, "set_attribute"):
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


def _record_exception(span, exc):
    if span is not None and hasattr(span, "record_exception"):
        try:
            span.record_exception(exc)
        except Exception:
            pass

"""v4: v3 retrieval widened, then reranked by a cross-encoder.

v3 leaves two failures that no amount of filtering or fusion tuning fixes:

  FM-1  Table chunks have no BM25 signal, so a chunk at dense rank 1 but absent
        from BM25's top-50 scores 1/(60+1) = 0.0164 and loses to a mediocre
        chunk that placed in both lists. Fusion cannot see that the table
        actually answers the question; a cross-encoder can, because it reads the
        query and the chunk together.

  FM-2  With a tight ticker+year filter, the largest sections (Item 7/8) fill
        every slot and short sections get crowded out. Reranking 20 candidates
        instead of 5 gives the short-section chunks a second chance.

So v4 does not change retrieval at all -- it widens the cut from 5 to 20 and
lets a stronger, slower model decide the final order.
"""

import logging
import time

import config
from retrieve.rerank import rerank
from retrieve.search_v3 import search_v3
from retrieve.tracing import get_tracer, record_results

log = logging.getLogger(__name__)


def search_v4(query, top_k=config.DEFAULT_TOP_K, ticker=None, year=None,
              section=None, exclude_sparse=True, exclude_by_ref=True,
              candidates=None, use_section_filter=None, use_clean_query=None):
    """Retrieve `candidates` chunks via v3, rerank them, return the best top_k."""
    n_candidates = candidates or config.RERANK_CANDIDATES
    # Never rerank fewer than we intend to return.
    n_candidates = max(n_candidates, top_k)

    tracer = get_tracer()
    started = time.perf_counter()

    with tracer.start_as_current_span("retrieve.v4") as root:
        _set(root, "openinference.span.kind", "CHAIN")
        _set(root, "input.value", query)
        _set(root, "retrieval.version", "v4")
        _set(root, "retrieval.top_k", top_k)
        _set(root, "retrieval.candidates", n_candidates)

        pool = search_v3(
            query,
            top_k=n_candidates,
            ticker=ticker,
            year=year,
            section=section,
            exclude_sparse=exclude_sparse,
            exclude_by_ref=exclude_by_ref,
            use_section_filter=use_section_filter,
            use_clean_query=use_clean_query,
        )

        with tracer.start_as_current_span("rerank.cross_encoder") as span:
            _set(span, "openinference.span.kind", "RERANKER")
            _set(span, "reranker.model", config.RERANK_MODEL)
            # The cross-encoder gets the ORIGINAL question, never clean_query.
            _set(span, "reranker.query", query)
            _set(span, "reranker.candidates_scored", len(pool))
            _set(span, "reranker.top_k", top_k)
            t0 = time.perf_counter()
            results = rerank(query, pool, top_k)
            rerank_ms = (time.perf_counter() - t0) * 1000
            _set(span, "reranker.latency_ms", round(rerank_ms, 2))
            if results:
                _set(span, "reranker.top_score", float(results[0].rerank_score))
                # How far the reranker had to reach -- the FM-1/FM-2 diagnostic.
                promoted = [r.metadata["pre_rerank"]["rrf_rank"] for r in results]
                _set(span, "reranker.promoted_from_ranks", str(promoted))
                _set(span, "reranker.deepest_promoted", max(promoted))
                _set(span, "reranker.rescued_beyond_top_k",
                     sum(1 for p in promoted if p > top_k))

        total_ms = (time.perf_counter() - started) * 1000
        _set(root, "retrieval.rerank_latency_ms", round(rerank_ms, 2))
        _set(root, "retrieval.total_latency_ms", round(total_ms, 2))
        _set(root, "retrieval.result_count", len(results))
        if results:
            _set(root, "retrieval.top_score", float(results[0].score))
        record_results(root, results)

    log.debug("search_v4(%r) reranked %d -> %d in %.1fms",
              query, len(pool), len(results), total_ms)
    return results


def _set(span, key, value):
    if span is not None and hasattr(span, "set_attribute"):
        try:
            span.set_attribute(key, value)
        except Exception:
            pass

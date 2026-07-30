"""Retrieval entry point.

v1 -- naive dense: embed the query, cosine search in Qdrant, return top-k.
v2 -- hybrid: dense + BM25, merged with Reciprocal Rank Fusion.

search() defaults to v1, so every existing call site keeps the exact behaviour
the v1 eval baseline was measured against.
"""

import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import config
from retrieve.embed import embed_query, get_embedder
from retrieve.result import SearchResult
from retrieve.tracing import get_tracer, record_results

log = logging.getLogger(__name__)

_client = None


def get_client():
    """Process-wide Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
    return _client


def warm_up(retrieval=None):
    """Load the model, open the connection, and (for v2) build the BM25 index."""
    get_embedder()
    get_client()
    retrieval = retrieval or config.DEFAULT_RETRIEVAL
    if retrieval in ("v2", "v3", "v4"):
        from retrieve.bm25 import get_index
        get_index()
    if retrieval == "v4":
        from retrieve.rerank import get_reranker
        get_reranker()


def build_filter(ticker=None, year=None, section=None,
                 exclude_sparse=True, exclude_by_ref=True):
    """Compose the Qdrant filter. Conditions are additive; None means unfiltered."""
    must, must_not = [], []

    if ticker is not None:
        must.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
    if year is not None:
        must.append(FieldCondition(key="year", match=MatchValue(value=int(year))))
    if section is not None:
        must.append(FieldCondition(key="section", match=MatchValue(value=section)))

    # must_not rather than matching False, so a chunk missing the flag entirely is
    # still retrievable.
    if exclude_sparse:
        must_not.append(FieldCondition(key="is_sparse", match=MatchValue(value=True)))
    if exclude_by_ref:
        must_not.append(FieldCondition(key="incorporated_by_reference",
                                       match=MatchValue(value=True)))

    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def describe_filters(ticker=None, year=None, section=None,
                     exclude_sparse=True, exclude_by_ref=True):
    """Human/trace-readable summary of the filters in effect."""
    return {
        "ticker": ticker,
        "year": year,
        "section": section,
        "exclude_sparse": exclude_sparse,
        "exclude_by_ref": exclude_by_ref,
    }


def dense_search(query, limit, ticker=None, year=None, section=None,
                 exclude_sparse=True, exclude_by_ref=True, tracer=None):
    """Embed and cosine-search. Returns (results, embed_ms, query_ms).

    Shared by v1 and by v2's dense arm, so the two can never diverge.
    """
    tracer = tracer or get_tracer()

    with tracer.start_as_current_span("embed.query") as embed_span:
        _set(embed_span, "openinference.span.kind", "EMBEDDING")
        _set(embed_span, "embedding.model_name", config.EMBEDDING_MODEL)
        _set(embed_span, "embedding.embeddings.0.embedding.text", query)
        t0 = time.perf_counter()
        vector = embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        _set(embed_span, "embedding.latency_ms", round(embed_ms, 2))

    with tracer.start_as_current_span("qdrant.query_points") as q_span:
        _set(q_span, "openinference.span.kind", "RETRIEVER")
        t0 = time.perf_counter()
        response = get_client().query_points(
            collection_name=config.COLLECTION_NAME,
            query=vector,
            limit=limit,
            query_filter=build_filter(ticker, year, section,
                                      exclude_sparse, exclude_by_ref),
            with_payload=True,
            with_vectors=False,
        )
        query_ms = (time.perf_counter() - t0) * 1000
        _set(q_span, "retrieval.latency_ms", round(query_ms, 2))
        _set(q_span, "retrieval.hits", len(response.points))

    return [SearchResult.from_point(p) for p in response.points], embed_ms, query_ms


def search(query, top_k=config.DEFAULT_TOP_K, ticker=None, year=None, section=None,
           exclude_sparse=True, exclude_by_ref=True, retrieval=None,
           candidates=None):
    """Return the top_k most relevant chunks to `query` as SearchResults.

    retrieval="v1" (default) is dense only; "v2" is hybrid dense+BM25 via RRF.
    """
    retrieval = retrieval or config.DEFAULT_RETRIEVAL
    if retrieval == "v4":
        from retrieve.search_v4 import search_v4
        return search_v4(query, top_k=top_k, ticker=ticker, year=year,
                         section=section, exclude_sparse=exclude_sparse,
                         exclude_by_ref=exclude_by_ref, candidates=candidates)
    if retrieval == "v3":
        from retrieve.search_v3 import search_v3
        return search_v3(query, top_k=top_k, ticker=ticker, year=year,
                         section=section, exclude_sparse=exclude_sparse,
                         exclude_by_ref=exclude_by_ref)
    if retrieval == "v2":
        from retrieve.hybrid import search_hybrid
        return search_hybrid(query, top_k=top_k, ticker=ticker, year=year,
                             section=section, exclude_sparse=exclude_sparse,
                             exclude_by_ref=exclude_by_ref)
    if retrieval != "v1":
        raise ValueError(f"unknown retrieval version {retrieval!r} (expected v1/v2/v3/v4)")

    tracer = get_tracer()
    filters = describe_filters(ticker, year, section, exclude_sparse, exclude_by_ref)
    started = time.perf_counter()

    with tracer.start_as_current_span("retrieve.search") as span:
        _set(span, "openinference.span.kind", "RETRIEVER")
        _set(span, "input.value", query)
        _set(span, "retrieval.version", "v1")
        _set(span, "retrieval.top_k", top_k)
        _set(span, "retrieval.collection", config.COLLECTION_NAME)
        _set(span, "retrieval.embedding_model", config.EMBEDDING_MODEL)
        for key, value in filters.items():
            _set(span, f"retrieval.filter.{key}",
                 "None" if value is None else value)

        try:
            results, embed_ms, query_ms = dense_search(
                query, top_k, ticker, year, section,
                exclude_sparse, exclude_by_ref, tracer=tracer)
        except Exception as exc:
            _record_exception(span, exc)
            raise

        total_ms = (time.perf_counter() - started) * 1000
        _set(span, "retrieval.embed_latency_ms", round(embed_ms, 2))
        _set(span, "retrieval.query_latency_ms", round(query_ms, 2))
        _set(span, "retrieval.total_latency_ms", round(total_ms, 2))
        _set(span, "retrieval.result_count", len(results))
        if results:
            _set(span, "retrieval.top_score", float(results[0].score))
        record_results(span, results)

    log.debug("search(%r) -> %d hits in %.1fms", query, len(results), total_ms)
    return results


def _set(span, key, value):
    if span is not None and hasattr(span, "set_attribute"):
        try:
            span.set_attribute(key, value)
        except Exception:  # tracing must never break retrieval
            pass


def _record_exception(span, exc):
    if span is not None and hasattr(span, "record_exception"):
        try:
            span.record_exception(exc)
        except Exception:
            pass

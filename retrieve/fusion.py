"""Reciprocal Rank Fusion.

RRF merges ranked lists using only rank position, never the raw scores -- which
is the whole point here, since BM25 scores and cosine similarities live on
incompatible scales and cannot be meaningfully added or averaged.

    rrf(d) = sum over lists of  1 / (k + rank(d))     ranks are 1-based
"""

import config


def chunk_key(result):
    """Identity of a chunk across retrievers. Mirrors the Qdrant point ID."""
    return (result.filing_id, result.section, result.chunk_index)


def reciprocal_rank_fusion(ranked_lists, top_k=5, k=None, weights=None):
    """Fuse ranked SearchResult lists.

    ranked_lists: {name: [SearchResult, ...]} each already sorted best-first.
    Returns [(SearchResult, rrf_score, contributions)], best first, where
    contributions maps retriever name -> (rank, original_score).
    """
    k = config.RRF_K if k is None else k
    weights = weights or {}

    fused = {}
    for name, results in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, result in enumerate(results, start=1):
            key = chunk_key(result)
            entry = fused.setdefault(key, {"result": result, "score": 0.0,
                                           "contributions": {}})
            entry["score"] += weight / (k + rank)
            entry["contributions"][name] = (rank, float(result.score))
            # Prefer whichever copy carries text (both do, but be explicit).
            if not entry["result"].text and result.text:
                entry["result"] = result

    ordered = sorted(fused.values(), key=lambda e: -e["score"])
    return [(e["result"], e["score"], e["contributions"]) for e in ordered[:top_k]]

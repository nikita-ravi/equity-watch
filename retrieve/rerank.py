"""Cross-encoder reranking.

A bi-encoder embeds the query and the chunk independently, so it can never model
how they interact -- which is why a table of bare numbers scores poorly against a
question that never names those numbers. A cross-encoder reads (query, chunk)
jointly and scores the pair directly, which is exactly the signal RRF is missing.

It is far too slow to run over the whole corpus, so it only re-scores the
candidates an earlier stage already retrieved.
"""

import logging
import time

import config

log = logging.getLogger(__name__)

_reranker = None


class Reranker:
    def __init__(self, model_name=None, batch_size=None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        self.batch_size = batch_size or config.RERANK_BATCH_SIZE
        started = time.perf_counter()
        self.model = CrossEncoder(self.model_name, device=config.TORCH_DEVICE)
        log.info("loaded reranker %s in %.1fs",
                 self.model_name, time.perf_counter() - started)

    def score(self, query, texts):
        """Score each (query, text) pair. Returns floats in [0, 1], higher = better.

        Chunks longer than the model's max sequence length are truncated by the
        tokenizer -- a 2,200-char chunk does not fit in 512 tokens, so the tail
        of a long chunk is not seen by the reranker.
        """
        if not texts:
            return []
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def get_reranker():
    """Process-wide reranker, loaded on first use."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def rerank(query, results, top_k, tracer=None):
    """Re-score `results` against `query` and return the best `top_k`.

    `query` should be the original question text: the cross-encoder is trained on
    natural language pairs, so the entity-stripped clean_query would deprive it
    of the same context it deprived the retriever of.
    """
    if not results:
        return []

    scores = get_reranker().score(query, [r.text for r in results])
    # `results` arrives in fusion order, so the index is the pre-rerank rank.
    for rank, (result, score) in enumerate(zip(results, scores), start=1):
        result.rerank_score = score
        # Keep what got this chunk into the pool, so the rescue is auditable.
        result.metadata = dict(result.metadata)
        result.metadata["pre_rerank"] = {"rrf_score": result.score, "rrf_rank": rank}
        result.score = score

    ordered = sorted(results, key=lambda r: -r.rerank_score)
    return ordered[:top_k]

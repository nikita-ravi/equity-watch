"""In-memory BM25 index over the sec_10k corpus.

The whole corpus is 9,858 chunks / ~22 MB of text, so it is cheaper to hold the
lexical index in process than to rebuild the collection with Qdrant sparse
vectors. The index is built once per process and reused.

Filtering happens after scoring but before top-n selection, so BM25's IDF
statistics stay corpus-wide (standard practice) while the returned candidates
respect the caller's filters.
"""

import logging
import re
import time

from qdrant_client import QdrantClient

import config
from retrieve.result import SearchResult

log = logging.getLogger(__name__)

# Keep numerals intact. A default word tokenizer shreds "383,285" into "383" and
# "285" and "$1.5" into "1" and "5", which is exactly the lexical signal dense
# retrieval already fails at -- losing it here would defeat the point of BM25.
_TOKEN = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    """Lexical index over every chunk in the collection."""

    def __init__(self, client=None, collection=None):
        self.collection = collection or config.COLLECTION_NAME
        self.client = client or QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
        self.payloads = []
        self.point_ids = []
        self._bm25 = None

    def build(self):
        from rank_bm25 import BM25Okapi

        started = time.perf_counter()
        offset = None
        corpus = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                self.point_ids.append(point.id)
                self.payloads.append(payload)
                corpus.append(tokenize(payload.get("text", "")))
            if offset is None:
                break

        self._bm25 = BM25Okapi(corpus)
        log.info("BM25 index built over %d chunks in %.1fs",
                 len(corpus), time.perf_counter() - started)
        return self

    def __len__(self):
        return len(self.payloads)

    def search(self, query, top_n=50, ticker=None, year=None, section=None,
               exclude_sparse=True, exclude_by_ref=True):
        """Return up to top_n (SearchResult, bm25_score) pairs, best first."""
        if self._bm25 is None:
            self.build()

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        ranked = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            payload = self.payloads[index]
            if not _passes(payload, ticker, year, section,
                           exclude_sparse, exclude_by_ref):
                continue
            ranked.append((score, index))

        ranked.sort(key=lambda pair: -pair[0])
        return [(_to_result(self.payloads[i], float(s)), float(s))
                for s, i in ranked[:top_n]]


def _passes(payload, ticker, year, section, exclude_sparse, exclude_by_ref):
    """Python mirror of the Qdrant filter, driven by the same field names."""
    if ticker is not None and payload.get("ticker") != ticker:
        return False
    if year is not None and payload.get("year") != int(year):
        return False
    if section is not None and payload.get("section") != section:
        return False
    # must_not semantics: a chunk missing the flag entirely stays retrievable.
    if exclude_sparse and payload.get("is_sparse") is True:
        return False
    if exclude_by_ref and payload.get("incorporated_by_reference") is True:
        return False
    return True


def _to_result(payload, score):
    payload = dict(payload)
    text = payload.pop("text", "")
    return SearchResult(
        text=text,
        ticker=payload.get("ticker", ""),
        year=payload.get("year"),
        section=payload.get("section", ""),
        score=score,
        metadata=payload,
    )


_index = None


def get_index():
    """Process-wide BM25 index, built on first use."""
    global _index
    if _index is None:
        _index = BM25Index().build()
    return _index

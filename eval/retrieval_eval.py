"""Scores retrieval quality against the ground-truth question set.

The primary metric is keyword_hit@k: of a question's answer_keywords, what
fraction appear anywhere in the top-k retrieved chunk texts. It is deliberately
generous -- it asks "did retrieval put the answer in front of the model", not
"did the model answer correctly" -- which is the right question for a layer that
has no LLM in it yet.
"""

import re
import statistics
import time
from collections import defaultdict

import config
from eval.dataset import QUESTIONS
from retrieve.search import search

# Chunk text carries newlines and runs of spaces from the filing; a keyword like
# "net sales" must still match when the filing broke it across a line.
_WHITESPACE = re.compile(r"\s+")


def normalize(text):
    return _WHITESPACE.sub(" ", (text or "")).lower()


class RetrievalEval:
    def __init__(self, questions=None, top_k=5, use_filters=False,
                 exclude_sparse=True, exclude_by_ref=True, retrieval=None):
        self.questions = list(questions if questions is not None else QUESTIONS)
        self.top_k = top_k
        self.use_filters = use_filters
        self.retrieval = retrieval or config.DEFAULT_RETRIEVAL
        self.exclude_sparse = exclude_sparse
        self.exclude_by_ref = exclude_by_ref

    # --- single question ------------------------------------------------------
    def evaluate_question(self, question):
        started = time.perf_counter()
        results = search(
            question.question,
            top_k=self.top_k,
            # use_filters=False is the v1-blind test: the retriever gets no hint
            # about which company or year the question is about.
            ticker=question.ticker if self.use_filters else None,
            year=question.year if self.use_filters else None,
            section=None,
            exclude_sparse=self.exclude_sparse,
            exclude_by_ref=self.exclude_by_ref,
            retrieval=self.retrieval,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        haystack = " ".join(normalize(r.text) for r in results)
        matched, missing = [], []
        for keyword in question.answer_keywords:
            (matched if normalize(keyword) in haystack else missing).append(keyword)

        total = len(question.answer_keywords)
        return {
            "id": question.id,
            "question": question.question,
            "ticker": question.ticker,
            "year": question.year,
            "section": question.section,
            "question_type": question.question_type,
            "difficulty": question.difficulty,
            # primary metric
            "keyword_hit": (len(matched) / total) if total else 0.0,
            "matched_keywords": matched,
            "missing_keywords": missing,
            # supporting metrics -- did we land on the right document at all?
            "ticker_hit": any(r.ticker == question.ticker for r in results),
            "year_hit": any(r.year == question.year for r in results),
            "section_hit": any(r.section == question.section for r in results),
            "top1_score": results[0].score if results else 0.0,
            "num_results": len(results),
            "latency_ms": round(latency_ms, 1),
            "retrieved": [{
                "rank": i,
                "ticker": r.ticker,
                "year": r.year,
                "section": r.section,
                "section_label": r.section_label,
                "score": r.score,
                "chunk_index": r.chunk_index,
                "filing_id": r.filing_id,
                "text": r.text,
            } for i, r in enumerate(results, start=1)],
        }

    # --- whole set ------------------------------------------------------------
    def evaluate_all(self, progress=None):
        per_question = []
        for i, question in enumerate(self.questions, start=1):
            record = self.evaluate_question(question)
            per_question.append(record)
            if progress:
                progress(i, len(self.questions), record)

        return {
            "config": {
                "top_k": self.top_k,
                "retrieval": self.retrieval,
                "use_filters": self.use_filters,
                "exclude_sparse": self.exclude_sparse,
                "exclude_by_ref": self.exclude_by_ref,
                "bm25_weight": config.BM25_WEIGHT,
                "embedding_model": config.EMBEDDING_MODEL,
                "collection": config.COLLECTION_NAME,
                "num_questions": len(self.questions),
            },
            "aggregate": aggregate(per_question),
            "by_ticker": group_by(per_question, "ticker"),
            "by_difficulty": group_by(per_question, "difficulty"),
            "by_question_type": group_by(per_question, "question_type"),
            "by_section": group_by(per_question, "section"),
            "failures": [r for r in per_question if r["keyword_hit"] < 0.5],
            "results": per_question,
        }


# --- metric helpers -----------------------------------------------------------
def _mean(values):
    return statistics.fmean(values) if values else 0.0


def aggregate(records):
    if not records:
        return {}
    return {
        "n": len(records),
        "keyword_hit": _mean([r["keyword_hit"] for r in records]),
        # A question counts as "perfect" only if every keyword surfaced.
        "perfect_rate": _mean([1.0 if r["keyword_hit"] == 1.0 else 0.0 for r in records]),
        "zero_rate": _mean([1.0 if r["keyword_hit"] == 0.0 else 0.0 for r in records]),
        "ticker_hit": _mean([1.0 if r["ticker_hit"] else 0.0 for r in records]),
        "year_hit": _mean([1.0 if r["year_hit"] else 0.0 for r in records]),
        "section_hit": _mean([1.0 if r["section_hit"] else 0.0 for r in records]),
        "top1_score": _mean([r["top1_score"] for r in records]),
        "latency_ms": _mean([r["latency_ms"] for r in records]),
    }


def group_by(records, key):
    buckets = defaultdict(list)
    for record in records:
        buckets[record[key]].append(record)
    return {name: aggregate(rows) for name, rows in sorted(buckets.items())}

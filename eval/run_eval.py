#!/usr/bin/env python
"""Run the retrieval eval and save full results as JSON.

    python eval/run_eval.py --top-k 5 --output eval/results/v1_unfiltered.json
    python eval/run_eval.py --top-k 5 --use-filters --output eval/results/v1_filtered.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                             # noqa: E402  (must come before retrieval imports)
from eval.dataset import QUESTIONS                        # noqa: E402
from eval.retrieval_eval import RetrievalEval             # noqa: E402
from retrieve.search import warm_up                       # noqa: E402
from retrieve.tracing import flush, init_tracing          # noqa: E402


def pct(value):
    return f"{value * 100:5.1f}%"


def print_table(title, rows, label_header):
    print(f"\n{title}")
    print(f"  {label_header:<14} {'n':>3}  {'kw_hit':>7} {'perfect':>8} {'zero':>6} "
          f"{'ticker':>7} {'year':>6} {'section':>8} {'top1':>6}")
    print("  " + "-" * 74)
    for name, m in rows:
        print(f"  {str(name):<14} {m['n']:>3}  {pct(m['keyword_hit']):>7} "
              f"{pct(m['perfect_rate']):>8} {pct(m['zero_rate']):>6} "
              f"{pct(m['ticker_hit']):>7} {pct(m['year_hit']):>6} "
              f"{pct(m['section_hit']):>8} {m['top1_score']:>6.3f}")


def print_summary(report):
    cfg = report["config"]
    agg = report["aggregate"]
    mode = "FILTERED (ticker+year passed)" if cfg["use_filters"] else "UNFILTERED (v1 blind)"

    print("\n" + "=" * 78)
    print(f"  RETRIEVAL EVAL — {mode}")
    print(f"  retrieval={cfg.get('retrieval', 'v1')}  top_k={cfg['top_k']}  "
          f"bm25_weight={cfg.get('bm25_weight', 1.0):.2f}  "
          f"model={cfg['embedding_model']}  "
          f"n={cfg['num_questions']}")
    print("=" * 78)

    print(f"\n  keyword_hit@{cfg['top_k']} (PRIMARY) : {pct(agg['keyword_hit'])}")
    print(f"  all keywords found       : {pct(agg['perfect_rate'])}")
    print(f"  no keywords found        : {pct(agg['zero_rate'])}")
    print(f"  ticker_hit               : {pct(agg['ticker_hit'])}")
    print(f"  year_hit                 : {pct(agg['year_hit'])}")
    print(f"  section_hit              : {pct(agg['section_hit'])}")
    print(f"  mean top1 score          : {agg['top1_score']:.3f}")
    print(f"  mean latency             : {agg['latency_ms']:.0f} ms")

    print_table("BY TICKER", sorted(report["by_ticker"].items()), "ticker")
    order = {"easy": 0, "medium": 1, "hard": 2}
    print_table("BY DIFFICULTY",
                sorted(report["by_difficulty"].items(),
                       key=lambda kv: order.get(kv[0], 9)), "difficulty")
    print_table("BY QUESTION TYPE", sorted(report["by_question_type"].items()), "type")
    # Section breakdown — key diagnostic: Item 1A/1B/1C section_hit should be
    # near 0% with standard RRF (FM-1/FM-3) and should improve with lower
    # bm25_weight (FM-1 fix) or --include-sparse (FM-3 fix).
    print_table("BY SECTION", sorted(report["by_section"].items()), "section")

    failures = report["failures"]
    print(f"\nFAILURES — keyword_hit < 0.5  ({len(failures)} of {agg['n']})")
    if not failures:
        print("  none")
    for f in failures:
        print(f"  {f['id']:<22} {pct(f['keyword_hit'])}  "
              f"{f['ticker']} {f['year']} {f['section']:<8} [{f['difficulty']}]")
        print(f"    q: {f['question'][:88]}")
        print(f"    missing: {f['missing_keywords']}")
        got = ", ".join(f"{r['ticker']} {r['year']} {r['section']}"
                        for r in f["retrieved"][:3])
        print(f"    top hits: {got or '(none)'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the retrieval eval.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval", choices=["v1", "v2", "v3", "v4"], default="v1",
                        help="v1=dense, v2=hybrid RRF, v3=+query parsing, v4=+cross-encoder rerank")
    parser.add_argument("--use-filters", action="store_true",
                        help="pass ticker+year to search() -- measures the ceiling")
    parser.add_argument("--output", default=None, help="path for the full JSON report")
    parser.add_argument("--include-sparse", action="store_true")
    parser.add_argument("--include-by-ref", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions (smoke test)")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="no per-question progress")
    parser.add_argument(
        "--bm25-weight", type=float, default=None, metavar="W",
        help="BM25 weight in RRF (default: config.BM25_WEIGHT=1.0). "
             "Values <0.80 break the intersection gate so dense-only chunks can win. "
             "Sweep: 0.25 0.5 0.75 1.0 to quantify FM-1 impact.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s",
                        stream=sys.stderr)
    if not args.no_trace:
        init_tracing()

    # Apply BM25 weight override before any retrieval module initialises.
    if args.bm25_weight is not None:
        config.BM25_WEIGHT = args.bm25_weight

    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    evaluator = RetrievalEval(
        questions=questions,
        top_k=args.top_k,
        use_filters=args.use_filters,
        exclude_sparse=not args.include_sparse,
        exclude_by_ref=not args.include_by_ref,
        retrieval=args.retrieval,
    )

    warm_up(retrieval=args.retrieval)

    def progress(i, total, record):
        if args.quiet:
            return
        mark = "OK  " if record["keyword_hit"] >= 0.5 else "FAIL"
        print(f"  [{i:>2}/{total}] {mark} {record['id']:<22} "
              f"kw={record['keyword_hit']:.2f} "
              f"missing={len(record['missing_keywords'])}", file=sys.stderr)

    report = evaluator.evaluate_all(progress=progress)
    flush()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwrote {args.output}", file=sys.stderr)

    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

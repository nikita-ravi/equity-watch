#!/usr/bin/env python
"""Diff two eval runs: what improved, what regressed, what stayed broken.

    python eval/compare.py eval/results/v1_unfiltered.json eval/results/v1_filtered.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = ["keyword_hit", "perfect_rate", "zero_rate", "ticker_hit", "year_hit",
           "section_hit", "top1_score"]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def pct(value):
    return f"{value * 100:5.1f}%"


def delta_str(delta, as_pct=True, width=7):
    if abs(delta) < 1e-9:
        return f"{'  --':>{width}}"
    body = f"{delta * 100:+.1f}pp" if as_pct else f"{delta:+.3f}"
    return f"{body:>{width}}"


def label(report, path):
    cfg = report.get("config", {})
    mode = "filtered" if cfg.get("use_filters") else "unfiltered"
    return f"{os.path.basename(path)} [{mode}, k={cfg.get('top_k')}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two eval result JSONs.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--show-unchanged", action="store_true")
    args = parser.parse_args(argv)

    base, cand = load(args.baseline), load(args.candidate)

    print("\n" + "=" * 84)
    print("  EVAL COMPARISON")
    print(f"    baseline  : {label(base, args.baseline)}")
    print(f"    candidate : {label(cand, args.candidate)}")
    print("=" * 84)

    print(f"\n  {'metric':<14} {'baseline':>10} {'candidate':>10} {'delta':>10}")
    print("  " + "-" * 48)
    for metric in METRICS:
        b, c = base["aggregate"].get(metric, 0), cand["aggregate"].get(metric, 0)
        as_pct = metric != "top1_score"
        b_s = pct(b) if as_pct else f"{b:.3f}"
        c_s = pct(c) if as_pct else f"{c:.3f}"
        print(f"  {metric:<14} {b_s:>10} {c_s:>10} "
              f"{delta_str(c - b, as_pct, 10)}")

    b_by = {r["id"]: r for r in base["results"]}
    c_by = {r["id"]: r for r in cand["results"]}

    improved, regressed, unchanged = [], [], []
    for qid in sorted(set(b_by) | set(c_by)):
        b_r, c_r = b_by.get(qid), c_by.get(qid)
        if b_r is None or c_r is None:
            print(f"\n  ! {qid} present in only one run -- skipped")
            continue
        delta = c_r["keyword_hit"] - b_r["keyword_hit"]
        row = (qid, b_r, c_r, delta)
        (improved if delta > 1e-9 else regressed if delta < -1e-9
         else unchanged).append(row)

    def dump(title, rows):
        print(f"\n{title} ({len(rows)})")
        if not rows:
            print("  none")
        for qid, b_r, c_r, delta in rows:
            print(f"  {qid:<22} {b_r['keyword_hit']:.2f} -> {c_r['keyword_hit']:.2f} "
                  f"({delta:+.2f})  {b_r['ticker']} {b_r['year']} {b_r['section']}")
            fixed = set(b_r["missing_keywords"]) - set(c_r["missing_keywords"])
            broke = set(c_r["missing_keywords"]) - set(b_r["missing_keywords"])
            if fixed:
                print(f"      now found : {sorted(fixed)}")
            if broke:
                print(f"      now missing: {sorted(broke)}")

    dump("IMPROVED", sorted(improved, key=lambda r: -r[3]))
    dump("REGRESSED", sorted(regressed, key=lambda r: r[3]))

    still = [r for r in unchanged if r[2]["keyword_hit"] < 0.5]
    print(f"\nSTILL FAILING in both ({len(still)})")
    if not still:
        print("  none")
    for qid, _b, c_r, _d in still:
        print(f"  {qid:<22} {c_r['keyword_hit']:.2f}  missing: {c_r['missing_keywords']}")

    if args.show_unchanged:
        dump("UNCHANGED", unchanged)

    print(f"\n  net: {len(improved)} improved, {len(regressed)} regressed, "
          f"{len(unchanged)} unchanged\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

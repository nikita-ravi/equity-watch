#!/usr/bin/env python
"""Answer-faithfulness eval: does the answer reflect what was retrieved?

The retrieval eval measures whether the right chunks came back. It cannot
measure whether the answer reflects them, which is the failure that actually
matters in finance -- a confidently wrong number passes keyword_hit as long as
the correct one also appears somewhere in the top-5.

This runs the agent over the ground-truth questions and applies the same
deterministic checks the guardrails apply at request time. Same rules, 65
answers instead of one.

TWO PHASES, deliberately separable
----------------------------------
Generating answers is slow, costs Groq calls, and is not perfectly repeatable.
Judging them is free and deterministic. Keeping them apart means a change to the
rules can be re-scored against answers that were already produced, rather than
paying for a fresh set -- and it means a judging bug cannot silently consume the
expensive half of the work.

    # generate and judge in one go
    python eval/faithfulness_eval.py --limit 10 --runs eval/results/runs.json \
                                     --output eval/results/faithfulness.json

    # re-judge those same answers after changing guardrails/rules.py
    python eval/faithfulness_eval.py --judge-only --runs eval/results/runs.json \
                                     --output eval/results/faithfulness.json

WHAT A FAILURE MEANS HERE
-------------------------
An `invented` figure is one that appears in no retrieved chunk, came from no
get_financials call, and cannot be derived from numbers that did. That is the
strongest claim this harness makes, and it is deliberately conservative: a
figure computed from two retrieved figures is recorded as `derived` and is NOT
a failure. Conflating the two would make the whole measurement meaningless.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import QUESTIONS                          # noqa: E402
from guardrails import (DERIVED, EXACT, INVENTED, RESCALED,  # noqa: E402
                        check_answer)


# --- phase 1: generate --------------------------------------------------------
def generate(questions, model=None, temperature=0, progress=True):
    """Run the agent over `questions`. Returns run records.

    A question that fails is recorded with its error rather than dropped, so the
    denominator stays honest -- a harness that silently skips failures reports a
    better pass rate than the system earns.
    """
    from agent.agent import run
    from retrieve.search import warm_up

    warm_up(retrieval="v4")
    runs = []
    for i, question in enumerate(questions, start=1):
        started = time.perf_counter()
        record = {
            "id": question.id,
            "question": question.question,
            "ticker": question.ticker,
            "year": question.year,
            "section": question.section,
            "question_type": question.question_type,
            "difficulty": question.difficulty,
        }
        try:
            answer, calls, _ = run(question.question, model=model,
                                   temperature=temperature)
            record.update(answer=answer, call_log=calls, error=None)
        except Exception as exc:                       # noqa: BLE001
            record.update(answer="", call_log=[], error=f"{type(exc).__name__}: {exc}")
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        runs.append(record)
        if progress:
            state = "ERROR" if record["error"] else f"{len(record['call_log'])} calls"
            print(f"  [{i:>3}/{len(questions)}] {question.id:<20} {state}",
                  file=sys.stderr)
    return runs


# --- phase 2: judge -----------------------------------------------------------
def judge(runs):
    """Apply the guardrail rules to each run. Returns per-run records."""
    judged = []
    for record in runs:
        out = {k: record[k] for k in
               ("id", "question", "ticker", "year", "section",
                "question_type", "difficulty")}
        if record.get("error"):
            out.update(error=record["error"], faithful=None, counts={},
                       findings=[], invented=[], answer="")
            judged.append(out)
            continue

        verdict = check_answer(record["answer"], record.get("call_log") or [])
        out.update(
            error=None,
            faithful=verdict.allowed,
            counts=dict(verdict.counts),
            findings=[{"rule": f.rule, "severity": f.severity,
                       "detail": f.detail, "evidence": f.evidence}
                      for f in verdict.findings],
            invented=[f.detail for f in verdict.blocking
                      if f.rule == "numeric_grounding"],
            answer=record["answer"],
        )
        judged.append(out)
    return judged


def aggregate(judged):
    scored = [r for r in judged if r["error"] is None]
    errors = [r for r in judged if r["error"] is not None]
    if not scored:
        return {"n": 0, "errors": len(errors)}

    def rate(predicate):
        return sum(1 for r in scored if predicate(r)) / len(scored)

    figures = defaultdict(int)
    for record in scored:
        for kind, count in record["counts"].items():
            figures[kind] += count

    checkable = sum(figures[k] for k in (EXACT, RESCALED, DERIVED, INVENTED))
    return {
        "n": len(scored),
        "errors": len(errors),
        # The headline: answers with no blocking finding at all.
        "faithful_rate": rate(lambda r: r["faithful"]),
        "invented_figure_rate": rate(
            lambda r: any(f["rule"] == "numeric_grounding" and f["severity"] == "block"
                          for f in r["findings"])),
        "bad_citation_rate": rate(
            lambda r: any(f["rule"] == "citations" and f["severity"] == "block"
                          for f in r["findings"])),
        "wrong_chunk_index_rate": rate(
            lambda r: any(f["rule"] == "citation_index" for f in r["findings"])),
        "no_citation_rate": rate(
            lambda r: any(f["rule"] == "citations_missing" for f in r["findings"])),
        "figures": {
            "checked": checkable,
            "exact": figures[EXACT],
            "rescaled": figures[RESCALED],
            "derived": figures[DERIVED],
            "invented": figures[INVENTED],
            "skipped": figures["skipped"],
            "grounded_share": ((figures[EXACT] + figures[RESCALED] + figures[DERIVED])
                               / checkable) if checkable else None,
        },
    }


def group_by(judged, key):
    buckets = defaultdict(list)
    for record in judged:
        buckets[record[key]].append(record)
    return {name: aggregate(rows) for name, rows in sorted(buckets.items())}


def report(judged, agg):
    print("\n" + "=" * 78)
    print("  ANSWER FAITHFULNESS")
    print("=" * 78)
    if not agg.get("n"):
        print("  no answers scored")
        return
    print(f"\n  answers scored           : {agg['n']}"
          + (f"  ({agg['errors']} errored)" if agg["errors"] else ""))
    print(f"  faithful (no blocks)     : {agg['faithful_rate'] * 100:5.1f}%")
    print(f"  with an invented figure  : {agg['invented_figure_rate'] * 100:5.1f}%")
    print(f"  citing a source never retrieved : {agg['bad_citation_rate'] * 100:5.1f}%")
    print(f"  citing a wrong chunk index      : {agg['wrong_chunk_index_rate'] * 100:5.1f}%")
    print(f"  citing nothing at all           : {agg['no_citation_rate'] * 100:5.1f}%")

    fig = agg["figures"]
    print(f"\n  figures checked          : {fig['checked']}"
          f"   ({fig['skipped']} skipped as years/ordinals)")
    print(f"    exact                  : {fig['exact']}")
    print(f"    rescaled               : {fig['rescaled']}")
    print(f"    derived (not a failure): {fig['derived']}")
    print(f"    INVENTED               : {fig['invented']}")
    if fig["grounded_share"] is not None:
        print(f"  grounded share           : {fig['grounded_share'] * 100:5.1f}%")

    failures = [r for r in judged if r["faithful"] is False]
    print(f"\n  UNFAITHFUL ANSWERS ({len(failures)})")
    for record in failures:
        print(f"    {record['id']:<20} {record['ticker']} FY{record['year']}")
        for finding in record["findings"]:
            if finding["severity"] == "block":
                print(f"      - {finding['rule']}: {finding['detail']}")

    errored = [r for r in judged if r["error"]]
    if errored:
        print(f"\n  ERRORED ({len(errored)})")
        for record in errored:
            print(f"    {record['id']:<20} {record['error'][:60]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N questions")
    parser.add_argument("--ticker", default=None, help="restrict to one ticker")
    parser.add_argument("--runs", default=None,
                        help="where to save (or load) raw agent runs")
    parser.add_argument("--judge-only", action="store_true",
                        help="re-judge an existing --runs file, no agent calls")
    parser.add_argument("--output", default=None, help="where to save results JSON")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.judge_only:
        if not args.runs or not os.path.exists(args.runs):
            parser.error("--judge-only needs an existing --runs file")
        runs = json.load(open(args.runs))["runs"]
        print(f"judging {len(runs)} saved runs (no agent calls)", file=sys.stderr)
    else:
        questions = list(QUESTIONS)
        if args.ticker:
            questions = [q for q in questions if q.ticker == args.ticker.upper()]
        if args.limit:
            questions = questions[:args.limit]
        print(f"running the agent over {len(questions)} questions", file=sys.stderr)
        runs = generate(questions, model=args.model)
        if args.runs:
            os.makedirs(os.path.dirname(os.path.abspath(args.runs)), exist_ok=True)
            json.dump({"runs": runs}, open(args.runs, "w"), indent=1, default=str)
            print(f"raw runs -> {args.runs}", file=sys.stderr)

    judged = judge(runs)
    agg = aggregate(judged)
    report(judged, agg)

    if args.output:
        payload = {
            "config": {"n_questions": len(judged), "model": args.model,
                       "judged_by": "guardrails.check_answer"},
            "aggregate": agg,
            "by_question_type": group_by(judged, "question_type"),
            "by_section": group_by(judged, "section"),
            "by_ticker": group_by(judged, "ticker"),
            "results": judged,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(payload, open(args.output, "w"), indent=1, default=str)
        print(f"\nresults -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

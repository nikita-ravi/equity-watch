#!/usr/bin/env python
"""Run one question through the agent and print the answer, tool calls and trace.

    python agent/run_agent.py "How did Apple's revenue change from FY2022 to FY2024?"

Requires GROQ_API_KEY. Set LANGCHAIN_API_KEY too and every run is traced to
LangSmith under the sec-rag-agent project.
"""

import argparse
import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before anything imports retrieve.tracing.
os.environ.setdefault("PHOENIX_ENABLED", "0")
# HF tokenizers fork-safety warning becomes a real hazard once LangGraph adds
# worker threads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_DEVICE", "cpu")


def load_env():
    """Load sec-rag/.env if present, then set LangSmith defaults."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            with open(env_path) as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"\''))

    # LangSmith reads either name depending on version; keep them in sync.
    if os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "sec-rag-agent")
        os.environ.setdefault("LANGSMITH_API_KEY", os.environ["LANGCHAIN_API_KEY"])
    else:
        # Tracing on without a key floods every run with 401 retries.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_TRACING", None)


load_env()

from agent.agent import run  # noqa: E402


def print_tool_calls(calls):
    print("\n" + "=" * 78)
    print(f"  TOOL CALLS ({len(calls)})")
    print("=" * 78)
    for i, call in enumerate(calls, start=1):
        proposed, used = call["proposed"], call["used"]
        print(f"\n  [{i}] search_10k(query={call['query']!r},")
        print(f"          ticker={proposed['ticker']!r}, year={proposed['year']!r})")
        if call["adjustments"]:
            for adj in call["adjustments"]:
                print(f"      VALIDATION: {adj['field']} {adj['proposed']!r} -> "
                      f"{adj['used']!r}  ({adj['reason']})")
        else:
            print(f"      validation: passed through "
                  f"(ticker={used['ticker']}, year={used['year']})")
        dropped = call.get("num_dropped_below_floor", 0)
        note = (f"  ({dropped} dropped below floor "
                f"{call.get('relevance_floor')})" if dropped else "")
        print(f"      -> {call['num_results']} chunks{note}")
        for r in call["results"][:3]:
            excerpt = " ".join(r["text"].split())[:74]
            print(f"         [{r['ticker']} | FY{r['year']} | {r['section']} | "
                  f"chunk {r['chunk_index']} | {r['score']:.3f}] {excerpt}...")


def trace_url():
    """Resolve the URL of the run that just finished.

    Spans are shipped by a background thread, so wait for the tracer queue to
    drain before asking LangSmith for the newest root run -- otherwise the run
    may not exist server-side yet.
    """
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return None
    project = os.environ.get("LANGCHAIN_PROJECT", "sec-rag-agent")
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass
    try:
        from langsmith import Client
        client = Client()
        runs = list(client.list_runs(project_name=project, is_root=True, limit=1))
        if runs:
            return client.get_run_url(run=runs[0])
    except Exception as exc:  # tracing must never break the run
        logging.getLogger(__name__).debug("could not resolve trace URL: %s", exc)
    return f"https://smith.langchain.com/  (project: {project})"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask the 10-K agent a question.")
    parser.add_argument("query", help="natural language question")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--no-guardrails", action="store_true",
                        help="skip the input and output checks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)-7s %(name)s: %(message)s",
                        stream=sys.stderr)

    # Load the embedder, BM25 index and reranker on the MAIN thread. Loading a
    # torch model lazily inside a LangGraph worker thread is what hung the first
    # runs.
    from retrieve.search import warm_up
    warm_up(retrieval="v4")

    print("\n" + "=" * 78)
    print(f"  QUERY: {args.query}")
    print("=" * 78)

    # Input checks run BEFORE the agent, so a question no filing can answer
    # costs nothing. A blocked question is answered with the reason rather
    # than silently redirected -- validate() would clamp an out-of-range year
    # and answer a different question than the one asked.
    if not args.no_guardrails:
        from guardrails import check_question
        gate = check_question(args.query)
        for finding in gate.findings:
            print(f"  [guardrail:{finding.rule}] {finding.detail}", file=sys.stderr)
        if not gate.allowed:
            print("\n" + "=" * 78)
            print("  REFUSED")
            print("=" * 78 + "\n")
            for finding in gate.blocking:
                print(finding.evidence or finding.detail)
            print()
            return 0

    try:
        answer, calls, _messages = run(args.query, model=args.model,
                                       temperature=args.temperature)
    except BaseException:
        print("\n" + "!" * 78, file=sys.stderr)
        print("  AGENT RUN FAILED", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        traceback.print_exc()
        from agent.tools import CALL_LOG
        print(f"\ntool calls completed before failure: {len(CALL_LOG)}", file=sys.stderr)
        for c in CALL_LOG:
            print(f"  - {c['query']!r} ticker={c['used']['ticker']} "
                  f"year={c['used']['year']} -> {c['num_results']} chunks", file=sys.stderr)
        # Re-raising here unwinds through torch's atexit handlers and aborts the
        # process (SIGABRT), which macOS reports as a crash. Exit cleanly.
        os._exit(1)

    print_tool_calls(calls)

    print("\n" + "=" * 78)
    print("  ANSWER")
    print("=" * 78 + "\n")
    print(answer)

    # Output checks verify what the prompt only asked for. Findings are
    # reported rather than suppressing the answer: a blocked answer with no
    # explanation is less useful than the answer plus the reason to doubt it.
    if not args.no_guardrails:
        from guardrails import check_answer, format_report
        verdict = check_answer(answer, calls)
        print("\n" + "-" * 78)
        print("  GUARDRAILS" + ("" if verdict.allowed else "  -- UNGROUNDED CLAIMS FOUND"))
        print("-" * 78)
        print(format_report(verdict))

    url = trace_url()
    print("\n" + "-" * 78)
    if url:
        print(f"  LangSmith trace: {url}")
    else:
        print("  LangSmith trace: disabled (LANGCHAIN_API_KEY not set)")
    print("-" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

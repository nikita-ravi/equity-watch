#!/usr/bin/env python
"""CLI for dense retrieval over the ingested 10-K corpus.

    python retrieve.py "What are Apple's main risk factors?"
    python retrieve.py "Apple risk factors" --top-k 10
    python retrieve.py "JPM credit risk" --ticker JPM --year 2023
    python retrieve.py "supply chain disruption" --section "Item 1A"

Sparse and incorporated-by-reference chunks are excluded by default; pass
--include-sparse / --include-by-ref to search them too.

Tracing goes to Phoenix. Add --ui to launch the Phoenix app and keep it open so
traces can be inspected and compared across queries.
"""

import argparse
import logging
import sys
import textwrap

import config
from retrieve.search import search, warm_up
from retrieve.tracing import flush, init_tracing, start_phoenix_ui

SNIPPET_CHARS = 300


def format_result(index, result, snippet_chars=SNIPPET_CHARS):
    text = " ".join(result.text.split())
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rstrip() + "..."
    body = textwrap.fill(f'"{text}"', width=96,
                         initial_indent="      ", subsequent_indent="      ")
    return f"  [{index}] {result.header()}\n{body}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Search the sec_10k collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="natural language query")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--retrieval", choices=["v1", "v2", "v3", "v4"], default="v1",
                        help="v1=dense, v2=hybrid RRF, v3=+query parsing, v4=+cross-encoder rerank")
    parser.add_argument("--ticker", default=None, help="e.g. AAPL")
    parser.add_argument("--year", type=int, default=None, help="e.g. 2024")
    parser.add_argument("--section", default=None, help='e.g. "Item 1A"')
    parser.add_argument("--include-sparse", action="store_true",
                        help="do not exclude is_sparse chunks")
    parser.add_argument("--include-by-ref", action="store_true",
                        help="do not exclude incorporated_by_reference chunks")
    parser.add_argument("--ui", action="store_true",
                        help="launch the Phoenix UI and hold it open")
    parser.add_argument("--no-trace", action="store_true", help="disable tracing")
    parser.add_argument("--snippet-chars", type=int, default=SNIPPET_CHARS)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )

    ui_url = None
    if args.ui:
        # Launch before init_tracing so the collector endpoint is already listening.
        ui_url = start_phoenix_ui()
    if not args.no_trace:
        init_tracing()

    warm_up(retrieval=args.retrieval)
    results = search(
        args.query,
        top_k=args.top_k,
        ticker=args.ticker,
        year=args.year,
        section=args.section,
        exclude_sparse=not args.include_sparse,
        exclude_by_ref=not args.include_by_ref,
        retrieval=args.retrieval,
    )

    active = [f"{k}={v}" for k, v in (("ticker", args.ticker), ("year", args.year),
                                      ("section", args.section)) if v is not None]
    print(f'\nquery: "{args.query}"'
          + (f"  |  filters: {', '.join(active)}" if active else "")
          + f"  |  {len(results)} results\n")
    if not results:
        print("  no matching chunks -- try relaxing the filters "
              "(--include-sparse / --include-by-ref)\n")
    for i, result in enumerate(results, start=1):
        print(format_result(i, result, args.snippet_chars))
        print()

    flush()

    if ui_url:
        print(f"Phoenix UI: {ui_url}  (project: {config.PHOENIX_PROJECT})")
        print("Ctrl-C to exit and shut the UI down.")
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

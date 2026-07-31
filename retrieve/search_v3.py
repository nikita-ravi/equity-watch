"""v3: structured query parsing in front of v2 hybrid retrieval.

v2 showed that filters are worth ~3pp on keyword_hit and, more importantly, take
year precision from ~40% of slots to 100%. v3's job is to derive those filters
from the question itself so callers do not have to supply them by hand.

Explicit arguments always win over parsed ones, so every v2 call site keeps
working unchanged.
"""

import logging
import time

import config
from retrieve.hybrid import search_hybrid
from retrieve.query_parser import parse_query
from retrieve.tracing import get_tracer

log = logging.getLogger(__name__)


def search_v3(query, top_k=config.DEFAULT_TOP_K, ticker=None, year=None,
              section=None, exclude_sparse=True, exclude_by_ref=True,
              use_section_filter=None, use_clean_query=None):
    """Parse the query into filters, then run hybrid retrieval underneath."""
    if use_section_filter is None:
        use_section_filter = config.V3_SECTION_FILTER
    if use_clean_query is None:
        use_clean_query = config.V3_USE_CLEAN_QUERY

    tracer = get_tracer()
    started = time.perf_counter()

    # One root span for the whole operation. Without it, query.parse and the
    # retrieve.search underneath land in two unrelated traces, so the UI shows a
    # rewritten query with no sign of the question it came from.
    with tracer.start_as_current_span("retrieve.v3") as root:
        _set(root, "openinference.span.kind", "CHAIN")
        _set(root, "input.value", query)
        _set(root, "retrieval.version", "v3")

        with tracer.start_as_current_span("query.parse") as span:
            parsed = parse_query(query)
            _set(span, "openinference.span.kind", "CHAIN")
            _set(span, "input.value", query)
            _set(span, "parse.ticker", parsed.ticker or "None")
            _set(span, "parse.year", parsed.year or "None")
            _set(span, "parse.section", parsed.section or "None")
            _set(span, "parse.clean_query", parsed.clean_query)
            for key, value in parsed.evidence.items():
                _set(span, f"parse.evidence.{key}", str(value))

        # Caller-supplied values take precedence over anything parsed.
        effective_ticker = ticker if ticker is not None else parsed.ticker
        effective_year = year if year is not None else parsed.year
        effective_section = section if section is not None else (
            parsed.section if use_section_filter else None)

        search_text = parsed.clean_query if use_clean_query else query

        # Both the question and the text actually searched, so the trace never
        # makes you guess which one produced the results.
        _set(root, "retrieval.original_query", query)
        _set(root, "retrieval.search_text", search_text)
        _set(root, "retrieval.used_clean_query", bool(use_clean_query))
        _set(root, "parse.ticker", parsed.ticker or "None")
        _set(root, "parse.year", parsed.year or "None")
        _set(root, "parse.section", parsed.section or "None")
        _set(root, "retrieval.applied_ticker", effective_ticker or "None")
        _set(root, "retrieval.applied_year", effective_year or "None")
        _set(root, "retrieval.applied_section", effective_section or "None")

        results = search_hybrid(
            search_text,
            top_k=top_k,
            ticker=effective_ticker,
            year=effective_year,
            section=effective_section,
            exclude_sparse=exclude_sparse,
            exclude_by_ref=exclude_by_ref,
        )
        _set(root, "retrieval.result_count", len(results))

    # Record what the parse actually did, so eval JSON and Phoenix traces can be
    # audited without re-running the parser.
    for result in results:
        result.metadata = dict(result.metadata)
        result.metadata["parsed"] = {
            "ticker": parsed.ticker,
            "year": parsed.year,
            "section": parsed.section,
            "clean_query": parsed.clean_query,
            "applied_ticker": effective_ticker,
            "applied_year": effective_year,
            "applied_section": effective_section,
            "evidence": parsed.evidence,
        }

    log.debug("search_v3(%r) parsed=%s -> %d hits in %.1fms",
              query, (parsed.ticker, parsed.year, parsed.section),
              len(results), (time.perf_counter() - started) * 1000)
    return results


def _set(span, key, value):
    if span is not None and hasattr(span, "set_attribute"):
        try:
            span.set_attribute(key, value)
        except Exception:
            pass

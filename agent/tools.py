"""The retrieval tool the agent calls.

Thin wrapper over the existing v4 pipeline. The only real logic here is deciding
whether the agent's ticker/year win over the query parser's, and making sure a
bad argument degrades into a broader search instead of an exception.
"""

import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import config
from agent.validation import validate
from retrieve.search import search

log = logging.getLogger(__name__)

# Every tool call, for the test harness to print and for debugging.
CALL_LOG = []


class RetrieveArgs(BaseModel):
    query: str = Field(
        description="The specific question to search for. Include the company "
                    "name and fiscal year in the text when you know them.")
    ticker: str = Field(
        default=None,
        description="Optional. One of AAPL, MSFT, GOOGL, JPM, BAC, XOM, WMT, "
                    "JNJ, PFE, HD. Omit if the query text already names the "
                    "company.")
    year: int = Field(
        default=None,
        description="Optional fiscal year: 2022, 2023 or 2024 only.")
    section: str = Field(
        default=None,
        description="Optional. Restrict retrieval to a specific 10-K section. "
                    "Use '1A' for Risk Factors, '7' for MD&A, '8' for Financial "
                    "Statements, '1' for Business, '1B' for Unresolved Staff "
                    "Comments, '9A' for Controls and Procedures. Omit when the "
                    "answer may span multiple sections.")


@tool("search_10k", args_schema=RetrieveArgs)
def search_10k(query, ticker=None, year=None, section=None):
    """Search SEC 10-K filings for 10 companies (AAPL, MSFT, GOOGL, JPM, BAC,
    XOM, WMT, JNJ, PFE, HD) across fiscal years 2022-2024.

    Returns verbatim excerpts from the filings with their source metadata. Call
    once per company and once per fiscal year -- this tool searches a single
    filing scope at a time and cannot compare across companies or years itself.
    """
    checked = validate(ticker, year)

    # Explicit args override the parser; when absent, v3's parser reads the
    # company and year out of the query text itself.
    raw = search(
        query,
        top_k=config.AGENT_TOP_K,
        ticker=checked.ticker,
        year=checked.year,
        section=section,
        retrieval="v4",
        candidates=config.AGENT_RERANK_CANDIDATES,
    )

    # Drop weak matches before the model sees them. Passing a 0.001-relevance
    # chunk to the LLM invites it to cite text it did not use.
    floor = config.RELEVANCE_FLOOR
    results = [r for r in raw
               if (r.rerank_score if r.rerank_score is not None else r.score) >= floor]
    dropped = len(raw) - len(results)

    record = {
        "query": query,
        "proposed": {"ticker": ticker, "year": year, "section": section},
        "used": {"ticker": checked.ticker, "year": checked.year, "section": section},
        "adjustments": checked.adjustments,
        "num_results": len(results),
        "num_dropped_below_floor": dropped,
        "relevance_floor": floor,
        "results": [{
            "ticker": r.ticker,
            "year": r.year,
            "section": r.section,
            "section_label": r.section_label,
            "chunk_index": r.chunk_index,
            "score": r.rerank_score if r.rerank_score is not None else r.score,
            "text": r.text,
        } for r in results],
    }
    CALL_LOG.append(record)
    log.info("search_10k(query=%r, ticker=%s, year=%s, section=%s) -> %d chunks | validation: %s",
             query, ticker, year, section, len(results), checked.summary())

    if not raw:
        return ("No chunks matched. The corpus only covers AAPL, MSFT, GOOGL, "
                "JPM, BAC, XOM, WMT, JNJ, PFE, HD for fiscal years 2022-2024.")

    if not results:
        scope = f"{checked.ticker or 'the corpus'}"
        if checked.year:
            scope += f" FY{checked.year}"
        return (f"No confident matches for {scope}: all {len(raw)} candidates scored "
                f"below the {floor} relevance floor. Do NOT answer this part of the "
                f"question from these results -- state that the filings did not "
                f"return confident evidence for it.")

    return _format(results, checked)


def _format(results, checked):
    """Render chunks so the model can quote them and cite them precisely."""
    lines = []
    if checked.adjustments:
        lines.append(f"[note: arguments adjusted -- {checked.summary()}]")
    for r in results:
        score = r.rerank_score if r.rerank_score is not None else r.score
        lines.append(
            f"[{r.ticker} | FY{r.year} | {r.section} — {r.section_label} | "
            f"chunk {r.chunk_index} | relevance {score:.3f}]\n{r.text}"
        )
    return "\n\n---\n\n".join(lines)


def reset_call_log():
    CALL_LOG.clear()


# --- corpus inventory ---------------------------------------------------------
_INVENTORY = None


def _load_inventory():
    """Scan the collection once and cache what it contains.

    Selective payload: the text field is ~22 MB across the corpus and is not
    needed to describe what is in it.
    """
    global _INVENTORY
    if _INVENTORY is not None:
        return _INVENTORY

    from qdrant_client import QdrantClient
    client = QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)

    by_ticker = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=2048,
            offset=offset,
            with_payload=["ticker", "year", "company", "section", "period_of_report"],
            with_vectors=False,
        )
        for point in points:
            p = point.payload or {}
            entry = by_ticker.setdefault(p.get("ticker"), {
                "company": p.get("company"), "years": {}})
            year = entry["years"].setdefault(p.get("year"), {
                "chunks": 0, "sections": set(), "period": p.get("period_of_report")})
            year["chunks"] += 1
            year["sections"].add(p.get("section"))
        if offset is None:
            break

    _INVENTORY = by_ticker
    return _INVENTORY


@tool("list_corpus")
def list_corpus() -> str:
    """List what this corpus actually contains: which companies, which fiscal
    years, how many chunks and sections per filing.

    Call this FIRST when the question is vague about which company or year to
    look at ("healthcare companies", "the banks", "the most recent filings"), or
    when you need to check whether something exists before searching for it.
    """
    inventory = _load_inventory()

    lines = ["SEC 10-K corpus inventory:", ""]
    lines.append(f"{'TICKER':<7} {'COMPANY':<28} {'FY':<6} {'CHUNKS':>7} "
                 f"{'SECTIONS':>9}  PERIOD END")
    lines.append("-" * 78)
    for ticker in config.COMPANIES:
        entry = inventory.get(ticker)
        if not entry:
            lines.append(f"{ticker:<7} (not in corpus)")
            continue
        for year in sorted(entry["years"]):
            data = entry["years"][year]
            lines.append(
                f"{ticker:<7} {str(entry['company'])[:27]:<28} {year:<6} "
                f"{data['chunks']:>7} {len(data['sections']):>9}  {data['period']}")

    lines.append("")
    lines.append("Fiscal-year labelling: 'year' above is the calendar year the "
                 "filing period ENDS in.")
    for ticker, offset in config.FISCAL_YEAR_OFFSET.items():
        examples = ", ".join(f"{ticker} fiscal {y - offset} = year {y}"
                             for y in config.YEARS)
        lines.append(f"  {ticker} names its fiscal year after the STARTING year, "
                     f"so its labels shift by {offset:+d}: {examples}.")
    lines.append("Nothing outside these companies and years exists in the corpus.")
    return "\n".join(lines)


# --- reported financials (XBRL) ----------------------------------------------
# Kept separate from search_10k on purpose. A figure retrieved as text has to
# survive being flattened out of a table, ranked, and then read correctly by the
# model; the same figure arrives here already typed and already reconciled
# against the statement it came from. Retrieval stays the right tool for
# narrative and for anything the consolidated statements do not carry --
# segment splits, store counts, bank-specific lines.
_FINANCIALS = {}


def _financials_for(ticker, year):
    """Extract (and cache) reconciled figures for one filing."""
    key = (ticker, int(year))
    if key in _FINANCIALS:
        return _FINANCIALS[key]

    from ingest.fetch import find_filings, init_edgar
    from ingest.financials import extract

    init_edgar(config.SEC_IDENTITY)
    refs = find_filings(ticker, [int(year)])
    if not refs:
        _FINANCIALS[key] = None
        return None
    ref = refs[0]
    _FINANCIALS[key] = extract(ref.filing.obj().financials, ticker, int(year),
                               ref.filing_id, ref.period_of_report)
    return _FINANCIALS[key]


class FinancialsArgs(BaseModel):
    ticker: str = Field(
        description="One of AAPL, MSFT, GOOGL, JPM, BAC, XOM, WMT, JNJ, PFE, HD.")
    year: int = Field(description="Fiscal year: 2022, 2023 or 2024 only.")


@tool("get_financials", args_schema=FinancialsArgs)
def get_financials(ticker, year):
    """Reported consolidated financial figures for ONE company and ONE fiscal
    year, taken from the filing's own XBRL data rather than from retrieved text.

    Use this for top-line figures -- revenue, net income, operating income,
    total assets, total liabilities, shareholders' equity, operating cash flow,
    capital expenditures. It is exact and needs no interpretation.

    It does NOT hold segment or product breakdowns (iPhone revenue, Intelligent
    Cloud revenue, YouTube ads), counts that live in narrative text (employees,
    stores, countries), or bank-specific lines (net interest income, provision
    for credit losses). Use search_10k for those.

    A metric listed as "could not be extracted" is a gap in this tool, NOT a
    statement about the filing -- the company may well report it. Fall back to
    search_10k for those, and never fill one in from memory.
    """
    checked = validate(ticker, year)
    if not checked.ticker or not checked.year:
        return ("get_financials needs both a valid ticker and a fiscal year "
                "(2022-2024). " + checked.summary())

    data = _financials_for(checked.ticker, checked.year)
    if data is None:
        return f"No 10-K on EDGAR for {checked.ticker} FY{checked.year}."

    usable = data.reconciled
    CALL_LOG.append({
        "tool": "get_financials",
        "used": {"ticker": checked.ticker, "year": checked.year},
        "filing_id": data.filing_id,
        "metrics": {k: v for k, v in usable.items()},
        "unavailable": [k for k, f in data.figures.items()
                        if f.status not in ("ok", "statement")],
    })

    if not usable:
        return (f"No figures could be reconciled for {checked.ticker} "
                f"FY{checked.year}. Do not state any number for this filing.")

    lines = []
    if checked.adjustments:
        # search_10k surfaces this; dropping it here would answer a different
        # question than the one asked, silently.
        lines.append(f"[note: arguments adjusted -- {checked.summary()}]")
    lines.append(f"{checked.ticker} FY{checked.year} reported figures "
                 f"(10-K {data.filing_id}, period {data.period_of_report}) "
                 f"-- US$ millions:")
    for metric in ("revenue", "net_income", "operating_income", "total_assets",
                   "total_liabilities", "stockholders_equity",
                   "operating_cash_flow", "capital_expenditures"):
        if metric in usable:
            lines.append(f"  {metric:22} {usable[metric] / 1e6:>14,.0f}")
    missing = [k for k, f in data.figures.items()
               if f.status not in ("ok", "statement")]
    if missing:
        # Deliberately NOT "not reported". Apple reports both operating cash
        # flow and capital expenditures; they are simply not extractable here.
        # Wording this as a fact about the filing would put a false statement
        # in the model's mouth.
        lines.append(f"  could not be extracted for this filing "
                     f"(try search_10k): {', '.join(sorted(missing))}")
    return "\n".join(lines)

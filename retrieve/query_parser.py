"""Pull ticker / year / section out of a natural language question.

Rules-based, not LLM-based: no LLM client is wired into this project yet, and
the entity space here is closed (10 known companies, 3 known years, 23 known
item sections), which is exactly the case where rules beat a model call.

The two subtleties that make this more than string matching are handled
explicitly below: fiscal-year labels that disagree with the corpus year tag, and
comparative questions that name two years.
"""

import datetime
import re
from dataclasses import dataclass, field

import config


@dataclass
class ParsedQuery:
    ticker: str = None
    year: int = None
    section: str = None
    clean_query: str = ""
    # What the parser keyed off, for tracing and debugging.
    evidence: dict = field(default_factory=dict)


# --- section intent -----------------------------------------------------------
# (phrase, section). Longest matching phrase wins, so "market risk" beats "risk"
# and "internal control" beats "control". Order breaks ties.
_SECTION_PHRASES = [
    # Item 7A must precede the generic risk and financial rules.
    ("quantitative and qualitative disclosures", "Item 7A"),
    ("interest rate sensitivity", "Item 7A"),
    ("interest rate risk", "Item 7A"),
    ("market risk", "Item 7A"),
    ("sensitivity", "Item 7A"),
    ("hedging", "Item 7A"),
    ("derivatives", "Item 7A"),
    ("quantitative", "Item 7A"),
    ("qualitative", "Item 7A"),

    ("risk factors", "Item 1A"),
    ("risks", "Item 1A"),
    ("risk", "Item 1A"),

    ("legal proceedings", "Item 3"),
    ("litigation", "Item 3"),
    ("lawsuit", "Item 3"),
    ("proceedings", "Item 3"),
    ("legal", "Item 3"),

    ("properties", "Item 2"),
    ("facilities", "Item 2"),
    ("headquarters", "Item 2"),
    ("locations", "Item 2"),

    ("internal control", "Item 9A"),
    ("internal controls", "Item 9A"),
    ("disclosure controls", "Item 9A"),
    ("controls and procedures", "Item 9A"),

    ("financial statements", "Item 8"),
    ("balance sheet", "Item 8"),
    ("cash flow", "Item 8"),
    ("cash flows", "Item 8"),
    ("notes to the financial statements", "Item 8"),

    ("management's discussion", "Item 7"),
    ("management discussion", "Item 7"),
    ("md&a", "Item 7"),
    ("financial condition", "Item 7"),
    ("financial results", "Item 7"),
    ("gross margin", "Item 7"),
    ("operating income", "Item 7"),
    ("net income", "Item 7"),
    ("revenues", "Item 7"),
    ("revenue", "Item 7"),
    ("earnings", "Item 7"),
    ("expenses", "Item 7"),
    ("margin", "Item 7"),
    ("profit", "Item 7"),
    ("income", "Item 7"),
    ("sales", "Item 7"),

    ("business", "Item 1"),
    ("operations", "Item 1"),
    ("overview", "Item 1"),
    ("segments", "Item 1"),
    ("products", "Item 1"),
    ("employees", "Item 1"),
    ("stores", "Item 2"),
]

# An explicit "Item 7A" / "item 1a" reference in the query beats any inference.
_EXPLICIT_ITEM = re.compile(r"\bitem\s+(\d{1,2}\s*[a-c]?)\b", re.IGNORECASE)

# "2023 Form 10-K" / "2023 annual report" names the FILING, so that year is the
# corpus year regardless of any other year mentioned or fiscal-label offset.
_FILING_REFERENCE = re.compile(
    r"\b((?:19|20)\d{2})\s+(?:form\s+)?10-?k\b|\b((?:19|20)\d{2})\s+annual\s+report\b",
    re.IGNORECASE,
)

_FISCAL_YEAR = re.compile(
    r"(?:fiscal(?:\s+year)?|fy)\s*((?:19|20)\d{2})", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_RELATIVE_YEAR = re.compile(
    r"\b(last\s+year|prior\s+year|previous\s+year|this\s+year|current\s+year)\b",
    re.IGNORECASE)


def parse_query(query):
    """Extract ticker, year and section from a natural language query."""
    text = query or ""
    lowered = text.lower()
    evidence = {}

    ticker, ticker_spans = _parse_ticker(text, lowered, evidence)
    year = _parse_year(text, lowered, ticker, evidence)
    section = _parse_section(lowered, evidence)

    return ParsedQuery(
        ticker=ticker,
        year=year,
        section=section,
        clean_query=_clean(text, ticker_spans),
        evidence=evidence,
    )


# --- ticker -------------------------------------------------------------------
def _parse_ticker(text, lowered, evidence):
    """Longest alias first, so "bank of america" is not shadowed by a shorter one."""
    candidates = []
    for ticker, aliases in config.COMPANY_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", lowered):
                candidates.append((len(alias), match.start(), match.end(), ticker, alias))

    # Bare ticker symbols ("AAPL", "JPM"). Checked after aliases but ranked the
    # same way; a symbol is an unambiguous mention.
    for ticker in config.COMPANIES:
        for match in re.finditer(rf"\b{re.escape(ticker.lower())}\b", lowered):
            candidates.append((len(ticker), match.start(), match.end(), ticker, ticker))

    if not candidates:
        return None, []

    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = candidates[0]
    evidence["ticker_matched"] = best[4]

    # Spans for every mention of the winning ticker, for clean_query removal.
    # Aliases overlap ("johnson & johnson" contains "johnson"), and deleting
    # overlapping ranges corrupts the surrounding text, so merge them first.
    spans = _merge_spans([(c[1], c[2]) for c in candidates if c[3] == best[3]])
    return best[3], spans


def _merge_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# --- year ---------------------------------------------------------------------
def _parse_year(text, lowered, ticker, evidence):
    """Resolve the corpus year tag the question is really about."""
    # 1. An explicit filing reference wins outright and takes no fiscal offset --
    #    "its 2023 Form 10-K" names the document, not a fiscal period.
    filing = _FILING_REFERENCE.search(text)
    if filing:
        year = int(filing.group(1) or filing.group(2))
        evidence["year_source"] = "filing_reference"
        return _clamp(year, evidence)

    fiscal = [int(y) for y in _FISCAL_YEAR.findall(text)]
    bare = [int(y) for y in _BARE_YEAR.findall(text)]
    years = fiscal or bare

    if years:
        # Comparative questions name two years ("2022 vs 2023"). The later one is
        # the filing being asked about; the earlier appears inside it as the
        # prior-year comparison column.
        year = max(years)
        evidence["year_source"] = "fiscal_label" if fiscal else "bare_year"
        evidence["years_seen"] = sorted(set(years))

        # Fiscal labels are the filer's own naming; bare years in a question are
        # usually calendar and already match the tag. Only shift labelled ones.
        if fiscal and ticker in config.FISCAL_YEAR_OFFSET:
            offset = config.FISCAL_YEAR_OFFSET[ticker]
            evidence["fiscal_offset"] = offset
            year += offset
        return _clamp(year, evidence)

    relative = _RELATIVE_YEAR.search(lowered)
    if relative:
        phrase = relative.group(1).lower()
        current = datetime.date.today().year
        year = current if phrase in ("this year", "current year") else current - 1
        evidence["year_source"] = f"relative:{phrase}"
        return _clamp(year, evidence)

    return None


def _clamp(year, evidence):
    """Keep the year inside the ingested range.

    Relative phrases resolve against today's date, which can land outside the
    corpus entirely; filtering to a year with no chunks would return nothing at
    all, which is strictly worse than filtering to the nearest ingested year.
    """
    low, high = min(config.YEARS), max(config.YEARS)
    if year < low or year > high:
        evidence["year_clamped_from"] = year
        return max(low, min(high, year))
    return year


# --- section ------------------------------------------------------------------
def _parse_section(lowered, evidence):
    explicit = _EXPLICIT_ITEM.search(lowered)
    if explicit:
        raw = re.sub(r"\s+", "", explicit.group(1)).upper()
        section = f"Item {raw}"
        if section in config.SECTION_LABELS:
            evidence["section_source"] = "explicit"
            return section

    best = None
    for phrase, section in _SECTION_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            if best is None or len(phrase) > best[0]:
                best = (len(phrase), phrase, section)

    if best is None:
        return None
    evidence["section_source"] = f"phrase:{best[1]}"
    return best[2]


# --- clean query --------------------------------------------------------------
def _clean(text, ticker_spans):
    """Strip the entity references the filters now handle.

    The company name and year are enforced by metadata filters, so leaving them
    in the embedded text only adds generic tokens that BM25 matches against every
    other filer -- the exact cross-company contamination v3 is meant to stop.
    """
    out = text
    for start, end in sorted(ticker_spans, reverse=True):
        out = out[:start] + " " + out[end:]

    out = _FILING_REFERENCE.sub(" ", out)
    out = _FISCAL_YEAR.sub(" ", out)
    out = _BARE_YEAR.sub(" ", out)
    out = _RELATIVE_YEAR.sub(" ", out)
    # Dangling possessives and connectives left behind by the removals.
    out = re.sub(r"\s*'s\b", "", out)
    out = re.sub(r"\b(?:in|for|during|of|at the end of)\s+(?:the\s+)?\?", "?", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([?.,])", r"\1", out)
    return out or text

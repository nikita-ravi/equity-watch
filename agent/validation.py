"""Coerce whatever the LLM proposes into arguments the retriever can actually use.

An LLM will confidently pass ticker="Apple", ticker="APPL", or year=2025. None of
those exist in the corpus, and a hard failure would end the run. Everything here
degrades instead: fuzzy-match the ticker, clamp the year, and if a value cannot
be salvaged, drop that filter and search unfiltered rather than erroring.
"""

import difflib
import logging
import re
from dataclasses import dataclass, field

import config

log = logging.getLogger(__name__)

VALID_TICKERS = set(config.COMPANIES)
VALID_YEARS = set(config.YEARS)

# Flattened alias -> ticker, reusing the v3 parser's mapping so the agent and the
# query parser can never disagree about what "Alphabet" means.
_ALIAS_TO_TICKER = {
    alias.lower(): ticker
    for ticker, aliases in config.COMPANY_ALIASES.items()
    for alias in aliases
}


@dataclass
class Validated:
    ticker: str = None
    year: int = None
    # (field, proposed, used, reason) for every value that was changed or dropped.
    adjustments: list = field(default_factory=list)

    def note(self, field_name, proposed, used, reason):
        self.adjustments.append({
            "field": field_name,
            "proposed": proposed,
            "used": used,
            "reason": reason,
        })
        log.info("validation: %s proposed=%r used=%r (%s)",
                 field_name, proposed, used, reason)

    def summary(self):
        if not self.adjustments:
            return "none"
        return "; ".join(
            f"{a['field']} {a['proposed']!r}->{a['used']!r} ({a['reason']})"
            for a in self.adjustments
        )


def validate(ticker=None, year=None):
    """Return a Validated with usable ticker/year, or None for either."""
    out = Validated()
    out.ticker = _validate_ticker(ticker, out)
    out.year = _validate_year(year, out)
    return out


def _validate_ticker(proposed, out):
    if proposed is None or (isinstance(proposed, str) and not proposed.strip()):
        return None

    raw = str(proposed).strip()
    upper = raw.upper()
    if upper in VALID_TICKERS:
        return upper

    # A company name rather than a symbol ("Apple", "Bank of America").
    lowered = raw.lower()
    if lowered in _ALIAS_TO_TICKER:
        resolved = _ALIAS_TO_TICKER[lowered]
        out.note("ticker", proposed, resolved, "matched company alias")
        return resolved

    # An alias embedded in a longer string ("Apple Inc.", "the Home Depot").
    for alias, ticker in sorted(_ALIAS_TO_TICKER.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            out.note("ticker", proposed, ticker, f"contains alias {alias!r}")
            return ticker

    # A typo'd symbol ("APPL") or a near-miss company name ("Microsft").
    pool = list(VALID_TICKERS) + list(_ALIAS_TO_TICKER)
    close = difflib.get_close_matches(lowered, [p.lower() for p in pool],
                                      n=1, cutoff=0.75)
    if close:
        match = close[0]
        resolved = _ALIAS_TO_TICKER.get(match, match.upper())
        if resolved in VALID_TICKERS:
            out.note("ticker", proposed, resolved, f"fuzzy match to {match!r}")
            return resolved

    # Unsalvageable -- search unfiltered rather than failing the tool call.
    out.note("ticker", proposed, None, "not in corpus; searching all companies")
    return None


def _validate_year(proposed, out):
    if proposed is None or (isinstance(proposed, str) and not str(proposed).strip()):
        return None

    try:
        value = int(str(proposed).strip())
    except (TypeError, ValueError):
        # "FY2023" or "fiscal 2023" rather than an int.
        digits = re.search(r"(19|20)\d{2}", str(proposed))
        if not digits:
            out.note("year", proposed, None, "unparseable; searching all years")
            return None
        value = int(digits.group(0))
        out.note("year", proposed, value, "parsed year out of string")

    if value in VALID_YEARS:
        return value

    low, high = min(VALID_YEARS), max(VALID_YEARS)
    clamped = max(low, min(high, value))
    out.note("year", proposed, clamped,
             f"outside corpus range {low}-{high}; clamped")
    return clamped

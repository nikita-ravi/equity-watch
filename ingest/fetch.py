"""Locate the right 10-K filing per company per fiscal year via edgartools."""

import logging
from dataclasses import dataclass

from edgar import Company, set_identity

log = logging.getLogger(__name__)

_identity_set = False


def init_edgar(identity):
    """Set the SEC User-Agent identity. Required before any EDGAR request."""
    global _identity_set
    if not _identity_set:
        set_identity(identity)
        _identity_set = True


@dataclass
class FilingRef:
    ticker: str
    company: str
    year: int
    filing_id: str          # accession number, e.g. 0000320193-24-000123
    period_of_report: str    # ISO date, e.g. 2024-09-28
    filing_date: str
    filing: object           # the edgartools Filing, for lazy section parsing


def find_filings(ticker, years):
    """Return one FilingRef per requested fiscal year that EDGAR actually has.

    Fiscal year is taken from period_of_report, so FY2024 means "the 10-K
    covering the period ending in 2024" regardless of when it was filed. Where a
    year has multiple 10-Ks (a refiling), the most recently filed one wins.
    Amendments (10-K/A) are excluded.
    """
    company = Company(ticker)
    company_name = company.name
    wanted = set(years)
    best = {}

    for filing in company.get_filings(form="10-K"):
        if filing.form != "10-K":  # drops 10-K/A
            continue
        period = getattr(filing, "period_of_report", None)
        if not period:
            continue
        try:
            year = int(str(period)[:4])
        except ValueError:
            continue
        if year not in wanted:
            continue
        incumbent = best.get(year)
        if incumbent is None or str(filing.filing_date) > str(incumbent.filing_date):
            best[year] = filing

    refs = []
    for year in sorted(years):
        filing = best.get(year)
        if filing is None:
            log.warning("[%s %s] no 10-K found on EDGAR for this fiscal year -- skipping",
                        ticker, year)
            continue
        refs.append(FilingRef(
            ticker=ticker,
            company=company_name,
            year=year,
            filing_id=filing.accession_no,
            period_of_report=str(filing.period_of_report),
            filing_date=str(filing.filing_date),
            filing=filing,
        ))
    return refs

"""Reported financial figures, taken from the filing's own XBRL tagging.

Retrieval answers narrative questions well and numeric ones only by accident: a
figure has to survive being flattened out of an Item 8 table into a chunk, be
retrieved, and then be read correctly off that flattened text. The same filing
already carries the number as structured, typed data, so for top-line figures
there is no reason to go through the text at all.

Scope, measured rather than assumed: of the 21 numerical questions in
eval/dataset.py this covers 7 outright. The rest are segment-level ("iPhone net
sales"), counts that live in narrative ("how many stores"), or bank-specific
lines (net interest income) -- none of which the consolidated statements hold.
This is a complement to retrieval, not a replacement for it.

WHY NOT companyfacts
--------------------
`Company.get_facts()` returns every tagged fact including dimensional
breakdowns -- segments, products, subsidiaries -- with no flag distinguishing a
consolidated total from a member. Selecting naively off it returned $43,715M for
Apple's FY2023 total assets, where the real figure is $352,583M; also $663M for
Pfizer and $21.4B for J&J revenue. Every one of those looked like a perfectly
ordinary number. The statement-level API used here is already dimension-filtered
and period-scoped.

THE VALIDATION GATE
-------------------
The library's normalised getters are right most of the time and quietly wrong
some of the time, by picking a neighbouring line:

    WMT  get_total_liabilities -> "Total liabilities, redeemable noncontrolling
         interest, and equity", which equals total ASSETS by the balance-sheet
         identity
    WMT  get_revenue -> "Net sales" (605,881), excluding $5.4B of membership
         income, against total revenues of 611,289
    PFE  get_revenue -> "Product revenues" (50,914) against total revenues of
         58,496

So nothing is trusted on the getter's word. Every figure is reconciled against
the statement text it should have come from, and a figure that does not
reconcile is stored as missing -- never as a number. A gap is visible and gets
fixed; a plausible wrong number is neither.
"""

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# metric -> (getter name, ordered regexes for the line it must reconcile
#            against, which statement to look in, comparison mode)
#
# Patterns are tried in order and the first that matches a line carrying a
# number wins, so the most specific total comes first. Two recurring traps
# shaped these:
#
#   The balance-sheet identity. "Total liabilities and equity" and WMT's "Total
#   liabilities, redeemable noncontrolling interest, and equity" both equal
#   total ASSETS. Matching either turns a wrong number into a confirmed one, so
#   the liabilities patterns exclude any line mentioning equity.
#
#   Consolidated vs attributable. XOM and PFE present "Net income (loss)
#   including noncontrolling interests" and "Net income before allocation to
#   noncontrolling interests" above the attributable figure the getters return.
#   The attributable line is matched first and the pre-allocation variants are
#   excluded outright.
#
# mode "abs" compares magnitudes: cash-flow statements present outflows in
# parentheses while the getters return them positive, which is a presentation
# convention and not a disagreement about the value.
# Every line that sits near "net income" on an income statement but is not it:
# the pre-allocation total, the slice belonging to minority holders, either half
# of a continuing/discontinued split, and the per-share figures.
_NOT_NEAR_NET_INCOME = (
    r"(?!.*(?:noncontrolling|before allocation|continuing operations"
    r"|discontinued operations|per share|per common share))"
)

METRICS = {
    "revenue": (
        "get_revenue",
        [r"^\s*Total revenues?\b", r"^\s*Total net sales\b",
         r"^\s*Sales to customers\b", r"^\s*Net sales\b", r"^\s*Revenues?\b"],
        "income", "exact",
    ),
    "net_income": (
        "get_net_income",
        # Attributable-to-the-registrant first (XOM reports the minority slice
        # on its own line directly beneath it, and WMT's consolidated total
        # sits directly above), then the plain line for filers with no split.
        [r"^\s*Consolidated net income attributable to" + _NOT_NEAR_NET_INCOME,
         r"^\s*Net income(?: \(loss\))? attributable to" + _NOT_NEAR_NET_INCOME,
         r"^\s*Net earnings attributable to" + _NOT_NEAR_NET_INCOME,
         r"^\s*Net income" + _NOT_NEAR_NET_INCOME + r"\b",
         r"^\s*Net earnings" + _NOT_NEAR_NET_INCOME + r"\b"],
        "income", "exact",
    ),
    "operating_income": (
        "get_operating_income",
        [r"^\s*Operating income\b", r"^\s*Income from operations\b"],
        "income", "exact",
    ),
    "total_assets": (
        "get_total_assets",
        [r"^\s*Total assets\b"],
        "balance", "exact",
    ),
    "total_liabilities": (
        "get_total_liabilities",
        [r"^\s*Total liabilities\s*:?\s*$", r"^\s*Total liabilities\b(?!.*equity)"],
        "balance", "exact",
    ),
    "stockholders_equity": (
        "get_stockholders_equity",
        # "(?!.*liabilit)" keeps "Total liabilities and stockholders' equity"
        # out -- that line is the balance-sheet identity and equals total
        # assets. The optional parenthetical absorbs HD's "Total stockholders'
        # (deficit) equity". "abs" because a deficit is presented unsigned
        # under a "(deficit)" label while the getter returns it negative: HD
        # FY2022 is -1,696 to the getter and 1,696 on the statement.
        [r"^\s*Total\b(?!.*liabilit).{0,40}?(?:share|stock)holders.{0,3}\s*"
         r"(?:\([^)]*\)\s*)?equity\b",
         r"^\s*Total equity\b"],
        "balance", "abs",
    ),
    "operating_cash_flow": (
        "get_operating_cash_flow",
        # The consolidated total only -- never the continuing/discontinued
        # split that PFE reports above it.
        [r"^\s*Net cash .{0,30}?operating activities\s*:?\s*$",
         r"^\s*Net cash .{0,30}?operating activities\b"
         r"(?!.*(?:continuing|discontinued))"],
        "cashflow", "abs",
    ),
    "capital_expenditures": (
        "get_capital_expenditures",
        [r"^\s*(?:Payments for a|A)dditions to property\b",
         r"^\s*Capital expenditures?\b",
         r"^\s*Purchases? of property\b",
         r"^\s*Payments for property\b"],
        "cashflow", "abs",
    ),
}

# Values in the rendered statements are in millions; the getters return units.
_SCALE = 1e6
# Relative tolerance when reconciling. Wide enough for rounding in the rendered
# statement, far too tight for a neighbouring line to slip through: the WMT and
# PFE errors are 0.9% and 13% adrift respectively.
_TOLERANCE = 0.005

# Statuses whose value may be used. "unreconciled", "absent" and "no_line" all
# mean "no number", and are deliberately indistinguishable to a caller.
_USABLE = ("ok", "statement")

# Metrics where an explicit total line on the statement outranks the getter.
# Only revenue so far: it is the one metric whose statement line names the total
# unambiguously ("Total revenues") while filers also report components directly
# above it, which is exactly the pair the getters confuse.
PREFER_STATEMENT = {"revenue"}


@dataclass
class Figure:
    metric: str
    value: float | None
    status: str          # "ok" | "statement" | "unreconciled" | "absent" | "no_line"
    statement_value: float | None = None
    note: str = ""


@dataclass
class FilingFinancials:
    ticker: str
    year: int
    filing_id: str
    period_of_report: str
    figures: dict = field(default_factory=dict)

    def get(self, metric):
        """The value for `metric`, or None when it did not reconcile."""
        figure = self.figures.get(metric)
        return figure.value if figure and figure.status in _USABLE else None

    @property
    def reconciled(self):
        return {k: v.value for k, v in self.figures.items() if v.status in _USABLE}


def _statement_text(financials, which):
    getter = {
        "income": "income_statement",
        "balance": "balance_sheet",
        "cashflow": "cashflow_statement",
    }[which]
    try:
        return str(getattr(financials, getter)())
    except Exception as exc:
        log.debug("could not render %s: %s", which, exc)
        return ""


def _first_number(line):
    """The first numeric column on a statement line, in millions.

    Parentheses mean negative, which matters for capital expenditures and for
    any line a filer presents as a deduction.
    """
    stripped = re.sub(r"^[^$\d(]*", "", line)
    match = re.search(r"(\(?)\$?\s*(-?[\d,]+(?:\.\d+)?)(\)?)", stripped)
    if not match:
        return None
    try:
        value = float(match.group(2).replace(",", ""))
    except ValueError:
        return None
    if match.group(1) and match.group(3):
        value = -value
    return value


def _line_value(text, patterns):
    """Find the first line matching any pattern and return its first number."""
    lines = text.splitlines()
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for line in lines:
            if compiled.search(line):
                value = _first_number(line)
                if value is not None:
                    return value, line.strip()
    return None, ""


def extract(financials, ticker, year, filing_id, period_of_report):
    """Extract and reconcile every metric for one filing."""
    out = FilingFinancials(ticker=ticker, year=year, filing_id=filing_id,
                           period_of_report=period_of_report)
    rendered = {}

    for metric, (getter_name, patterns, which, mode) in METRICS.items():
        prefer_statement = metric in PREFER_STATEMENT
        if which not in rendered:
            rendered[which] = _statement_text(financials, which)
        text = rendered[which]

        try:
            raw = getattr(financials, getter_name)()
        except Exception as exc:
            out.figures[metric] = Figure(metric, None, "absent", note=str(exc)[:80])
            continue
        if not isinstance(raw, (int, float)):
            out.figures[metric] = Figure(metric, None, "absent",
                                         note="getter returned no value")
            continue

        truth, line = _line_value(text, patterns)
        if truth is None:
            # The filing does not present this line under any name we know.
            # Recorded as a gap, not as the getter's unverified number.
            out.figures[metric] = Figure(metric, None, "no_line",
                                         note="no matching line in statement")
            continue

        expected = truth * _SCALE
        got = float(raw)
        if mode == "abs":
            got, expected = abs(got), abs(expected)
        agrees = abs(got - expected) <= _TOLERANCE * max(abs(expected), 1.0)
        if agrees:
            out.figures[metric] = Figure(metric, float(raw), "ok", truth)
        elif prefer_statement:
            # The two disagree and the statement line is the one this metric
            # explicitly asked for, so the statement wins and the getter's
            # number is discarded. This is not a tie-break: every observed
            # disagreement is the getter returning a *component* of the line
            # the pattern matched -- WMT "Net sales" (605,881) under "Total
            # revenues" (611,289), PFE "Product revenues" (50,914) under
            # "Total revenues" (58,496). The disagreement is still logged,
            # because a new one means a pattern is pointing at the wrong line.
            out.figures[metric] = Figure(
                metric, expected, "statement", truth,
                note=f"getter disagreed ({got:,.0f}); took statement line "
                     f"{line[:60]!r}")
            log.warning("[%s %s] %s: getter=%s statement=%s -- took statement (%s)",
                        ticker, year, metric, f"{got:,.0f}", f"{expected:,.0f}",
                        line[:60])
        else:
            out.figures[metric] = Figure(
                metric, None, "unreconciled", truth,
                note=f"getter={got:,.0f} statement={expected:,.0f} "
                     f"line={line[:60]!r}")
            log.warning("[%s %s] %s did not reconcile: getter=%s statement=%s (%s)",
                        ticker, year, metric, f"{got:,.0f}",
                        f"{expected:,.0f}", line[:60])

    return out

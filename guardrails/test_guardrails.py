"""Tests for the guardrails. Run: python -m guardrails.test_guardrails

Deliberately dependency-free (no pytest) so the checks can be verified in a
clean environment. The cases that matter most are the ones where a naive
implementation gets it wrong:

  - a derived figure ("a decrease of $3,293 million") must NOT be reported as
    invented; that conflation is what makes this kind of check useless
  - a figure written with a scale word must match a bare figure in a table,
    because a 10-K states scale once in a heading and then omits it
  - a percentage must NOT match another percentage on significant digits: 8%
    is not evidence for 80%, and accepting it would pass a tenfold error
"""

import sys

from guardrails import (DERIVED, EXACT, INVENTED, RESCALED, check_answer,
                        check_question, classify, extract_numbers)

CHUNKS = [
    {"results": [
        {"ticker": "AAPL", "year": 2022, "section": "Item 8", "chunk_index": 44,
         "text": "Total net sales $394,328 $365,817 $274,515 Net income 99,803"},
        {"ticker": "AAPL", "year": 2022, "section": "Item 7", "chunk_index": 0,
         "text": "Total net sales increased 8% or $28.5 billion during 2022"},
    ]},
    {"results": [
        {"ticker": "AAPL", "year": 2024, "section": "Item 8", "chunk_index": 11,
         "text": "Total net sales $391,035 $383,285 $394,328"},
    ]},
]
GROUNDED = [394328.0, 365817.0, 274515.0, 99803.0, 391035.0, 383285.0, 8.0]

_passed, _failed = 0, []


def check(name, got, want):
    global _passed
    if got == want:
        _passed += 1
    else:
        _failed.append(f"{name}: got {got!r}, want {want!r}")


def classify_one(text):
    for number in extract_numbers(text):
        if not number.trivial:
            return classify(number, GROUNDED)[0]
    return "no-number"


# --- numeric classification ---------------------------------------------------
check("bare figure in a chunk", classify_one("$391,035"), EXACT)
check("figure with scale word", classify_one("$394,328 million"), EXACT)
check("rounded to billions", classify_one("$394.3 billion"), RESCALED)
check("difference of two figures", classify_one("a decrease of $3,293 million"), DERIVED)
check("sum of two figures", classify_one("$785,363 million combined"), DERIVED)
check("percentage change", classify_one("fell 0.8%"), DERIVED)
check("invented figure", classify_one("revenue of $450,000 million"), INVENTED)
check("tenfold percentage error", classify_one("grew 80%"), INVENTED)
check("year is skipped", classify_one("in 2023"), "no-number")
check("ordinal is skipped", classify_one("the 3 sources"), "no-number")

# --- output checks ------------------------------------------------------------
GOOD = """Apple's net sales fell from $394,328 million in FY2022 to $391,035
million in FY2024, a decrease of $3,293 million or 0.8%.

Sources:
[AAPL | FY2022 | Item 8 | chunk 44] "Total net sales $394,328"
[AAPL | FY2024 | Item 8 | chunk 11] "Total net sales $391,035"
"""
verdict = check_answer(GOOD, CHUNKS)
check("grounded answer allowed", verdict.allowed, True)
check("grounded answer has no blocks", len(verdict.blocking), 0)
check("no invented figures", verdict.counts.get(INVENTED), 0)

BAD_NUMBER = """Net sales were $412,900 million in FY2024.

Sources:
[AAPL | FY2024 | Item 8 | chunk 11] "x"
"""
verdict = check_answer(BAD_NUMBER, CHUNKS)
check("invented figure blocks", verdict.allowed, False)

BAD_CITE = """Net sales were $391,035 million in FY2024.

Sources:
[AAPL | FY2023 | Item 8 | chunk 99] "never retrieved"
"""
verdict = check_answer(BAD_CITE, CHUNKS)
check("fabricated citation blocks", verdict.allowed, False)

NO_CITE = """Net sales were $391,035 million in FY2024."""
verdict = check_answer(NO_CITE, CHUNKS)
check("missing citations warns, does not block", verdict.allowed, True)
check("missing citations is flagged",
      any(f.rule == "citations_missing" for f in verdict.findings), True)

WRONG_INDEX = """Net sales were $391,035 million in FY2024.

Sources:
[AAPL | FY2024 | Item 8 | chunk 77] "right filing and section, wrong chunk"
"""
verdict = check_answer(WRONG_INDEX, CHUNKS)
check("wrong chunk index warns, does not block", verdict.allowed, True)
check("wrong chunk index has its own rule",
      any(f.rule == "citation_index" for f in verdict.findings), True)

# Sources block quotes chunk text verbatim; its numbers must not be counted.
check("sources block excluded from counts",
      check_answer(GOOD, CHUNKS).counts.get(EXACT), 2)

# --- input checks -------------------------------------------------------------
for question, allowed in [
    ("What are Apple's main risk factors?", True),
    ("What was JPMorgan's net income for full-year 2022?", True),
    ("Compare Apple and Microsoft R&D in 2024", True),
    ("What does Alphabet's 10-K say about AI risk?", True),
    ("Should I buy Apple stock?", False),
    ("Is Pfizer a good investment right now?", False),
    ("Will AAPL stock go up next quarter?", False),
    ("Give me a price target for XOM", False),
    ("What was Apple's revenue in 2019?", False),
]:
    check(f"input: {question[:38]}", check_question(question).allowed, allowed)

check("out-of-corpus company warns but allows",
      check_question("What was Tesla's revenue in 2023?").allowed, True)
check("out-of-corpus company is flagged",
      any(f.rule == "scope" for f in check_question("What was Tesla's revenue in 2023?").findings),
      True)

# --- report -------------------------------------------------------------------
print(f"\n{_passed} passed, {len(_failed)} failed")
for failure in _failed:
    print(f"  FAIL  {failure}")
sys.exit(1 if _failed else 0)

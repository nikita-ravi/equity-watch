"""Checks applied to what the model wrote, against what the tools returned.

The system prompt already *asks* for grounding. This *verifies* it. That gap is
the whole reason this module exists: an instruction is a request, and a request
is not a guarantee.

Two rules, both deterministic:

  numeric_grounding  every figure in the answer must appear in a retrieved
                     chunk, come from get_financials, or be derivable from
                     numbers that do

  citations          every source the answer cites must be a chunk that was
                     actually retrieved on this turn

Neither consults an LLM. A model that invented $391,035M will confirm it when
asked to check itself, because the error and the check share the same faulty
premise -- so the check is arithmetic and set membership instead.
"""

import re

from guardrails.rules import (DERIVED, EXACT, INVENTED, RESCALED, Verdict,
                              classify, extract_numbers)

# The citation format the system prompt demands:
#   [AAPL | FY2022 | Item 8 | chunk 44] "excerpt"
#   [AAPL | FY2023 | reported financials | revenue] 383,285
_CITATION = re.compile(
    r"\[\s*(?P<ticker>[A-Z]{1,5})\s*\|\s*FY(?P<year>\d{4})\s*\|\s*"
    r"(?P<section>[^|\]]+?)\s*(?:\|\s*(?P<locator>[^\]]+?)\s*)?\]",
    re.IGNORECASE,
)
_CHUNK_INDEX = re.compile(r"chunk\s*(\d+)", re.IGNORECASE)


def grounded_values(call_log):
    """Every number the tools actually put in front of the model this turn.

    Drawn from the retrieved chunk texts and from any get_financials figures,
    because both are legitimate sources for a number in the answer.
    """
    values = []
    for entry in call_log or []:
        for result in entry.get("results") or []:
            for number in extract_numbers(result.get("text", "")):
                values.extend(number.candidates)
        for value in (entry.get("metrics") or {}).values():
            if isinstance(value, (int, float)):
                # get_financials reports units; statements are in millions, so
                # both readings are legitimate for the model to have quoted.
                values.append(float(value))
                values.append(float(value) / 1e6)
    return values


def retrieved_chunks(call_log):
    """{(ticker, year, section, chunk_index)} actually returned this turn."""
    keys = set()
    for entry in call_log or []:
        for result in entry.get("results") or []:
            keys.add((
                str(result.get("ticker", "")).upper(),
                int(result.get("year") or 0),
                str(result.get("section", "")).strip().lower(),
                result.get("chunk_index"),
            ))
    return keys


def check_numeric_grounding(answer, call_log, verdict=None):
    """Classify every figure in `answer`. Invented figures block."""
    verdict = verdict or Verdict()
    values = grounded_values(call_log)

    counts = {EXACT: 0, RESCALED: 0, DERIVED: 0, INVENTED: 0, "skipped": 0}
    # Strip the Sources block first: it quotes chunk text verbatim, so its
    # numbers are grounded by construction and would pad the counts.
    body = _strip_sources(answer)

    for number in extract_numbers(body):
        if number.trivial:
            counts["skipped"] += 1
            continue
        kind, why = classify(number, values)
        counts[kind] += 1
        if kind == INVENTED:
            verdict.add("numeric_grounding", "block",
                        f"{number.raw} is not supported by anything retrieved",
                        evidence=why)
        elif kind == RESCALED:
            verdict.add("numeric_grounding", "info",
                        f"{number.raw} matched only on significant digits",
                        evidence=why)

    if not values and counts[INVENTED]:
        # Nothing was retrieved at all, so every figure is unsupported. Say that
        # once rather than once per number.
        verdict.add("numeric_grounding", "block",
                    "the answer states figures but no chunks were retrieved")

    verdict.counts.update(counts)
    return verdict


def check_citations(answer, call_log, verdict=None):
    """Every cited chunk must have actually been retrieved this turn."""
    verdict = verdict or Verdict()
    available = retrieved_chunks(call_log)
    seen = 0

    for match in _CITATION.finditer(answer or ""):
        section = (match.group("section") or "").strip().lower()
        # Citations of get_financials name the metric, not a chunk; those are
        # validated by the numeric rule instead.
        if "financial" in section and "item" not in section:
            continue
        seen += 1

        locator = match.group("locator") or ""
        index_match = _CHUNK_INDEX.search(locator) or _CHUNK_INDEX.search(section)
        key = (
            match.group("ticker").upper(),
            int(match.group("year")),
            section,
            int(index_match.group(1)) if index_match else None,
        )
        if key in available:
            continue
        # Fall back to a looser match: the right filing and section, even if the
        # chunk index was mangled. Worth a warning, not a block.
        loose = {(t, y, s) for t, y, s, _ in available}
        if (key[0], key[1], key[2]) in loose:
            # The filing and section were retrieved but not that chunk index.
            # A distinct rule, not a generic citation warning: a garbled index
            # on a real source is a different failure from citing a source
            # that was never retrieved, and from citing nothing at all.
            # Collapsing them into one counter hides which is happening.
            verdict.add("citation_index", "warn",
                        f"cited {match.group(0)} with a chunk index that was "
                        f"not retrieved")
        else:
            verdict.add("citations", "block",
                        f"cited {match.group(0)}, which was not retrieved "
                        f"on this turn")

    if seen == 0 and available:
        verdict.add("citations_missing", "warn",
                    "chunks were retrieved but the answer cites none of them")
    return verdict


def check_answer(answer, call_log):
    """Run every output rule. Returns a Verdict."""
    verdict = Verdict()
    check_numeric_grounding(answer, call_log, verdict)
    check_citations(answer, call_log, verdict)
    return verdict


def _strip_sources(answer):
    """Drop the trailing Sources block, which quotes chunks verbatim."""
    match = re.search(r"\n\s*sources?\s*:", answer or "", re.IGNORECASE)
    return answer[:match.start()] if match else (answer or "")


def format_report(verdict):
    """One-screen summary, for the CLI."""
    counts = verdict.counts
    lines = []
    if counts:
        lines.append(
            f"  figures: {counts.get(EXACT, 0)} exact, "
            f"{counts.get(RESCALED, 0)} rescaled, "
            f"{counts.get(DERIVED, 0)} derived, "
            f"{counts.get(INVENTED, 0)} unsupported "
            f"({counts.get('skipped', 0)} skipped as years/ordinals)")
    for finding in verdict.findings:
        mark = {"block": "BLOCK", "warn": "warn ", "info": "info "}[finding.severity]
        lines.append(f"  [{mark}] {finding.rule}: {finding.detail}")
        if finding.evidence:
            lines.append(f"          {finding.evidence}")
    if not lines:
        lines.append("  clean")
    return "\n".join(lines)

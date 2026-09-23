"""Deterministic guardrails over the agent's input and output.

The retrieval layer already refuses to hand the model weak evidence (the 0.3
relevance floor) and refuses to hand it an unverified number (the financials
reconciliation gate). What was missing was anything that inspected the ANSWER.
This package closes that: the system prompt asks for grounding, and these
checks verify it.

    from guardrails import check_question, check_answer

    verdict = check_question(q)
    if not verdict.allowed:
        ...                                  # refuse, with the reason

    answer, call_log, _ = agent.run(q)
    verdict = check_answer(answer, call_log)  # every figure, every citation
"""

from guardrails.input_check import ADVICE_RESPONSE, check_question
from guardrails.output_check import (check_answer, check_citations,
                                     check_numeric_grounding, format_report,
                                     grounded_values)
from guardrails.rules import (DERIVED, EXACT, INVENTED, RESCALED, Finding,
                              Number, Verdict, classify, extract_numbers)

__all__ = [
    "check_question", "check_answer", "check_numeric_grounding",
    "check_citations", "format_report", "grounded_values",
    "classify", "extract_numbers", "Verdict", "Finding", "Number",
    "EXACT", "RESCALED", "DERIVED", "INVENTED", "ADVICE_RESPONSE",
]

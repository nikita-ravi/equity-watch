"""Checks applied to the question before the agent runs.

Two rules, both rule-based for the same reason the query parser is: the space
being checked is closed and enumerable, so a classifier would be guessing at
something that can simply be written down.

  scope     the corpus is 10 companies and 3 fiscal years. A question about a
            company or year outside that is answerable only from the model's
            training data, which is exactly what this project exists not to do.

  advice    "should I buy Apple?" is not a retrieval question. Nothing in a
            10-K answers it, and a system that reads filings and then offers a
            recommendation is doing something different from what it claims.

Deliberately NOT here: prompt-injection and jailbreak filtering. The corpus is
public, audited SEC filings; the injection surface is theoretical, and a
classifier for it would be copied from a system with a different threat model
rather than built for this one.
"""

import re

import config
from guardrails.rules import Verdict

# Asking for a recommendation, a rating, or a prediction. These are checked
# against the question only -- an ANSWER may legitimately contain "should" when
# quoting a filing's forward-looking language.
_ADVICE = re.compile(
    r"\b(should i (buy|sell|invest|hold|own|add)"
    # "is it / is Pfizer / is this a good investment" -- the subject varies, so
    # allow any short subject between the verb and the judgement.
    r"|is\s+\S+(\s+\S+)?\s+a\s+(good|bad|solid|safe|smart)\s+"
    r"(buy|sell|investment|stock|time|pick|bet)"
    r"|(good|bad) (investment|buy|stock) (right now|today|now)"
    r"|worth (buying|investing|a buy|owning)"
    r"|(recommend|advise)\s+(a\s+|me\s+|buying|selling|investing)"
    r"|what should i (buy|invest|do with|pick)"
    r"|price target"
    r"|will\s+.{0,25}(stock|share|price).{0,25}(go up|go down|rise|fall|beat|outperform)"
    r"|(predict|forecast)\s+.{0,25}(price|stock|return|performance)"
    r"|(out|under)perform the market)\b",
    re.IGNORECASE,
)

ADVICE_RESPONSE = (
    "This system reports what SEC 10-K filings say; it does not give investment "
    "advice or price predictions. Ask what a filing discloses -- its risk "
    "factors, reported figures, or management's discussion -- and I will answer "
    "from the filing text."
)

# A capitalised token that looks like a ticker. Matched only when the question
# is otherwise unattributable, so "the Company" style phrasing is unaffected.
_TICKER_LIKE = re.compile(r"\b([A-Z]{2,5})\b")
# Any capitalised word, for spotting a company the corpus does not hold.
_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]{2,})\b")
# Capitalised only because they open a sentence.
_SENTENCE_START = {
    "what", "which", "when", "where", "how", "why", "who", "does", "did",
    "compare", "list", "show", "tell", "give", "summarise", "summarize",
    "describe", "explain", "according", "based", "the", "for", "from", "in",
    "is", "are", "was", "were", "can", "could", "would", "should", "risk",
    "risks", "revenue", "revenues", "net", "total", "cash", "item",
}
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

_ALIASES = {
    alias.lower(): ticker
    for ticker, aliases in config.COMPANY_ALIASES.items()
    for alias in aliases
}


def check_question(question):
    """Screen a question before it reaches the agent. Returns a Verdict."""
    verdict = Verdict()
    text = question or ""

    if _ADVICE.search(text):
        verdict.add("advice", "block",
                    "asks for investment advice or a prediction, which no "
                    "filing answers",
                    evidence=ADVICE_RESPONSE)

    _check_scope(text, verdict)
    return verdict


def _check_scope(text, verdict):
    """Flag companies and years the corpus cannot speak to."""
    lowered = text.lower()

    # Years first: a year outside the corpus is unambiguous.
    years = {int(m.group(0)) for m in _YEAR.finditer(text)}
    outside = sorted(y for y in years if y not in set(config.YEARS))
    if outside and not (years & set(config.YEARS)):
        asked = ", ".join(str(y) for y in outside)
        verdict.add("scope", "block",
                    f"asks about {asked}; the corpus covers "
                    f"{min(config.YEARS)}-{max(config.YEARS)} only",
                    # Refusing is deliberate. validate() would CLAMP an
                    # out-of-range year to the nearest one in the corpus and
                    # answer about that instead -- a different question than
                    # the one asked, answered without the user noticing.
                    evidence=(
                        f"This corpus holds 10-K filings for fiscal years "
                        f"{min(config.YEARS)}-{max(config.YEARS)} only, so there "
                        f"is nothing on file for {asked}. Ask about a year in "
                        f"that range and I will answer from the filing."))

    # Rather than enumerate every company NOT in the corpus -- an open set --
    # detect the absence of one that IS. A question that names a proper noun as
    # its subject but matches none of the ten is asking about something this
    # corpus cannot answer, whether that noun is "Tesla" or "Rivian".
    if any(alias in lowered for alias in _ALIASES):
        return
    if any(t in set(config.COMPANIES) for t in _TICKER_LIKE.findall(text)):
        return

    subjects = [w for w in _PROPER_NOUN.findall(text)
                if w.upper() not in _STOPWORD_CAPS and w.lower() not in _SENTENCE_START]
    if subjects:
        verdict.add("scope", "warn",
                    f"names {', '.join(sorted(set(subjects))[:3])}, which is not "
                    f"one of the ten companies in the corpus",
                    evidence=f"corpus: {', '.join(config.COMPANIES)}")


# Capitalised tokens that are not tickers and would otherwise be flagged.
_STOPWORD_CAPS = {
    "SEC", "EDGAR", "MD", "A", "I", "US", "USA", "GAAP", "CEO", "CFO", "R",
    "AI", "EPS", "FY", "Q1", "Q2", "Q3", "Q4", "IT", "ESG", "IP", "TV",
}

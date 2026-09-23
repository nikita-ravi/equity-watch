"""The checks, as code.

Everything here is deterministic. That is the point: the failure this project
most needs to catch is a confidently wrong number, and a model that invented
$391,035M will happily confirm it when asked to check its own work. Self-critique
fails precisely where factual errors live, so the number check is arithmetic
rather than a second opinion.

Numbers are compared two ways, and the verdict records which one matched:

  EXACT     the same canonical value, within a relative tolerance. "$394,328"
            in the answer against "394,328" in a chunk.

  RESCALED  the same significant digits at a different magnitude. A 10-K table
            headed "in millions" prints 394,328; an answer may write that as
            "$394.3 billion". These are the same figure, so refusing to match
            them would flag every correctly-rounded restatement as invented.
            Matching on the mantissa is deliberately permissive -- it is the
            looser of the two, and it is labelled so a reader can tell.

A number that matches neither is not immediately a hallucination. It may be
DERIVED: computed from two numbers that ARE grounded. "Revenue fell $3,293M" is
the difference of two retrieved figures and is the model doing arithmetic, not
inventing data. Derived numbers are counted separately and never reported as
invented -- conflating the two was the single easiest way to make this check
useless.
"""

import re
from dataclasses import dataclass, field

# A numeric token, optionally with currency, separators, decimals, a percent
# sign, or a trailing scale word. Ordered so the longest form wins.
_NUMBER = re.compile(
    r"""
    (?P<neg>\(|-)?                     # parenthesised or signed negative
    \$?\s?
    (?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234,567.89
             |\d+\.\d+                        # 1234.56
             |\d+)                            # 1234
    \)?
    \s*
    (?P<scale>billion|bn|b\b|million|mm|m\b|thousand|k\b|trillion|tn|t\b)?
    (?P<pct>\s*%|\s*percent)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}

# Corpus fiscal years and the calendar years a filing legitimately mentions.
# These appear in almost every chunk and every answer; treating them as
# financial figures would drown the real signal.
_YEAR = re.compile(r"^(19|20)\d\d$")

# Below this, a bare integer is far more likely to be a rank, a list marker, an
# item number or a count of sources than a reported figure. Percentages and
# decimals are exempt -- "3.4%" is a real claim.
_TRIVIAL_INTEGER_MAX = 12

EXACT = "exact"
RESCALED = "rescaled"
DERIVED = "derived"
INVENTED = "invented"

# Relative tolerance for an exact match: absorbs rounding in a rendered
# statement without letting a neighbouring line slip through.
_TOLERANCE = 0.005
# Significant digits that must agree for a rescaled match.
_MANTISSA_DIGITS = 3


@dataclass(frozen=True)
class Number:
    """One numeric token, with everything needed to judge and to explain it.

    `candidates` exists because a 10-K states scale once, in a table heading
    ("in millions"), and then prints bare numbers underneath. A chunk therefore
    carries 394,328 while an answer about the same figure writes "$394,328
    million" or "$394.3 billion". Comparing only the scaled value makes those
    three different numbers, and -- worse -- breaks derivation: "a decrease of
    $3,293 million" would not be recognised as 394,328 - 391,035, so a correct
    piece of arithmetic would be reported as invented. Carrying both the scaled
    and unscaled readings and matching on either avoids that.
    """
    raw: str            # exactly as written, for the report
    value: float        # canonical value after sign, separators and scale
    candidates: tuple   # every reading worth matching on
    is_pct: bool
    is_year: bool

    @property
    def trivial(self):
        """True for tokens that carry no factual claim worth checking."""
        if self.is_year:
            return True
        if self.is_pct:
            return False
        return float(self.value).is_integer() and abs(self.value) <= _TRIVIAL_INTEGER_MAX


def extract_numbers(text):
    """Every numeric token in `text`, in order of appearance."""
    out = []
    for match in _NUMBER.finditer(text or ""):
        digits = match.group("value")
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue

        bare = digits.replace(",", "")
        is_year = bool(_YEAR.match(bare)) and "," not in digits and "." not in digits

        scale = (match.group("scale") or "").lower().strip()
        multiplier = _SCALES.get(scale, 1.0) if (scale and not is_year) else 1.0
        sign = -1.0 if match.group("neg") else 1.0

        unscaled = sign * value
        scaled = sign * value * multiplier
        # Both readings when a scale word was present, one when it was not.
        candidates = (scaled,) if multiplier == 1.0 else (scaled, unscaled)

        out.append(Number(
            raw=match.group(0).strip(),
            value=scaled,
            candidates=candidates,
            is_pct=bool(match.group("pct")),
            is_year=is_year,
        ))
    return out


def _mantissa(value, digits=_MANTISSA_DIGITS):
    """Significant digits of `value`, with magnitude discarded.

    394,328 -> 394;  394.3e9 -> 394.  This is what lets a figure printed in a
    "in millions" table match the same figure written as billions in prose.
    """
    value = abs(float(value))
    if value == 0:
        return 0
    while value >= 10 ** digits:
        value /= 10.0
    while value < 10 ** (digits - 1):
        value *= 10.0
    return int(value)


def _close(a, b):
    return abs(a - b) <= _TOLERANCE * max(abs(a), abs(b), 1.0)


def classify(number, grounded_values):
    """How (or whether) `number` is supported by `grounded_values`.

    Every reading in `number.candidates` is tried before giving a verdict, so a
    figure written with a scale word matches a bare figure in a table.

    Returns EXACT, RESCALED, DERIVED or INVENTED, plus a short explanation.
    """
    for candidate in number.candidates:
        for value in grounded_values:
            if _close(candidate, value):
                return EXACT, f"matches {value:,.6g} in a retrieved chunk"

    # Rescaling is NOT applied to percentages. A currency figure printed in a
    # "in millions" table is the same fact written as billions in prose, so
    # matching on significant digits is right. A percentage is not: 0.8% and 8%
    # are different claims, and accepting one as evidence for the other would
    # let a tenfold error through -- the exact failure this guardrail exists to
    # catch. Percentages must be exact or derived.
    if not number.is_pct:
        for candidate in number.candidates:
            target = _mantissa(candidate)
            for value in grounded_values:
                if _mantissa(value) == target:
                    return RESCALED, (f"same significant digits as {value:,.6g} "
                                      f"in a retrieved chunk, at a different magnitude")

    for candidate in number.candidates:
        derivation = _derive(candidate, grounded_values, is_pct=number.is_pct)
        if derivation:
            return DERIVED, derivation

    return INVENTED, "does not appear in any retrieved chunk"


def _pct_close(a, b):
    """Percentages are rounded hard in prose: an answer writes 0.8% for 0.842%.
    A purely relative tolerance would call that invented, so allow a tenth of a
    point in absolute terms as well."""
    return abs(a - b) <= 0.1 or _close(a, b)


def _derive(target, values, is_pct=False, limit=60):
    """Try to reach `target` from a pair of grounded values.

    Covers the arithmetic a grounded answer legitimately performs: a change
    between two periods, a total, a ratio, and a percentage change. Bounded to
    the largest few values so the pair loop stays cheap on a long answer.
    """
    pool = sorted({v for v in values if v}, key=lambda v: -abs(v))[:limit]
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if _close(target, a - b):
                return f"= {a:,.6g} - {b:,.6g} (both retrieved)"
            if _close(target, b - a):
                return f"= {b:,.6g} - {a:,.6g} (both retrieved)"
            if _close(target, a + b):
                return f"= {a:,.6g} + {b:,.6g} (both retrieved)"
            near = _pct_close if is_pct else _close
            if b and near(target, (a - b) / abs(b) * 100):
                return f"= percentage change from {b:,.6g} to {a:,.6g} (both retrieved)"
            if b and near(target, abs(a - b) / abs(b) * 100):
                return f"= percentage change between {b:,.6g} and {a:,.6g} (both retrieved)"
            if b and near(target, a / b * 100):
                return f"= {a:,.6g} / {b:,.6g} as a percentage (both retrieved)"
            if b and _close(target, a / b):
                return f"= {a:,.6g} / {b:,.6g} (both retrieved)"
    return None


@dataclass
class Finding:
    rule: str
    severity: str        # "block" | "warn" | "info"
    detail: str
    evidence: str = ""


@dataclass
class Verdict:
    allowed: bool = True
    findings: list = field(default_factory=list)
    # Populated by the numeric check so a caller can report coverage.
    counts: dict = field(default_factory=dict)

    def add(self, rule, severity, detail, evidence=""):
        self.findings.append(Finding(rule, severity, detail, evidence))
        if severity == "block":
            self.allowed = False
        return self

    @property
    def blocking(self):
        return [f for f in self.findings if f.severity == "block"]

    def summary(self):
        if not self.findings:
            return "clean"
        return "; ".join(f"{f.rule}: {f.detail}" for f in self.findings)

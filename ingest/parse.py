"""Extract and clean the named Item sections out of a 10-K.

Only Item sections are extracted, which is also how exhibits, XBRL data files and
the boilerplate cover page get excluded -- they simply are not items. (Item 15 is
the *exhibit index*, which is a genuine item and is kept.)
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

from config import OPTIONAL_SECTIONS, SECTIONS, SPARSE_CHAR_THRESHOLD
from ingest.chunk import split_sentences

log = logging.getLogger(__name__)

_INCORPORATED_BY_REFERENCE = re.compile(
    r"incorporated\s+(?:herein\s+)?by\s+reference", re.IGNORECASE
)
# Only the opening of a section decides whether it *is* a cross-reference. Item 1
# and Item 15 mention incorporation by reference in passing while still carrying
# real disclosure; Part III items lead with it because it is all they contain.
_BY_REFERENCE_WINDOW = 600

# The "Item 7A." style prefix at the very start of a section.
_ITEM_PREFIX = re.compile(r"^\s*item\s+\d+\s*[A-Z]?\s*[\.\:\)\-]?\s*", re.IGNORECASE)


# "None.", "Not applicable", "N/A" -- the whole section is a null answer.
_NULL_ANSWER = re.compile(
    r"^[\[\(]?(none|not\s*applicable|n\s*/\s*a|omitted|reserved)[\]\)]?[\.\s]*$",
    re.IGNORECASE,
)
# A single sentence longer than this is prose, not a one-line null answer -- it is
# usually an unpunctuated table that the sentence splitter cannot break up.
_SINGLE_SENTENCE_MAX = 600

# Running page footer, e.g. "Apple Inc. | 2022 Form 10-K | 19". These bleed into
# whichever section straddles a page break and make null answers look substantive.
_PAGE_FOOTER = re.compile(
    r"^.{0,80}\|\s*(?:19|20)\d\d\s*Form\s+10-K\s*\|\s*\d{1,4}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# The signature block follows the last item and is not part of it.
_SIGNATURE_BLOCK = re.compile(r"^\s*SIGNATURES?\s*$", re.MULTILINE)


@dataclass
class Section:
    item: str
    label: str
    part: str
    text: str
    is_sparse: bool
    incorporated_by_reference: bool


def extract_sections(filing_ref):
    """Return the Sections present in this filing, in canonical item order.

    Sections absent from the filing are reported by the second return value
    rather than fabricated -- there is no text to embed for a section that the
    registrant never filed. Sections that are present but thin ("None.") are
    kept, flagged is_sparse.
    """
    tenk = filing_ref.filing.obj()
    available = set(getattr(tenk, "items", []) or [])

    sections, missing = [], []
    for item, label, part, optional in SECTIONS:
        raw = None
        if item in available:
            try:
                raw = tenk[item]
            except Exception as exc:  # a malformed item should not kill the filing
                log.warning("[%s %s] failed to read %s: %s",
                            filing_ref.ticker, filing_ref.year, item, exc)

        text = clean_text(raw)
        if not text:
            if item not in OPTIONAL_SECTIONS:
                missing.append(item)
            # Optional sections (Item 1C pre-2023, 9C, 16) are skipped silently.
            continue

        body = _strip_heading(text)
        sections.append(Section(
            item=item,
            label=label,
            part=part,
            text=text,
            is_sparse=is_sparse(text),
            incorporated_by_reference=bool(
                _INCORPORATED_BY_REFERENCE.search(body[:_BY_REFERENCE_WINDOW])
            ),
        ))

    return sections, missing


def clean_text(raw):
    """Normalise EDGAR whitespace without touching the substance of the text.

    Financial tables in Item 8 flatten messily here; that is expected and the
    text is ingested as-is rather than being repaired or dropped.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = text.replace(" ", " ").replace("​", "")
    text = _PAGE_FOOTER.sub("", text)
    # Everything from the signature block onward belongs to no item.
    signature = _SIGNATURE_BLOCK.search(text)
    if signature:
        text = text[:signature.start()]
    # Collapse runs of spaces/tabs, and 3+ newlines down to a paragraph break,
    # while preserving single newlines that carry table structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_sparse(text):
    """True for null answers, one-liners, and anything under the char threshold."""
    body = _strip_heading(text)
    if not body or _NULL_ANSWER.match(body):
        return True
    if len(body) < SPARSE_CHAR_THRESHOLD:
        return True
    return len(split_sentences(body)) <= 1 and len(body) < _SINGLE_SENTENCE_MAX


def _strip_heading(text):
    """Drop the leading "Item 1A." item number before inspecting a section.

    The heading stays in the stored text -- it is useful retrieval signal -- but
    it must not make an empty section look substantive. Only the item number is
    removed, not the title: EDGAR sometimes runs the title straight into the
    first sentence ("Executive CompensationThe information required by..."), so
    anything that assumes the heading is its own line will eat the whole body.
    """
    return _ITEM_PREFIX.sub("", text, count=1).strip()

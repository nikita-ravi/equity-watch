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
    # Two idioms mean the same thing and both must match. "Incorporated by
    # reference" is the common one. "Reference is made to ..." is ExxonMobil's,
    # and missing it left XOM FY2022/FY2023 Item 7 and Item 8 in the corpus as
    # 265- and 734-character stubs that pointed at an exhibit -- retrievable,
    # unflagged, and indistinguishable from a real section to everything
    # downstream.
    r"incorporated\s+(?:herein\s+)?by\s+reference"
    r"|reference\s+is\s+made\s+to",
    re.IGNORECASE,
)
# Only the opening of a section decides whether it *is* a cross-reference. Item 1
# and Item 15 mention incorporation by reference in passing while still carrying
# real disclosure; Part III items lead with it because it is all they contain.
_BY_REFERENCE_WINDOW = 600

# The "Item 7A." style prefix at the very start of a section.
_ITEM_PREFIX = re.compile(r"^\s*item\s+\d+\s*[A-Z]?\s*[\.\:\)\-]?\s*", re.IGNORECASE)

# A *heading* for another item, found inside a section's own text -- the signal
# that the upstream sectioner failed to close this section and ran on into the
# next one. Three independent guards keep prose cross-references ("included in
# Part II, Item 8 of this Form 10-K") from matching:
#   1. MULTILINE ^  -- the heading must start a line.
#   2. a delimiter immediately after the item number; a cross-reference reads
#      "Item 8 of this", with no "." or ":" to match.
#   3. _HEADING_TITLE -- real headings are followed by a capitalised title.
# Only a heading for a LATER item truncates, so a back-reference cannot.
_ITEM_HEADING = re.compile(
    r"^[ \t]*item[ \t]+(\d{1,2})[ \t]*([A-C]?)[ \t]*[\.\:\)\-]",
    re.IGNORECASE | re.MULTILINE,
)
# At least four consecutive capitals within the next stretch of the line, e.g.
# "MINE SAFETY DISCLOSURES" or "MARKET FOR REGISTRANT'S COMMON EQUITY". Applied
# with .match(text, pos), which anchors at pos on its own -- a leading "^" would
# anchor to the start of the string instead and never fire.
_HEADING_TITLE = re.compile(r"[ \t]*[A-Z][A-Z'’\- ]{3,}")

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

        text = truncate_at_next_item(clean_text(raw), item)
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

    sections = recover_incorporated_sections(
        sections, label=f"[{filing_ref.ticker} {filing_ref.year}]")
    return sections, missing


# Landmarks inside an annual report's appended "Financial Section". These are
# standard headings, not one filer's invention, but they are only consulted when
# Item 7/Item 8 have already been confirmed as cross-reference stubs -- see
# recover_incorporated_sections.
_FINANCIAL_SECTION = re.compile(r"^FINANCIAL SECTION\s*$", re.MULTILINE)
_AUDIT_REPORT = re.compile(
    r"^REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM", re.MULTILINE
)


def recover_incorporated_sections(sections, label=""):
    """Re-attribute a Financial Section that landed under the wrong item.

    Some filers answer Item 7 and Item 8 with a pointer ("Reference is made to
    the Financial Section of this report") and then append that Financial
    Section after the last item. The sectioner attributes the whole appendix to
    whichever item precedes it -- for XOM FY2022/FY2023 that is Item 16, which
    came back holding ~292k characters while Item 7 and Item 8 held a one-line
    stub each. Every MD&A question against those filings then retrieved chunks
    labelled Item 16, so `section` filtering could not reach the content and the
    agent's section parameter pointed at a stub.

    Narrowly triggered: it does nothing unless Item 7 or Item 8 is already
    flagged incorporated_by_reference *and* a donor section carries both
    landmarks. Returns the sections unchanged in every other case.
    """
    by_item = {s.item: s for s in sections}
    stubbed = [i for i in ("Item 7", "Item 8")
               if i in by_item and by_item[i].incorporated_by_reference]
    if not stubbed:
        return sections

    donor = max((s for s in sections if s.item not in ("Item 7", "Item 8")),
                key=lambda s: len(s.text), default=None)
    if donor is None:
        return sections

    start = _FINANCIAL_SECTION.search(donor.text)
    if not start:
        return sections
    # The audit report opens the financial statements; take the first one at or
    # after the MD&A so the internal-control report later on cannot win.
    audit = _AUDIT_REPORT.search(donor.text, start.end())
    if not audit:
        return sections

    mdna_text = donor.text[start.start():audit.start()].strip()
    stmts_text = donor.text[audit.start():].strip()
    kept = donor.text[:start.start()].strip()
    if not mdna_text or not stmts_text:
        return sections

    rebuilt = []
    for s in sections:
        if s.item == "Item 7":
            rebuilt.append(_respawn(s, mdna_text))
        elif s.item == "Item 8":
            rebuilt.append(_respawn(s, stmts_text))
        elif s.item == donor.item:
            # What actually belonged to the donor item, usually a null answer.
            rebuilt.append(_respawn(s, kept, by_ref=s.incorporated_by_reference))
        else:
            rebuilt.append(s)

    log.info("%s recovered Financial Section from %s -> Item 7 (%d chars) + "
             "Item 8 (%d chars); %s kept %d chars",
             label, donor.item, len(mdna_text), len(stmts_text),
             donor.item, len(kept))
    return rebuilt


def _respawn(section, text, by_ref=False):
    """A copy of `section` carrying new text, with the derived flags redone."""
    return Section(
        item=section.item,
        label=section.label,
        part=section.part,
        text=text,
        is_sparse=is_sparse(text) if text else True,
        incorporated_by_reference=by_ref,
    )


def _item_key(number, letter):
    return f"Item {number}{(letter or '').upper()}"


# Canonical document order, derived from config so the two cannot drift.
_ITEM_ORDER = {item: i for i, (item, _, _, _) in enumerate(SECTIONS)}


def truncate_at_next_item(text, item):
    """Cut `text` where the *next* item's heading begins.

    edgartools closes a section at the following item heading, but misses the
    no-space form some filers use ("ITEM 5.MARKET FOR REGISTRANT'S ..."). When
    it misses, the section runs on and swallows everything after it: WMT FY2024
    Item 4 (Mine Safety, "Not applicable.") came back as 87,852 characters
    holding the whole of Part II through the cash flow statement, duplicating
    content that Item 5/7/8 also extracted correctly.

    Returns the text unchanged when no later item's heading is found, which is
    the overwhelmingly common case.
    """
    here = _ITEM_ORDER.get(item)
    if here is None:
        return text

    for match in _ITEM_HEADING.finditer(text):
        # The section's own heading opens the text; never cut on it.
        if match.start() == 0:
            continue
        other = _ITEM_ORDER.get(_item_key(match.group(1), match.group(2)))
        if other is None or other <= here:
            continue
        if not _HEADING_TITLE.match(text, match.end()):
            continue
        return text[:match.start()].rstrip()

    return text


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

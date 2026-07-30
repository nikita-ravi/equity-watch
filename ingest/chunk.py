"""Sentence-aware, token-budgeted chunking.

Chunks are built by greedily packing whole sentences up to CHUNK_SIZE_TOKENS,
then starting the next chunk with a tail of trailing sentences worth roughly
CHUNK_OVERLAP_TOKENS. Token counts come from the embedding model's own
tokenizer, so a chunk never overflows what the model actually encodes.

Chunking is always scoped to a single section -- the caller passes one section's
text at a time, so chunks can never straddle a section boundary.
"""

import re

# Abbreviations that end in a period but do not end a sentence. Without this the
# splitter shreds 10-K prose ("U.S. federal law", "Apple Inc. designs...").
_ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "lp", "plc", "no", "nos", "vs", "etc",
    "approx", "est", "fig", "mr", "mrs", "ms", "dr", "jr", "sr", "st", "prof",
    "gov", "dept", "univ", "cf", "al", "ie", "eg", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sept", "sep", "oct", "nov", "dec",
}

# Break after ., ! or ? (plus optional closing quote/bracket) when followed by
# whitespace and something that plausibly starts a new sentence.
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])["\'’”)\]]*\s+(?=["\'‘“(\[]*[A-Z0-9])')

# A trailing single-letter or abbreviation token, e.g. "U.S." or "Inc."
_TRAILING_TOKEN = re.compile(r'([A-Za-z]{1,5})\.["\'’”)\]]*\s*$')


def split_sentences(text):
    """Split text into sentences, tolerating the abbreviations common in 10-Ks."""
    pieces = _SENTENCE_BREAK.split(text)
    sentences = []
    for piece in pieces:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            # False break -- glue it back onto the previous sentence.
            sentences[-1] = sentences[-1] + " " + piece
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def _ends_with_abbreviation(sentence):
    match = _TRAILING_TOKEN.search(sentence)
    if not match:
        return False
    token = match.group(1)
    if token.lower() in _ABBREVIATIONS:
        return True
    # Single capital letter before the period: an initial ("J. P. Morgan") or
    # the tail of a dotted acronym ("U.S.").
    return len(token) == 1 and token.isupper()


def chunk_text(text, count_tokens, encode_tokens, decode_tokens,
               chunk_size=512, overlap=64):
    """Chunk `text` into a list of strings, each at most `chunk_size` tokens.

    count_tokens(str) -> int
    encode_tokens(str) -> list[int]       (used only to split oversized units)
    decode_tokens(list[int]) -> str
    """
    text = (text or "").strip()
    if not text:
        return []

    if count_tokens(text) <= chunk_size:
        return [text]

    # Units are sentences, except that any single sentence too big to fit in a
    # chunk (long financial tables flattened into one "sentence") is hard-split
    # on token windows first.
    units = []
    for sentence in split_sentences(text):
        n = count_tokens(sentence)
        if n <= chunk_size:
            units.append((sentence, n))
        else:
            for piece in _split_oversized(sentence, encode_tokens, decode_tokens,
                                          chunk_size, overlap):
                units.append((piece, count_tokens(piece)))

    chunks = []
    current, current_tokens = [], 0
    for unit in units:
        sentence, n = unit
        if current and current_tokens + n > chunk_size:
            chunks.append(" ".join(s for s, _ in current))
            current, current_tokens = _overlap_tail(current, overlap)
        current.append(unit)
        current_tokens += n
    if current:
        chunks.append(" ".join(s for s, _ in current))

    return [c for c in (chunk.strip() for chunk in chunks) if c]


def _overlap_tail(current, overlap):
    """Return the trailing sentences of `current` worth ~`overlap` tokens."""
    if overlap <= 0:
        return [], 0
    tail, total = [], 0
    for unit in reversed(current):
        # Never let the overlap alone fill the next chunk.
        if total >= overlap:
            break
        tail.insert(0, unit)
        total += unit[1]
    return tail, total


def _split_oversized(sentence, encode_tokens, decode_tokens, chunk_size, overlap):
    """Hard-split a single oversized sentence into overlapping token windows."""
    ids = encode_tokens(sentence)
    stride = max(1, chunk_size - overlap)
    pieces = []
    for start in range(0, len(ids), stride):
        window = ids[start:start + chunk_size]
        if not window:
            break
        piece = decode_tokens(window).strip()
        if piece:
            pieces.append(piece)
        if start + chunk_size >= len(ids):
            break
    return pieces

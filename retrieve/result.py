"""The shape a search hit takes on its way out of the retrieval layer."""

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    text: str
    ticker: str
    year: int
    section: str            # e.g. "Item 1A"
    score: float            # ranking score -- see note below
    metadata: dict = field(default_factory=dict)
    # Set by v4 only. The cross-encoder's relevance score in [0, 1]. When it is
    # set, `score` mirrors it so ordering and score agree; the RRF score that
    # got the chunk into the candidate pool stays in metadata["fusion"].
    rerank_score: float = None

    # --- convenience accessors over the ingest payload ------------------------
    @property
    def section_label(self):
        return self.metadata.get("section_label", "")

    @property
    def company(self):
        return self.metadata.get("company", self.ticker)

    @property
    def part(self):
        return self.metadata.get("part", "")

    @property
    def filing_id(self):
        return self.metadata.get("filing_id", "")

    @property
    def chunk_index(self):
        return self.metadata.get("chunk_index")

    @classmethod
    def from_point(cls, point):
        """Build from a Qdrant ScoredPoint."""
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        return cls(
            text=text,
            ticker=payload.get("ticker", ""),
            year=payload.get("year"),
            section=payload.get("section", ""),
            score=point.score,
            # Everything except the text, so metadata stays a faithful copy of the
            # rest of the ingest payload rather than a hand-picked subset.
            metadata=payload,
        )

    def header(self):
        """"AAPL 2024 | Item 1A - Risk Factors | score: 0.847" """
        label = f" — {self.section_label}" if self.section_label else ""
        return (f"{self.ticker} {self.year} | {self.section}{label} "
                f"| score: {self.score:.3f}")

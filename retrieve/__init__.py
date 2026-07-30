"""Retrieval layer: embed query -> dense cosine search -> ranked chunks."""

from retrieve.result import SearchResult
from retrieve.search import search

__all__ = ["SearchResult", "search"]

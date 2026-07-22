"""Public interface for knowledge-base retrieval."""

from .store import SearchResult, build_index, search

__all__ = ["SearchResult", "build_index", "search"]

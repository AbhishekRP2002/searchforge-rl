"""The provider contract. Pure data and a protocol — no HTTP, no verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    """One normalised search hit. The intersection every provider can supply."""

    title: str
    url: str
    excerpt: str
    date: str | None
    rank: int


@dataclass(frozen=True)
class PageContent:
    """One normalised fetched page."""

    url: str
    title: str
    text: str
    date: str | None


class ProviderError(Exception):
    """A provider call failed. `retryable` decides adapter-level retry, so auth
    failures (401/403) never spin"""

    def __init__(
        self, message: str, *, status: int | None = None, retryable: bool = False
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, num_results: int) -> list[SearchResult]: ...

    async def fetch(self, url: str) -> PageContent: ...

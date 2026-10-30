"""The provider contract. Pure data and a protocol — no HTTP, no verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Protocol


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

    supports_date_filter: ClassVar[bool]
    """True only when the provider documents a native filter for *both* bounds.

    This is the capability table for `date_filter="native"`: a provider with
    half a filter (say, a start date only) would answer a two-sided window
    differently from every other arm, so it counts as unsupported."""

    async def search(
        self,
        query: str,
        num_results: int,
        *,
        after: date | None = None,
        before: date | None = None,
    ) -> list[SearchResult]: ...

    async def fetch(self, url: str) -> PageContent: ...


def refuse_native_dates(name: str, after: date | None, before: date | None) -> None:
    """Guard for adapters without a native date filter. The tool server refuses
    such a configuration before any call; this catches a direct caller, so a
    filter is never silently dropped on one arm."""
    if after or before:
        raise ValueError(f"{name} has no native date filter; use date_filter='emulated'")

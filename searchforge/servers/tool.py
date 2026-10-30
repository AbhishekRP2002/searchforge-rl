"""MCP tool server exposing `web_search` and `web_fetch`.

Both tools are pure functions of their arguments: no vf.State subclass, so the
framework skips its state channel entirely and parallel tool calls cannot race.
The separate server reads provider credentials from its own environment or an
untraced local env file; the harness runtime never receives them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import verifiers.v1 as vf
from dotenv import dotenv_values
from pydantic import model_validator
from verifiers.v1.utils.decorators import discover_decorated

from searchforge.caps import cap_excerpts, cap_page, page_count, slice_page
from searchforge.dates import check_window, parse_bound, parse_result_date, within
from searchforge.providers.base import ProviderError, SearchProvider
from searchforge.providers.exa import ExaProvider
from searchforge.providers.firecrawl import FirecrawlProvider
from searchforge.providers.keenable import KeenableProvider
from searchforge.providers.parallel import ParallelProvider
from searchforge.providers.serper import SerperProvider
from searchforge.providers.tavily import TavilyProvider
from searchforge.providers.tinyfish import TinyFishProvider
from searchforge.render import render_page, render_results

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

PROVIDERS: dict[str, tuple[str, Callable[[str], SearchProvider]]] = {
    "serper": ("SERPER_API_KEY", SerperProvider),
    "exa": ("EXA_API_KEY", ExaProvider),
    "firecrawl": ("FIRECRAWL_API_KEY", FirecrawlProvider),
    "tavily": ("TAVILY_API_KEY", TavilyProvider),
    "parallel": ("PARALLEL_API_KEY", ParallelProvider),
    "tinyfish": ("TINYFISH_API_KEY", TinyFishProvider),
    "keenable": ("KEENABLE_API_KEY", KeenableProvider),
}

PROVIDER_KEY_ENV = {name: variable for name, (variable, _) in PROVIDERS.items()}

logger = logging.getLogger("searchforge.tools")


def configure_logging() -> None:
    """Called from `__main__` only — a library never configures logging for its host.

    This process needs its own setup because verifiers' `setup_logging` installs its
    stdlib->loguru bridge on the `verifiers.v1` logger inside the *eval* process, and
    under the default rich dashboard it also sets `logging.lastResort = None`. Neither
    reaches here, so without this every record below is discarded silently.

    stderr keeps diagnostics out of whatever stdout carries. `level` is the severity
    axis, independent of the stream: INFO here means INFO and above, not errors only.
    """
    logging.basicConfig(
        level=os.environ.get("SEARCHFORGE_LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


BUDGET_EXHAUSTED = (
    "Tool call budget exhausted for this task. Answer with what you have."
)
PARALLEL_REFUSED = (
    "Parallel tool calls are disabled for this task. Issue one call at a time "
    "and wait for its result before making the next."
)
UNAVAILABLE = (
    "SEARCH_PROVIDER_UNAVAILABLE: the provider remained unavailable after retries. "
    "Try a different query or answer with what you have."
)
RATE_LIMITED = (
    "SEARCH_PROVIDER_RATE_LIMITED: the provider remained rate-limited after "
    "retries. Answer with what you have."
)


PAGINATION_OFF = (
    "Pagination is disabled for this task. Call web_fetch with a url only."
)
FETCH_DISABLED = "web_fetch is disabled for this task. Use web_search only."


DATE_FILTER_OFF = (
    "Date filtering is disabled for this task. Call web_search with a query only."
)


def _window_header(after: date | None, before: date | None, mode: str) -> str:
    """One line naming the window and how it was applied. It lands in the tool
    message, so the trace records which mode served each dated call."""
    low = after.isoformat() if after else "any"
    high = before.isoformat() if before else "any"
    return f"Date filter ({mode}): published {low} to {high}"


def _recoverable_failure(exc: ProviderError) -> str:
    if not exc.retryable:
        raise exc
    return RATE_LIMITED if exc.status == 429 else UNAVAILABLE


DateFilterMode = Literal["off", "native", "emulated"]


def check_date_filter(provider: str, mode: DateFilterMode) -> None:
    """Refuse a native date filter on a provider that cannot serve one.

    Called at taskset load and again when the tool server builds a provider, so a
    misconfigured comparison fails before its first rollout rather than letting
    one arm quietly search without the window every other arm applies.
    """
    if mode != "native":
        return
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(
            f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}"
        )
    if not getattr(spec[1], "supports_date_filter", False):
        native = sorted(
            name
            for name, (_, construct) in PROVIDERS.items()
            if getattr(construct, "supports_date_filter", False)
        )
        raise ValueError(
            f"date_filter='native' but {provider!r} has no native filter for both "
            f"bounds. Use date_filter='emulated' for every arm, or compare only "
            f"providers that support it: {native}"
        )


class WebToolsetConfig(vf.ToolsetConfig):
    colocated: bool = False
    """Run outside the agent runtime so provider credentials stay server-side."""

    env_file: Path | None = Path(".env")
    """Local, untraced credentials file read only by the tool server."""

    provider: str = "serper"
    num_results: int = 5
    result_format: str = "markdown"
    include_answer_box: bool = False
    """Serper's answerBox is Google's own direct answer — a capability other
    providers lack, not better retrieval. Off by default."""
    max_tool_calls: int = 10

    enable_fetch: bool = True
    """Whether `web_fetch` is advertised at all. Off reproduces a search-only
    harness, which is what Exa's published experiment used: the model sees result
    excerpts and nothing else (`configs/exa-replication.toml`)."""

    max_page_chars: int | None = 8000
    """Characters of fetched page text the model sees, head-kept, with a visible
    `[truncated: showing N of M chars]` marker when anything was cut. Applied here
    on the normalised text, never in an adapter, so every arm is cut by the same
    code at the same number. `None` is uncapped.

    Why 8,000: measured over 29 live fetches, 20 were at most 24,627 characters and
    the rest jumped to 107k-162k, with a median rollout near 20k tokens. Eight
    thousand characters (~2k tokens) keeps several fetches inside a 32k training
    context, sits inside the 3k-50k range published search environments use, and
    caps the bimodal tail that made every rollout truncate."""

    max_excerpt_chars: int | None = 2000
    """Characters per search-result excerpt, head-kept with an ellipsis. Matches
    Exa's per-result limit. It rarely binds today (whole result lists measure
    ~1.3k characters) and exists so no provider can return page-length excerpts
    and win on volume. `None` is uncapped."""

    fetch_pagination: bool = False
    """Whether `web_fetch` takes a `page` argument. On, a long page is served in
    `max_page_chars`-sized pages, each marked `page n of m`, and each page is one
    provider call charged to the budget -- there is no cache yet, so the page is
    re-fetched. Off, the argument is not advertised at all."""

    date_filter: DateFilterMode = "off"
    """Whether `web_search` takes optional `after`/`before` dates (YYYY-MM-DD).

    - `off`: the tool advertises `query` only, exactly as before this knob existed.
    - `native`: each provider applies the window with its own documented filter.
      A provider without one is refused at load (`check_date_filter`); it is never
      silently searched unfiltered.
    - `emulated`: no provider receives the dates. Every arm's results are filtered
      after the fact on their own `date` field, the same code for every provider.
      An undated or unreadable date is dropped, so an arm whose provider dates
      fewer results returns fewer hits -- that is the cost of identical treatment,
      and it shows up in the logs as `dropped=`.

    Mixing modes across arms of one comparison is a different experiment, not a
    provider comparison.
    """

    allow_parallel_tool_calls: bool = False
    """Whether the agent may have more than one tool call in flight at once.

    Real deployments want parallel search: it hides latency and gathers more per
    turn. It is off by default here anyway, for two reasons that only matter
    while training.

    The shared `max_tool_calls` budget already caps total spend, so parallelism
    cannot widen the search space -- it spends the same budget faster. What it
    does change is the incentive. An agent that can fan out is rewarded for
    shotgunning broad queries rather than composing a precise one, and query
    precision is the behaviour this environment exists to train. It also
    arrives as one large token burst, which matters when the reference
    `seq_len` is 4,096.

    Turn it on to study that trade-off; it is a knob, not a verdict. Note that
    verifiers 0.3.1 exposes no `parallel_tool_calls` sampling field, so this is
    enforced here at the tool server rather than on the model request: the
    model may still *emit* parallel calls, and the ones beyond the first are
    refused with a message instead of being served.
    """


    @model_validator(mode="after")
    def _pagination_needs_a_page_size(self) -> WebToolsetConfig:
        if self.fetch_pagination and self.max_page_chars is None:
            raise ValueError("fetch_pagination needs max_page_chars, the page size")
        return self


def build_provider(config: WebToolsetConfig) -> SearchProvider:
    spec = PROVIDERS.get(config.provider)
    if spec is None:
        raise ValueError(
            f"unknown provider {config.provider!r}; expected one of {sorted(PROVIDERS)}"
        )
    check_date_filter(config.provider, config.date_filter)
    variable, construct = spec
    key = os.environ.get(variable)
    if not key and config.env_file:
        key = dotenv_values(config.env_file).get(variable)
    if not key:
        raise RuntimeError(
            f"{variable} is missing from the provider server. Export it in that "
            f"process or set WebToolsetConfig.env_file to a local .env file."
        )
    if construct is SerperProvider:
        # ponytail: one branch for the one provider carrying an extra config flag.
        # A factory per provider would be more code than the branch it removes.
        return SerperProvider(key, include_answer_box=config.include_answer_box)
    return construct(key)


class WebToolset(vf.Toolset[WebToolsetConfig]):
    TOOL_PREFIX = "web"  # the model sees `web_search` / `web_fetch`

    def __init__(self, config: WebToolsetConfig) -> None:
        super().__init__(config)
        self._calls = 0
        self._in_flight = asyncio.Lock()

    def _spend(self) -> bool:
        # No await between the check and the increment, so concurrent callers
        # cannot both pass the budget test on the same remaining call.
        if self._calls >= self.config.max_tool_calls:
            return False
        self._calls += 1
        return True

    def _claim(self) -> bool:
        """Take the single in-flight slot, when parallelism is disabled.

        A refused call is deliberately not charged to the budget: the agent is
        being told how to call, not being punished for it, and charging would
        make the refusal itself an incentive to avoid tools.
        """
        if self.config.allow_parallel_tool_calls:
            return True
        if self._in_flight.locked():
            return False
        return True

    def _hidden_tools(self) -> set[str]:
        """Method names left unregistered: the unused variant of each tool whose
        schema depends on config, and `fetch` entirely when it is disabled."""
        hidden = {"search" if self.config.date_filter != "off" else "search_dated"}
        if not self.config.enable_fetch:
            hidden |= {"fetch", "fetch_paged"}
        else:
            hidden.add("fetch" if self.config.fetch_pagination else "fetch_paged")
        return hidden

    def register(self, mcp: MCPServer) -> None:
        """Advertise exactly one variant of each tool, or none of `fetch`.

        `search`/`search_dated` are both named `search` for the model, and
        `fetch`/`fetch_paged` both `fetch`; the MCP server builds each schema
        from the method signature. Registering only one keeps optional arguments
        out of the schema entirely when their knob is off, so a default run
        advertises exactly what it did before the knobs existed.

        This repeats the base `Toolset.register` loop, which verifiers (0.4.0)
        still offers no public hook to filter, so `discover_decorated` is an
        internal import. The real-server schema test fails if either changes.
        """
        for fn in discover_decorated(self, "tool"):
            if fn.__name__ in self._hidden_tools():
                continue
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )

    @vf.tool
    async def search(self, query: str) -> str:
        """Search the web for a query. Returns ranked results with titles, URLs
        and text excerpts."""
        return await self._search(query, None, None)

    @vf.tool(name="search")
    async def search_dated(
        self, query: str, after: str | None = None, before: str | None = None
    ) -> str:
        """Search the web for a query. Returns ranked results with titles, URLs
        and text excerpts. Optionally restrict results to pages published on or
        after `after` and/or on or before `before`, each a date in YYYY-MM-DD form."""
        if self.config.date_filter == "off":
            # Unreachable through MCP (`register` hides it); guards direct callers.
            return DATE_FILTER_OFF
        start, problem = parse_bound(after, "after")
        if problem:
            return problem
        end, problem = parse_bound(before, "before")
        if problem:
            return problem
        if problem := check_window(start, end):
            return problem
        return await self._search(query, start, end)

    async def _search(self, query: str, after: date | None, before: date | None) -> str:
        if not self._claim():
            logger.info("search refused: parallel tool calls are disabled")
            return PARALLEL_REFUSED
        if not self._spend():
            logger.info("search budget_exhausted after %d calls", self._calls)
            return BUDGET_EXHAUSTED
        dated = after is not None or before is not None
        native = dated and self.config.date_filter == "native"
        started = time.perf_counter()
        try:
            async with self._in_flight:
                provider = build_provider(self.config)
                if native:
                    results = await provider.search(
                        query, self.config.num_results, after=after, before=before
                    )
                else:
                    results = await provider.search(query, self.config.num_results)
        except ProviderError as exc:
            logger.warning(
                "search provider=%s status=%s retryable=%s query=%r",
                self.config.provider,
                exc.status,
                exc.retryable,
                query[:80],
            )
            return _recoverable_failure(exc)
        results = cap_excerpts(results, self.config.max_excerpt_chars)
        dropped = 0
        if dated and not native:
            kept = [r for r in results if within(parse_result_date(r.date), after, before)]
            dropped = len(results) - len(kept)
            results = kept
        rendered = render_results(results, self.config.result_format)
        if dated:
            rendered = f"{_window_header(after, before, self.config.date_filter)}\n\n{rendered}"
        logger.info(
            "search provider=%s call=%d/%d hits=%d chars=%d ms=%.0f date_filter=%s "
            "after=%s before=%s dropped=%d query=%r",
            self.config.provider,
            self._calls,
            self.config.max_tool_calls,
            len(results),
            len(rendered),
            (time.perf_counter() - started) * 1000,
            self.config.date_filter if dated else "none",
            after,
            before,
            dropped,
            query[:80],
        )
        return rendered

    @vf.tool
    async def fetch(self, url: str) -> str:
        """Retrieve the readable text of one web page by its URL."""
        return await self._fetch(url, None)

    @vf.tool(name="fetch")
    async def fetch_paged(self, url: str, page: int = 1) -> str:
        """Retrieve the readable text of one web page by its URL. Long pages are
        split into numbered pages; pass `page` (1, 2, ...) to read further."""
        if not self.config.fetch_pagination:
            # Unreachable through MCP (`register` hides it); guards direct callers.
            return PAGINATION_OFF
        if page < 1:
            return f"page must be 1 or greater, got {page}."
        return await self._fetch(url, page)

    async def _fetch(self, url: str, page_number: int | None) -> str:
        if not self.config.enable_fetch:
            return FETCH_DISABLED
        if not url.startswith(("http://", "https://")):
            return f"Not a fetchable URL: {url!r}. Pass an http or https URL."
        if not self._claim():
            logger.info("fetch refused: parallel tool calls are disabled")
            return PARALLEL_REFUSED
        if not self._spend():
            logger.info("fetch budget_exhausted after %d calls", self._calls)
            return BUDGET_EXHAUSTED
        started = time.perf_counter()
        try:
            async with self._in_flight:
                page = await build_provider(self.config).fetch(url)
        except ProviderError as exc:
            logger.warning(
                "fetch provider=%s status=%s retryable=%s url=%s",
                self.config.provider,
                exc.status,
                exc.retryable,
                url[:120],
            )
            return _recoverable_failure(exc)
        raw = len(page.text)
        limit = self.config.max_page_chars
        if page_number is None:
            text = cap_page(page.text, limit)
        else:
            assert limit is not None  # the config validator guarantees a page size
            text = slice_page(page.text, limit, page_number)
            if text is None:
                return (
                    f"{url} has {page_count(raw, limit)} page(s); "
                    f"page {page_number} does not exist."
                )
        rendered = render_page(replace(page, text=text))
        # `raw` against `chars` is the load-bearing pair: before the cap, one
        # 161,937-char fetch consumed ~16x a 4k sequence budget. Both are logged so
        # the cap's bite is visible per provider while the run happens.
        logger.info(
            "fetch provider=%s call=%d/%d raw=%d chars=%d page=%s ms=%.0f url=%s",
            self.config.provider,
            self._calls,
            self.config.max_tool_calls,
            raw,
            len(rendered),
            page_number,
            (time.perf_counter() - started) * 1000,
            url[:120],
        )
        return rendered

if __name__ == "__main__":
    configure_logging()
    WebToolset.run()

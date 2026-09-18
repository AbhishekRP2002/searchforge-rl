"""MCP tool server exposing `web_search` and `web_fetch`.

Both tools are pure functions of their arguments: no vf.State subclass, so the
framework skips its state channel entirely and parallel tool calls cannot race.
The separate server reads provider credentials from its own environment or an
untraced local env file; the harness runtime never receives them.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import verifiers.v1 as vf
from dotenv import dotenv_values

from searchforge.providers.base import ProviderError, SearchProvider
from searchforge.providers.exa import ExaProvider
from searchforge.providers.serper import SerperProvider
from searchforge.render import render_page, render_results

PROVIDER_KEY_ENV = {"serper": "SERPER_API_KEY", "exa": "EXA_API_KEY"}

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
UNAVAILABLE = (
    "SEARCH_PROVIDER_UNAVAILABLE: the provider remained unavailable after retries. "
    "Try a different query or answer with what you have."
)
RATE_LIMITED = (
    "SEARCH_PROVIDER_RATE_LIMITED: the provider remained rate-limited after "
    "retries. Answer with what you have."
)


def _recoverable_failure(exc: ProviderError) -> str:
    if not exc.retryable:
        raise exc
    return RATE_LIMITED if exc.status == 429 else UNAVAILABLE


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


def build_provider(config: WebToolsetConfig) -> SearchProvider:
    if config.provider not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"unknown provider {config.provider!r}; "
            f"expected one of {sorted(PROVIDER_KEY_ENV)}"
        )
    variable = PROVIDER_KEY_ENV[config.provider]
    key = os.environ.get(variable)
    if not key and config.env_file:
        key = dotenv_values(config.env_file).get(variable)
    if not key:
        raise RuntimeError(
            f"{variable} is missing from the provider server. Export it in that "
            f"process or set WebToolsetConfig.env_file to a local .env file."
        )
    if config.provider == "serper":
        return SerperProvider(key, include_answer_box=config.include_answer_box)
    return ExaProvider(key)


class WebToolset(vf.Toolset[WebToolsetConfig]):
    TOOL_PREFIX = "web"  # the model sees `web_search` / `web_fetch`

    def __init__(self, config: WebToolsetConfig) -> None:
        super().__init__(config)
        self._calls = 0

    def _spend(self) -> bool:
        if self._calls >= self.config.max_tool_calls:
            return False
        self._calls += 1
        return True

    @vf.tool
    async def search(self, query: str) -> str:
        """Search the web for a query. Returns ranked results with titles, URLs
        and text excerpts."""
        if not self._spend():
            logger.info("search budget_exhausted after %d calls", self._calls)
            return BUDGET_EXHAUSTED
        started = time.perf_counter()
        try:
            results = await build_provider(self.config).search(
                query, self.config.num_results
            )
        except ProviderError as exc:
            logger.warning(
                "search provider=%s status=%s retryable=%s query=%r",
                self.config.provider, exc.status, exc.retryable, query[:80],
            )
            return _recoverable_failure(exc)
        rendered = render_results(results, self.config.result_format)
        logger.info(
            "search provider=%s call=%d/%d hits=%d chars=%d ms=%.0f query=%r",
            self.config.provider, self._calls, self.config.max_tool_calls,
            len(results), len(rendered), (time.perf_counter() - started) * 1000,
            query[:80],
        )
        return rendered

    @vf.tool
    async def fetch(self, url: str) -> str:
        """Retrieve the readable text of one web page by its URL."""
        if not url.startswith(("http://", "https://")):
            return f"Not a fetchable URL: {url!r}. Pass an http or https URL."
        if not self._spend():
            logger.info("fetch budget_exhausted after %d calls", self._calls)
            return BUDGET_EXHAUSTED
        started = time.perf_counter()
        try:
            page = await build_provider(self.config).fetch(url)
        except ProviderError as exc:
            logger.warning(
                "fetch provider=%s status=%s retryable=%s url=%s",
                self.config.provider, exc.status, exc.retryable, url[:120],
            )
            return _recoverable_failure(exc)
        rendered = render_page(page)
        # `chars` is the load-bearing field: pages are uncapped by decision, and one
        # 161,937-char fetch consumed ~16x the spec's whole sequence budget. This is
        # the number that makes that visible while it happens instead of afterwards.
        logger.info(
            "fetch provider=%s call=%d/%d chars=%d ms=%.0f url=%s",
            self.config.provider, self._calls, self.config.max_tool_calls,
            len(rendered), (time.perf_counter() - started) * 1000, url[:120],
        )
        return rendered


if __name__ == "__main__":
    configure_logging()
    WebToolset.run()

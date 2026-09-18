"""Shared HTTP mechanics for provider adapters; provider contracts stay separate."""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from searchforge.providers.base import ProviderError

TIMEOUT = 30.0
RETRY_ATTEMPTS = 3
RETRY_WAIT = wait_exponential_jitter(initial=0.5, max=8.0)
"""One policy for every provider. `post_json` reads these as module globals at call
time rather than binding them as default arguments, so monkeypatching RETRY_WAIT
here actually takes effect — a default would bind once, at import.
"""


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> httpx.Response:
    """POST with the provider policy: retry transport faults, 429 and 5xx."""
    logger = logging.getLogger(f"searchforge.providers.{provider}")
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
        retry=retry_if_exception(_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                raise ProviderError(
                    f"{provider} transport error: {exc}", retryable=True
                ) from exc
            if response.status_code >= 400:
                raise ProviderError(
                    f"{provider} returned {response.status_code}: {response.text[:200]}",
                    status=response.status_code,
                    retryable=response.status_code == 429
                    or response.status_code >= 500,
                )
            return response
    raise RuntimeError("unreachable")

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

QueryValue = str | int | float | bool | None
"""What httpx will URL-encode into a query string. Narrower than the JSON body
type, because a nested structure has no defined encoding in a query parameter."""

TIMEOUT = 30.0
RETRY_ATTEMPTS = 3
RETRY_WAIT = wait_exponential_jitter(initial=0.5, max=8.0)
"""One policy for every provider. `request_json` reads these as module globals at call
time rather than binding them as default arguments, so monkeypatching RETRY_WAIT
here actually takes effect — a default would bind once, at import.
"""


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


async def request_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    headers: dict[str, str],
    method: str = "POST",
    payload: dict[str, object] | None = None,
    params: dict[str, QueryValue] | None = None,
) -> httpx.Response:
    """Issue one provider request under the shared policy: retry transport faults,
    429 and 5xx; never retry an auth failure.

    `method`/`params` exist because the contract is not POST-only: TinyFish's search
    and Keenable's fetch are GET endpoints that carry their arguments in the query
    string. Routing them through the same function keeps one retry policy for every
    provider rather than letting the GET adapters grow their own.
    """
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
                response = await client.request(
                    method, url, headers=headers, json=payload, params=params
                )
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

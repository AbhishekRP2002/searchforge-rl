"""The only module that reads vf.Trace. Keeps verifiers coupling in one place.

Evidence is reconstructed from the trace rather than accumulated in vf.State:
the trace is durable, race-free, and holds exactly the content the agent saw.
"""

from __future__ import annotations

import verifiers.v1 as vf

from searchforge.evidence import urls_in_text

NATIVE_SEARCH_ITEM_TYPES = {"web_search_call", "web_search_tool_result"}


def tool_result_text(trace: vf.Trace) -> str:
    """Every tool result this episode produced, concatenated."""
    parts = []
    for message in trace.tool_messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif content:
            parts.append(str(content))
    return "\n\n".join(parts)


def retrieved_urls(trace: vf.Trace) -> set[str]:
    return urls_in_text(tool_result_text(trace))


def tool_call_count(trace: vf.Trace) -> int:
    return len(trace.tool_messages)


def tool_call_counts(trace: vf.Trace) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in trace.tool_messages:
        name = message.name or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def native_search_citations(trace: vf.Trace) -> list[str]:
    """Provider-side web search bypasses our toolset entirely: the search runs on
    the model provider's infrastructure, so those actions are not tokens our
    policy emitted and our tool results are empty. Non-zero voids a comparison run.

    Native items live on `AssistantMessage.provider_state`, the opaque per-provider
    item list. `Trace.calls` holds ModelCall records — timing, usage, finish reason,
    no message — so the messages are the only route to this evidence.
    """
    found: list[str] = []
    for message in trace.assistant_messages:
        for item in message.provider_state or []:
            if item.get("type") in NATIVE_SEARCH_ITEM_TYPES:
                found.append(item.get("type", ""))
            # Anthropic sends content as a str; only the Responses-style list of
            # parts carries annotations.
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                for annotation in part.get("annotations") or []:
                    if annotation.get("type") == "url_citation":
                        found.append(annotation.get("url", ""))
    return found

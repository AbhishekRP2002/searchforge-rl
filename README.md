# SearchForge

SearchForge is a Verifiers v1 environment for paired evaluation of web-search
agents across interchangeable search providers. It exposes two MCP tools—
`web_search` and `web_fetch`—and records provider-attributable rewards,
diagnostics and traces.

Seven providers ship: **Serper**, **Exa**, **Firecrawl**, **Tavily**,
**Parallel**, **TinyFish** and **Keenable**. Each supplies both tools and
normalises to the same five search fields (`title`, `url`, `excerpt`, `date`,
`rank`) and four page fields (`url`, `title`, `text`, `date`). Any two of them
can form a paired comparison arm.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Fill `.env` locally. It is ignored by git:

```dotenv
SERPER_API_KEY=...
EXA_API_KEY=...
FIRECRAWL_API_KEY=...
TAVILY_API_KEY=...
PARALLEL_API_KEY=...
TINYFISH_API_KEY=...
KEENABLE_API_KEY=...
PRIME_API_KEY=...
```

Only the providers you actually run need a value; a blank key skips that
provider's live check rather than failing it. Provider keys are read only by the
separate MCP server. They are not forwarded to the harness runtime or serialized
into traces. `PRIME_API_KEY` is used by the documented model/judge endpoint.

## Design defaults

- Any two providers form a paired comparison over the same `source_id` rows.
- The harness has `network_allow=[]`; provider HTTP originates from the separate
  tool server.
- One retry policy covers every provider, in `providers/transport.py`: transport
  faults, 429 and 5xx retry up to three attempts with exponential jitter.
  Authentication faults are never retried.
- Every adapter calls its provider with ordinary default settings. Two exceptions
  are deliberate and named in code, because leaving the default would have capped
  or advantaged one arm alone: Keenable's server-side `max_chars` (50,000 by
  default) is raised out of the way, and Parallel's `max_results` (10 by default)
  is set explicitly to the configured `num_results`.
- Provider features that answer the question for the agent stay off: Serper's
  `answerBox`, Tavily's `include_answer`, Exa's `deep` modes and Parallel's
  `objective` are capability asymmetries, not retrieval quality.
- There are deliberately no snippet or page character caps. Token and tool-result
  behavior is measured before any truncation policy is chosen.
- The system prompt states the enforced shared tool-call budget. MCP injects tool
  descriptions and argument schemas at runtime, so the prompt does not duplicate
  them.

## Offline verification

```bash
uv run pytest
uv run eval searchforge --dry-run --no-push \
  --env.taskset.providers '["serper", "exa"]'
```

The default dataset is a package asset at
`searchforge/data/tasks_v0.jsonl`, so installed wheels do not depend on a repository
checkout.

## Live provider checks

These tests are excluded from the default suite and skip a provider whose key is
empty:

```bash
uv run pytest -m live tests/live -v
```

## Full comparative live gate

```bash
uv run --env-file .env eval searchforge \
  --env.agent.harness.id null \
  --env.agent.runtime.type docker \
  --env.agent.runtime.allow '[]' \
  --env.taskset.providers '["serper", "exa"]' \
  -m openai/gpt-5.4-nano \
  -n 8 -r 1 -c 1 --no-push
```

Validate the resulting real trace file:

```bash
uv run python scripts/live_gate.py \
  outputs/<run>/traces.jsonl serper exa
```

The gate checks both provider arms, MCP advertisement, actual use of both tools,
judge execution, successful trace serialization, zero native-search bypasses and
classified failures. Docker/network-policy behavior must also be retained in the
run logs; configuration alone is not proof of enforcement.

## Matrix analysis

`scripts/matrix.py` reads the Verifiers v1 serialized shape, including
`rewards.<name>.score`, `task.data.source_id`, metrics and errors:

```bash
uv run python scripts/matrix.py \
  model-a=serper=outputs/serper/traces.jsonl \
  model-a=exa=outputs/exa/traces.jsonl
```

The implementation plan and current gate status are in
[Plan 1](docs/superpowers/plans/2026-09-17-evaluation-environment.md) and the
[implementation ledger](docs/LEDGER.md).

# SearchForge

SearchForge is a Verifiers v1 environment for paired evaluation of web-search
agents across Serper and Exa. It exposes two MCP tools—`web_search` and
`web_fetch`—and records provider-attributable rewards, diagnostics and traces.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Fill `.env` locally. It is ignored by git:

```dotenv
SERPER_API_KEY=...
EXA_API_KEY=...
PRIME_API_KEY=...
```

Provider keys are read only by the separate MCP server. They are not forwarded to
the harness runtime or serialized into traces. `PRIME_API_KEY` is used by the
documented model/judge endpoint.

## Design defaults

- Serper and Exa form a paired comparison over the same `source_id` rows.
- The harness has `network_allow=[]`; provider HTTP originates from the separate
  tool server.
- Serper retries transport faults, 429 and 5xx responses up to three attempts with
  exponential jitter. Exa applies the same policy. Authentication faults are never
  retried.
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

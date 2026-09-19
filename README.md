# SearchForge

**A packaged environment for training and evaluating web-search agents with RL,
where the search backend is a measured variable rather than a fixed assumption.**

## Why this exists

Teams training search agents with RL tune the model, the prompt, the reward and
the algorithm — and treat the search tool as fixed plumbing. It is not. The
backend an agent trains on shapes how fast it learns, how many calls it makes
per rollout, and how well it performs once deployed.

Exa published a controlled experiment making exactly this point: same model
(Qwen3-4B), same prompt, same training data, same RL algorithm, same grader —
only the search API differed.

| Measurement | Exa | SERP |
| --- | --- | --- |
| SimpleQA pass@1 | 0.767 | 0.692 |
| Search results containing the answer | 36.1% | 32.6% |
| Search calls to reach equal performance | baseline | +62% |
| Training tokens to match final performance | −69% | baseline |

That result may well be right. But it is a vendor benchmark where the vendor's
product is the treatment arm, the control is an unnamed "SERP (Google proxy)",
and the grader is the vendor's own. Nobody outside can re-run it.

SearchForge is the apparatus to re-run it — for any model, against any provider,
with the grader and the reward under your control.

## What it gives you

- **A provider-attributable baseline, with no GPU.** Run your model against
  several search backends over the same questions and get accuracy, retrieval
  quality, tool usage, token cost and failure rates per arm. This is the primary
  entry point and it is cheap.
- **A paired experimental design by default.** Every question runs on every
  provider and arms join on `source_id`, so task difficulty cancels and the test
  runs on discordant pairs. Unpaired sampling at realistic n cannot resolve a
  7-point effect; pairing can, at the same cost.
- **An RL environment, not just a benchmark.** The same taskset, tools, rewards
  and traces feed prime-rl for training a small open-weight model, then
  re-evaluating the checkpoint in the same harness.
- **Rewards that are hard to game.** One reference-based judge at weight 1.0,
  a truncation penalty, and everything else measured at weight 0 — because RL
  optimises whatever you score, including paths you did not intend.

## The knobs

The search provider is the first configurable variable, and the one this
project exists to measure. It is not the only one, and more will follow.

| Config path | What it swaps | Status |
| --- | --- | --- |
| `env.taskset.providers` | Search backend, and the comparison arms | **Seven shipped** |
| `env.taskset.task.tools.*` | Result count, rendering, call budget, answer-box policy | Shipped |
| `env.taskset.task.judges` | Judge model and prompt | Shipped |
| `env.taskset.task.rewards` | Your reward by import path; re-weight ours | Shipped |
| `env.taskset.task.metrics` | Diagnostics, same mechanism | Shipped |
| `model` | Any endpoint | Shipped |
| `env.agent.harness.id` | `null`, `bash`, `codex`, `claude_code`, your own | Shipped |
| `env.agent.runtime.type` | `subprocess`, `docker`, `prime`, `modal` | Shipped |
| Task mixture and sources | Which benchmarks, in what proportion | Dataset design open |
| Training recipe | prime-rl integration | Next milestone |

**The discipline:** any run may change anything. A run *labelled as a provider
comparison* asserts that everything except the provider matched its sibling, and
says so loudly when it did not.

## Status

Evaluation works end to end and is proven against live providers. Training is
not yet wired.

| | |
| --- | --- |
| Environment, tools, rewards, judge, trace analysis | Working |
| Seven provider adapters, live-verified | Working |
| Paired Serper × Exa baseline on the seed set | Produced — a null result at n=8 |
| Task mixture from five published benchmarks | Built, 300 rows, **not yet re-validated** |
| prime-rl training | Not started |

## The task mixture

Questions are borrowed; the **distribution is designed here**, and that is what
separates a dataset from a benchmark. Every row carries `source_dataset`, so any
result slices by where the question came from — the difference between "this
provider is better" and "this provider is better on multi-hop Wikipedia
questions".

`searchforge/data/tasks_v1.jsonl` — 300 rows, built reproducibly from seed 0:

| Source | Rows | Hops | Why it is in the mixture |
| --- | --- | --- | --- |
| SimpleQA | 60 | 1 | Single-hop: raw retrieval, no planning |
| HotpotQA | 60 | 2 | Two-hop bridge and comparison |
| MuSiQue | 60 | 2–3+ | Harder composition, hop count in the source id |
| 2WikiMultihopQA | 60 | 2–3+ | Multi-hop over a different Wikipedia slice |
| FRAMES | 60 | 3+ | Multiple constraints per question |

Sizing is not arbitrary. McNemar on paired samples needs roughly 277 pairs to
detect a 7.5-point effect at 80% power assuming a 20% discordance rate; 300
clears that. The 8-row seed set (`tasks_v0.jsonl`, still the packaged default
for smoke runs) could not — it produced 7 ties out of 7.

```bash
# regenerate, or change the mixture
uv run --with datasets python -m scripts.build_taskset --hotpotqa 100 --simpleqa 40

# run against it
uv run eval searchforge --env.taskset.data_path searchforge/data/tasks_v1.jsonl
```

**Two honest caveats.**

*Axes are derived, never guessed.* `hops` and `answer_type` come from what the
source actually determines — MuSiQue encodes hops in its id, SimpleQA ships a
native answer type, 2Wiki gives evidence triples. `popularity` and
`evidence_depth` need an external signal or a pilot run, so they are written as
`unknown` rather than fabricated. A made-up axis is worse than a missing one,
because the analysis would slice on it and believe the result.

*Rows are not re-validated.* Spec section 9 requires every borrowed item to be
checked against the live web and stamped. `validated_on` is empty, and stays
empty until that pass runs.

BrowseComp has a loader but is **off by default and not shipped**. OpenAI
publishes it encrypted so the questions stay out of training corpora; writing
decrypted rows into a public repository would help contaminate it. Generate a
local sample with `--browsecomp N` and leave the output untracked. It is also a
poor training signal — built so answers are hard to find, so at a 10-call budget
most rollouts fail and a group with all-zero rewards yields no GRPO advantage.

## Providers

Seven ship, each supplying both tools and normalising to the same five search
fields (`title`, `url`, `excerpt`, `date`, `rank`) and four page fields (`url`,
`title`, `text`, `date`). Any two form a paired comparison arm.

| Provider | Search | Fetch |
| --- | --- | --- |
| Serper | Google SERP | `scrape.serper.dev` |
| Exa | neural index, `type=auto` | contents API |
| Firecrawl | `/v2/search` | `/v2/scrape` |
| Tavily | `/search` | `/extract` |
| Parallel | `/v1/search` | `/v1/extract` |
| TinyFish | `api.search.tinyfish.ai` | `api.fetch.tinyfish.ai` |
| Keenable | `/v1/search` | `/v1/fetch` |

Adding one is a file plus a registry entry; the shared contract suite runs
against it automatically.

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

Datasets are package assets under `searchforge/data/`, so installed wheels do
not depend on a repository checkout. `tasks_v0.jsonl` (8 rows) stays the default
because it keeps a smoke run cheap; pass `--env.taskset.data_path` to select the
300-row `tasks_v1.jsonl` mixture.

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

## Project documents

The spec, the implementation plans and the ledger live under `docs/` and are
deliberately untracked (`.gitignore`), so they are present in a working
checkout but not in the repository:

- `docs/spec/searchforge-v0.md` — the technical specification
- `docs/superpowers/plans/` — per-phase implementation plans
- `docs/LEDGER.md` — what is implemented, what is proven live, and every
  locked decision with its reason

The ledger is the authority on status. This README describes intent; the ledger
records which gates have actually been met.

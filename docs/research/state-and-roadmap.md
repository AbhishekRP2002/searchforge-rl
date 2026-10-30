# SearchForge-RL: where it stands, what others do, what to build next

*Research note, 2026-10-30. Covers `main` at `bd06111`. No code was changed to write it.*

**Contents**

1. [Current state](#1-current-state)
2. [Pending work](#2-pending-work)
3. [Outside landscape](#3-outside-landscape)
4. [Recommendations and roadmap](#4-recommendations-and-roadmap)

**Short version.** SearchForge is a small, careful evaluation environment
(about 1,500 lines of library and script code, 200+ tests) for web-search agents.
Its one idea is that the search backend is an experimental variable. It runs every
question against several search providers and compares them pair by pair. The
evaluation path works end to end. Training is not connected yet. Three gaps stand
between it and a credible result: page text has no length limit, so every rollout
overflows a 4k-token training context; no saved fingerprint proves that two arms
differed only in the provider; and search results are never cached, so runs are
expensive and cannot be repeated exactly. Nobody else in the field treats "which
search API?" as a controlled variable. That is a real niche, and the roadmap
below builds on it.

---

## 1. Current state

### 1.1 What it is, in one paragraph

SearchForge is a **task set plugin for Prime Intellect's `verifiers` library
(pinned at 0.3.1, the `verifiers.v1` API)**. `verifiers` handles the agent loop,
model calls, sandboxes, trace files and the eval command (`uv run eval`).
SearchForge supplies four things:

- **questions**: JSONL rows
- **two tools**: `web_search` and `web_fetch`, served from an MCP server
- **a scorer**: an LLM judge that compares the answer with a gold reference
- **analysis scripts**: they read the trace files and compare providers

### 1.2 Package layout

```
searchforge/
  __init__.py         exports the taskset and the judge under the plugin id "searchforge"
  taskset.py          task rows -> Task objects; rewards and metrics          (185 lines)
  servers/tool.py     MCP server: web_search / web_fetch, budget, parallel gate (257)
  providers/          one adapter per search API + shared HTTP retry layer
    base.py           SearchResult / PageContent dataclasses, ProviderError, Protocol
    transport.py      one retry policy for every provider (3 tries, jittered backoff)
    serper.py exa.py firecrawl.py tavily.py parallel.py tinyfish.py keenable.py
  render.py           results -> markdown (default) or JSON text the model sees
  evidence.py         URL normalising, citation grounding, "is the answer in the text"
  trace_reader.py     the only module that reads a vf.Trace
  judge.py prompts.py system prompt, judge prompt, judge config
  data/tasks_v0.jsonl 8 hand-made seed questions (default, for smoke runs)
  data/tasks_v1.jsonl 300 questions sampled from 5 public benchmarks
scripts/
  build_taskset.py    rebuilds tasks_v1 from Hugging Face datasets, seed 0
  matrix.py           per-arm summary table + paired provider comparison
  live_gate.py        fail-closed checks that a live run is valid
configs/eval.toml     launch defaults: null harness, docker runtime, serper+exa
```

### 1.3 Start to finish: one evaluation run

Take one question from `tasks_v1.jsonl`:

```json
{"source_id": "simpleqa:ba166cd1c47e",
 "question": "What was the original name of the village of Berrien Springs, Michigan?",
 "answer": "Wolf's Prairie", "hops": 1, "answer_type": "entity",
 "source_dataset": "simpleqa", "recency": "unknown", "popularity": "unknown",
 "evidence_depth": "unknown", "validated_on": ""}
```

You run `uv run eval @ configs/eval.toml --env.taskset.providers '["serper","exa"]'`.

1. **Load (`taskset.py:SearchForgeTaskset.load`).** It reads the rows and keeps
   those inside `hops_min..hops_max`. It then builds the **cross-product of rows ×
   providers**, so this question becomes two tasks, `simpleqa:ba166cd1c47e::serper`
   and `...::exa`. Each task gets the same system prompt (`prompts.py`), which
   states the tool budget (10 calls by default). Each task also gets its own copy
   of the tool config with `provider` set.
2. **Tools (`servers/tool.py`).** For each rollout, `verifiers` starts a
   `WebToolset` MCP server **outside** the agent's sandbox (`colocated=False`).
   Only that server reads the provider API keys, so keys never enter the sandbox
   or the traces. The sandbox itself has `network_allow=[]`. The model sees two
   tools:
   - `web_search(query)`: calls the provider, gets back `SearchResult(title, url,
     excerpt, date, rank)` for 5 hits, and renders them as compact markdown.
   - `web_fetch(url)`: gets back `PageContent(url, title, text, date)` and returns
     the **full page text, uncapped** (this is deliberate for now; see 1.6).
   - Both tools share one budget counter (`max_tool_calls=10`). Calls past the
     budget get "budget exhausted, answer now".
   - Parallel calls are refused by default, and a refused call is not charged.
   - Provider faults are retried 3× (`transport.py`). They then come back to the
     model as readable messages, `SEARCH_PROVIDER_UNAVAILABLE` or
     `..._RATE_LIMITED`. An auth error is raised, not hidden.
3. **Agent loop.** `verifiers` runs it. With the `null` harness, the model calls
   the MCP tools directly. Deadlines are 900 s for the agent and 300 s for scoring
   (`taskset.py:32-42`). These were added after a stuck container sat idle for
   about 7 hours.
4. **Scoring.** These read the finished trace, not live state:

   | Name | Weight | What it measures |
   |---|---|---|
   | `searchforge` (judge) | **1.0** | `gpt-5.4-nano` sees question + gold + answer → yes/no. Reference-based, so it checks the fact, not plausibility. |
   | `context_overflow` | **−0.25** | 1 if the rollout was truncated (any length or turn limit). |
   | `answer_in_results` | 0 | Did the gold answer appear anywhere in the tool output? Separates retrieval failure from reasoning failure. |
   | `citations_grounded` | 0 | Share of URLs in the answer that this episode actually retrieved. No citations scores 1.0. |
   | `citation_presence` | 0 | Did the answer cite anything? |
   | metrics | — | tool calls, search calls, fetch calls, total tokens, `native_search_calls` (a canary: if non-zero, the model used its provider's built-in search and the run is void). |

   Weight-0 rewards are logged but do not affect training. That is deliberate.
   Anything that is scored gets optimised, including ways around the scorer.
5. **Output.** `verifiers` writes `outputs/<run>/traces.jsonl`. Each line holds
   one episode, and each episode holds a list of traces.
6. **Analysis.**
   - `scripts/live_gate.py traces.jsonl serper exa` fails closed unless all of
     these hold: both arms ran, both tools were advertised and used, the judge
     ran on every trace, no native search happened, and no failure went
     unclassified.
   - `scripts/matrix.py model=serper=a.jsonl model=exa=b.jsonl` prints one row
     per arm. It shows accuracy with a Wilson 95% interval, answer-in-results,
     "extraction gap" (answer-in-results minus accuracy), citation stats, mean
     calls and tokens, and failure rate. It then prints a **paired** comparison
     joined on `source_id`: the delta with its CI, wins, losses and ties, and an
     exact McNemar p-value.

### 1.4 How evidence and retrieval work

- **One contract for every provider.** All seven adapters reduce to 5 search
  fields and 4 page fields. Any two providers can be compared because the model
  sees the same shape.
- **Provider features that "answer for you" are off** in every arm: Serper's
  answerBox, Tavily's `include_answer`, Exa's deep modes and Parallel's
  `objective`. Two defaults were changed on purpose to keep arms equal: Keenable's
  server-side 50k-character cap is lifted, and Parallel's `max_results` is set to
  `num_results`.
- **Grounding is checked against the episode's own retrievals**, never a gold
  URL list. A gold URL list would favour whichever provider built the dataset
  (`evidence.py` docstring).
- **There is no cache.** Every call goes to the live API. Each call also opens a
  fresh `httpx.AsyncClient` (`serper.py:27`, and the same in the other adapters),
  so connections are not reused.

### 1.5 The task mixture

300 rows, 60 from each source, built with seed 0 by `scripts/build_taskset.py`:

| Source | Hops recorded | Notes |
|---|---|---|
| SimpleQA | 1 | native answer type kept |
| HotpotQA (distractor, val) | 2 | yes/no answers dropped |
| MuSiQue (val, answerable) | 2–3 (from id) | |
| 2WikiMultihopQA (val) | 2–3 (from evidence count) | yes/no dropped |
| FRAMES (test) | 2–3 (from link count) | some answers are sentences, e.g. "Yes, … are both Italian" |

The labels are honest: `popularity` and `evidence_depth` are `unknown` on all 300
rows, `recency` is `unknown` on 120, and `validated_on` is empty on all 300. No
axis was invented. The `tasks_v0` seed set has every label filled in but only 8
rows. BrowseComp has a loader that is off by default and never committed, to
avoid leaking the encrypted test set.

### 1.6 Measured facts already recorded in the repo

From the README and code comments, based on 16 live rollouts and 29 fetches:

- Search output is small: median 1,304 characters, max 1,561.
- Fetch output is bimodal: 20 of 29 fetches are ≤24.6k characters, and the rest
  jump to 107k–162k.
- The median rollout is about 20k tokens. The reference training `seq_len` is
  4,096, so **every rollout would truncate and take the −0.25 penalty**. A GRPO
  group with identical rewards produces no learning signal. This is the main
  blocker for training.
- The 8-row paired Serper × Exa baseline was a null result: 7 ties out of 7
  paired questions.

### 1.7 About the "recent commits" named in the brief

I searched every branch on `origin` (`main`, `claude/relaxed-einstein-wofl0f`,
`feat/evaluation-environment` and `feat/provider-adapter-coverage`) for these
commits: *retrieval diagnostic split, self-splitting analysis cells, comparison
fingerprint, source slicing, verifier network-notice fix, lint sweep*. **None
exist.** The last real commits are:

| Commit | What it did |
|---|---|
| `9bc76e0` | Gate parallel tool calls; add agent and scoring deadlines |
| `4a5ef69` | 300-row mixture from 5 benchmarks |
| `07e4bfe`, `ed9b479`, `bd06111` | README rewrites, adding the "where this is going" section |
| `d3dc10a`, `8974694`, `c35e7f4` | Five more providers, one transport, one contract test suite |
| `d4326d0` | Fix: read episodes, not traces, from `traces.jsonl` |

Several named items are *planned* in the README but **not built**:

- **Comparison fingerprint.** The README says `task.config` is `null` in traces,
  so `matrix.py` cannot check that arms matched.
- **Source slicing.** Every row has `source_dataset`, but `matrix.py` never groups
  by it.
- **Retrieval diagnostic split.** `answer_in_results` and `extraction_gap`
  exist. A per-source or per-hop split of them does not.
- **Lint sweep.** No lint config is committed. Running ruff gives 10 findings
  (below).

They may sit on a local branch that was never pushed. They are listed as pending
work in section 2.

### 1.8 Test results

The run: `uv sync --extra dev && uv run pytest` on Python 3.13, with no network
needed. **203 passed, 1 skipped, 7 deselected**, in about 9–15 s.

- **Skipped (1):** `tests/test_provider_contract.py:195`. For Serper, "serper
  always returns a page object". This is an expected skip inside the
  parameterised contract suite, not a gap.
- **Deselected (7):** `tests/live/`, which runs only with `-m live`. Run
  explicitly, all 7 **skipped**, because this container has no provider keys.
  So the live adapters were **not** verified here.
- **Dry run:** `uv run eval searchforge --dry-run --no-push --env.taskset.providers
  '["serper","exa"]'` resolves its config fine. Note that without `@
  configs/eval.toml`, the output directory name shows the `bash` harness. The
  README's dry-run command therefore does not exercise the null-harness config
  that real runs use.
- **Lint (not part of CI):** `ruff check .` reports 10 findings:
  - 4 unused imports in tests
  - 3 unsorted import blocks
  - 1 `typing.Callable` that should be `collections.abc.Callable`
  - `SIM103` at `servers/tool.py:173`
  - `ISC004` at `scripts/matrix.py:232`

  8 can be fixed automatically. None is a bug.
- **No CI.** The repo has no `.github/workflows`.

---

## 2. Pending work

There are no `TODO`, `FIXME`, `XXX` or `NotImplementedError` markers in the
code. The one marker is a `ponytail:` note at `servers/tool.py:142`, which is a
deliberate design note. The pending work below comes from the README, unused
fields, and gaps found by reading the code.

Priority key: **P0** blocks a credible result or training. **P1** is needed for
the project to be useful to others. **P2** is good to have.

| # | Pri | Item | Evidence (file:line) | Why it matters |
|---|---|---|---|---|
| 1 | P0 | **Cap page text, the same way for every arm.** No cap exists; fetches reach 162k chars. | `servers/tool.py:236-249`, README "Context pruning…" | Every rollout truncates at a 4k `seq_len`, so there is no GRPO signal. |
| 2 | P0 | **Comparison fingerprint.** Write a hash of the full toolset, judge and prompt config into each trace, and make `matrix.py` refuse to compare arms whose hashes differ in more than `provider`. | README "Today's gap: … `task.config` is `null`"; `scripts/matrix.py:121-182` checks only `source_id` | Without it, a "provider comparison" can silently differ in other ways. This is the project's core promise. |
| 3 | P0 | **Validate the 300 rows against the live web** and fill `validated_on`. | all 300 rows have `validated_on: ""`; README "Rows are not re-validated" | Stale gold answers turn into fake provider differences. |
| 4 | P0 | **Run the 300-row paired baseline** (at least Serper × Exa) and publish it. | README status table | Only an 8-row null result exists today. |
| 5 | P0 | **prime-rl training recipe.** | README knobs table: "Next milestone" | This is what makes it an RL environment and not only an eval. |
| 6 | P1 | **Slice results by `source_dataset` and `hops`** in `matrix.py`. | `matrix.py:62-108` only aggregates the whole cell | The README promises "better on multi-hop Wikipedia questions"; the tool cannot show that yet. |
| 7 | P1 | **`matrix.py` handles only exactly 2 providers per model.** | `scripts/matrix.py:226-229` raises otherwise | Seven providers ship. N-way comparison needs pairwise output with a multiple-comparison correction. |
| 8 | P1 | **Search/fetch cache**, keyed on (provider, endpoint, args) and stored on disk. | none exists; every call is live (`servers/tool.py:198,236`) | Cost, rate limits, reproducibility, and replay for training. |
| 9 | P1 | **Reuse HTTP clients.** A new `httpx.AsyncClient` is made per call. | `providers/serper.py:27,57` (and the others); `build_provider` per call at `servers/tool.py:198,236` | TLS setup on every call adds latency under load. |
| 10 | P1 | **Fields that are loaded but never used:** `recency`, `popularity`, `evidence_depth`, `answer_type` and `validated_on`. Only `hops` filters, and nothing slices. | `taskset.py:57-66`; `build_taskset.py` writes `unknown` | Either fill them from a pilot run, or drop them from the claims. |
| 11 | P1 | **`answer_in_results` is a plain substring match.** It misses long FRAMES answers ("Yes, X and Y are both Italian"; up to 54 words) and different date formats. | `evidence.py:57-64`; 11 FRAMES answers over 5 words | Under-counts retrieval success, so `extraction_gap` is biased per source. |
| 12 | P1 | **FRAMES yes/no-style answers are not filtered** (HotpotQA and 2Wiki are). | `build_taskset.py:130-143` vs `:150,186` | Inconsistent filtering: "Yes, …" answers grade close to a coin flip. |
| 13 | P1 | **`eval()` on dataset metadata.** | `build_taskset.py:118` | This runs code from a downloaded dataset. Use `ast.literal_eval`. |
| 14 | P1 | **CI + lint config.** No workflow and no ruff/pyright config; 10 ruff findings. | repo root; `servers/tool.py:173`, `scripts/matrix.py:232`, tests | Keeps the contract suite green for contributors. |
| 15 | P1 | **Live adapter checks are manual only.** | `tests/live/test_live_providers.py:25` skips without keys | Provider APIs drift. A scheduled job with keys would catch it. |
| 16 | P2 | **Dry-run doc command skips `configs/eval.toml`**, so it resolves the `bash` harness. | README "Offline verification" | A smoke run should use the real config. |
| 17 | P2 | **`live_gate.py` fails any run with a single failed trace.** | `scripts/live_gate.py:68-69` | Fine for a gate. At 300 × 7 arms, add a failure-rate threshold. |
| 18 | P2 | **`list` answers.** The schema allows `str \| list[str]`, but all 300 rows are strings, and two are classed `list` by shape only. | `taskset.py:55`, `build_taskset.py:54-71` | List grading paths go untested on real data. |
| 19 | P2 | **RL algorithm knob** (GRPO only; verl and other estimators). | README knobs table: "Planned" | Long-term. |
| 20 | P2 | **Interpretability analyses** (did query wording change after training, stopping behaviour, fetch vs re-search). | README "Interpretability…" | The data is already in the traces. Only scripts are missing. |
| 21 | P2 | **Scale check at 32B.** | README "Scale is a validity question" | Long-term research question. |

---

<!-- PART3 -->

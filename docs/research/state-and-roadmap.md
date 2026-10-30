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

## 3. Outside landscape

*Sources were checked in October 2026, from repos, papers and official docs.
"(unverified)" marks claims I could not confirm from a primary source.*

### 3.1 Environment frameworks: how each one defines an environment

| Framework | How an environment is defined | Served as | Sandboxing | Search-relevant |
|---|---|---|---|---|
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | `reset() → obs`, `step(a) → obs, reward, terminated, truncated, info` | in-process | none | The shape most LLM environment APIs copy |
| [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | Multi-agent Gym (turn-based or parallel) | in-process | none | — |
| [verifiers](https://github.com/PrimeIntellect-ai/verifiers) (what SearchForge uses) | v1: `Taskset.load() → Task` objects. A Task has `@vf.reward` and `@vf.metric` methods scored on a `Trace`. A **Harness** is where the model runs (null, bash, Claude Code, Codex), and Toolsets are MCP servers. | **ZMQ + msgpack env server** (`verifiers/v1/serve`, confirmed in 0.3.1). Model traffic goes through an interception server. | subprocess, docker (proxy-enforced network policy), Prime sandboxes, Modal | Ships `wiki_search` and a `web_search_interception` example. **0.4.0 is out (Oct 12 2026); SearchForge pins 0.3.1** |
| [Environments Hub](https://app.primeintellect.ai/dashboard/environments) / [prime-environments](https://github.com/PrimeIntellect-ai/prime-environments) | Published verifiers packages; `prime env push`, `prime eval run owner/env` | Hub registry | via verifiers | Already lists `browsecomp_plus`, `simpleqa_verified_2`, `search_r1_ish`, `hle_web_py`, `deep_consult` |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | Consumes verifiers envs directly | Async trainer (FSDP2) / inference (vLLM) split, `max_off_policy_steps` | — | Wiki Search multi-turn example |
| [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) | Dataset (JSONL) + agent + **resources server** (`seed_session`, tool endpoints, `verify() → reward`) | **3 FastAPI services**: model, agent, resources | Docker, Daytona, ECS, Apptainer, … | **Closest overlap**: `google_search`, `tavily_search` (LLM judge), `browsecomp_advanced_harness`, `deepsearchqa`, `widesearch` |
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta + HF) | Gym-style `reset/step/state` | **WebSocket**. Each environment is a Docker image you can push to HF Spaces; MCP tools at `/mcp` | Docker / HF Spaces | Hosts BrowserGym. Used by TRL, SkyRL, ART, Unsloth. Marked "early development" |
| [TRL](https://github.com/huggingface/trl) | `GRPOTrainer(environment_factory=Env)`: public methods with docstrings become tools; `rollout_func` is the escape hatch | in-process or OpenEnv | via OpenEnv/Harbor | — |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Subclass `AgentInstanceBase` with async `reset`/`step` → reward, feedback text, done | in-process, on Ray | none | — |
| [verl](https://github.com/volcengine/verl) | `AgentLoopBase.run()`; tools as `BaseTool` in a YAML registry; `rollout.mode=async` | in-process, on Ray | Sandbox Fusion (code) | Search-R1 recipe built in |
| [SkyRL / SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL) | `BaseTextEnv.init/step` returning OpenAI messages + reward | in-process; Harbor integration | via Harbor | `envs/search` (Search-R1) |
| [rLLM](https://github.com/rllm-org/rllm) | `@rllm.rollout def solve(task) → Episode` + evaluator; wraps 10+ CLI harnesses | — | Docker, Daytona, Modal | 60+ benchmarks incl. search |
| [ART](https://github.com/OpenPipe/ART) (OpenPipe) | You write the rollout in plain Python; **RULER** = LLM judge that ranks a group, needing no reference; MCP·RL | client → training backend | none | ART·E email-search agent |
| [Atropos](https://github.com/NousResearch/atropos) (Nous) | `BaseEnv` microservices push scored groups to a trajectory API | HTTP | — | deep-research env. **Repo archived** |
| [GEM](https://github.com/axon-rl/gem) | Gym API for LLMs, async vectorised, tool wrappers | in-process | — | Search tool wrapper; adopted by [ROLL](https://github.com/alibaba/ROLL) |
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | Agents run unchanged; a gateway captures model calls for training | gateway + rollout controller | k8s Jobs | — |
| [AReaL](https://github.com/inclusionAI/AReaL) / [slime](https://github.com/THUDM/slime) | Trainers; AReaL wraps black-box agents via a `base_url` proxy | — | — | AReaL powers ASearcher |
| [AgentGym / -RL](https://github.com/WooooDyy/AgentGym) | One **HTTP EnvServer** per env (`reset`, `step`, `observation`) | HTTP | per env | SearchQA env |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym), [WebArena](https://github.com/web-arena-x/webarena), [WebArena-Verified](https://github.com/ServiceNow/webarena-verified) | Gym over Playwright; self-hosted sites; WebArena-Verified uses deterministic evaluators and offline replay of network traces | in-process + Docker sites | Docker | Browsing, not search APIs |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym), [OpenHands](https://github.com/OpenHands/OpenHands), [Harbor](https://github.com/laude-institute/harbor) | Container tasks with tests; Harbor is becoming the shared task format (verifiers, TRL, SkyRL, rLLM, NeMo-Gym and OpenEnv all read it) | Docker / cloud | Docker, Daytona, Modal | — |
| [Reasoning Gym](https://github.com/open-thought/reasoning-gym) | Procedural generators + `score_answer` | in-process | — | single turn, no tools |

**Search-agent recipes** (the closest prior art):

| Project | Search backend during RL | Reward | Notes |
|---|---|---|---|
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1) ([paper](https://arxiv.org/abs/2503.09516)) | Local Wikipedia-2018 + E5 + FAISS, top-3, served at `POST /retrieve` | Exact match | Masking retrieved tokens out of the loss matters (0.431 vs 0.343) |
| [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch) ([paper](https://arxiv.org/abs/2505.04588)) | A fine-tuned LLM **simulates** the search engine | EM/F1 | No API cost; press reports ~88% cheaper (unverified) |
| [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | Local retrieval, then web | outcome | Two-stage RL |
| [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) ([paper](https://arxiv.org/abs/2504.03160)) | **Real web** (Serper/Bing) + browsing LLM | F1 | Bursts of 4,096 concurrent requests, a 50-node CPU tool cluster, caching, rate-limit pain |
| [ASearcher](https://github.com/inclusionAI/ASearcher) ([paper](https://arxiv.org/abs/2508.07976)) | Local wiki or web | F1 + LLM judge | 100+ tool calls and 400k tokens per trajectory; fully async; training locally transfers to the web |
| [Tongyi DeepResearch / WebSailor family](https://github.com/Alibaba-NLP/DeepResearch) | Serper + page visitor | LLM judge | Synthetic data + agentic mid-training + GRPO |
| [MiroFlow](https://github.com/MiroMindAI/MiroFlow) | Tools as MCP servers | — | Eval framework; GAIA, BrowseComp, xBench |

### 3.2 Serving and scale: what others do

**How environments are exposed.**
- **HTTP/FastAPI services** are the norm: NeMo-Gym, AgentGym and Atropos.
- **OpenEnv** uses WebSocket and **verifiers** uses ZMQ. I found no framework
  that documents gRPC.
- The pattern that is winning is a **three-way split: model server ↔ agent loop
  ↔ tool/verifier server**. The pieces communicate by HTTP, and each piece scales
  on its own. SearchForge already has the tool-server piece: its MCP server runs
  outside the sandbox.

**Keeping rollout separate from training.**
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl),
  [AReaL](https://arxiv.org/abs/2505.24298) and
  [PipelineRL](https://arxiv.org/abs/2509.19128v2) all run rollouts
  asynchronously, on GPUs separate from training, with a bound on how stale the
  policy may get. Each reports roughly 2–2.8× better throughput than synchronous
  training.
- [Kimi K2](https://arxiv.org/abs/2507.20534) stays synchronous. It uses
  "partial rollouts" instead: very long trajectories are paused and resumed in the
  next step.

**Sandboxing.**
- Code and terminal environments need it: Docker, Modal, E2B, Daytona, Prime
  sandboxes or k8s. Kimi K2 ran over 10k sandboxes at once.
- **Search agents mostly don't need one**, because the tools are API calls. The
  real isolation need is the one SearchForge already enforces: credentials stay in
  the tool server, and the harness gets no network egress.

**Search backends at scale.** There are three approaches, and serious projects
use more than one:

1. **Live APIs.** DeepResearcher and Tongyi do this. It is realistic but costly,
   rate-limited and impossible to repeat exactly. DeepResearcher had to add caching
   and a CPU cluster.
2. **A frozen corpus with a local retriever.** Search-R1 uses Wikipedia-2018 +
   E5.
   - [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) offers ~100k curated
     documents with gold evidence.
   - [DeepResearchGym](https://arxiv.org/abs/2505.19253) offers ClueWeb22 +
     FineWeb behind `/search` and `/fetch`.
   - This is cheap and deterministic, and ASearcher reports that training this way
     transfers to the live web.
3. **A simulated search engine** ([ZeroSearch](https://arxiv.org/abs/2505.04588)).

**Bottlenecks people report:**
- **Long-tail rollouts** leave GPUs idle (ASearcher: 100× token spread).
- **Tool latency, rate limits and anti-crawl blocking** (DeepResearcher).
- **Context length**: ASearcher keeps only the last 25k characters, and trains
  the model to summarise pages.
- **Unmasked tool tokens destabilise training.**
- **Reward hacking.** Live search can find the answer key itself. Anthropic
  [documented](https://www.anthropic.com/engineering/eval-awareness-browsecomp) a
  model finding and decrypting BrowseComp answers from GitHub.

### 3.3 How benchmarks get plugged in

**The Search-R1 "standard 7"**
([paper §4](https://arxiv.org/abs/2503.09516)):
- **Datasets**: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue and Bamboogle.
- **Training split**: train on NQ + HotpotQA; the other five are out-of-domain.
- **Metric**: Exact Match.
- **Where it's reused**: SkyRL, verl and the Prime `search_r1_ish` env.
- **Why it matters here**: it is the comparison table reviewers expect.

**Harder agentic sets**:
- **Benchmarks**: FRAMES (824 questions), BrowseComp (1,266, encrypted),
  BrowseComp-Plus, GAIA (text subset), xbench-DeepSearch (encrypted) and HLE.
- **How they're reported**: usually with an LLM judge, as Avg@k or Pass@k
  ([ASearcher](https://arxiv.org/abs/2508.07976)).

**Grader conventions to copy rather than reinvent:**
- **SimpleQA**
  ([simpleqa_eval.py](https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py))
  grades three ways: CORRECT, INCORRECT or NOT_ATTEMPTED. It reports
  accuracy-given-attempted and an F1 score, which rewards a model for abstaining
  rather than guessing.
  - SearchForge's judge is binary, and its prompt *forbids* abstaining. That is a
    deliberate choice, but it should be stated whenever results are compared with
    SimpleQA numbers.
- **BrowseComp**
  ([browsecomp_eval.py](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py))
  extracts an exact answer, has a judge return `correct: yes|no`, and reports
  confidence for calibration. The data is XOR-encrypted with a per-row canary;
  this stops crawlers, not determined models.

**Eval harnesses:**

| Harness | Defines a task as | Agentic tools? | Search benchmarks included |
|---|---|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | YAML + Jinja prompt, generate/loglikelihood | No | QA sets only, closed-book |
| [inspect_ai](https://inspect.aisi.org.uk/) / [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) | dataset + solver/agent (`react()`) + scorer; sandboxes | **Yes**, with a built-in `web_search()` tool (Tavily, Exa, Google, provider-native, …) | `browse_comp`, `gaia`, `assistant_bench`, `mind2web`, `hle`; SimpleQA-verified (unverified) |
| [lighteval](https://github.com/huggingface/lighteval) | Custom tasks/metrics; **uses inspect-ai as its preferred backend** | through inspect | — |
| [OpenAI simple-evals](https://github.com/openai/simple-evals) | Python eval classes | minimal | SimpleQA, BrowseComp graders. **Deprecated Jul 2025**, kept as a reference |
| [HELM](https://github.com/stanford-crfm/helm) | scenarios + metrics | not that I found | — |
| [Prime Hub evals](https://docs.primeintellect.ai/tutorials-environments/getting-started) | the verifiers environment itself | Yes | see 3.1 |
| [liquid4all/search_evals](https://github.com/liquid4all/search_evals) | compares **search APIs** on BrowseComp, SimpleQA, FRAMES, HLE | Yes | Closest to SearchForge's question, but eval only, with no pairing statistics and no RL |

---

## 4. Recommendations and roadmap

### 4.1 Where SearchForge stands against the field

**What it already does better than most:**

- **It treats the search backend as the experimental variable.**
  - Every framework above treats search as fixed plumbing.
  - NeMo-Gym has `google_search` and `tavily_search` as *separate environments*,
    with no paired comparison between them.
  - `search_evals` compares APIs but has no paired statistics and no training.
  - **No one ships a paired, same-question, provider A/B test with McNemar and
    a fail-closed validity gate.** That is SearchForge's niche.
- **Good measurement habits:**
  - weight-0 diagnostics, so diagnostics are not optimised
  - a native-search canary, which voids runs where the model bypassed the tools
  - grounding checked against the episode's own retrievals, never gold URLs
  - "answer-for-you" provider features switched off in every arm
  - axis labels that say `unknown` rather than a guess
- **Sound security defaults.** Keys live only in the tool server, the harness has
  no egress, and the judge prompt treats its inputs as untrusted.
- **It sits on a maintained framework.** It is already on the verifiers v1 API,
  so it gets prime-rl, the Hub, sandboxes and the ZMQ env server without
  building them.

**What is missing, compared with the field:**

| Gap | Who has it | Size of the gap |
|---|---|---|
| A training run of any size | everyone in §3.1 recipes | Large: blocked by #1 (page cap) |
| Cache / replay / frozen-corpus mode | Search-R1, BrowseComp-Plus, DeepResearchGym, DeepResearcher (cache) | Medium |
| Fingerprinted, self-validating comparisons | none do it fully; WebArena-Verified does offline replay | Medium, and it is *the* differentiator |
| Standard benchmark reporting (Search-R1 7 with EM; SimpleQA 3-way; BrowseComp-Plus) | Search-R1, ASearcher, inspect_evals | Small–medium |
| Distribution: listed on the Hub, an OpenEnv/HTTP adapter, PyPI | prime-environments, OpenEnv, NeMo-Gym | Small |
| Answer-leak blocking (HF dataset pages, eval repos) | verifiers interception hooks, Anthropic guidance | Small |
| CI, lint, scheduled live checks | most mature repos | Small |
| Token and cost accounting per arm in dollars | search_evals (partly) | Small |

### 4.2 Positioning

**"The controlled-experiment layer for search in agent RL."** In practice that
means:

1. Bring your model and the search APIs you are considering.
2. Get a paired, fingerprinted, statistically honest comparison, both at eval
   time and after a short RL run.
3. Optionally, train against a cached or frozen backend so results can be
   repeated exactly.

Two groups would use it:

- **Teams choosing a search API** for an agent product. This is eval-only, needs
  no GPU and serves a large audience.
- **Researchers** who need to say which part of a search-agent gain came from the
  retriever.

The search providers themselves are a third audience. They currently publish
vendor-run benchmarks, such as the Exa study the README quotes, and an
independent harness is useful to them too.

Do **not** compete on being a general RL framework, a sandbox host or a browser
environment. The field is crowded there, and verifiers, NeMo-Gym and OpenEnv
already do it well.

### 4.3 Roadmap

Sizes: **S** ≈ ≤2 days, **M** ≈ 1–2 weeks, **L** ≈ 3–6 weeks, all for one
person.

#### Near term (next ~4–6 weeks): make one comparison credible, and train once

| # | Item | Size | Notes |
|---|---|---|---|
| N1 | **Equal page cap + head/tail truncation marker** in `web_fetch`, set by config, recorded in the trace | S | Measure at 4k, 8k and 16k chars against the logged fetch-size distribution. The same cap applies to every arm. Unblocks training (§2 #1). |
| N2 | **Comparison fingerprint.** Hash the toolset config minus `provider`, plus the judge model and prompt hash, system prompt hash, data file hash, model, harness and verifiers version. Store it in each trace's `info`. `matrix.py` refuses to compare arms with different hashes. | S–M | The project's core promise (§2 #2). |
| N3 | **Search/fetch cache** (SQLite or disk, keyed on provider + endpoint + normalised args + date). Modes: `off` / `read-write` / `replay-only`. | M | Cuts cost and rate-limit pain. Makes runs repeatable and replay-only CI possible. Record cache hits per trace so a comparison can say "both arms were live". |
| N4 | **Slice `matrix.py` by `source_dataset` and `hops`, and support N providers** (all pairs, with Holm correction) | S | §2 #6–7. |
| N5 | **Validate the 300 rows** (one pass with two providers + judge; flag disagreements by hand) and stamp `validated_on` | M | §2 #3. |
| N6 | **Publish the 300-row Serper × Exa (+1–2 more) baseline**: a results doc with cost per arm | S (once N1–N5 are done) | First real result; the thing to share. |
| N7 | **Hygiene**: CI (pytest + ruff + dry run with `configs/eval.toml`), `ast.literal_eval`, reuse httpx clients, filter FRAMES yes/no answers, fuzzy `answer_in_results` | S | §2 #9, #11–16. |
| N8 | **Bump to verifiers 0.4.x** and re-run the suite | S | Stays on the maintained line before others depend on it. |
| N9 | **First small prime-rl run** (Qwen3-1.7B/4B, one provider, cached backend, 4k–8k seq_len) | M–L | Proves the reward gives a learning signal after N1. Then repeat on the second provider: that is the Exa experiment, re-run in the open. |

#### Mid term (~2–4 months): make it easy to adopt

| # | Item | Size | Notes |
|---|---|---|---|
| M1 | **Publish on the Prime Environments Hub** (`prime env push`), with the 300-row mixture and the 8-row smoke set | S | Users who already run prime-rl get it for free. |
| M2 | **A standalone HTTP search-tool server** (FastAPI): `POST /search`, `POST /fetch`, `GET /providers`, with a cache, budget and per-call logs. The MCP server becomes a thin wrapper around it. | M | Lets any framework use the provider layer: verl `BaseTool`, TRL `environment_factory`, NeMo-Gym resources, inspect `web_search` custom provider. Also the base for an OpenEnv image. |
| M3 | **Adapters** (thin, one file each): OpenEnv Docker image; TRL `environment_factory` class; inspect_ai task + tool; NeMo-Gym resources server | M total | Each adapter reuses M2. Ship two first: inspect (eval users) and TRL/OpenEnv (training users). |
| M4 | **Benchmark packs** as `build_taskset` sources: Search-R1 7 (NQ, TriviaQA, PopQA, Bamboogle loaders + EM/F1 metrics at weight 0), SimpleQA-verified, BrowseComp-Plus (local only, never committed), GAIA text subset | M | Report EM next to the judge so results line up with the literature. Add a three-way SimpleQA grader as an optional judge. |
| M5 | **Frozen-backend arm**: a "provider" backed by BrowseComp-Plus or Wiki-2018 E5 (local retriever) or DeepResearchGym | M | Gives a deterministic control arm and cheap training; answers "does training on a frozen corpus transfer to live APIs?" for each provider. |
| M6 | **Answer-leak guard**: block fetch/search hits on HF dataset pages, known eval repos and the simple-evals blob; count blocked hits per arm as a metric | S | Guards against reward hacking. Blocking is identical across arms, so it stays fair. |
| M7 | **Cost and latency per arm**: $ per call table per provider, p50/p95 latency from the tool logs, shown in the matrix | S | Product-relevant: "Exa is +x points at $y more per 1k questions". |
| M8 | **Docs site + PyPI release** (`pip install searchforge`): quickstart (eval without a GPU in 5 minutes), "add a provider", "add a benchmark", "how to read the matrix" | M | |

#### Long term (4+ months): research results and scale

| # | Item | Size | Notes |
|---|---|---|---|
| L1 | **Train on provider A, evaluate on B** (the full matrix) at 4B and 32B | L | The headline research question: do provider effects survive at scale and across backends? |
| L2 | **Trace-based behaviour analysis**: how query wording changes after training, when the agent stops, fetch vs re-search, all per provider | M | The data already exists (README "Interpretability"). |
| L3 | **More knobs, each fingerprinted**: harness, reward shape, RL estimator (GRPO vs others, verl backend), mixture weights, page cap, parallel tools | L | Keep the rule: one knob per labelled comparison. |
| L4 | **Scale design for live training**: a pool of tool servers per provider with per-key token buckets, a shared cache service (Redis/S3), async rollouts via prime-rl with a per-turn deadline, partial rollouts for the long tail | L | DeepResearcher's lessons. Live search becomes the bottleneck long before GPUs do. |
| L5 | **A public leaderboard** of model × provider × benchmark slice, re-run on a schedule, with frozen and live arms | L | The "useful product" end state; vendors and users can both point at it. |
| L6 | **Context-management tools** (`web_fetch` with `page`/`find` arguments, or a summarise step) as a fingerprinted knob | M | The ASearcher-style fix for long pages, measured rather than assumed. |

### 4.4 Scaling design, in one picture

```
         trainer (prime-rl)          eval CLI / inspect / TRL / OpenEnv
                │  async, off-policy-bounded         │
                ▼                                    ▼
        verifiers env server (ZMQ)  ── or ──  adapters (M3)
                │ per rollout: Task + MCP toolset
                ▼
   SearchForge tool service (M2: HTTP + MCP wrapper)
     ├─ budget, parallel gate, page cap (N1), leak guard (M6)
     ├─ cache: off | read-write | replay-only (N3) ── shared store (L4)
     ├─ per-provider client pools + rate limiters (L4)
     └─ providers: 7 live APIs | frozen corpus (M5)
                │
                ▼
   traces.jsonl (+ fingerprint N2) → live_gate → matrix (slices, N-way, cost)
```

The rule that keeps comparisons honest at scale: **everything inside the tool
service applies identically to every arm, and is written into the
fingerprint.**

### 4.5 Top 5 next actions

1. **Add an equal, configurable page cap to `web_fetch`** (N1). It unblocks
   training and is logged in every trace.
2. **Write the comparison fingerprint** into every trace, and make `matrix.py`
   refuse arms that do not match (N2).
3. **Add a search/fetch cache with a replay-only mode** (N3), so runs are
   cheaper and can be repeated.
4. **Validate the 300 rows, then publish the paired 300-row baseline**, sliced
   by source and hops (N4–N6).
5. **List the environment on the Prime Environments Hub and run the first small
   prime-rl training run** on a cached backend (M1, N9).

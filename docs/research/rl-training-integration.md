# Adding RL training to SearchForge

*Research note, 2026-10-30. Docs only, no code changes. Sources were read from
repo code and docs at their `main` branches as of late October 2026. Claims I
could not confirm from a primary source are marked (unverified).*

## The question

The end-user flow should be: **"evaluate my model on providers A and B"** →
**"train on A, train on B"** → **"evaluate both checkpoints on A and B"**, with
as little setup as possible. Everything except the search provider stays fixed.
That is the Exa experiment, re-run in the open.

## Short answer

- **Primary path: prime-rl.** SearchForge is already a `verifiers` taskset, and
  prime-rl trains `verifiers` tasksets directly. Its `wiki-search` example is
  almost exactly our case: Qwen3-4B, LoRA, multi-turn tool calls, an LLM judge,
  and 2 GPUs. Integration is a config file plus a few fixes in SearchForge, not
  new framework code.
- **Secondary path: an OpenEnv adapter.** One OpenEnv wrapper around our tool
  server would let users train with TRL, AgileRL, SkyRL or ART, including on a
  single GPU.
- **Worth noting:** Exa ran its own experiment on Tinker. SkyRL can run
  Tinker-API scripts on local GPUs, so a Tinker recipe is a possible later
  addition. It is the closest way to match Exa's setup exactly.
- **Blockers, in order:**
  1. bump to `verifiers` ≥ 0.4.0, which prime-rl requires
  2. cap page text so rollouts fit the sequence length
  3. cache search and fetch calls
  4. allow a training runtime that doesn't need Docker per rollout

  None of these is a training feature. All four are application-level fixes the
  project needs anyway.

---

## 1. What the published experiments actually did

| Work | Framework | Model | Hardware | Steps / batch / group | Context and tool budget | Reward | Search backend |
|---|---|---|---|---|---|---|---|
| **Exa, "How Search Quality Shapes RL Outcomes"** ([blog](https://exa.ai/blog/rl-search-outcomes), May 2026) | **Tinker**, LoRA | Qwen3-4B-Instruct-2507 | hosted (Tinker) | ~100 steps shown; 1.58B (Exa) vs 1.89B (SERP) tokens to step 100 | **search tool only: 5 results, each snippet ≤ 2,000 chars**. Chosen because "long multi-hop trajectories quickly saturat[ed] the context window" | **Dr. GRPO**. Binary SimpleQA LLM grader, **−0.25 if context is exceeded**. Exact-substring reward was dropped because the agent gamed it with verbose or hedged answers | Exa vs "SERP" (Google proxy), live |
| **Search-R1** ([paper](https://arxiv.org/abs/2503.09516), App. B) | verl | Qwen2.5-3B/7B (14B in the appendix) | **1 node, 8×H100** | 500 steps; batch 512; GRPO group 5 | seq 4,096; response ≤ 500; retrieved text ≤ 500 tokens; ≤ 4 actions; top-3 passages | Exact match; retrieved tokens masked out of the loss | Local Wikipedia-2018 + E5 |
| **DeepResearcher** ([paper](https://arxiv.org/abs/2504.03160) §3.2, §5.1) | verl | Qwen2.5-7B-Instruct | (unverified); 50-node CPU cluster just for tools | 256 prompts × 16 rollouts per step | ≤ 10 tool calls; long pages read in segments by a separate reading agent | F1, −1 for bad format; observations masked | **Live** (Serper/Bing). Search cache with a 7-day TTL; retries for rate limits (e.g. 200/s) |
| **ASearcher** ([paper](https://arxiv.org/abs/2508.07976) §3–4) | AReaL (fully async) | Qwen2.5-7B/14B; QwQ-32B | 16 nodes recommended for 7B; **~16k H800-hours** for QwQ-32B | batch 128 (7B/14B), 64 (32B) | 32 turns (7B/14B), 128 (32B); keeps the last 25k chars of history; the **agent summarises pages itself** | F1 × format (base models); LLM judge (QwQ); zero-advantage groups filtered out | Local wiki or live web |

Three things carry over directly:

1. **Every one of them limits how much tool text reaches the model**: snippet
   caps, retrieved-token caps, segmented reading or summarisation. SearchForge
   is the only one with uncapped `web_fetch`.
2. **Every one masks tool output out of the loss.** prime-rl does this by
   default (see §2.1).
3. **Exa's reward design is essentially ours:** a binary LLM judge plus a −0.25
   context penalty. Our judge prompt is reference-based like theirs. Their
   harness had **no fetch tool**, so we need to state that difference whenever
   we compare results with theirs.

---

## 2. Frameworks, checked against their code and docs

### 2.1 prime-rl (primary)

Repo: [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), v0.10.0, main @ `a566e5d`.

| Question | Answer | Source |
|---|---|---|
| How does it load an env? | The orchestrator imports a `verifiers` taskset by id, e.g. `env.taskset.id = "wiki-search"`. A non-built-in id must be an **installed package**; Hub ids `owner/name` resolve to `name`. So `searchforge` just has to be installed in prime-rl's venv. | `examples/basic/wiki-search/rl.toml`; `verifiers/v1/utils/loaders.py::_import_plugin` |
| Which `verifiers` version? | **`verifiers[harbor]>=0.4.0`**. We pin `0.3.1`, so the bump is a hard prerequisite. | `pyproject.toml` |
| Config files | One `rl.toml` holds trainer, orchestrator and inference settings; `uv run rl @ rl.toml` launches all three. Separate `eval.toml` for baselines. | `docs/overview.md`, wiki-search README |
| Minimum hardware | Defaults to **1 inference GPU + 1 trainer GPU**. Wiki-search (Qwen3-4B, LoRA) is documented on 2 GPUs with `--deployment.num-train-gpus 1 --deployment.num-infer-gpus 1 --orchestrator.batch-size 128`. "Single-GPU runs are supported for debugging." | `docs/scaling.md`, `docs/overview.md`, wiki-search README |
| LoRA | `[trainer.model.lora]` (rank, alpha, targets). With LoRA, weight sync uses the filesystem instead of NCCL. | `docs/advanced.md` |
| Async / off-policy | Async by default: inference samples with π(n−1) while the trainer makes π(n). `orchestrator.max_staleness` (default 8) masks tokens that are too stale, and rollouts are never cancelled. | `docs/algorithms.md` §Async |
| Multi-turn tools | The orchestrator drives multi-turn rollouts **without re-tokenising across turns**, using the `renderers` package. Tool calls use vLLM's tool parser (`tool_call_parser = "hermes"` for Qwen). MCP toolsets come in through the `verifiers` harness. | `docs/overview.md`, wiki-search README |
| Tool tokens in the loss | **Masked out by default.** Env observation tokens are excluded unless the algorithm is `echo`. | `docs/algorithms.md` |
| Truncation | Truncated rollouts are trained like any other; `is_truncated` is logged per env. I found no setting that drops them, so our −0.25 `context_overflow` reward is what shapes the behaviour (unverified that no such setting exists). | `orchestrator/metrics.py`, `docs/training.md` |
| No-signal groups | Samples with zero advantage are filtered out. A curriculum gate (`advantage_range`) or `difficulty_pool` sampler keeps groups that carry signal. Directly useful because many of our questions are all-correct or all-wrong. | `docs/algorithms.md`, wiki-search `rl.toml` |
| Length control | `algo.length_penalty` (`linear`): output tokens, input tokens and turns, normalised per group. The GLM search recipe uses `num_input_tokens_weight = 0.1`. | `docs/algorithms.md`, `examples/advanced/glm-4.5-air/search.toml` |
| Search examples | `basic/wiki-search` (4B, LoRA, `seq_len = 4096`, group 16, 200 steps). `advanced/glm-4.5-air/search.toml`: a live Serper search through the `rlm` harness, 131k context, with history compaction at 98k tokens and BrowseComp eval every 20 steps. | repo |
| Hosted option | **Prime Lab hosted training**: LoRA, any `verifiers` env, TOML config, submitted via the Prime CLI. **Private beta.** | [docs](https://docs.primeintellect.ai/hosted-training/what-is-lab) |
| Monitoring | `uv run dashboard` shows a per-token trace viewer with tool and loss masks. Runs also stream to Prime Lab unless `--no-monitors.prime` is set. | `docs/training.md` |

**Fit for SearchForge: very high.** Our taskset, judge, rewards and metrics carry
over unchanged. Most of the work is a config file.

### 2.2 AgileRL

Repo: [AgileRL/AgileRL](https://github.com/AgileRL/AgileRL), main @ `aa8afcc`.

| Question | Answer |
|---|---|
| LLM algorithms | GRPO, GSPO, CISPO, PPO-LLM, REINFORCE-LLM, SFT, DPO (`agilerl/algorithms/`) |
| Multi-turn and tools | Yes. Every LLM env is reached **through OpenEnv**: text `reset()`/`step()`, an optional `tools` attribute, or an OpenEnv `MCPEnvironment` whose tools are found through `list_tools`. Tool-call parsers are chosen per model family. It runs in-process or remotely (`env_url`, one WebSocket session per rollout). (`docs/llm_finetuning/environments.rst`) |
| Hardware | Single-GPU "colocated rollout": trainer and vLLM share one GPU via vLLM sleep/wake. Multi-GPU through `torchrun`, full replicas or FSDP2. QLoRA supported. |
| Its unique angle | **Evolutionary HPO**: `GRPO.population(...)` trains a population and mutates hyperparameters. Hosted on their "Arena" service. |
| Async | No fully async off-policy pipeline documented for LLMs; collection appears synchronous (unverified). |
| Fit | **Medium.** Usable through an OpenEnv adapter. HPO could tune the RL recipe, but it must tune *once* and freeze the result for both arms. Tuning per provider would break the comparison. Less mature for long agentic rollouts than prime-rl. |

### 2.3 Others

| Framework | How an external env or tool server plugs in | Multi-turn | Min hardware | Effort to wire SearchForge |
|---|---|---|---|---|
| **TRL** `GRPOTrainer` ([docs](https://github.com/huggingface/trl/blob/main/docs/source/openenv.md)) | `environment_factory=EnvClass`: public methods become tools and the trainer runs the tool loop. Or `rollout_func` for full control. OpenEnv, OpenReward and Harbor all use the same contract. | Yes. `max_completion_length` covers **the whole conversation, tool results included** | **1 GPU** with `vllm_mode="colocate"` | **Low–medium**: a class with `search`/`fetch` methods calling our providers, a reward function wrapping our judge. Loses our trace analysis unless we rebuild it. |
| **verl** ([search tool docs](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)) | `BaseTool` subclasses in a YAML tool registry; agent loop at `verl/experimental/agent_loop` | Yes (SGLang multi-turn) | Search-R1: 8×H100. The docs example uses 8×H20 (unverified) | **Medium–high**. The reference `SearchTool` class has been **removed** from verl, so the tool must be user-written. Strongest scale story; heaviest setup. |
| **SkyRL** ([repo](https://github.com/NovaSky-AI/SkyRL)) | `skyrl-gym` `BaseTextEnv.init/step`; ships a Search-R1 `search` env. **Implements the Tinker API on local GPUs.** Harbor integration. | Yes | multi-GPU typical (unverified minimum) | **Medium.** The interesting part is running a Tinker script locally (§2.4). |
| **ART** (OpenPipe, [repo](https://github.com/OpenPipe/ART)) | You write the rollout in plain Python against an OpenAI-compatible client; the ART server trains LoRA with GRPO. Inference pauses while training runs. **RULER** is a judge that needs no reference answer. MCP·RL trains against any MCP server. | Yes | 1 GPU (Unsloth + vLLM), or W&B Serverless RL | **Low.** Our agent loop plus our judge as the reward. Synchronous, so throughput is low; fine for small LoRA studies. **Don't use RULER for this project**: we have gold answers, and a judge without them scores plausibility. |
| **rLLM** ([repo](https://github.com/rllm-org/rllm)) | `@rllm.rollout def solve(task) -> Episode` plus an evaluator. Backends: verl, **tinker**, fireworks. | Yes | tinker backend: no local GPU | **Low–medium.** Search community projects exist ("Cut the Bill, Keep the Turns"). |
| **Tinker** ([cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)) | Low-level `sample` / `forward_backward` / `optim_step` API; you write the loop. The cookbook has a **Search-R1 "search_tool" recipe**. | Yes | **No local GPU** (hosted). SkyRL can run the same API locally. | **Medium.** This is **exactly how Exa ran its experiment**, which makes it the most faithful replication path. Paid service. |

### 2.4 How the frameworks relate

```
                       ┌── prime-rl (local GPUs)          ← primary
SearchForge taskset ───┤
  (verifiers v1)       └── Prime Lab hosted training      ← same config, no GPUs (private beta)

SearchForge tool server ── OpenEnv adapter ──┬── TRL GRPOTrainer (1 GPU, colocate)
  (providers, cap, cache)                    ├── AgileRL (1 GPU, HPO)
                                             ├── SkyRL / ART
                                             └── HF Spaces deploy

Tinker loop (exact Exa replication) ──┬── Tinker hosted
                                      └── SkyRL Tinker backend (local GPUs)
```

One OpenEnv adapter opens four frameworks, which makes it the best second
investment. A Tinker recipe is the third.

---

## 3. Recommendation

### 3.1 Primary path: prime-rl, with the same taskset for eval and training

The arguments:

- **Zero translation.** The same taskset, judge, rewards and metrics, and the
  same trace format that `matrix.py` already reads.
- **A matching recipe exists.** `wiki-search` is Qwen3-4B, LoRA, multi-turn
  tools, a judge and 2 GPUs.
- **Async with stale-token masking**, which matters for long-tail search
  rollouts.
- **Zero-advantage filtering and a difficulty curriculum** are built in. They
  answer our README's "no GRPO signal" worry directly.
- **The same config scales up.** It can run on Prime Lab hosted training, and on
  multiple nodes, without being rewritten.

### 3.2 Secondary path: an OpenEnv adapter, plus a TRL single-GPU recipe

This is for users who aren't on the Prime stack. The OpenEnv server would
expose `search`/`fetch` as MCP tools with the same provider layer, cap, cache
and budget, and its `step()` would run our judge. On top of it, ship one TRL
recipe (`vllm_mode="colocate"`, Qwen3-1.7B/4B, LoRA). The judge and cap
settings stay identical, so the two paths' results remain comparable.

### 3.3 Optional third path: a Tinker recipe

This mirrors Exa's setup most closely: Tinker, LoRA, Qwen3-4B-Instruct-2507 and
Dr. GRPO. It runs on Tinker hosted or on local GPUs via SkyRL. Worth doing once
the first two paths work, specifically to say "we reproduced their setup and
here's what changed".

---

## 4. Integration plan for this repo

### 4.1 Prerequisites (application code, before any training)

| # | Change | Why training needs it |
|---|---|---|
| P1 | **Bump to `verifiers` ≥ 0.4.0** | prime-rl requires it. The date-filter branch also imports a `verifiers` internal helper, which needs checking after the bump. |
| P2 | **Equal page cap on `web_fetch`** (and possibly snippet length), configurable and recorded in the trace | Our median rollout is ~20k tokens; wiki-search uses `seq_len = 4096`. Exa capped snippets at 2,000 chars and Search-R1 capped retrieved text at 500 tokens. Uncapped pages mean every rollout truncates and the group has no signal. |
| P3 | **Search/fetch cache**, per provider, with a recorded hit/miss flag | In a GRPO group of 8–16 rollouts on the same question, the first query is often identical, so a cache cuts cost and rate-limit failures sharply. It also means identical queries in a group see identical results, which is fairer. DeepResearcher used a 7-day cache. |
| P4 | **A training runtime that doesn't start a container per rollout** | `configs/eval.toml` uses `docker` because only Docker enforces `network_allow = []`; `subprocess` errors on that policy. With the `null` harness the agent runs no code, so the tool server is the only route to the network. A `training` profile could relax the policy for `subprocess`. Needs testing before relying on it. |
| P5 | **Comparison fingerprint** in every trace | The paired train-A / train-B design is only valid if the fingerprints differ in `provider` alone. |
| P6 | **Separate train and eval data** | `tasks_v1.jsonl` uses validation and test splits. Training needs train splits (HotpotQA, MuSiQue and 2Wiki have them; Exa trained on MuSiQue + HotpotQA). Add `tasks_train_v1.jsonl` built by `build_taskset.py --split train`, with deduplication against the eval rows. |

### 4.2 Files to add

```
configs/train/
  smoke.toml           # Qwen3-0.6B/1.7B, 1+1 GPU, 20 steps, tasks_v0, cache on (install check)
  qwen3-4b-lora.toml   # the paired-experiment recipe; provider is the only field changed per arm
configs/eval/
  checkpoint.toml      # evaluate a checkpoint on every provider (the 2×2 swap)
scripts/train_pair.py  # thin wrapper: one config, N providers → N runs with identical seeds,
                       # writes run manifest + fingerprint; never edits the recipe per arm
docs/training.md       # "eval → train → re-eval" in 5 commands
```

User-facing flow, without a new CLI:

```bash
# 0. inside a prime-rl checkout
uv pip install -e /path/to/searchforge          # or: prime env install <owner>/searchforge
# 1. baseline across providers
uv run eval @ configs/eval/checkpoint.toml --env.taskset.providers '["serper","exa"]'
# 2. train one arm per provider (same seed, same recipe)
uv run rl @ configs/train/qwen3-4b-lora.toml --env.taskset.providers '["serper"]' --run.name serper
uv run rl @ configs/train/qwen3-4b-lora.toml --env.taskset.providers '["exa"]'    --run.name exa
# 3. evaluate each checkpoint on both providers, then compare
uv run python scripts/matrix.py ...
```

A `searchforge train` command can wrap step 2 later. Ship the documented
commands first.

### 4.3 Smallest viable experiment

| Setting | Value | Why |
|---|---|---|
| Model | Qwen3-4B-Instruct-2507, LoRA rank 8–16 | Same as Exa and wiki-search |
| GPUs | 2 (1 inference, 1 trainer), H100 80 GB or A100 80 GB | The documented wiki-search 2-GPU mode |
| `seq_len` | 16k (not 4k) after P2 | Exa's 5 × 2k-char snippets alone are about 2.5k tokens per search; multi-hop needs room |
| Batch / group | 128 rollouts per step, group 8 → 16 questions per step | The 2-GPU wiki-search setting |
| Steps | 100 per arm (Exa's plots run to about step 100) | |
| Train data | HotpotQA + MuSiQue train (Exa's choice), a few thousand rows | |
| Algorithm | GRPO with a zero-advantage gate; Dr. GRPO-style normalisation if available (unverified that prime-rl has Dr. GRPO by name) | |
| Reward | Unchanged: judge 1.0, `context_overflow` −0.25, everything else weight 0 | |
| Eval | 2×2 swap (train on A or B × evaluate on A or B), pass@k for k = 1…8, with our paired McNemar | Matches Exa's Figure 6 |

**Rough cost per arm.** These are estimates only, so check current prices.

- **Tool calls.** 128 rollouts per step × about 4 calls each × 100 steps ≈
  **51k calls**. Before caching, that is about $15–50 on a cheap SERP API and
  about $250 at around $5 per 1k (unverified prices). Cache hits inside each
  group should cut it substantially.
- **Judge.** 12.8k calls to a nano-class model, a few dollars.
- **GPUs.** 2 GPUs for an estimated 4–10 hours, about $20–60 on rented H100s
  (unverified throughput).

So a full paired run is roughly **$100–700**, dominated by search API price.
That price difference is itself a result worth reporting.

### 4.4 Keeping "train on A / train on B" fair

1. **One recipe file, one changed field.** `train_pair.py` takes `--providers`
   and refuses any other difference. The fingerprint (P5) proves it.
2. **The same seed, data order, step count and LoRA initialisation.** Ideally
   use **≥ 2 seeds per arm**: RL variance at n = 1 can be as large as the effect
   we're looking for.
3. **The same cap, cache mode and budget for both arms.** Record cache hit rates
   per arm, because a provider that returns more stable results gets more hits.
   That is a real difference, but it should be visible, not hidden.
4. **Compare at equal steps and at equal compute.** Exa's headline is "69% fewer
   tokens to equal performance", so log tokens per step per arm.
5. **Evaluate every checkpoint on every provider**, on held-out questions, with
   the same judge and pass@k. Exa's swap result (Exa at inference helps both
   models) is the part most worth reproducing.
6. **Keep the leak guard on.** The live web can surface benchmark answer pages.
   Block known dataset and answer-key domains equally in both arms.

---

## 5. Comparison table

| | prime-rl | Prime Lab hosted | TRL (OpenEnv) | AgileRL (OpenEnv) | ART | Tinker (or SkyRL) | verl |
|---|---|---|---|---|---|---|---|
| Uses our taskset directly | **yes** | **yes** | adapter | adapter | rewrite loop | rewrite loop | rewrite as `BaseTool` |
| Multi-turn tools | yes | yes | yes | yes | yes | yes (manual) | yes |
| Tool tokens masked | default | default | via mask | (unverified) | (unverified) | manual | yes |
| Async / off-policy | **yes** (staleness bound) | yes | no | no (unverified) | no | manual | yes (agent loop) |
| Min GPUs | 2 (1 for debug) | 0 | **1** | **1** | **1** | 0 / own GPUs | 8 (typical) |
| LoRA | yes | yes (default) | yes | yes / QLoRA | yes | yes | yes |
| Zero-advantage filtering | built in | built in | (unverified) | (unverified) | manual | manual | manual |
| Matches Exa's setup | close | close | partial | partial | partial | **exact** | partial |
| Effort for us | **S** (config + prerequisites) | S (beta access) | M | M | M | M | L |

**Plan:**
1. Do P1–P6.
2. Add `configs/train/smoke.toml` and run it on 2 GPUs.
3. Run the paired Qwen3-4B experiment on prime-rl.
4. Add the OpenEnv adapter with one TRL single-GPU recipe.
5. Optionally, add a Tinker recipe for an exact replication of Exa's setup.

# Page capping and search-environment design: what others do

*Research note, 2026-10-30. Docs only; no code changed. For every repo I read the
source code as well as the docs, at the default-branch commit cloned on this date.
"(unverified)" marks claims I could not check against a primary source.*

**The short answer**

- **Most search-RL environments cap tool output by character count** and keep
  the start of the text. The numbers that appear repeatedly:
  - 2–5k characters per search result
  - 3–50k characters per fetched page
  - ~100k characters as a "safety" cap
- **A few take a different route:**
  - **pagination**: ASearcher, DeepResearcher
  - **section navigation**: verifiers `wiki_search`
  - **an LLM summariser between the page and the agent**: Tongyi DeepResearch,
    WebExplorer, DeepResearcher
- **Exa's study had no fetch tool.** The search tool returned 5 results, each
  cut to at most 2,000 characters, inside a 32k-token context. Their "future
  work" names content fetching and context pruning, which is exactly the gap
  SearchForge is hitting now.
- **Recommendation for SearchForge:** cut every fetched page at 8,000
  characters, keeping the start. Apply the cut after the provider's text has been
  normalised, so every provider is cut the same way. Add a visible
  `[truncated: showing 8,000 of N chars]` marker and two recorded metrics, and
  use a 32k context for training, not 4k. Offer pagination as an optional switch,
  and keep summarisers out of the default (section 4).

---

## 1. How existing environments cap search and fetch output

| Environment | What is capped | Limit | Method | Model controls it? | Source |
|---|---|---|---|---|---|
| **Exa RL study** | each search result | 5 results × ≤2,000 chars; **no fetch tool**; 32k context, 2,048 tokens per generation, ≤10 assistant messages | truncate | no | [exa.ai/blog/rl-search-outcomes](https://exa.ai/blog/rl-search-outcomes) (appendix) |
| **tinker-cookbook `search_tool`** (Exa's starting point) | results per query, trajectory | 3 results (Chroma over Wikipedia), 5 turns, 32k trajectory tokens | top-k + trajectory limit | no | [`recipes/search_tool/train.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/search_tool/train.py) |
| **prime-environments `search_r1_ish`** | the whole search output | ≤5 results, then **≤5,000 chars total** + "Results truncated for length". With the Exa backend, results carry full page text, so the first result fills most of the space | start of text | `num_results` (capped at 5) | [`search_r1_ish.py:65,118-120`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/search_r1_ish/search_r1_ish.py) |
| **prime `webexplorer`** | search, browse | top-10 Serper snippets; a fetched page (Jina) is cut to **50,000 chars**, then an LLM answers a `query` about it (max 4,000 output tokens) | truncate + **query-focused LLM summary** | yes: `browse(url, query)` | [`webexplorer/tools.py:42,102-115`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/webexplorer/webexplorer/tools.py) |
| **prime `browsecomp_plus`** | search, document | search returns **document IDs only** (k=5 by default, model may ask for up to 20); `get_document` returns **≤3,000 chars** | start of text | `k` | [`browsecomp_plus.py:245,311`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/browsecomp_plus/browsecomp_plus.py) |
| **prime `hle_web_py`** | search | 5 Exa results, **highlights only**, no fetch | provider extraction | no | [`hle_web_py.py:34-46`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/hle_web_py/hle_web_py.py) |
| **prime `web_voyager`** (browser) | accessibility tree | 3,000 / 5,000 chars; last 5 screenshots kept | start of text + "[...truncated...]" | no | [`web_voyager.py:374-376,580,592`](https://github.com/PrimeIntellect-ai/prime-environments/blob/main/environments/web_voyager/web_voyager.py) |
| **verifiers `wiki_search`** | page reading | no character cap; the model lists a page's sections, then reads one | **navigate by section** | yes: `view_sections`, `read_section` | [`wiki_search/servers/wiki.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/wiki_search/wiki_search/servers/wiki.py) |
| **verifiers `compact`** | conversation history | after each tool result the model must save notes; the next turn sees only those notes | **model-written memory** | yes | [`compact/program.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/compact/compact/program.py) |
| **Search-R1** | each observation | top-3 Wikipedia passages; observation cut to **500 tokens** (`max_obs_length`); retrieved tokens are **masked out of the loss** | token cut, start of text | no | [`train_ppo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/train_ppo.sh), [`generation.py:87-89`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py) |
| **ASearcher** | search docs, pages, history | top-5 docs × ≤5,000 chars. A page is cut to ≤250k chars, then **split into 10k-char pages served one per call** (reasoning agent) or 25k chunks sent to a summary step (search agent). History is trimmed to the **last 25,000 chars** | truncate + **paginate** + history window | yes, by calling again | [`train/search_agent.py:148-173`](https://github.com/inclusionAI/ASearcher/blob/main/ASearcher/train/search_agent.py), [`train/reasoning_agent.py:268,420-427`](https://github.com/inclusionAI/ASearcher/blob/main/ASearcher/train/reasoning_agent.py) |
| **DeepResearcher** | pages | top-10 search results; pages are read in **40,960-char viewports** by a separate reading agent, which extracts the useful parts and decides whether to read the next page | **paginate + LLM extraction** | the reading agent decides | [`web_search_agent.py:34`](https://github.com/GAIR-NLP/DeepResearcher/blob/main/scrl/handler/web_search_agent/web_search_agent.py), [`reading_agent.py:45-90`](https://github.com/GAIR-NLP/DeepResearcher/blob/main/scrl/handler/reading_agent/reading_agent.py) |
| **Tongyi DeepResearch / WebSailor `visit`** | pages | page (via Jina) cut to **95k tokens**, then an extractor LLM returns evidence + summary for the given `goal`. If that fails, it retries at 25,000 chars | truncate + **goal-conditioned LLM summary** | yes: `visit(url, goal)` | [`inference/tool_visit.py:18-32,197-214`](https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/tool_visit.py) |
| **NeMo-Gym `browsecomp_advanced_harness`** | search, browse, bash | search ≤**30,000 chars per call**, split across the queries in it; browse ≤30,000 split across URLs. Optional disk mode saves pages (≤2 MB each) and gives the model a read-only `bash` tool, whose output is the **first 8 KB + last 4 KB**. Has a URL blocklist for leaked answers | budget split + **grep the pages from disk** | yes, in disk mode | [`browsecomp_advanced_harness/app.py:87-97,173,311,574-602`](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/browsecomp_advanced_harness/app.py) |
| **MiroFlow** | any tool result, history | **≤100,000 chars** ("≈25k tokens") + "[Result truncated]"; `keep_tool_result=K` keeps only the last K tool results (−1 = keep all) | truncate + sliding window | no | [`utils/io_utils.py:134-136`](https://github.com/MiroMindAI/MiroFlow/blob/main/src/utils/io_utils.py), [`llm/provider_client_base.py:115-171`](https://github.com/MiroMindAI/MiroFlow/blob/main/src/llm/provider_client_base.py) |
| **inspect_ai** | all tool output; Google pages | **16 KiB** default `max_tool_output`. The Google provider keeps ≤2,000 words per page and an LLM drops irrelevant pages (3 results). `web_browser` shows the accessibility tree and lets the model scroll | truncate; LLM relevance filter; scroll | scroll only | [`_web_search/_google.py:60,197-214`](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/tool/_tools/_web_search/_google.py), [`_cli/eval.py:875`](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_cli/eval.py) |
| **OpenAI simple-evals (BrowseComp, SimpleQA)** | — | grader only; no agent or tools | — | — | [simple-evals](https://github.com/openai/simple-evals) |
| **SearchForge today** | search, fetch | search ≈1.3k chars (max 1.6k); **fetch has no cap** (up to 162k chars) | none | no | `servers/tool.py` |

**Patterns:**

1. **Cutting characters from the start of the text is the default.** Everyone
   who returns raw page text cuts it from the start. Only NeMo-Gym's bash output
   keeps the end as well.
2. **For long tasks, the model controls how much it reads.** It either pages
   through (ASearcher, DeepResearcher, inspect scroll), navigates
   (`wiki_search`), or greps (NeMo-Gym disk mode). Each extra page costs a turn.
3. **Summarisers appear in research-agent systems** (Tongyi, WebExplorer,
   DeepResearcher). They add a second model to the loop. That model's quality and
   cost become part of the environment.
4. **The cap is almost always visible.** A marker says that something was cut.
   SearchForge should do the same, and also log the original length.

---

## 2. The Exa study, in detail

Source: [How Search Quality Shapes RL Outcomes](https://exa.ai/blog/rl-search-outcomes)
(Sol Kim and Nitya Sridhar, 13 May 2026). The figures in the "Disclosed" table
are quoted from the post and its appendix. An
[Exa webinar](https://exa.ai/webinars/rl-search-outcomes) walks through the same
setup.

### Disclosed

| Item | Value |
|---|---|
| Model | Qwen3-4B-Instruct-2507, **LoRA rank 32**, trained on Tinker |
| Arms | Exa vs "SERP" ("as a proxy for Google results"); one model trained per arm |
| Tools | **Search only.** "Each search tool call returns 5 live web results, each truncated to a snippet of maximum 2,000 characters." Chosen "with model context limits in mind" after long multi-hop runs "quickly saturat[ed] the context window" |
| Content fetching | **None.** Listed under Future Work, together with "context pruning" |
| Prompt | Adapted from Search-R1 and tinker-cookbook `search_env.py`; "Answer:" prefix, 1–5 words (full text in the appendix) |
| Context | **32k tokens**, **2,048 tokens per generation**, **≤10 assistant messages** per rollout |
| Training data | MuSiQue + HotpotQA |
| Algorithm | **Dr. GRPO**; batch 64 × group 16; **100 steps**; constant LR 2.35×10⁻⁴; async with max staleness 2 (older rollouts dropped); constant-reward groups dropped |
| Reward | Binary. They tried exact substring match first, but it was gamed with wordy answers and lists of alternatives, so they **switched to the SimpleQA LLM grader**. **−0.25 penalty** when the context limit is exceeded |
| Eval | pass@k (unbiased estimator), **200 questions × 200 rollouts** per benchmark. Six sets: SimpleQA, 2Wiki, HotpotQA, FRAMES, MuSiQue, BrowseComp. Baseline: Qwen3-235B-A22B with the same tools |
| Headline numbers | SimpleQA pass@1 **0.767 vs 0.692**; 20% more tokens for SERP to reach step 100 (1.89B vs 1.58B); Exa needs **69% fewer tokens** (0.58B vs 1.89B), **62% fewer search calls** (93k vs 248k) and **58% fewer turns** (146k vs 350k) to reach equal performance |
| Mechanism evidence | Results contain the answer **36.1±2.4% vs 32.6±2.3%** of the time (~240k rollouts per engine). SERP returns <5 results on **6.74%** of calls and 0 results on 0.19% (Exa: 0.005%) |
| Cross-engine | Swapping the engine at eval time: Exa-trained agents beat SERP-trained agents on either engine |

### Not disclosed: what SearchForge must decide for itself

1. **Which SERP API.** Serper, SerpAPI, Bright Data or another; and settings such
   as location, language and number of results.
2. **How each arm's "snippet" was built.** Google SERP snippets are naturally
   ~150–300 characters. Exa can return up to 2,000 characters of page text or
   highlights. If the SERP arm was not padded with page text, the Exa arm saw up to
   ~5–10× more text per result. That would make text volume a confound,
   separate from ranking quality.
   - This is my inference, not something the post says. It matters because
     SearchForge's own adapters differ the same way: Exa returns highlights,
     Serper returns a short snippet.
3. **Exa settings.** Search type (auto/neural/fast), contents mode (text,
   highlights or summary), livecrawl, and whether Snapshot (date pinning) was
   used.
4. **Grader model** behind the SimpleQA grader prompt, and how NOT_ATTEMPTED was
   scored.
5. **Training run count and seeds.** One run per arm appears to have been used,
   so run-to-run variance is unknown. Small gaps such as MuSiQue 0.311 vs 0.307
   have no reported significance.
6. **Train-set size and filtering.** How MuSiQue and HotpotQA were sampled and
   filtered (e.g. dropping yes/no answers), and how the 200 eval questions were
   picked.
7. **How "answer present in results" was matched** (substring? normalised?).
8. **Leak controls.** How BrowseComp was decrypted, and whether any domain
   blocklist stopped the agent from finding answer keys.
9. **Tool schema details.** One query per call or several; whether parallel
   calls were allowed.
10. **Code.** Training code and traces are not released.

### What this means for a faithful re-run in SearchForge

- An **"Exa replication" preset**:
  - `web_fetch` off
  - `num_results=5`
  - a **2,000-char cap per result excerpt**
  - ≤9 tool calls, which leaves room for the answer in the 10-message limit
  - 32k context, 2,048 tokens per turn
  - SimpleQA three-way grader
  - −0.25 overflow penalty (SearchForge already has this)
- **Then a second preset that adds `web_fetch` with the cap below.** That tests
  whether Exa's result survives once the agent can read whole pages, which is the
  part their study did not cover.

---

## 3. Relevant papers

| Paper | Tool interface | Content limits | Context management | Reward | Lesson for SearchForge |
|---|---|---|---|---|---|
| [Search-R1](https://arxiv.org/abs/2503.09516) | `<search>` tags → local Wiki-2018 E5 retriever | top-3 passages; 500-token observation | none (≤4 turns) | exact match | Mask tool tokens out of the loss (the paper reports a clear gain, 0.431 vs 0.343). Short observations are enough for Wikipedia-style QA |
| [Empirical study of reasoning-search RL](https://arxiv.org/abs/2505.15117) | same as Search-R1 | same | same | EM (+ format) | **Engine quality changes RL dynamics.** With random results the agent stops searching; with BM25 it searches more and less efficiently; with E5 it searches selectively. This is the offline version of Exa's claim and supports **search-call count as a primary outcome** |
| [Retrieval, Reward, and Training Protocols](https://arxiv.org/abs/2605.27881) | local retrieval | — | — | outcome vs process | Fixing a **coverage gap in the corpus** gave bigger gains than any algorithm change. Simple outcome reward is competitive. Content coverage, not algorithm, dominates |
| [ZeroSearch](https://arxiv.org/abs/2505.04588) | an LLM simulates the engine | simulated docs | — | EM | Live search is expensive and non-deterministic. A frozen or simulated control arm is cheap |
| [DeepResearcher](https://arxiv.org/abs/2504.03160) | real web search (Serper/Bing) + browse | top-10 results; 40,960-char viewports | reading agent extracts and pages down | F1 | Training on the real web works but needs caching, rate-limit handling and a CPU tool cluster. The reading agent counts as part of the environment |
| [ASearcher](https://arxiv.org/abs/2508.07976) | search + access URL | 5k chars per doc; pages 10k–25k chars per chunk | history window of the last 25k chars; up to 100+ calls | F1 + LLM judge | Pagination plus a history window is what allows long searches at a fixed context size |
| [WebSailor](https://arxiv.org/abs/2507.02592) / [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | `search`, `visit(url, goal)` | 95k tokens → summariser | the summary is the observation | LLM judge | Goal-conditioned summaries keep the context small, but the summariser's quality becomes part of what you measure |
| [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) | search over a fixed 100k-doc corpus | Prime env caps documents at 3k chars | — | LLM judge | A fixed corpus gives a deterministic **control arm** next to live providers |
| [Environment-aware IR](https://arxiv.org/abs/2606.16817) (details unverified beyond the abstract) | several local retrievers | — | — | retrieval metrics | RL learns **retriever-specific query styles that do not transfer** between retrievers. Expect provider-specific query habits, and measure them (query length, quoting, operators) |

---

## 4. A cap for SearchForge's `web_fetch`

### The data we have

| Measure | Value | In tokens (~4 chars per token) |
|---|---|---|
| search output | median 1,304 chars, max 1,561 | ~330–400 |
| fetch, typical | 20 of 29 fetches ≤24,627 chars | ≤~6k |
| fetch, tail | 107k, 162k chars | 27k–40k |
| median rollout | ~20k tokens | — |
| reference `seq_len` | 4,096 | — |

**4,096 cannot hold even one typical fetch.** Search-R1 fits in 4k only because
each observation is 500 tokens over at most 4 turns. Exa used 32k. The context
target needs to change as well as the cap.

### Options compared

| Option | Same treatment for every provider? | Simple for users? | Keeps evidence? | Cost | Verdict |
|---|---|---|---|---|---|
| **A. Fixed char cap from the start + `[truncated: showing X of N chars]`** | yes, if applied after normalisation | yes, one number | loses the end of long pages; infoboxes and lead text usually come first | free | **Default** |
| B. Token cap | depends on the tokenizer, so differs between models | needs a tokenizer in the tool server | same as A | small | No. Characters are model-neutral; report approximate tokens instead |
| C. Pagination `web_fetch(url, page=n)` | yes, same page size for every provider | one more argument | full page reachable, one page per call | each page uses a tool call | **Optional switch** (off by default; on for long-search training) |
| D. `find` / query-focused extraction | yes if done locally (BM25 over chunks). **No** if it uses provider-native extraction (Exa highlights, Tavily `query`), because that is a provider feature | more arguments | best evidence for its size | local CPU | Mid-term. Only as a local, identical ranker |
| E. LLM summariser (Tongyi, WebExplorer style) | **only if** the same fixed model and prompt serve every arm, and are in the fingerprint. Even then its errors depend on the page content, so it adds noise that differs by provider | needs a key, adds cost | depends on the summariser | one LLM call per fetch | **Not the default.** An optional research switch at most |

### Recommendation

**Default (applies to every arm identically, in `servers/tool.py` after
`render_page`):**
- **`max_page_chars = 8000`**, keeping the start of the text.
  - That is ~2k tokens. It keeps ~1/3 of a typical page and the opening of every
    long one. Ten calls of fetches then fit in a 32k context.
  - Worst case: system prompt + question ≈ 1k, 10 fetches × 2k = 20k, plus
    generations ≈ 3–5k.
- **Marker:** `[truncated: showing 8,000 of 161,937 chars]`, so the model knows
  more exists. When pagination is on, the marker also says how to request the
  next page.
- **Record two numbers per fetch:** `fetch_chars_raw` and `fetch_truncated`, as
  weight-0 metrics. Truncation rate per provider then becomes a reported
  diagnostic, not a hidden effect.
- **A per-result excerpt cap of 2,000 chars** on search results, matching Exa.
  It almost never binds today, but it protects against a provider that returns
  page-length excerpts.
- **Training context: 32k tokens**, with 2,048 tokens per generation, matching
  Exa. Drop the 4,096 reference.

**Two settings, both written into the trace and the comparison fingerprint:**
1. `max_page_chars`: an integer, or `None` for uncapped (eval-only studies).
2. `fetch_pagination: bool`: adds `page` to `web_fetch`. Pages are
   `max_page_chars` long and each costs one budget call. Off by default; turn it
   on for BrowseComp-style tasks.

**A preset for users who want "short and cheap":** `max_page_chars=2000` with
`max_tool_calls=4`. That fits 4k–8k contexts in Search-R1 style.

**Not recommended as default:**
- **Token caps**, which depend on the model's tokenizer.
- **Keeping the end of pages too**, because web page endings are mostly
  navigation and footers.
- **Provider-native "relevant passages" for fetch**, which is a capability
  asymmetry and the same reason answerBox is off.
- **LLM summarisers.** They add a second model whose behaviour differs by
  provider.

**Fairness rule, unchanged:** the cap runs on SearchForge's normalised
`PageContent.text`, never inside an adapter. Every arm is cut by the same code at
the same number.

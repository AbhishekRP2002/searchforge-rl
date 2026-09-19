"""SearchForge taskset: web research tasks scored by a reference judge."""

import json
from itertools import product
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from searchforge.evidence import answer_present, grounded_fraction, urls_in_text
from searchforge.judge import SearchForgeJudgeConfig
from searchforge.prompts import SYSTEM_TEMPLATE
from searchforge.servers.tool import PROVIDER_KEY_ENV, WebToolset, WebToolsetConfig
from searchforge.trace_reader import (
    native_search_citations,
    retrieved_urls,
    tool_call_count,
    tool_call_counts,
    tool_result_text,
)

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "tasks_v0.jsonl"


def load_rows(path: Path) -> list[dict]:
    """Read one JSON object per line. Blank lines are skipped."""
    if not path.exists():
        raise FileNotFoundError(f"task dataset not found at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


AGENT_TIMEOUT = 900.0
"""Seconds for one agent solve attempt. Verifiers leaves every TaskTimeout slot
None, which means no deadline at any layer: one wedged container idled roughly
seven hours before its run could finalize. Sized for the worst observed rollout
(10 tool calls, several multi-second fetches, a 68k-token context) with room to
spare, so it fires on a hang and never on a slow-but-working episode."""

SCORING_TIMEOUT = 300.0
"""Seconds for judge execution. One model call per trace; a deadline here stops
a hung grader holding the whole run open after the agent has finished."""


class SearchForgeData(vf.TaskData):
    network_allow: list[str] = Field(default_factory=list)
    """The harness has no direct web egress; the separate MCP server owns it."""

    timeout: vf.TaskTimeout = vf.TaskTimeout(
        agent=AGENT_TIMEOUT, scoring=SCORING_TIMEOUT
    )
    """`setup` and `finalize` stay None: both are trivial here, and an agent
    config timeout still overrides any of these (`agent.py` prefers its own
    value and falls back to the task's)."""

    question: str
    answer: str | list[str]

    hops: int
    recency: str
    answer_type: str
    popularity: str
    evidence_depth: str

    source_dataset: str
    source_id: str
    validated_on: str
    provider: str = "serper"


class SearchForgeTaskConfig(vf.TaskConfig):
    tools: WebToolsetConfig = Field(default_factory=WebToolsetConfig)
    judges: vf.Judges = Field(default_factory=lambda: [SearchForgeJudgeConfig()])
    """Replaceable with --env.taskset.task.judges."""


class SearchForgeTask(vf.Task[SearchForgeData, vf.State, SearchForgeTaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.source_id}::{self.data.provider}"

    @classmethod
    def toolsets(cls, config: SearchForgeTaskConfig) -> list[vf.Toolset]:
        tools = config.tools
        if tools.env_file:
            tools = tools.model_copy(
                update={"env_file": tools.env_file.expanduser().resolve()}
            )
        return [WebToolset(tools)]

    @vf.reward(weight=-0.25)
    async def context_overflow(self, trace: vf.Trace) -> float:
        # `is_truncated`, not `stop_condition == "truncated"`: no stop condition
        # uses that string. The real values are max_turns / max_input_tokens /
        # max_output_tokens / max_total_tokens / context_length, plus a
        # length-finished final response, which is exactly what is_truncated folds.
        return float(trace.is_truncated)

    @vf.reward(weight=0.0)
    async def citations_grounded(self, trace: vf.Trace) -> float:
        cited = urls_in_text(trace.last_reply or "")
        return grounded_fraction(cited, retrieved_urls(trace))

    @vf.reward(weight=0.0)
    async def citation_presence(self, trace: vf.Trace) -> float:
        """Keep citation rate separate from grounding precision."""
        return float(bool(urls_in_text(trace.last_reply or "")))

    @vf.reward(weight=0.0)
    async def answer_in_results(self, trace: vf.Trace) -> float:
        """Did the provider surface the answer at all? Separates retrieval
        failure from reasoning failure — the matrix's most actionable diagnostic."""
        return float(answer_present(self.data.answer, tool_result_text(trace)))

    @vf.metric
    async def num_tool_calls(self, trace: vf.Trace) -> float:
        return float(tool_call_count(trace))

    @vf.metric
    async def num_search_calls(self, trace: vf.Trace) -> float:
        return float(tool_call_counts(trace).get("web_search", 0))

    @vf.metric
    async def num_fetch_calls(self, trace: vf.Trace) -> float:
        return float(tool_call_counts(trace).get("web_fetch", 0))

    @vf.metric
    async def tokens_consumed(self, trace: vf.Trace) -> float:
        return float(trace.num_total_tokens)

    @vf.metric
    async def native_search_calls(self, trace: vf.Trace) -> float:
        """Canary. Non-zero means the model searched through its own provider and
        bypassed our toolset, which voids this run for comparison."""
        return float(len(native_search_citations(trace)))


class SearchForgeConfig(vf.TasksetConfig):
    data_path: Path = DEFAULT_DATA_PATH
    providers: list[str] = Field(default_factory=lambda: ["serper"])
    """Cross-product: every question runs on every provider listed."""
    hops_min: int = 1
    hops_max: int = 99
    task: SearchForgeTaskConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=SearchForgeTaskConfig
    )


class SearchForgeTaskset(
    vf.Taskset[SearchForgeTask, SearchForgeConfig]  # pyright: ignore[reportInvalidTypeArguments]
):
    def load(self) -> list[SearchForgeTask]:
        unknown = sorted(set(self.config.providers) - PROVIDER_KEY_ENV.keys())
        if unknown:
            raise ValueError(
                f"unknown provider(s) {unknown}; expected {sorted(PROVIDER_KEY_ENV)}"
            )
        rows = [
            row
            for row in load_rows(self.config.data_path)
            if self.config.hops_min <= row["hops"] <= self.config.hops_max
        ]
        tasks: list[SearchForgeTask] = []
        for idx, (row, provider) in enumerate(product(rows, self.config.providers)):
            config = self.config.task.model_copy(
                update={
                    "tools": self.config.task.tools.model_copy(
                        update={"provider": provider}
                    )
                }
            )
            tasks.append(
                SearchForgeTask(
                    SearchForgeData(
                        idx=idx,
                        provider=provider,
                        system_prompt=SYSTEM_TEMPLATE.format(
                            max_tool_calls=config.tools.max_tool_calls
                        ),
                        prompt=row["question"],
                        **row,
                    ),
                    config,
                )
            )
        return tasks

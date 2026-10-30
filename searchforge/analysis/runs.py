"""Read finished runs and split them into provider arms.

One SearchForge run already holds every provider it was given: the taskset is
the cross-product of rows x providers. So an arm is not a file. It is the set of
traces in a run that share a model and a provider, and the run's directory
supplies the resolved config that produced them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RESOLVED_CONFIG = Path("configs") / "resolved" / "eval.json"
"""Where `vf-eval` writes the fully resolved config, relative to the run dir."""


def read_traces(path: Path) -> list[dict]:
    """Every trace in a run, flattened out of the episodes that hold them.

    `traces.jsonl` is one *Episode* per line, not one Trace: an episode wraps
    `traces: list[Trace]` because a rollout can involve several agents. A
    single-agent run has exactly one trace per line, which is what makes reading
    the wrong level look like it works until the fields come back empty.
    """
    episodes = (
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )
    return [trace for episode in episodes for trace in episode.get("traces", [])]


@dataclass
class Run:
    path: Path
    """The run directory (the parent of `traces.jsonl`)."""
    traces: list[dict]
    config: dict | None
    """The resolved `vf-eval` config, or None when only a trace file was given."""

    @property
    def name(self) -> str:
        return self.path.name


def load_run(path: Path | str) -> Run:
    """Accept a run directory or a `traces.jsonl` path."""
    path = Path(path)
    traces_path = path / "traces.jsonl" if path.is_dir() else path
    if not traces_path.exists():
        raise FileNotFoundError(f"no traces.jsonl at {traces_path}")
    run_dir = traces_path.parent
    config_path = run_dir / RESOLVED_CONFIG
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return Run(path=run_dir, traces=read_traces(traces_path), config=config)


def trace_provider(trace: dict) -> str:
    return str(trace.get("task", {}).get("data", {}).get("provider", "unknown"))


def trace_model(trace: dict, run: Run) -> str:
    model = (trace.get("agent") or {}).get("config", {}).get("model")
    if not model and run.config:
        model = run.config.get("model")
    return str(model or "unknown")


def source_id(trace: dict) -> str:
    return str(trace["task"]["data"]["source_id"])


@dataclass
class Arm:
    name: str
    provider: str
    model: str
    run: Run
    traces: list[dict] = field(default_factory=list)

    @property
    def scored(self) -> list[dict]:
        """Traces that finished. Failed rollouts are counted, never scored."""
        return [trace for trace in self.traces if trace.get("ok", False)]


def split_arms(runs: list[Run]) -> list[Arm]:
    """One arm per (run, model, provider), named by provider where that is unique.

    A name only grows a suffix when two arms would otherwise share it, so the
    common case -- one run, several providers -- reads as plain provider names.
    """
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for index, run in enumerate(runs):
        for trace in run.traces:
            key = (index, trace_model(trace, run), trace_provider(trace))
            grouped.setdefault(key, []).append(trace)

    providers = [provider for (_, _, provider) in grouped]
    models = {model for (_, model, _) in grouped}
    arms = []
    for (index, model, provider), traces in grouped.items():
        name = provider
        if providers.count(provider) > 1:
            name = f"{provider}@{model}" if len(models) > 1 else f"{provider}@{runs[index].name}"
        arms.append(Arm(name, provider, model, runs[index], traces))
    names = [arm.name for arm in arms]
    for arm in arms:
        # Same provider and model in two runs with identical dir names: last resort.
        if names.count(arm.name) > 1:
            arm.name = f"{arm.provider}@{arm.model}@{arm.run.path}"
    return sorted(arms, key=lambda arm: arm.name)

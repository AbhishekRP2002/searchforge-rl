"""What each arm was run with, so a comparison can prove it changed one thing.

A provider comparison asserts that its arms differ in the provider and nothing
else. This module turns each arm into a flat dict of everything that could move
a result -- the resolved config with the provider removed, plus what the traces
themselves record -- and diffs those dicts. A difference outside the provider
is reported by exact key, never averaged away.
"""

from __future__ import annotations

import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path

from searchforge.analysis.runs import Arm

IGNORED = (
    # The provider is the variable under test, and the arm list names all of them.
    "env.taskset.task.tools.provider",
    "env.taskset.providers",
    "env.taskset.providers.*",
    # Run bookkeeping: where output goes and how it is displayed.
    "run.*",
    "output_dir",
    "push",
    "verbose",
    "rich.*",
    # Which rows ran (`-n`, shuffle, skip). Pairing joins on source_id and lists
    # every question only one arm has, so a subset is visible without refusing.
    "select.*",
    # Throughput knobs: they change how fast a run goes, not what an episode sees.
    "max_concurrent",
    "env.max_concurrent_agents",
    "env.interception.*",
    # Credentials and transport plumbing, which differ per machine.
    "env.taskset.task.tools.env_file",
    "*api_key_var",
    "*headers*",
    "*.timeout.connect",
    "*.timeout.read",
    "*.timeout.write",
    "*.timeout.pool",
    # Replaced by a content hash below: the same file on two machines has two paths.
    "env.taskset.data_path",
)
"""Dotted-key patterns that cannot change what an episode sees or how it is scored."""

ABSENT = "<absent>"


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Nested dicts and lists to dotted keys: `judges.0.model`."""
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = ((str(i), v) for i, v in enumerate(value))
    else:
        return {prefix: value}
    flat: dict[str, object] = {}
    for key, child in items:
        flat.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _ignored(key: str) -> bool:
    return any(fnmatchcase(key, pattern) for pattern in IGNORED)


def _sha(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _dataset_identity(config: dict) -> str:
    """The data file's content hash; its basename when the file is not on this machine."""
    data_path = (config.get("env") or {}).get("taskset", {}).get("data_path")
    if not data_path:
        return ABSENT
    path = Path(data_path)
    if path.is_file():
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    return f"unreadable:{path.name}"


def _distinct(values: list[object]) -> str:
    return " | ".join(sorted({json.dumps(v, sort_keys=True) for v in values})) or ABSENT


def arm_fields(arm: Arm) -> dict[str, object]:
    """Everything about an arm that could move its result, provider excluded."""
    fields: dict[str, object] = {}
    if arm.run.config is not None:
        fields.update(
            (key, value)
            for key, value in flatten(arm.run.config).items()
            if not _ignored(key)
        )
        fields["dataset"] = _dataset_identity(arm.run.config)
    else:
        fields["config"] = ABSENT

    # From the traces themselves. These are what actually happened, which is the
    # stronger evidence: `current_date: null` in the config, for example, only
    # resolves to a real date in the system prompt the episodes received.
    traces = arm.traces
    agent = [(trace.get("agent") or {}).get("config", {}) for trace in traces]
    fields["trace.verifiers_version"] = _distinct(
        [(trace.get("verifiers") or {}).get("version") for trace in traces]
    )
    fields["trace.model"] = _distinct([config.get("model") for config in agent])
    fields["trace.sampling"] = _distinct([config.get("sampling") for config in agent])
    fields["trace.harness"] = _distinct(
        [(config.get("harness") or {}).get("id") for config in agent]
    )
    fields["trace.system_prompt"] = _distinct(
        [_sha(trace["task"]["data"].get("system_prompt") or "") for trace in traces]
    )
    fields["trace.tool_schemas"] = _distinct(
        [_sha(sorted(trace.get("tools") or [], key=lambda t: str(t.get("name")))) for trace in traces]
    )
    fields["trace.reward_weights"] = _distinct(
        [
            {name: (reward or {}).get("weight") for name, reward in (trace.get("rewards") or {}).items()}
            for trace in traces
            if trace.get("ok", False)
        ]
    )
    return fields


def fingerprint(fields: dict[str, object]) -> str:
    return _sha(fields)


def differences(arms: list[Arm]) -> dict[str, dict[str, object]]:
    """Keys whose value is not the same across every arm, with each arm's value."""
    per_arm = {arm.name: arm_fields(arm) for arm in arms}
    keys = set().union(*per_arm.values()) if per_arm else set()
    diff: dict[str, dict[str, object]] = {}
    for key in sorted(keys):
        values = {name: fields.get(key, ABSENT) for name, fields in per_arm.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            diff[key] = values
    return diff


def allowed(key: str, allow: list[str]) -> bool:
    """`--allow-diff model` covers `model` and everything under `model.`."""
    return any(key == prefix or key.startswith(prefix + ".") for prefix in allow)

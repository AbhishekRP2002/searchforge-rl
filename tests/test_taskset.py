from searchforge.taskset import SearchForgeConfig, SearchForgeTaskset


def _tasks(**kwargs):
    return list(SearchForgeTaskset(SearchForgeConfig(**kwargs)).load())


def test_single_provider_yields_one_task_per_row():
    tasks = _tasks(providers=["serper"])
    assert len({t.data.source_id for t in tasks}) == len(tasks)


def test_two_providers_yield_the_cross_product():
    """Paired design: the same question runs on every provider, so task
    difficulty cancels in the comparison. Spec section 11."""
    single = _tasks(providers=["serper"])
    paired = _tasks(providers=["serper", "exa"])

    assert len(paired) == 2 * len(single)
    assert {t.data.provider for t in paired} == {"serper", "exa"}


def test_every_question_appears_once_per_provider():
    by_provider = {}
    for task in _tasks(providers=["serper", "exa"]):
        by_provider.setdefault(task.data.provider, set()).add(task.data.source_id)

    assert by_provider["serper"] == by_provider["exa"]


def test_idx_is_unique_across_the_cross_product():
    """verifiers keys task identity off `data.idx`: `example_id`, the
    `rollout_number` counter and GEPA's `tasks_by_idx` all index by it. One idx per
    question would collapse both arms of a pair into a single reported example."""
    tasks = _tasks(providers=["serper", "exa"])
    assert len({t.data.idx for t in tasks}) == len(tasks)


def test_idx_collision_does_not_hide_behind_a_single_provider():
    """The single-provider default cannot collide, so it proves nothing — this is
    why the bug survived the rest of the suite."""
    tasks = _tasks(providers=["serper"])
    assert len({t.data.idx for t in tasks}) == len(tasks)


def test_task_key_is_unique_and_names_the_provider():
    tasks = _tasks(providers=["serper", "exa"])
    keys = [t.key for t in tasks]

    assert len(keys) == len(set(keys))
    assert any(k.endswith("::exa") for k in keys)
    assert any(k.endswith("::serper") for k in keys)


def test_task_key_joins_back_to_the_source_id():
    task = _tasks(providers=["serper"])[0]
    assert task.key.split("::")[0] == task.data.source_id


def test_provider_is_passed_through_to_the_toolset_config():
    task = _tasks(providers=["exa"])[0]
    assert task.toolsets(task.config)[0].config.provider == "exa"


def test_prompt_carries_the_question_and_system_prompt_names_both_tools():
    task = _tasks(providers=["serper"])[0]

    assert task.data.question in task.data.prompt
    assert "web_search" in task.data.system_prompt
    assert "web_fetch" in task.data.system_prompt


def test_system_prompt_gives_no_task_specific_strategy():
    """Naming the tools is an affordance. Telling the model how to attack a
    particular question is a task hint and would confound the comparison."""
    task = _tasks(providers=["serper"])[0]
    lowered = task.data.system_prompt.lower()

    for hint in ("multi-hop", "multihop", "first search for", "intermediate entity"):
        assert hint not in lowered


def test_system_prompt_does_not_gate_answering_on_confidence():
    """Answer avoidance — withholding rather than risking a wrong answer — is the
    measured collapse mode for this agent shape (arXiv:2602.19526 s4.2). "When you
    are confident" sets a higher bar than sufficiency and licenses withholding."""
    lowered = _tasks(providers=["serper"])[0].data.system_prompt.lower()

    assert "confident" not in lowered
    assert "enough information" in lowered


def test_runtime_env_does_not_expose_provider_keys_to_the_harness(monkeypatch):
    """Provider credentials belong to the separate MCP server, not the agent."""
    monkeypatch.setenv("SERPER_API_KEY", "secret")
    task = _tasks(providers=["serper"])[0]

    assert task.runtime_env() == {}


def test_runtime_env_is_empty_when_the_key_is_unset(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert _tasks(providers=["serper"])[0].runtime_env() == {}


def test_runtime_env_never_forwards_any_provider_key(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "s")
    monkeypatch.setenv("EXA_API_KEY", "e")
    task = _tasks(providers=["serper"])[0]

    assert task.runtime_env() == {}


def test_tool_server_is_separate_from_the_harness_runtime():
    task = _tasks(providers=["serper"])[0]
    assert task.toolsets(task.config)[0].config.colocated is False


def test_tool_server_env_file_is_resolved_before_launch():
    task = _tasks(providers=["serper"])[0]
    env_file = task.toolsets(task.config)[0].config.env_file

    assert env_file is not None
    assert env_file.is_absolute()


def test_harness_has_a_concrete_deny_by_default_network_policy():
    task = _tasks(providers=["serper"])[0]

    assert task.data.network_allow == []


def test_mixture_filters_the_loaded_rows():
    only_multi = _tasks(providers=["serper"], hops_min=2)
    assert only_multi
    assert all(t.data.hops >= 2 for t in only_multi)


def test_unknown_provider_fails_before_a_rollout_starts():
    import pytest

    with pytest.raises(ValueError, match="unknown provider"):
        _tasks(providers=["altavista"])

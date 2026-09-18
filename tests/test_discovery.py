"""The taskset id `searchforge` must resolve to exactly one vf.Taskset subclass.

verifiers resolves a taskset id by importing the module named after it and reading
`__all__` — see verifiers.v1.utils.loaders._import_plugin / _plugin_class. If this
test fails, `uv run eval searchforge` cannot work.
"""

from verifiers.v1.utils.loaders import taskset_class


def test_searchforge_taskset_resolves():
    cls = taskset_class("searchforge")
    assert cls.__name__ == "SearchForgeTaskset"


def test_toolset_prefix_is_web():
    from searchforge.servers.tool import WebToolset

    assert WebToolset.TOOL_PREFIX == "web"


def test_toolset_declares_no_state():
    """A State subclass would add two HTTP round trips per tool call and a
    last-write-wins race on parallel calls. v0 tools are pure functions."""
    from verifiers.v1.state import State, state_cls

    from searchforge.servers.tool import WebToolset

    assert state_cls(WebToolset) is State


def test_searchforge_judge_resolves():
    """The judge id and the taskset id are both `searchforge`. Asserting the
    configured id is not enough — that passed while the package exported no Judge
    subclass at all, and the failure only surfaced in a live run."""
    from verifiers.v1.utils.loaders import judge_class

    assert judge_class("searchforge").__name__ == "SearchForgeJudge"

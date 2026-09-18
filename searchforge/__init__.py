from searchforge.judge import SearchForgeJudge
from searchforge.taskset import SearchForgeTaskset

# Both plugin loaders resolve the id "searchforge" to this package and filter
# `__all__` by base class, so the taskset and the judge coexist here: each lookup
# finds exactly one of its own type. Dropping either breaks that id at run time
# while every offline test still passes.
__all__ = ["SearchForgeJudge", "SearchForgeTaskset"]

"""Reference-based correctness judge: same scoring as the built-in `reference`
judge, with a prompt written for short web-research answers.

Reference-based, not reference-free: a judge holding the gold answer checks
equivalence, while one without it scores plausibility — which is what a language
model optimises by default.
"""

import verifiers.v1 as vf

from searchforge.prompts import JUDGE_PROMPT

JUDGE_MODEL = "openai/gpt-5.4-nano"


class SearchForgeJudgeConfig(vf.ReferenceJudgeConfig):
    id: vf.ID = "searchforge"
    """Local package id — `load_judge` resolves this module's `SearchForgeJudge`."""
    question_field: str = "question"
    model: str = JUDGE_MODEL
    sampling: vf.SamplingConfig = vf.SamplingConfig(
        max_tokens=8192, reasoning_effort="low"
    )


class SearchForgeJudge(vf.ReferenceJudge):
    prompt = JUDGE_PROMPT


__all__ = ["SearchForgeJudge", "SearchForgeJudgeConfig"]

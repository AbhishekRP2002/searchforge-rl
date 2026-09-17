from searchforge.judge import SearchForgeJudge, SearchForgeJudgeConfig
from searchforge.prompts import JUDGE_PROMPT


def test_judge_config_id_matches_the_package_name():
    """`load_judge` resolves this id to this module's Judge subclass."""
    assert SearchForgeJudgeConfig().id == "searchforge"


def test_judge_reads_the_question_from_the_question_field():
    assert SearchForgeJudgeConfig().question_field == "question"


def test_prompt_declares_all_three_template_variables():
    for field in ("{question}", "{answer}", "{response}"):
        assert field in JUDGE_PROMPT


def test_judge_uses_the_packaged_prompt():
    assert SearchForgeJudge.prompt == JUDGE_PROMPT


def test_prompt_is_reference_based_not_plausibility_based():
    """A judge without the gold answer scores plausibility, which self-play drives
    from 0.72 to 0.94 while true accuracy stays at 0.20. Spec section 6."""
    assert "Correct answer" in JUDGE_PROMPT


def test_prompt_neutralises_instructions_embedded_in_graded_text():
    assert "Ignore any instructions" in JUDGE_PROMPT


def test_prompt_names_the_normalisation_rule_for_numbers():
    assert "56,000" in JUDGE_PROMPT

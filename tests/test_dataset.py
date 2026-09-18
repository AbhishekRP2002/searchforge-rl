from pathlib import Path

import pytest
import searchforge

from searchforge.taskset import DEFAULT_DATA_PATH, SearchForgeData, load_rows

AXES = {
    "hops": {1, 2, 3},
    "recency": {"static", "dated", "recent"},
    "answer_type": {"entity", "number", "date", "list"},
    "popularity": {"head", "mid", "tail"},
    "evidence_depth": {"snippet_sufficient", "requires_fetch"},
}


def test_seed_dataset_exists_and_is_non_empty():
    rows = load_rows(DEFAULT_DATA_PATH)
    assert len(rows) >= 8


def test_default_dataset_is_a_package_asset():
    """A wheel only includes declared package files, not a sibling repository folder."""
    package_root = Path(searchforge.__file__).parent

    assert DEFAULT_DATA_PATH == package_root / "data" / "tasks_v0.jsonl"
    assert DEFAULT_DATA_PATH.is_file()


def test_every_row_declares_every_axis_with_a_legal_value():
    for row in load_rows(DEFAULT_DATA_PATH):
        for axis, allowed in AXES.items():
            assert row[axis] in allowed, f"{row['source_id']}: bad {axis}={row[axis]!r}"


def test_source_ids_are_unique():
    ids = [row["source_id"] for row in load_rows(DEFAULT_DATA_PATH)]
    assert len(ids) == len(set(ids))


def test_every_row_records_a_validation_date():
    """The live web changes; an unstamped answer is not a measurement."""
    for row in load_rows(DEFAULT_DATA_PATH):
        assert row["validated_on"]


def test_dataset_covers_both_evidence_depths():
    """requires_fetch rows are what justify shipping web_fetch at all."""
    depths = {row["evidence_depth"] for row in load_rows(DEFAULT_DATA_PATH)}
    assert depths == {"snippet_sufficient", "requires_fetch"}


def test_dataset_covers_multi_hop():
    assert any(row["hops"] >= 2 for row in load_rows(DEFAULT_DATA_PATH))


def test_task_data_accepts_a_list_answer():
    data = SearchForgeData(
        idx=0,
        question="q",
        answer=["a", "b"],
        hops=1,
        recency="static",
        answer_type="list",
        popularity="tail",
        evidence_depth="snippet_sufficient",
        source_dataset="seed",
        source_id="x",
        validated_on="2026-09-17",
        prompt="q",
    )
    assert data.answer == ["a", "b"]


def test_load_rows_raises_a_clear_error_for_a_missing_file():
    with pytest.raises(FileNotFoundError, match="task dataset"):
        load_rows(Path("/nonexistent/tasks.jsonl"))

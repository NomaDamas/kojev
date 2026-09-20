"""Fixture-level tests for the KoBEST/KLUE benchmark contract."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from kojev import bench
from kojev.bench import (
    KLUE_TASKS,
    KOBEST_CONFIGS,
    TASK_COLUMNS,
    BenchmarkError,
    BenchmarkItem,
    MetricPayload,
    RandomDecisionModel,
    assert_no_kobest_contamination,
    calibration_metrics,
    map_klue_row,
    map_kobest_row,
    metric_report,
    render_metric_tables,
    render_task_grid,
)
from kojev.schema import JsonValue, Question, QuestionType

if TYPE_CHECKING:
    from pathlib import Path


def test_maps_all_kobest_configs_to_typed_questions() -> None:
    rows: dict[str, dict[str, JsonValue]] = {
        "boolq": {
            "idx": "b-1",
            "paragraph": "오늘은 맑다.",
            "question": "오늘 날씨가 맑은가?",
            "label": 1,
        },
        "copa": {
            "idx": "c-1",
            "premise": "비가 왔다.",
            "question": "왜?",
            "alternative_1": "땅이 젖었다.",
            "alternative_2": "해가 떴다.",
            "label": 0,
        },
        "wic": {
            "idx": "w-1",
            "word": "은행",
            "context_1": "은행에 돈을 맡겼다.",
            "context_2": "강둑에 은행나무가 있다.",
            "label": 0,
        },
        "hellaswag": {
            "idx": "h-1",
            "context": "그는 문을 열었다.",
            "ending": ["밖으로 나갔다.", "잠들었다.", "노래했다.", "웃었다."],
            "label": 0,
        },
        "sentineg": {
            "idx": "s-1",
            "sentence": "이 영화는 재미있다.",
            "label": 1,
        },
    }

    mapped = {config: map_kobest_row(config, rows[config]) for config in KOBEST_CONFIGS}

    assert set(mapped) == {"boolq", "copa", "wic", "hellaswag", "sentineg"}
    assert mapped["boolq"].questions[0].type is QuestionType.NOUL
    assert mapped["copa"].questions[0].type is QuestionType.CHOICE
    assert mapped["copa"].questions[0].gold == 0
    assert mapped["wic"].questions[0].type is QuestionType.NOUL
    assert mapped["hellaswag"].questions[0].options == rows["hellaswag"]["ending"]
    assert mapped["sentineg"].questions[0].type is QuestionType.NOUL


def test_maps_klue_validation_tasks_and_rounds_sts_gold() -> None:
    ynat = map_klue_row("ynat", {"guid": "y-1", "title": "새 소식", "label": 3})
    nli = map_klue_row(
        "nli",
        {"guid": "n-1", "premise": "A", "hypothesis": "B", "label": 2},
    )
    sts = map_klue_row(
        "sts",
        {
            "guid": "t-1",
            "sentence1": "같다",
            "sentence2": "비슷하다",
            "labels": {"real-label": 3.6},
        },
    )

    assert set(KLUE_TASKS) == {"ynat", "nli", "sts"}
    assert ynat.questions[0].type is QuestionType.CHOICE
    assert nli.questions[0].options == ["함의", "중립", "모순"]
    assert sts.questions[0].type is QuestionType.SCORE
    assert sts.questions[0].gold == 4


def test_metrics_report_accuracy_macro_f1_brier_and_fifteen_bin_ece() -> None:
    metrics = calibration_metrics(
        gold=(1, 0, 1, 0),
        probabilities=(
            (0.1, 0.9),
            (0.8, 0.2),
            (0.6, 0.4),
            (0.3, 0.7),
        ),
    )

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(0.5)
    assert metrics.brier == pytest.approx(0.45)
    assert 0.0 <= metrics.ece_15 <= 1.0


def test_metric_report_includes_kind_aggregates() -> None:
    items = (
        BenchmarkItem(
            task="boolq",
            kind=QuestionType.NOUL,
            example_id="1",
            state="a",
            question=map_kobest_row(
                "boolq",
                {
                    "idx": "1",
                    "paragraph": "a",
                    "question": "b",
                    "label": 1,
                },
            ).questions[0],
        ),
    )
    report = metric_report(items, RandomDecisionModel(seed=0))
    tasks = report["tasks"]
    kinds = report["kinds"]
    assert tasks["boolq"]["count"] == 1
    assert kinds["noul"]["count"] == 1


def test_contamination_assert_rejects_training_manifest_overlap(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _ = manifest.write_text(
        json.dumps({"source": "skt/kobest_v1", "id": "kobest-42"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="contamination"):
        assert_no_kobest_contamination(manifest, {"kobest-42"})


def test_contamination_assert_rejects_malformed_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "malformed.jsonl"
    _ = manifest.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="malformed manifest"):
        assert_no_kobest_contamination(manifest, set())


@pytest.mark.parametrize(
    ("contents", "expected_error", "message"),
    [
        (
            json.dumps({"source": "skt/kobest_v1", "id": "kobest-42"}) + "\n",
            AssertionError,
            "contamination",
        ),
        ("{not-json}\n", BenchmarkError, "malformed manifest"),
    ],
)
def test_cli_training_manifest_rejects_invalid_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
    expected_error: type[AssertionError | BenchmarkError],
    message: str,
) -> None:
    manifest = tmp_path / "training-manifest.jsonl"
    _ = manifest.write_text(contents, encoding="utf-8")

    def fake_load_task(
        dataset: str, task: str, split: str, limit: int
    ) -> tuple[BenchmarkItem, ...]:
        assert split in {"test", "validation"}
        assert limit == 1
        example_id = "kobest-42" if dataset == "skt/kobest_v1" else f"{task}-1"
        question = Question(
            type=QuestionType.NOUL,
            instructions="질문",
            options=["아니오", "예"],
            gold=0,
            meta={"task": task, "id": example_id},
        )
        return (BenchmarkItem(task, QuestionType.NOUL, example_id, "state", question),)

    monkeypatch.setattr(bench, "_load_task", fake_load_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kojev.bench",
            "--model",
            "random",
            "--limit",
            "8",
            "--training-manifest",
            str(manifest),
        ],
    )

    with pytest.raises(expected_error, match=message):
        bench.main()


def test_random_model_repeats_exactly_for_same_input() -> None:
    example = map_kobest_row(
        "sentineg",
        {"idx": "repeat-1", "sentence": "좋다", "label": 1},
    )
    question = example.questions[0]
    model = RandomDecisionModel(seed=7)
    first = model.decide(example.state, (question,))
    second = model.decide(example.state, (question,))
    assert first == second


def _task_payload(accuracy: float) -> MetricPayload:
    return {
        "count": 10,
        "accuracy": accuracy,
        "macro_f1": accuracy,
        "brier": 0.25,
        "ece_15": 0.1,
    }


def _full_tasks(offset: float) -> dict[str, MetricPayload]:
    names = ("boolq", "copa", "wic", "hellaswag", "sentineg", "ynat", "nli", "sts")
    return {
        name: _task_payload(offset + index * 0.01) for index, name in enumerate(names)
    }


def test_task_grid_has_eight_task_columns_and_five_model_rows() -> None:
    """Acceptance: RESULTS.md has >=5 model rows x >=8 task columns."""
    models = (
        ("main", _full_tasks(0.50)),
        ("rlcd", _full_tasks(0.51)),
        ("control", _full_tasks(0.40)),
        ("distill", _full_tasks(0.49)),
        ("english", _full_tasks(0.20)),
    )
    table = render_task_grid(models)
    header = table.splitlines()[0]
    assert header.count("|") == 10
    for task in TASK_COLUMNS:
        assert f"| {task} |" in header
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "model" not in line and "---" not in line
    ]
    assert len(rows) == 5
    assert "0.500" in table
    assert "0.200" in table


def test_task_grid_rejects_a_model_missing_a_task() -> None:
    tasks = _full_tasks(0.5)
    del tasks["copa"]
    with pytest.raises(BenchmarkError, match="copa"):
        _ = render_task_grid((("main", tasks),))


def test_grid_cli_writes_results_from_named_json_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    paths: list[str] = []
    for name, offset in (
        ("main", 0.50),
        ("rlcd", 0.51),
        ("control", 0.40),
        ("distill", 0.49),
        ("english", 0.20),
    ):
        path = reports / f"{name}.json"
        _ = path.write_text(
            json.dumps({"tasks": _full_tasks(offset)}), encoding="utf-8"
        )
        paths.extend(["--grid", f"{name}={path}"])
    results = tmp_path / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", ["kojev.bench", *paths, "--results", str(results)])
    bench.main()
    table = results.read_text(encoding="utf-8")
    assert "## Accuracy" in table
    assert "## Brier" in table
    assert table.count("| main |") == 4
    headers = [line for line in table.splitlines() if "| boolq |" in line]
    assert headers
    assert headers[0].count("|") == 10


def test_metric_tables_include_f1_brier_and_ece_sections() -> None:
    """Plan RESULTS tables: per-task acc, F1, Brier, ECE — not accuracy alone."""
    models = (
        ("main", _full_tasks(0.50)),
        ("rlcd", _full_tasks(0.51)),
        ("control", _full_tasks(0.40)),
        ("distill", _full_tasks(0.49)),
        ("english", _full_tasks(0.20)),
    )
    document = render_metric_tables(models)
    for heading in ("## Accuracy", "## Macro-F1", "## Brier", "## ECE-15"):
        assert heading in document
    assert document.count("| model |") == 4
    assert "0.250" in document  # brier from the fixture
    assert "0.100" in document  # ece_15 from the fixture

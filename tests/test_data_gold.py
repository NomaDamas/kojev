"""Mapper and validation tests for the Korean gold corpus builder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kojev import data_gold
from kojev.data_gold import (
    build_korquad_pair,
    map_nli_row,
    map_nsmc_row,
    map_sts_row,
    map_unsmile_row,
    validate_jsonl,
)
from kojev.schema import Example, JsonValue, QuestionType

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_maps_nsmc_label_to_choice_and_noul_gold() -> None:
    example = map_nsmc_row({"document": "좋은 영화다", "label": 1}, split="train")

    assert example.source == "e9t/nsmc"
    assert example.questions[0].type is QuestionType.CHOICE
    assert example.questions[0].gold == 1
    assert example.questions[1].gold == 1


def test_maps_nli_label_to_entailment_neutral_contradiction() -> None:
    example = map_nli_row(
        {
            "premise": "비가 온다.",
            "hypothesis": "우산이 필요하다.",
            "label": 0,
        },
        split="val",
    )

    assert example.questions[0].options == ["함의", "중립", "모순"]
    assert example.questions[0].gold == 0


def test_maps_sts_real_label_to_rounded_score_and_binary_gold() -> None:
    example = map_sts_row(
        {
            "sentence1": "오늘은 맑다.",
            "sentence2": "날씨가 좋다.",
            "labels": {"real-label": 4.5, "binary-label": 1},
        },
        split="train",
    )

    assert example.questions[0].type is QuestionType.SCORE
    assert example.questions[0].gold == 4
    assert example.questions[1].gold == 1


def test_korquad_negative_pair_flips_answerability_gold() -> None:
    positive = build_korquad_pair(
        {"context": "서울은 수도이다.", "question": "수도는?", "id": "a"},
        split="train",
        negative_question="부산은 어디인가?",
    )
    negative = build_korquad_pair(
        {"context": "부산은 항구다.", "question": "수도는?", "id": "b"},
        split="train",
        negative_question="수도는?",
    )

    assert positive.questions[0].gold == 1
    assert negative.questions[0].gold == 0


def test_unsmile_emits_one_noul_per_category_and_single_label_choice() -> None:
    example = map_unsmile_row(
        {
            "문장": "깨끗한 글",
            "여성/가족 혐오": 0,
            "남성 혐오": 1,
            "clean": 0,
        },
        split="train",
    )

    assert len(example.questions) == 2
    assert [question.type for question in example.questions] == [
        QuestionType.NOUL,
        QuestionType.CHOICE,
    ]
    assert example.questions[0].gold == 1
    assert example.questions[1].gold == 1


def test_validation_reports_malformed_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    valid: dict[str, JsonValue] = {
        "state": "상태",
        "questions": [
            {
                "type": "noul",
                "instructions": "참인가?",
                "options": ["아니오", "예"],
                "gold": 1,
                "meta": {},
            }
        ],
        "source": "fixture",
        "split": "train",
    }
    _ = path.write_text(
        json.dumps(valid, ensure_ascii=False)
        + "\nnot json\n"
        + json.dumps(valid, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    errors = validate_jsonl(path)

    assert errors == [f"{path}:2: invalid JSON"]


def test_every_source_contributes_disjoint_train_val_and_test_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gold test split must span sources, not only datasets that happen to
    # ship an upstream "test" split.
    rows: list[dict[str, JsonValue]] = [
        {"document": f"리뷰 {index}", "label": index % 2} for index in range(12_000)
    ]

    def fake_rows(_config: object, _split: str) -> Iterator[dict[str, JsonValue]]:
        yield from rows

    monkeypatch.setattr(data_gold, "_iter_source_rows", fake_rows)
    config = {"source": "e9t/nsmc", "config": None, "splits": ("train",)}

    carved = data_gold._source_split_rows(config)  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert len(carved["test"]) == 1000
    assert len(carved["val"]) == 500
    assert len(carved["train"]) == 8000
    seen = [
        json.dumps(row, sort_keys=True)
        for split in ("train", "val", "test")
        for row in carved[split]
    ]
    assert len(seen) == len(set(seen))


def test_source_split_is_deterministic_under_the_split_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, JsonValue]] = [
        {"document": f"문장 {index}", "label": index % 2} for index in range(3_000)
    ]

    def fake_rows(_config: object, _split: str) -> Iterator[dict[str, JsonValue]]:
        yield from rows

    monkeypatch.setattr(data_gold, "_iter_source_rows", fake_rows)
    config = {"source": "e9t/nsmc", "config": None, "splits": ("train",)}

    first = data_gold._source_split_rows(config)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    second = data_gold._source_split_rows(config)  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert first == second


def test_validation_reports_line_number_for_blank_state(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    question: dict[str, JsonValue] = {
        "type": "noul",
        "instructions": "참인가?",
        "options": ["아니오", "예"],
        "gold": 1,
        "meta": {},
    }
    good: dict[str, JsonValue] = {
        "state": "좋은 영화",
        "questions": [question],
        "source": "fixture",
        "split": "train",
    }
    blank: dict[str, JsonValue] = {**good, "state": ""}
    _ = path.write_text(
        json.dumps(good, ensure_ascii=False)
        + "\n"
        + json.dumps(blank, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    errors = validate_jsonl(path)

    assert len(errors) == 1
    assert errors[0].startswith(f"{path}:2:")
    assert "non-blank state" in errors[0]


def test_validation_rejects_example_with_no_questions() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        _ = Example.model_validate(
            {
                "state": "상태",
                "questions": [],
                "source": "fixture",
                "split": "train",
            }
        )

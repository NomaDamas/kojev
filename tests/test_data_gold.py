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
from kojev.schema import Example, JsonValue, Question, QuestionType

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
    """Substituting a DIFFERENT question makes the pair unanswerable.

    This test previously asserted the inverse: that a differing
    `negative_question` stayed answerable (gold=1) and an identical one became
    unanswerable (gold=0). That is backwards, and it is why the defect survived
    review. The built corpus settled it: under those semantics every KorQuAD
    question in every split carried gold=1, 18,993 in total, because negatives
    are always built from a DIFFERENT title's question.
    """
    answerable = build_korquad_pair(
        {"context": "서울은 수도이다.", "question": "수도는?", "id": "a"},
        split="train",
    )
    unanswerable = build_korquad_pair(
        {"context": "서울은 수도이다.", "question": "수도는?", "id": "b"},
        split="train",
        negative_question="부산은 어디인가?",
    )

    assert answerable.questions[0].gold == 1
    assert unanswerable.questions[0].gold == 0
    # The unanswerable pair must actually ask the substituted question.
    assert "부산은 어디인가?" in unanswerable.state
    assert answerable.state != unanswerable.state


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

    # One positive noul per active category, a matched negative noul so the
    # sub-task is answerable both ways, then the single-label choice. The
    # negative count deliberately changed this expectation from 2 to 3.
    nouls = [q for q in example.questions if q.type is QuestionType.NOUL]
    choices = [q for q in example.questions if q.type is QuestionType.CHOICE]
    assert len(nouls) == 2
    assert len(choices) == 1
    assert sorted(q.gold for q in nouls if q.gold is not None) == [0, 1]
    assert choices[0].gold == 1


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


def test_split_state_isolation_gives_held_out_splits_priority() -> None:
    # Upstream corpora contain duplicate texts, so the same state can land on
    # both sides of a row-level carve. Held-out splits must win.
    def example(state: str, split: str) -> Example:
        return Example(
            state=state,
            questions=[
                Question(
                    type=QuestionType.NOUL,
                    instructions="긍정이다.",
                    options=["아니오", "예"],
                    gold=1,
                    meta={},
                )
            ],
            source="e9t/nsmc",
            split=split,
        )

    shared = "중복된 리뷰"
    buckets = {
        "test": [example(shared, "test"), example("테스트 전용", "test")],
        "val": [example(shared, "val"), example("검증 전용", "val")],
        "train": [example(shared, "train"), example("학습 전용", "train")],
    }
    per_source = {
        "e9t/nsmc:test": list(buckets["test"]),
        "e9t/nsmc:val": list(buckets["val"]),
        "e9t/nsmc:train": list(buckets["train"]),
    }
    summary: dict[str, dict[str, int | str]] = {
        key: {"states": 0, "questions": 0, "dropped_blank_state": 0}
        for key in per_source
    }

    data_gold._enforce_split_state_isolation(buckets, per_source, summary)  # pyright: ignore[reportPrivateUsage]

    test_states = {x.state for x in buckets["test"]}
    val_states = {x.state for x in buckets["val"]}
    train_states = {x.state for x in buckets["train"]}
    assert shared in test_states
    assert shared not in val_states
    assert shared not in train_states
    assert not test_states & val_states
    assert not test_states & train_states
    assert not val_states & train_states
    assert summary["e9t/nsmc:train"]["dropped_cross_split_state"] == 1
    assert summary["e9t/nsmc:val"]["dropped_cross_split_state"] == 1
    assert summary["e9t/nsmc:test"]["dropped_cross_split_state"] == 0


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


def test_korquad_negative_asks_the_substituted_question_and_flips_gold() -> None:
    """An unanswerable KorQuAD pair must ask the OTHER question and answer no.

    Measured on the built corpus, every KorQuAD question in every split carried
    gold=1: 15,995 train, 999 val, 1,999 test. The answerability task was
    degenerate, and since the negative reused the row's own question the
    negative example was a byte-identical duplicate of its positive.
    """
    row = {
        "id": "q-1",
        "title": "A",
        "context": "고양이는 포유류이다.",
        "question": "고양이는 무엇인가?",
    }

    positive = build_korquad_pair(row, split="train")
    negative = build_korquad_pair(
        row, split="train", negative_question="에펠탑은 어디에 있는가?"
    )

    assert positive.questions[0].gold == 1
    assert "고양이는 무엇인가?" in positive.state

    # The negative must ask the substituted question and be unanswerable.
    assert negative.questions[0].gold == 0, "negative kept the answerable gold"
    assert "에펠탑은 어디에 있는가?" in negative.state, (
        "negative never asked the substituted question"
    )
    assert negative.state != positive.state, "negative duplicates its positive"


def test_unsmile_emits_negative_nouls_for_inactive_categories() -> None:
    """UnSmile nouls must be answerable both ways.

    Measured on the built corpus, every UnSmile noul carried gold=1: 8,610 in
    train and 519 in val. Nouls were emitted only for ACTIVE categories, so the
    honest answer was always 예 and the sub-task taught nothing. The choice
    questions were healthy, which is why the defect was not obvious from the
    source-level numbers alone.
    """
    example = map_unsmile_row(
        {
            "문장": "깨끗한 글",
            "여성/가족": 0,
            "남성": 1,
            "성소수자": 0,
            "인종/국적": 0,
            "연령": 0,
            "지역": 0,
            "종교": 0,
            "기타 혐오": 0,
            "악플/욕설": 0,
            "clean": 0,
        },
        split="train",
    )

    nouls = [q for q in example.questions if q.type is QuestionType.NOUL]
    golds = {q.gold for q in nouls}
    assert 1 in golds, "no positive noul emitted"
    assert 0 in golds, "every noul is answerable 예; the sub-task is degenerate"

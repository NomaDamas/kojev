from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kojev.data_ood import (
    assert_no_train_instruction_overlap,
    build_ood_examples,
    reversed_monotone_gold,
)
from kojev.schema import Example, Question, QuestionType

if TYPE_CHECKING:
    from pathlib import Path


def test_reversed_monotone_rule_flips_polarity() -> None:
    assert reversed_monotone_gold(0, 2) == 1
    assert reversed_monotone_gold(1, 2) == 0


def test_ood_probes_use_rule_gold_and_six_families(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    gold.mkdir()
    example = Example(
        state="좋은 영화",
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="원래 질문",
                options=["부정", "긍정"],
                gold=1,
            )
        ],
        source="e9t/nsmc",
        split="test",
    )
    _ = (gold / "test.jsonl").write_text(
        example.model_dump_json() + "\n", encoding="utf-8"
    )

    built = build_ood_examples(gold)

    assert built
    assert {item.source for item in built} == {"e9t/nsmc"}
    assert all(item.split == "test" for item in built)
    assert any(
        question.type is QuestionType.SCORE
        for item in built
        for question in item.questions
    )


def test_train_instruction_overlap_is_a_hard_failure(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    ood = tmp_path / "ood.jsonl"
    payload = {
        "state": "상태",
        "questions": [
            {
                "type": "noul",
                "instructions": "침범 질문",
                "options": ["아니오", "예"],
                "gold": 1,
            }
        ],
        "source": "fixture",
        "split": "train",
    }
    _ = train.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _ = ood.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="침범 질문"):
        assert_no_train_instruction_overlap(train, ood)

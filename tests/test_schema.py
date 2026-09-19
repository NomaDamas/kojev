"""Schema JSONL round-trip and option-count validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kojev.schema import (
    Example,
    Question,
    QuestionType,
    confidence,
    read_jsonl,
    write_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path


def _mixed_example() -> Example:
    return Example(
        state="리뷰: 배송이 빠르고 포장이 단단했다.",
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="이 리뷰의 극성은?",
                options=["부정", "긍정"],
                gold=1,
                meta={"family": "polarity"},
            ),
            Question(
                type=QuestionType.SCORE,
                instructions="만족도를 고르시오",
                options=["매우 불만", "불만", "보통", "만족", "매우 만족"],
                gold=4,
                meta={},
            ),
            Question(
                type=QuestionType.NOUL,
                instructions="이 리뷰는 긍정적이다",
                options=["아니오", "예"],
                gold=1,
                meta={"kind": "noul"},
            ),
        ],
        source="unit",
        split="train",
    )


def test_round_trips_mixed_example_through_jsonl_when_written_then_read(
    tmp_path: Path,
) -> None:
    # Given a mixed choice/score/noul example
    example = _mixed_example()
    path = tmp_path / "examples.jsonl"

    # When it is written and read back as JSONL
    write_jsonl(path, [example])
    loaded = read_jsonl(path)

    # Then field values are preserved
    assert loaded == [example]
    parsed = loaded[0]
    assert parsed.state == example.state
    assert parsed.source == "unit"
    assert parsed.split == "train"
    assert parsed.questions[0].type == QuestionType.CHOICE
    assert parsed.questions[0].options == ["부정", "긍정"]
    assert parsed.questions[0].gold == 1
    assert parsed.questions[1].type == QuestionType.SCORE
    assert parsed.questions[1].gold == 4
    assert parsed.questions[2].type == QuestionType.NOUL
    assert parsed.questions[2].gold == 1


def test_rejects_score_question_when_option_count_is_one() -> None:
    # Given a score question with a single level
    # When it is constructed
    # Then validation fails
    with pytest.raises(ValueError, match="score"):
        _ = Question(
            type=QuestionType.SCORE,
            instructions="만족도",
            options=["유일"],
            gold=0,
            meta={},
        )


def test_rejects_score_question_when_option_count_is_eleven() -> None:
    with pytest.raises(ValueError, match="score"):
        _ = Question(
            type=QuestionType.SCORE,
            instructions="만족도",
            options=[str(i) for i in range(11)],
            gold=0,
            meta={},
        )


def test_rejects_choice_question_when_option_count_exceeds_255() -> None:
    with pytest.raises(ValueError, match="choice"):
        _ = Question(
            type=QuestionType.CHOICE,
            instructions="고르시오",
            options=[str(i) for i in range(256)],
            gold=0,
            meta={},
        )


def test_rejects_example_when_state_is_empty() -> None:
    with pytest.raises(ValueError, match="non-blank state"):
        _ = Example(
            state="",
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
            split="train",
        )


def test_rejects_example_when_state_is_only_whitespace() -> None:
    with pytest.raises(ValueError, match="non-blank state"):
        _ = Example(
            state="   \n\t ",
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
            split="train",
        )


def test_confidence_is_one_when_distribution_is_peaked() -> None:
    assert confidence((1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_confidence_is_zero_when_distribution_is_uniform() -> None:
    assert confidence((0.5, 0.5)) == pytest.approx(0.0)

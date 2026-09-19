"""Property and unit tests for gold-preserving Korean augmentation."""

# ruff: noqa: S311

from __future__ import annotations

import random

import pytest

from kojev.augment import (
    INSTRUCTION_TEMPLATES,
    PARAPHRASE_TEMPLATES,
    augment_example,
    drop_distractor,
    negate_noul,
    paraphrase_instruction,
    shuffle_options,
    swap_score_synonyms,
)
from kojev.schema import Example, Question, QuestionType


def _gold(question: Question) -> int:
    assert question.gold is not None
    return question.gold


def _choice(gold: int = 2) -> Question:
    return Question(
        type=QuestionType.CHOICE,
        instructions="이 리뷰의 주제는 무엇인가?",
        options=["배송", "가격", "품질", "서비스"],
        gold=gold,
        meta={"family": "choice"},
    )


def _score() -> Question:
    return Question(
        type=QuestionType.SCORE,
        instructions="만족도를 고르시오",
        options=["매우 부정", "부정", "보통", "긍정", "매우 긍정"],
        gold=4,
        meta={"family": "score"},
    )


def _noul(gold: int = 1) -> Question:
    return Question(
        type=QuestionType.NOUL,
        instructions="이 리뷰는 긍정적이다",
        options=["아니오", "예"],
        gold=gold,
        meta={"family": "noul"},
    )


def test_shuffle_options_remaps_gold_to_the_same_option() -> None:
    question = _choice()

    transformed = shuffle_options(question, random.Random(7))

    assert transformed.options[_gold(transformed)] == question.options[_gold(question)]
    assert sorted(transformed.options) == sorted(question.options)


def test_paraphrase_instruction_has_eight_deterministic_templates() -> None:
    question = _choice()

    assert len(PARAPHRASE_TEMPLATES) >= 8
    assert (
        len(
            {
                paraphrase_instruction(question, random.Random(seed)).instructions
                for seed in range(32)
            }
        )
        >= 8
    )


def test_paraphrase_instruction_uses_final_consonant_josa() -> None:
    consonant = Question(
        type=QuestionType.CHOICE,
        instructions="감정",
        options=["기쁨", "슬픔"],
        gold=0,
        meta={},
    )
    vowel = Question(
        type=QuestionType.CHOICE,
        instructions="주제",
        options=["배송", "가격"],
        gold=0,
        meta={},
    )

    assert "감정은" in paraphrase_instruction(consonant, random.Random(0)).instructions
    assert "주제는" in paraphrase_instruction(vowel, random.Random(0)).instructions


def test_each_question_family_has_eight_fixed_instruction_templates() -> None:
    for question_type in QuestionType:
        assert len(INSTRUCTION_TEMPLATES[question_type]) >= 8


def test_paraphrase_instruction_uses_both_eun_neun_and_i_ga_particles() -> None:
    question = Question(
        type=QuestionType.CHOICE,
        instructions="감정",
        options=["기쁨", "슬픔"],
        gold=0,
        meta={},
    )

    instructions = {
        paraphrase_instruction(question, random.Random(seed)).instructions
        for seed in range(128)
    }

    assert any("감정은" in instruction for instruction in instructions)
    assert any("감정이" in instruction for instruction in instructions)


def test_score_paraphrase_removes_existing_instruction_suffix() -> None:
    question = _score()

    transformed = paraphrase_instruction(question, random.Random(0))

    assert "만족도" in transformed.instructions
    assert "고르시오고르시오" not in transformed.instructions


def test_score_synonyms_cover_similarity_levels_without_reordering() -> None:
    question = Question(
        type=QuestionType.SCORE,
        instructions="유사도",
        options=["전혀 다름", "다름", "보통", "비슷함", "완전히 같음"],
        gold=3,
        meta={},
    )

    transformed = swap_score_synonyms(question, random.Random(0))

    assert transformed.gold == question.gold
    assert transformed.options != question.options
    assert len(transformed.options) == len(question.options)


def test_drop_distractor_never_drops_the_gold_option() -> None:
    question = _choice(gold=1)

    transformed = drop_distractor(question, random.Random(11))

    assert len(transformed.options) == 3
    assert transformed.options[_gold(transformed)] == question.options[_gold(question)]


def test_drop_distractor_is_noop_below_four_options() -> None:
    question = Question(
        type=QuestionType.CHOICE,
        instructions="무엇인가?",
        options=["아니오", "예", "모름"],
        gold=1,
        meta={},
    )

    assert drop_distractor(question, random.Random(3)) == question


def test_score_synonyms_preserve_order_and_gold_index() -> None:
    question = _score()

    transformed = swap_score_synonyms(question, random.Random(13))

    assert transformed.gold == question.gold
    assert len(transformed.options) == len(question.options)
    assert transformed.options[0] != question.options[0]


def test_negating_noul_flips_gold_exactly() -> None:
    question = _noul(gold=1)

    transformed = negate_noul(question)

    assert transformed.instructions == "이 리뷰는 긍정적이 아니다"
    assert transformed.gold == 0
    assert transformed.options == question.options


def test_augment_example_preserves_gold_across_all_question_kinds() -> None:
    example = Example(
        state="배송이 빨랐다.",
        questions=[_choice(), _score(), _noul()],
        source="unit",
        split="train",
    )

    transformed = augment_example(example, random.Random(0), probability=1.0)

    for original, candidate in zip(
        example.questions,
        transformed.questions,
        strict=True,
    ):
        assert candidate.gold is not None
        assert original.gold is not None
        if original.type is QuestionType.CHOICE:
            assert candidate.options[_gold(candidate)] in original.options
        elif original.type is QuestionType.SCORE:
            assert candidate.gold == original.gold
        else:
            assert candidate.gold == 1 - original.gold


def test_invalid_option_input_is_rejected_by_schema_before_augmentation() -> None:
    with pytest.raises(ValueError, match="choice"):
        _ = Question(
            type=QuestionType.CHOICE,
            instructions="고르시오",
            options=["하나"],
            gold=0,
            meta={},
        )


def test_one_thousand_deterministic_examples_preserve_gold_for_every_transform() -> (
    None
):
    # Given 1,000 deterministically generated valid questions
    for seed in range(1000):
        rng = random.Random(seed)
        choice = Question(
            type=QuestionType.CHOICE,
            instructions=f"항목{seed}의 분류는 무엇인가?",
            options=[f"선택지{index}" for index in range(4 + seed % 4)],
            gold=seed % (4 + seed % 4),
            meta={},
        )
        score = Question(
            type=QuestionType.SCORE,
            instructions="만족도",
            options=["매우 부정", "부정", "보통", "긍정", "매우 긍정"],
            gold=seed % 5,
            meta={},
        )
        noul = _noul(gold=seed % 2)

        # When every transform is applied with a deterministic seed
        shuffled = shuffle_options(choice, rng)
        paraphrased = paraphrase_instruction(choice, rng)
        dropped = drop_distractor(choice, rng)
        swapped = swap_score_synonyms(score, rng)
        negated = negate_noul(noul)

        # Then the semantic gold answer remains the same (or flips exactly).
        assert choice.options[_gold(choice)] == shuffled.options[_gold(shuffled)]
        assert set(dropped.options).issubset(choice.options)
        assert choice.options[_gold(choice)] in dropped.options
        assert paraphrased.gold == choice.gold
        assert swapped.gold == score.gold
        assert negated.gold == 1 - _gold(noul)


def test_negative_gold_invariant_probe_catches_a_dropped_gold_option() -> None:
    # Given a deliberately broken transform that removes the gold option
    original = _choice(gold=2)
    broken = original.model_copy(
        update={
            "options": [original.options[0], original.options[1], original.options[3]],
            "gold": 2,
        }
    )

    # When the invariant is checked
    def assert_preserved(candidate: Question) -> None:
        assert original.options[_gold(original)] in candidate.options
        assert candidate.options[_gold(candidate)] == original.options[_gold(original)]

    # Then the probe fails rather than allowing a silently corrupted label.
    with pytest.raises(AssertionError):
        assert_preserved(broken)

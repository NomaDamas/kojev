"""Gold-preserving Korean question augmentation transforms."""

from __future__ import annotations

import random  # noqa: TC003
from dataclasses import dataclass
from typing import Final, assert_never, override

from kojev.schema import Example, Question, QuestionType

_HANGUL_START: Final[int] = 0xAC00
_HANGUL_END: Final[int] = 0xD7A3
_JONGSEONG_COUNT: Final[int] = 28
_MIN_DISTRACTOR_OPTIONS: Final[int] = 4
_PARAPHRASE_TEMPLATES: Final[tuple[str, ...]] = (
    "{topic}{eun_neun} 무엇인가?",
    "{topic}{eun_neun} 무엇인지 고르시오.",
    "다음 중 {topic}{eun_neun} 가장 알맞은 것을 고르시오.",
    "{topic}{eun_neun} 어떻게 분류되는지 선택하시오.",
    "아래 선택지에서 {topic}{eun_neun} 골라 주세요.",
    "{topic}{eun_neun} 해당하는 답을 고르시오.",
    "{topic}{eun_neun} 판단해 보시오.",
    "무엇이 {topic}{i_ga} 맞는 설명인가?",
)

INSTRUCTION_TEMPLATES: Final[dict[QuestionType, tuple[str, ...]]] = dict.fromkeys(
    QuestionType, _PARAPHRASE_TEMPLATES
)
PARAPHRASE_TEMPLATES: Final[tuple[str, ...]] = _PARAPHRASE_TEMPLATES

SCORE_SYNONYMS: Final[tuple[tuple[str, str], ...]] = (
    ("전혀 다름", "완전히 다름"),
    ("다름", "다른 편"),
    ("매우 부정", "아주 나쁨"),
    ("부정", "나쁨"),
    ("보통", "중간"),
    ("비슷함", "비슷한 편"),
    ("긍정", "좋음"),
    ("매우 긍정", "아주 좋음"),
    ("완전히 같음", "완전히 동일함"),
    ("매우 낮음", "아주 낮음"),
    ("매우 높음", "아주 높음"),
)
SCORE_SYNONYMS_ALT: Final[tuple[tuple[str, str], ...]] = (
    ("전혀 다름", "아예 다름"),
    ("다름", "차이가 있음"),
    ("매우 부정", "최악"),
    ("부정", "나쁨"),
    ("보통", "중간 정도"),
    ("비슷함", "닮은 편"),
    ("긍정", "좋음"),
    ("매우 긍정", "최상"),
    ("완전히 같음", "완전히 일치함"),
    ("매우 낮음", "낮은 편"),
    ("매우 높음", "높은 편"),
)


@dataclass(frozen=True, slots=True)
class AugmentationError(ValueError):
    """Invalid augmentation configuration."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the structured error reason."""
        return self.reason


def _has_final_consonant(text: str) -> bool:
    for character in reversed(text):
        codepoint = ord(character)
        if _HANGUL_START <= codepoint <= _HANGUL_END:
            return (codepoint - _HANGUL_START) % _JONGSEONG_COUNT != 0
    return False


def _eun_neun(text: str) -> str:
    return "은" if _has_final_consonant(text) else "는"


def _i_ga(text: str) -> str:
    return "이" if _has_final_consonant(text) else "가"


def _topic(instructions: str) -> str:
    topic = instructions.rstrip(" ?.!。")
    for suffix in ("고르시오", "선택하시오", "골라 주세요", "판단해 보시오"):
        if topic.endswith(suffix):
            topic = topic[: -len(suffix)].rstrip()
            break
    return topic.removesuffix("이다").rstrip()


def shuffle_options(question: Question, rng: random.Random) -> Question:
    """Shuffle choice options and remap the gold index."""
    if question.type is not QuestionType.CHOICE:
        return question
    order = list(range(len(question.options)))
    rng.shuffle(order)
    options = [question.options[index] for index in order]
    gold = None if question.gold is None else order.index(question.gold)
    return question.model_copy(update={"options": options, "gold": gold})


def paraphrase_instruction(question: Question, rng: random.Random) -> Question:
    """Replace the instruction with one of eight fixed Korean templates."""
    topic = _topic(question.instructions)
    template = rng.choice(INSTRUCTION_TEMPLATES[question.type])
    return question.model_copy(
        update={
            "instructions": template.format(
                topic=topic,
                eun_neun=_eun_neun(topic),
                i_ga=_i_ga(topic),
            )
        }
    )


def drop_distractor(question: Question, rng: random.Random) -> Question:
    """Drop one non-gold choice when at least four options are available."""
    if (
        question.type is not QuestionType.CHOICE
        or len(question.options) < _MIN_DISTRACTOR_OPTIONS
    ):
        return question
    candidates = [
        index for index in range(len(question.options)) if index != question.gold
    ]
    dropped = rng.choice(candidates)
    options = [
        option for index, option in enumerate(question.options) if index != dropped
    ]
    gold = None if question.gold is None else question.gold - (question.gold > dropped)
    return question.model_copy(update={"options": options, "gold": gold})


def swap_score_synonyms(question: Question, rng: random.Random) -> Question:
    """Replace score labels with ordered synonyms without changing indices."""
    if question.type is not QuestionType.SCORE:
        return question
    replacements: dict[str, str] = dict(
        SCORE_SYNONYMS if rng.randrange(2) == 0 else SCORE_SYNONYMS_ALT
    )
    options = [replacements.get(option, option) for option in question.options]
    return question.model_copy(update={"options": options})


def negate_noul(question: Question) -> Question:
    """Negate a noul statement and flip its binary gold label."""
    if question.type is not QuestionType.NOUL:
        return question
    instruction = question.instructions
    if instruction.endswith("이다"):
        instruction = f"{instruction[:-2]}이 아니다"
    elif instruction.endswith("다"):
        instruction = f"{instruction[:-1]}지 않다"
    else:
        instruction = f"{instruction}이 아니다"
    gold = None if question.gold is None else 1 - question.gold
    return question.model_copy(update={"instructions": instruction, "gold": gold})


def augment_example(
    example: Example,
    rng: random.Random,
    probability: float = 0.7,
) -> Example:
    """Apply independently sampled gold-preserving transforms."""
    if not 0.0 <= probability <= 1.0:
        raise AugmentationError(
            reason="augmentation probability must be between 0 and 1"
        )
    questions: list[Question] = []
    for question in example.questions:
        transformed = question
        if rng.random() < probability:
            match question.type:
                case QuestionType.CHOICE:
                    transformed = shuffle_options(transformed, rng)
                    transformed = paraphrase_instruction(transformed, rng)
                    transformed = drop_distractor(transformed, rng)
                case QuestionType.SCORE:
                    transformed = paraphrase_instruction(transformed, rng)
                    transformed = swap_score_synonyms(transformed, rng)
                case QuestionType.NOUL:
                    transformed = negate_noul(transformed)
                case unreachable:
                    assert_never(unreachable)
        questions.append(transformed)
    return example.model_copy(update={"questions": questions})

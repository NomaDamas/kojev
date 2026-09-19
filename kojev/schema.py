"""Typed-decision schema: state plus choice, score, and noul questions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final, Self, assert_never, override

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

type JsonValue = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
)

_CHOICE_MIN: Final = 2
_CHOICE_MAX: Final = 255
_SCORE_MIN: Final = 2
_SCORE_MAX: Final = 10
_NOUL_COUNT: Final = 2
_CONFIDENCE_MIN_K: Final = 2


@dataclass(frozen=True, slots=True)
class SchemaError(ValueError):
    """Malformed typed-decision input."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the structured reason."""
        return self.reason

    @classmethod
    def bad_option_count(cls, kind: QuestionType, got: int) -> SchemaError:
        """Reject an illegal option/level cardinality."""
        match kind:
            case QuestionType.CHOICE:
                expected = f"{_CHOICE_MIN}..{_CHOICE_MAX} options"
            case QuestionType.SCORE:
                expected = f"{_SCORE_MIN}..{_SCORE_MAX} ordered levels"
            case QuestionType.NOUL:
                expected = f"{_NOUL_COUNT} options"
            case unreachable:
                assert_never(unreachable)
        return cls(reason=f"{kind} questions require {expected}, got {got}")

    @classmethod
    def bad_gold(cls, gold: int, count: int) -> SchemaError:
        """Reject a gold index outside the option list."""
        return cls(reason=f"gold {gold} is outside 0..{count - 1}")

    @classmethod
    def confidence_k(cls) -> SchemaError:
        """Reject a confidence call with fewer than two probabilities."""
        return cls(reason="confidence requires K >= 2")


class QuestionType(StrEnum):
    """Closed set of typed-decision primitives."""

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class Question(BaseModel):
    """One typed question attached to a state."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    type: QuestionType
    instructions: str
    options: list[str]
    gold: int | None = None
    meta: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_options_and_gold(self) -> Self:
        count = len(self.options)
        match self.type:
            case QuestionType.CHOICE:
                legal = _CHOICE_MIN <= count <= _CHOICE_MAX
            case QuestionType.SCORE:
                legal = _SCORE_MIN <= count <= _SCORE_MAX
            case QuestionType.NOUL:
                legal = count == _NOUL_COUNT
            case unreachable:
                assert_never(unreachable)
        if not legal:
            raise SchemaError.bad_option_count(self.type, count)
        if self.gold is not None and not 0 <= self.gold < count:
            raise SchemaError.bad_gold(self.gold, count)
        return self


class Example(BaseModel):
    """A state with one or more typed questions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    state: str
    questions: list[Question]
    source: str
    split: str


def confidence(probs: Sequence[float]) -> float:
    """Return ``1 - H(p) / ln K`` for a categorical distribution ``p``.

    Entropy uses the natural log so a peaked distribution scores 1 and a
    uniform distribution scores 0.
    """
    kind_count = len(probs)
    if kind_count < _CONFIDENCE_MIN_K:
        raise SchemaError.confidence_k()
    entropy = 0.0
    for prob in probs:
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return 1.0 - entropy / math.log(kind_count)


def write_jsonl(path: Path, examples: Sequence[Example]) -> None:
    """Write examples as UTF-8 JSONL, one object per line."""
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = f"{example.model_dump_json()}\n"
            _ = handle.write(payload)


def read_jsonl(path: Path) -> list[Example]:
    """Parse a JSONL file of examples, skipping blank lines."""
    parsed: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                parsed.append(Example.model_validate_json(line))
    return parsed

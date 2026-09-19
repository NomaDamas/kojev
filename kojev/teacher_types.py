"""Public teacher result and parser types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, override

_ANSWER_LINE: Final = re.compile(
    (r"^\s*(?:a|q|question)?\s*([1-9]\d*)\s*(?::|=|-)\s*"
     r"(?:option\s*)?([0-9]+)\b"),
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BudgetReached:
    """Normal result when the durable budget refusal marker is reached."""

    marker: str = "BUDGET_REACHED"


class MissingApiKeyError(RuntimeError):
    """Raised when no OpenRouter credential is configured."""

    @override
    def __str__(self) -> str:
        """Return the missing-credential diagnostic."""
        return "OPENROUTER_API_KEY is required"


@dataclass(frozen=True, slots=True)
class TeacherRequestError(RuntimeError):
    """Raised when OpenRouter retries are exhausted."""

    status_code: int

    @override
    def __str__(self) -> str:
        """Return the exhausted-retries diagnostic."""
        return f"OpenRouter request exhausted retries: HTTP {self.status_code}"


def parse_teacher_answers(raw: str, question_count: int) -> dict[int, int]:
    """Parse only line-oriented indexed answers, ignoring prose and injections."""
    answers: dict[int, int] = {}
    for line in raw.splitlines():
        match = _ANSWER_LINE.match(line)
        if match is None:
            continue
        question_index = int(match.group(1))
        option_index = int(match.group(2))
        if 1 <= question_index <= question_count:
            answers[question_index] = option_index
    return answers

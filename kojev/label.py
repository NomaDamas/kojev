"""Teacher-label goldless Korean question families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import anyio
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict

from kojev.schema import Example, Question, QuestionType, read_jsonl
from kojev.teacher import INITIAL_CONCURRENCY, BudgetReached, TeacherClient

if TYPE_CHECKING:
    from collections.abc import Sequence

LABEL_SOURCE: Final = "teacher:qwen3-vl-8b-instruct"
WAVE_SIZE: Final = INITIAL_CONCURRENCY
RETRY_COUNT: Final = 2
TOPICS: Final[tuple[str, ...]] = (
    "과학",
    "역사",
    "사회",
    "문화",
    "경제",
    "정치",
    "환경",
    "기술",
)
DIFFICULTIES: Final[tuple[str, ...]] = ("쉬움", "보통", "어려움")


class LedgerRecord(BaseModel):
    """Minimal typed view of the existing teacher ledger."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    kind: str
    request_id: str | None = None
    candidate_id: str | None = None


class OutputLine(BaseModel):
    """Resume view of one labeled example, schema-valid or legacy extra keys."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    state: str
    request_id: str | None = None
    candidate_id: str | None = None
    questions: list[Question] = []

    def provider_request_id(self) -> str | None:
        """Prefer a top-level id, else the stamped question meta."""
        if self.request_id is not None:
            return self.request_id
        if not self.questions:
            return None
        value = self.questions[0].meta.get("provider_request_id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class LabeledRecord:
    """One serialized teacher-labeled example."""

    example: Example
    request_id: str
    cost_usd: float
    candidate_id: str = ""


class CliArgs(argparse.Namespace):
    """Typed mutable namespace populated by argparse."""

    gold: Path = Path("data/gold/train.jsonl")
    output: Path = Path("data/distill/train.jsonl")
    ledger: Path = Path("data/distill/ledger.jsonl")
    limit: int | None = None
    korquad_only: bool = False


def _goldless(example: Example) -> bool:
    """Return whether every question is genuinely unlabeled."""
    return all(question.gold is None for question in example.questions)


def _korquad_family(context: str, split: str) -> Example:
    """Build three goldless questions over one KorQuAD context."""
    return Example(
        state=f"지문: {context}",
        source="KorQuAD/squad_kor_v1:goldless",
        split=split,
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="이 지문의 주제를 고르시오.",
                options=list(TOPICS),
            ),
            Question(
                type=QuestionType.NOUL,
                instructions="이 지문은 수치/통계를 포함한다.",
                options=["아니오", "예"],
            ),
            Question(
                type=QuestionType.SCORE,
                instructions="지문의 난이도",
                options=list(DIFFICULTIES),
            ),
        ],
    )


def _gold_train_family(example: Example) -> Example:
    """Build fresh reading-comprehension questions over a gold train state."""
    return Example(
        state=example.state,
        source=f"{example.source}:gold-train-goldless",
        split="train",
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="지문의 핵심 주장에 가장 가까운 설명을 고르시오.",
                options=["핵심 주장", "세부 정보", "반대 주장", "관련 없는 설명"],
            ),
            Question(
                type=QuestionType.NOUL,
                instructions="지문에서 직접 확인할 수 있는 내용이 있다.",
                options=["아니오", "예"],
            ),
            Question(
                type=QuestionType.SCORE,
                instructions="지문의 독해 난이도",
                options=list(DIFFICULTIES),
            ),
        ],
    )


def build_goldless_families(
    examples: Sequence[Example], *, include_gold_train: bool = False
) -> list[Example]:
    """Keep only unlabeled inputs, optionally deriving fresh train questions."""
    if include_gold_train:
        return [
            _gold_train_family(example)
            for example in examples
            if example.split == "train"
        ]
    return [example for example in examples if _goldless(example)]


def build_korquad_families(limit: int | None = None) -> list[Example]:
    """Load KorQuAD train contexts without reading or writing data/gold."""
    dataset = load_dataset("KorQuAD/squad_kor_v1", split="train")
    families: list[Example] = []
    for row in dataset:
        context = row["context"]
        if not isinstance(context, str):
            message = "KorQuAD context must be text"
            raise TypeError(message)
        families.append(_korquad_family(context, "train"))
    return families if limit is None else families[:limit]


def _candidate_id(example: Example) -> str:
    """Return a stable ID for one state and question family."""
    requested = example.questions[0].meta.get("request_id")
    if isinstance(requested, str):
        return requested
    payload = {
        "state": example.state,
        "questions": [
            {
                "type": question.type,
                "instructions": question.instructions,
                "options": question.options,
            }
            for question in example.questions
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _issued_records(
    ledger_path: Path, output_path: Path
) -> tuple[set[str], set[str], set[str]]:
    """Read completed provider and deterministic candidate IDs."""
    request_ids: set[str] = set()
    candidate_ids: set[str] = set()
    states: set[str] = set()
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                record = LedgerRecord.model_validate_json(line)
            except ValueError:
                continue
            if record.kind == "usage":
                if record.request_id is not None:
                    request_ids.add(record.request_id)
                if record.candidate_id is not None:
                    candidate_ids.add(record.candidate_id)
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                output = OutputLine.model_validate_json(line)
            except ValueError:
                continue
            states.add(output.state)
            candidate_ids.add(
                output.candidate_id
                or _candidate_id(
                    Example(
                        state=output.state,
                        questions=output.questions,
                        source="resume",
                        split="train",
                    )
                )
            )
    return request_ids, candidate_ids, states


def _question_prompt(question: Question) -> str:
    """Make the existing constrained teacher client see each option."""
    options = " ".join(
        f"{index}: {option}" for index, option in enumerate(question.options)
    )
    return (
        f"{question.instructions} 선택지: {options} "
        "반드시 정확히 다음 형식으로 답하세요: A1: 0"
    )


async def _ask_with_fallback(
    client: TeacherClient, state: str, questions: Sequence[Question]
) -> tuple[dict[int, int], str, float] | None:
    """Ask together first, then isolate questions when parsing is incomplete."""
    result = await client.ask(
        state, [_question_prompt(question) for question in questions]
    )
    if isinstance(result, BudgetReached):
        return None
    expected = range(1, len(questions) + 1)
    if all(index in result.answers for index in expected):
        return result.answers, result.request_id, result.cost_usd
    answers: dict[int, int] = {}
    request_id = result.request_id
    total_cost = result.cost_usd
    for index, question in enumerate(questions, 1):
        single = await client.ask(state, [_question_prompt(question)])
        if isinstance(single, BudgetReached) or 1 not in single.answers:
            return None
        answers[index] = single.answers[1]
        total_cost += single.cost_usd
    return answers, request_id, total_cost


async def _label_with_retries(
    client: TeacherClient, example: Example
) -> LabeledRecord | None:
    """Retry timeout failures without allowing one item to abort its wave."""
    candidate_id = _candidate_id(example)
    for _ in range(RETRY_COUNT):
        try:
            labeled = await _ask_with_fallback(client, example.state, example.questions)
        except TimeoutError:
            continue
        if labeled is None:
            return None
        answers, provider_request_id, cost_usd = labeled
        labeled_questions = [
            question.model_copy(
                update={
                    "gold": answers[index],
                    "meta": {
                        **question.meta,
                        "label_source": LABEL_SOURCE,
                        "request_id": candidate_id,
                        "provider_request_id": provider_request_id,
                    },
                }
            )
            for index, question in enumerate(example.questions, 1)
        ]
        return LabeledRecord(
            example=example.model_copy(update={"questions": labeled_questions}),
            request_id=provider_request_id,
            cost_usd=cost_usd,
            candidate_id=candidate_id,
        )
    return None


class LabelRunner:
    """Run teacher requests in bounded waves and append durable labels."""

    def __init__(
        self, *, client: TeacherClient, output_path: Path, ledger_path: Path
    ) -> None:
        """Store the existing teacher client and append-only destinations."""
        self._client: TeacherClient = client
        self._output_path: Path = output_path
        self._ledger_path: Path = ledger_path

    def _append_records(self, records: Sequence[LabeledRecord]) -> None:
        """Append one completed wave without rewriting prior output."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                _ = handle.write(f"{record.example.model_dump_json()}\n")

    async def _run_wave(
        self, examples: Sequence[Example], issued: set[str], candidate_ids: set[str]
    ) -> list[LabeledRecord]:
        """Run exactly one bounded wave and wait for all its tasks."""
        records: list[LabeledRecord] = []
        lock = anyio.Lock()

        async def label(example: Example) -> None:
            record = await _label_with_retries(self._client, example)
            if record is None:
                return
            async with lock:
                candidate_id = record.candidate_id or _candidate_id(example)
                if record.request_id in issued or candidate_id in candidate_ids:
                    return
                issued.add(record.request_id)
                candidate_ids.add(candidate_id)
                records.append(
                    record
                    if record.candidate_id == candidate_id
                    else LabeledRecord(
                        example=record.example,
                        request_id=record.request_id,
                        cost_usd=record.cost_usd,
                        candidate_id=candidate_id,
                    )
                )

        async with anyio.create_task_group() as task_group:
            for example in examples:
                _ = task_group.start_soon(label, example)
        return records

    async def run(self, examples: Sequence[Example]) -> int:
        """Label candidates in bounded waves and return new record count."""
        issued, candidate_ids, completed_states = _issued_records(
            self._ledger_path, self._output_path
        )
        candidates = [
            example
            for example in examples
            if (
                _goldless(example)
                and example.state not in completed_states
                and _candidate_id(example) not in candidate_ids
                and _candidate_id(example) not in issued
            )
        ]
        labeled_count = 0
        for start in range(0, len(candidates), WAVE_SIZE):
            wave = candidates[start : start + WAVE_SIZE]
            records = await self._run_wave(wave, issued, candidate_ids)
            self._append_records(records)
            labeled_count += len(records)
        return labeled_count


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--gold", type=Path, default=Path("data/gold/train.jsonl"))
    _ = parser.add_argument(
        "--output", type=Path, default=Path("data/distill/train.jsonl")
    )
    _ = parser.add_argument(
        "--ledger", type=Path, default=Path("data/distill/ledger.jsonl")
    )
    _ = parser.add_argument("--limit", type=int)
    _ = parser.add_argument("--korquad-only", action="store_true")
    return parser.parse_args(namespace=CliArgs())


async def _run(args: CliArgs) -> int:
    korquad = build_korquad_families(args.limit)
    gold_train = (
        []
        if args.korquad_only
        else build_goldless_families(read_jsonl(args.gold), include_gold_train=True)
    )
    examples = [*korquad, *gold_train]
    async with TeacherClient(
        ledger_path=args.ledger, api_key=os.environ.get("OPENROUTER_API_KEY")
    ) as client:
        count = await LabelRunner(
            client=client, output_path=args.output, ledger_path=args.ledger
        ).run(examples)
    _ = sys.stdout.write(
        json.dumps(
            {"labeled": count, "output": str(args.output), "ledger": str(args.ledger)}
        )
        + "\n"
    )
    return 0


def main() -> int:
    """Run the labeling CLI."""
    return anyio.run(_run, _parse_args())


if __name__ == "__main__":
    sys.exit(main())

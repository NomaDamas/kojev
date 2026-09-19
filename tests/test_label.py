"""Deterministic tests for teacher labeling and restart-safe output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from kojev.label import LabelRunner, OutputLine, build_goldless_families
from kojev.schema import Example, Question, QuestionType
from kojev.teacher import TeacherClient

if TYPE_CHECKING:
    from pathlib import Path


class QuestionOutput(BaseModel):
    """Typed serialized question projection used by the test boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    meta: dict[str, str]


class LabeledOutput(BaseModel):
    """Typed serialized output projection used by the test boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    label_source: str
    questions: list[QuestionOutput]


def _response(request_id: str, content: str = "A1: 1\nA2: 0\nA3: 2") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": request_id,
            "choices": [{"message": {"content": content}, "logprobs": None}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        },
    )


def _example(*, gold: int | None = None, split: str = "train") -> Example:
    return Example(
        state="지문: 한국의 산림은 다양한 생태계를 이룬다.",
        source="test/source",
        split=split,
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="지문의 주제를 고르시오.",
                options=[
                    "과학",
                    "환경",
                    "경제",
                    "역사",
                    "문화",
                    "정치",
                    "사회",
                    "스포츠",
                ],
                gold=gold,
            ),
            Question(
                type=QuestionType.NOUL,
                instructions="이 지문은 수치/통계를 포함한다.",
                options=["아니오", "예"],
                gold=gold if gold in {0, 1} else None,
            ),
            Question(
                type=QuestionType.SCORE,
                instructions="지문의 난이도",
                options=["쉬움", "보통", "어려움"],
                gold=gold if gold in {0, 1, 2} else None,
            ),
        ],
    )


@pytest.mark.anyio
async def test_resume_dedupes_completed_request_ids_and_stamps_source(
    tmp_path: Path,
) -> None:
    # Given a provider that repeats an already-issued ID after restart
    ledger = tmp_path / "ledger.jsonl"
    output = tmp_path / "train.jsonl"
    completed = _example()
    transport = httpx.MockTransport(lambda request: _response("req-existing"))
    client = TeacherClient(
        api_key="test-key",
        ledger_path=ledger,
        http_client=httpx.AsyncClient(transport=transport),
    )
    runner = LabelRunner(client=client, output_path=output, ledger_path=ledger)

    # When the runner is invoked twice with the same provider request ID
    first = await runner.run([completed])
    second = await runner.run([completed])

    # Then the second invocation appends no duplicate and every question is stamped
    assert first == 1
    assert second == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    labeled = LabeledOutput.model_validate(records[0])
    assert labeled.label_source == "teacher:qwen3-vl-8b-instruct"
    assert all(
        question.meta["label_source"] == "teacher:qwen3-vl-8b-instruct"
        for question in labeled.questions
    )
    assert OutputLine.model_validate(records[0]).request_id == "req-existing"
    assert len(records) == 1
    await client.aclose()


def test_refuses_families_with_dataset_gold() -> None:
    # Given a family whose questions already contain dataset gold
    gold_family = _example(gold=1)

    # When goldless families are selected directly
    selected = build_goldless_families([gold_family])

    # Then no existing gold family is sent to the teacher
    assert selected == []


@pytest.mark.anyio
async def test_ledger_is_append_only_and_request_ids_are_unique(tmp_path: Path) -> None:
    # Given an empty ledger and two distinct goldless states
    ledger = tmp_path / "ledger.jsonl"
    output = tmp_path / "train.jsonl"
    examples = [
        _example(),
        _example().model_copy(update={"state": "지문: 한강은 서울을 흐른다."}),
    ]
    request_count = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return _response(f"req-{request_count}")

    client = TeacherClient(
        api_key="test-key",
        ledger_path=ledger,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    runner = LabelRunner(client=client, output_path=output, ledger_path=ledger)

    # When both states are labeled
    _ = await runner.run(examples)

    # Then the ledger has only appended usage records and no duplicate request IDs
    text = ledger.read_text()
    lines = text.splitlines()
    request_ids = [json.loads(line)["request_id"] for line in lines]
    assert len(lines) == 2
    assert len(request_ids) == len(set(request_ids))
    assert text == "".join(f"{line}\n" for line in lines)
    await client.aclose()


def test_gold_train_states_get_fresh_unlabeled_reading_questions() -> None:
    # Given a gold train state with existing dataset answers
    example = _example(gold=1)

    # When a separate reading-comprehension family is derived
    families = build_goldless_families([example], include_gold_train=True)

    # Then the original gold is not copied into the new questions
    assert len(families) == 1
    assert all(question.gold is None for question in families[0].questions)

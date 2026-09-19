"""End-to-end contract tests for the local System One endpoint."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from kojev.serve import app

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
else:
    warnings.filterwarnings("ignore", message=".*BlockingPortal alias is deprecated.*")
    TestClient = importlib.import_module("fastapi.testclient").TestClient


class AnswerBody(BaseModel):
    """Typed JSON answer used by assertions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    probabilities: list[float]
    confidence: float
    choice: int | None = None
    score: float | None = None
    noul: float | None = None


class ResponseBody(BaseModel):
    """Typed JSON success response used by assertions."""

    model: str
    answers: dict[str, AnswerBody]
    usage: dict[str, int]


class ErrorBody(BaseModel):
    """Typed JSON error response used by assertions."""

    error_type: str
    message: str


_ERROR_ADAPTER = TypeAdapter(dict[str, ErrorBody])


@pytest.fixture
def client() -> TestClient:
    """Return the local API client."""
    return TestClient(app)


def _request(question: dict[str, object]) -> dict[str, object]:
    return {
        "model": "tiny-random",
        "state": "배송이 빠르고 포장이 단단하다",
        "questions": question,
    }


def test_mixed_questions_return_typed_answers_and_normalized_probabilities(
    client: TestClient,
) -> None:
    # Given one request containing choice, score, and noul questions
    payload = _request(
        {
            "choice": {
                "type": "choice",
                "instructions": "평가는?",
                "criteria": ["나쁨", "좋음"],
            },
            "score": {
                "type": "score",
                "instructions": "만족도는?",
                "options": ["낮음", "높음"],
            },
            "noul": {
                "type": "noul",
                "instructions": "긍정적인가?",
                "criteria": {"아니오": "negative", "예": "positive"},
            },
        }
    )

    # When the local endpoint evaluates the request
    response = client.post("/v1/systemone", json=payload)

    # Then all typed answer fields and normalized probabilities are present
    assert response.status_code == 200
    body = ResponseBody.model_validate_json(response.content)
    assert set(body.answers) == {"choice", "score", "noul"}
    for answer in body.answers.values():
        assert sum(answer.probabilities) == pytest.approx(1.0)
        assert isinstance(answer.confidence, float)
    assert isinstance(body.answers["choice"].choice, int)
    assert isinstance(body.answers["score"].score, float)
    assert isinstance(body.answers["noul"].noul, float)
    assert isinstance(body.usage["input_tokens"], int)


def test_choice_with_three_hundred_options_returns_422(client: TestClient) -> None:
    # Given a choice question exceeding the schema's maximum cardinality
    payload = _request(
        {
            "too_many": {
                "type": "choice",
                "instructions": "선택",
                "options": [str(index) for index in range(300)],
            }
        }
    )

    # When the malformed request is submitted
    response = client.post("/v1/systemone", json=payload)

    # Then the endpoint returns a structured validation failure
    assert response.status_code == 422
    detail = _ERROR_ADAPTER.validate_json(response.content)["detail"]
    assert detail.error_type
    assert detail.message


def test_question_missing_type_returns_422(client: TestClient) -> None:
    # Given a question without its discriminating type
    payload = _request(
        {"broken": {"instructions": "평가는?", "options": ["나쁨", "좋음"]}}
    )

    # When the malformed request is submitted
    response = client.post("/v1/systemone", json=payload)

    # Then the endpoint returns a structured validation failure
    assert response.status_code == 422
    detail = _ERROR_ADAPTER.validate_json(response.content)["detail"]
    assert detail.error_type
    assert detail.message


def test_validation_error_body_contains_error_type_and_message(
    client: TestClient,
) -> None:
    # Given a question with an invalid option count
    payload = _request(
        {
            "broken": {
                "type": "noul",
                "instructions": "긍정적인가?",
                "options": ["예"],
            }
        }
    )

    # When the malformed request is submitted
    response = client.post("/v1/systemone", json=payload)

    # Then both machine-readable error fields are non-empty
    detail = _ERROR_ADAPTER.validate_json(response.content)["detail"]
    assert isinstance(detail.error_type, str)
    assert isinstance(detail.message, str)
    assert detail.error_type
    assert detail.message

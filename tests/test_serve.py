"""End-to-end contract tests for the local System One endpoint."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, ClassVar, cast

import pytest
import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

import kojev.serve as serve_module
from kojev.encoder import (
    CheckpointProvenance,
    KoJevModel,
    SpanCollator,
    Tokenizer,
)
from kojev.schema import Example, Question, QuestionType
from kojev.serve import app

if TYPE_CHECKING:
    from pathlib import Path

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


def _saved_checkpoint(directory: Path) -> tuple[Path, tuple[float, ...]]:
    """Save a tiny checkpoint and return its path plus the model's own output.

    The returned probabilities are what a correctly restored server must
    reproduce. Serving a checkpoint with a freshly initialised head, or with a
    tokenizer other than the saved one, yields different numbers while still
    returning a well-formed 200 response.
    """
    unk = "[UNK]"
    pad = "[PAD]"
    vocab = {unk: 0, pad: 1}
    # The option surfaces must be IN the vocabulary: if they both fell back to
    # [UNK] their pooled features would be identical, the grouped softmax would
    # return (0.5, 0.5) for any head, and the test could not discriminate.
    for token in ("배송", "빠르다", "좋음", "나쁨", "아니오", "예"):
        vocab[token] = len(vocab)
    backend = HFTokenizer(WordLevel(vocab, unk_token=unk))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=unk, pad_token=pad
    )
    backbone = ModernBertModel(
        ModernBertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=128,
            pad_token_id=1,
        )
    )
    collator = SpanCollator(cast("Tokenizer", cast("object", tokenizer)), max_length=64)
    _ = backbone.resize_token_embeddings(len(tokenizer))
    model = KoJevModel(backbone, head_hidden_size=16)
    # Force a DISTINCTLY non-uniform head. An untrained random head on this tiny
    # symmetric input returns about (0.5, 0.5) whichever weights it holds, so a
    # naive equality assertion could not tell the saved model apart from a
    # re-initialised one. Skewing the output layer makes the difference visible.
    with torch.no_grad():
        _ = model.scorer.output_layer.weight.fill_(0.0)
        _ = model.scorer.output_layer.weight[0, 0].fill_(60.0)
        _ = model.scorer.output_layer.bias.fill_(0.0)
    example = Example(
        state="배송 빠르다",
        source="synthetic",
        split="test",
        questions=[
            Question(
                type=QuestionType.NOUL,
                instructions="배송이 빠르다",
                options=["아니오", "예"],
            )
        ],
    )
    answers = model.decide(example, collator)
    model.save_checkpoint(
        directory,
        collator,
        CheckpointProvenance(
            temperature=1.0,
            model_name="synthetic/modernbert",
            seed=0,
            train_path="train.jsonl",
            val_path="val.jsonl",
        ),
    )
    probabilities = tuple(answers[0].probabilities or ())
    # Guard the guard: if the fixture is not skewed the test proves nothing.
    assert probabilities, "fixture produced no probabilities"
    # The comparison below uses abs=1e-6, so the fixture only has to be far
    # enough from uniform that a DIFFERENT head would move it by more than that.
    # Exactly (0.5, 0.5) would be head-independent and prove nothing.
    assert abs(probabilities[0] - 0.5) > 1e-3, (
        f"fixture is too uniform to discriminate: {probabilities}"
    )
    return directory, probabilities


def test_served_checkpoint_reproduces_the_saved_models_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A served checkpoint must be the SAVED model, not a re-initialised one.

    Regression guard: serve.py previously built the runtime with
    KoJevModel.from_pretrained(), which allocates a fresh random head, and paired
    it with a WhitespaceTokenizer instead of the checkpoint's tokenizer. Both
    faults are silent: the endpoint still answers 200 with normalised
    probabilities that are simply not the trained model's.
    """
    checkpoint, expected = _saved_checkpoint(tmp_path / "checkpoint")
    monkeypatch.setenv("KOJEV_CKPT", str(checkpoint))
    serve_module._load_runtime.cache_clear()  # pyright: ignore[reportPrivateUsage]

    runtime = serve_module._load_runtime()  # pyright: ignore[reportPrivateUsage]
    example = Example(
        state="배송 빠르다",
        source="synthetic",
        split="test",
        questions=[
            Question(
                type=QuestionType.NOUL,
                instructions="배송이 빠르다",
                options=["아니오", "예"],
            )
        ],
    )
    answers = runtime.model.decide(example, runtime.collator)
    served = tuple(answers[0].probabilities or ())

    serve_module._load_runtime.cache_clear()  # pyright: ignore[reportPrivateUsage]
    assert served == pytest.approx(expected, abs=1e-6)

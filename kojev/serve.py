"""Local Jev-compatible System One HTTP serving shim."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import ClassVar, Final, Self, assert_never

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from transformers import ModernBertConfig, ModernBertModel

from kojev.encoder import KoJevModel, SpanCollator, load_checkpoint
from kojev.schema import Example, Question, QuestionType

_CKPT_ENV: Final = "KOJEV_CKPT"
_TINY_RANDOM: Final = "tiny-random"
_MISSING_OPTIONS: Final = "question requires options or criteria"
_BOTH_OPTIONS: Final = "question accepts options or criteria, not both"
_QUESTION_OPTIONS_ERROR: Final = "question_options"


class WireQuestion(BaseModel):
    """Jev question payload converted to schema options at the boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: QuestionType
    instructions: str
    options: list[str] | None = None
    criteria: list[str] | dict[str, str] | None = None

    @model_validator(mode="after")
    def _require_options(self) -> Self:
        if self.options is None and self.criteria is None:
            raise PydanticCustomError(_QUESTION_OPTIONS_ERROR, _MISSING_OPTIONS)
        if self.options is not None and self.criteria is not None:
            raise PydanticCustomError(_QUESTION_OPTIONS_ERROR, _BOTH_OPTIONS)
        return self

    def option_values(self) -> list[str]:
        """Return criteria or options in the schema's ordered option format."""
        if self.options is not None:
            return self.options
        if isinstance(self.criteria, dict):
            return list(self.criteria)
        return self.criteria if self.criteria is not None else []


class WireRequest(BaseModel):
    """Jev request payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str
    state: str
    questions: dict[str, WireQuestion] = Field(min_length=1)


class WhitespaceTokenizer:
    """Small tokenizer used by the local checkpoint smoke bundle."""

    vocab: dict[str, int]

    def __init__(self) -> None:
        """Create an initially empty whitespace vocabulary."""
        self.vocab = {}

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        """Add marker tokens and return the number newly registered."""
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                added += 1
        return added

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode whitespace-separated tokens into stable integer ids."""
        del add_special_tokens
        return [
            self.vocab.setdefault(token, len(self.vocab) + 1) for token in text.split()
        ]


class Runtime:
    """Loaded model and collation state shared by requests."""

    model: KoJevModel
    collator: SpanCollator

    def __init__(self, model: KoJevModel, collator: SpanCollator) -> None:
        """Store the model in inference mode and its collator."""
        self.model = model.eval()
        self.collator = collator


def _tiny_runtime() -> Runtime:
    config = ModernBertConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    model = KoJevModel(ModernBertModel(config), head_hidden_size=32)
    return Runtime(model, SpanCollator(WhitespaceTokenizer(), max_length=128))


@functools.cache
def _load_runtime() -> Runtime:
    checkpoint = os.environ.get(_CKPT_ENV, _TINY_RANDOM)
    if checkpoint == _TINY_RANDOM:
        return _tiny_runtime()
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        msg = f"checkpoint does not exist: {checkpoint_path}"
        raise FileNotFoundError(msg)
    # load_checkpoint, NOT from_pretrained: from_pretrained allocates a FRESH
    # head at the default width and would pair it with a whitespace tokenizer,
    # so the endpoint would answer 200 with probabilities from an untrained head
    # over wrongly tokenized input. load_checkpoint restores the saved head
    # weights and the saved tokenizer together.
    model, collator, _ = load_checkpoint(checkpoint_path)
    return Runtime(model, collator)


app = FastAPI(title="KoJev local System One")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the compact Jev validation error shape."""
    return JSONResponse(
        status_code=422,
        content={"detail": {"error_type": "validation_error", "message": str(exc)}},
    )


@app.post("/v1/systemone", response_model=None)
def system_one(payload: WireRequest) -> JSONResponse | dict[str, object]:
    """Evaluate typed questions using the checkpoint configured for this process."""
    runtime = _load_runtime()
    try:
        questions = [
            Question(
                type=question.type,
                instructions=question.instructions,
                options=question.option_values(),
            )
            for question in payload.questions.values()
        ]
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": {"error_type": "schema_error", "message": str(exc)}},
        )
    example = Example(
        state=payload.state,
        questions=questions,
        source="serve",
        split="inference",
    )
    answers = runtime.model.decide(example, runtime.collator)
    encoded: dict[str, dict[str, object]] = {}
    for name, answer in zip(payload.questions, answers, strict=True):
        fields: dict[str, object] = {
            "probabilities": list(answer.probabilities),
            "confidence": answer.confidence,
        }
        match answer.type:
            case QuestionType.CHOICE:
                fields["choice"] = answer.choice
            case QuestionType.SCORE:
                fields["score"] = answer.score
            case QuestionType.NOUL:
                fields["noul"] = answer.noul
            case unreachable:
                assert_never(unreachable)
        encoded[name] = fields
    input_tokens = len(payload.state.split()) + sum(
        len(question.instructions.split()) + len(question.option_values())
        for question in payload.questions.values()
    )
    return {
        "model": payload.model,
        "answers": encoded,
        "usage": {"input_tokens": input_tokens},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8930)

"""Typed OpenRouter response and ledger boundary models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Provider message content."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    content: str = ""


class TopLogprob(BaseModel):
    """One provider-ranked alternative token."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobToken(BaseModel):
    """One token and its provider-ranked alternatives."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob]


class Logprobs(BaseModel):
    """Provider logprobs content list."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    content: list[LogprobToken]


class Choice(BaseModel):
    """Provider completion choice."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
    message: Message
    logprobs: Logprobs | None = None


class Usage(BaseModel):
    """Provider-reported token counts."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponsePayload(BaseModel):
    """Validated OpenRouter response payload."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    id: str
    choices: list[Choice] = Field(min_length=1)
    usage: Usage


class LedgerLine(BaseModel):
    """One append-only ledger record used for budget totals."""

    kind: str
    cost_usd: float = 0.0

"""Span-pooling encoder behavior on a tiny deterministic CPU fixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import pytest
import torch
from torch import Tensor, nn
from transformers import ModernBertConfig, ModernBertModel

from kojev.encoder import KoJevModel, SpanCollator
from kojev.schema import Example, Question, QuestionType

if TYPE_CHECKING:

    def _train_step(_loss: Tensor, _optimizer: torch.optim.Optimizer) -> None: ...
else:

    def _train_step(loss: Tensor, optimizer: torch.optim.Optimizer) -> None:
        loss.backward()
        optimizer.step()


class TinyTokenizer:
    """Whitespace tokenizer with a mutable special-token vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                added += 1
        return added

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [
            self.vocab.setdefault(token, len(self.vocab) + 1) for token in text.split()
        ]


@dataclass(frozen=True, slots=True)
class TinyConfig:
    """Minimal backbone configuration."""

    hidden_size: int


@dataclass(frozen=True, slots=True)
class TinyOutput:
    """Minimal transformer output."""

    last_hidden_state: Tensor


class TinyBackbone(nn.Module):
    """Small deterministic hidden-state provider for fast head tests."""

    config: TinyConfig
    embedding: nn.Embedding

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = TinyConfig(hidden_size)
        self.embedding = nn.Embedding(512, hidden_size)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> TinyOutput:
        del attention_mask
        return TinyOutput(self.embedding.forward(input_ids))


def _examples() -> list[Example]:
    return [
        Example(
            state="배송이 빠르다",
            questions=[
                Question(
                    type=QuestionType.CHOICE,
                    instructions="평가는?",
                    options=["나쁨", "좋음"],
                    gold=1,
                ),
                Question(
                    type=QuestionType.NOUL,
                    instructions="긍정적인가?",
                    options=["아니오", "예"],
                    gold=1,
                ),
            ],
            source="fixture",
            split="train",
        ),
        Example(
            state="포장이 단단하다",
            questions=[
                Question(
                    type=QuestionType.SCORE,
                    instructions="만족도는?",
                    options=["낮음", "높음"],
                    gold=1,
                )
            ],
            source="fixture",
            split="train",
        ),
    ]


def _tiny_model() -> tuple[KoJevModel, SpanCollator]:
    tokenizer = TinyTokenizer()
    collator = SpanCollator(tokenizer, max_length=128)
    model = KoJevModel(TinyBackbone())
    return model, collator


def test_collator_marks_spans_without_marker_hidden_state_readout() -> None:
    # Given examples and a tokenizer with task marker support
    model, collator = _tiny_model()
    batch = collator(_examples())

    # When the encoder consumes the packed sequence
    output = model.forward(batch)

    # Then every option receives one scalar and marker positions are not pooled
    assert batch.input_ids.ndim == 2
    assert output.logits.shape == (6,)
    assert all(start < end for _, start, end in batch.question_spans)
    assert all(start < end for _, start, end in batch.option_spans)
    assert all(
        marker_id not in batch.question_token_ids for marker_id in collator.marker_ids
    )


def test_grouped_softmax_sums_to_one_for_each_question() -> None:
    # Given a mixed batch with three question groups
    model, collator = _tiny_model()
    output = model.forward(collator(_examples()))

    # When probabilities are grouped by question
    grouped = output.probabilities

    # Then each group is a normalized categorical distribution
    assert grouped[0:2].sum().item() == pytest.approx(1.0)
    assert grouped[2:4].sum().item() == pytest.approx(1.0)
    assert grouped[4:6].sum().item() == pytest.approx(1.0)


def test_tiny_modernbert_overfits_sixteen_examples() -> None:
    # Given a random ModernBERT backbone and repeated labeled examples
    tokenizer = TinyTokenizer()
    collator = SpanCollator(tokenizer, max_length=128)
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
    model = KoJevModel(ModernBertModel(config))
    examples = _examples() * 8
    batch = collator(examples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    # When the head and tiny random backbone train on CPU
    losses: list[float] = []
    for _ in range(200):
        optimizer.zero_grad()
        output = model.forward(batch)
        assert output.loss is not None
        _train_step(output.loss, optimizer)
        losses.append(output.loss.item())
        if losses[-1] < 0.01:
            break

    # Then the mechanics can memorize the fixture within the required budget
    assert losses[-1] < 0.01


def test_decide_returns_schema_valid_mixed_answers() -> None:
    # Given a mixed choice, score, and noul example
    model, collator = _tiny_model()
    _ = model.eval()
    example = Example(
        state="배송이 빠르고 포장이 단단하다",
        questions=[
            _examples()[0].questions[0],
            _examples()[1].questions[0],
            _examples()[0].questions[1],
        ],
        source="fixture",
        split="test",
    )

    # When the model produces typed decisions
    answers = model.decide(example, collator)

    # Then every answer has the input cardinality and valid distribution shape
    assert len(answers) == 3
    assert answers[0].type is QuestionType.CHOICE
    assert len(answers[0].probabilities) == 2
    assert answers[1].type is QuestionType.SCORE
    assert answers[1].score is not None
    assert answers[2].type is QuestionType.NOUL
    assert answers[2].noul is not None
    assert sum(answers[2].probabilities) == pytest.approx(1.0)


def test_schema_rejects_three_hundred_choice_options_before_model() -> None:
    # Given a choice question above the schema's legal cardinality
    # When the input is constructed
    # Then schema validation rejects it before collation or model execution
    with pytest.raises(ValueError, match="255"):
        _ = Question(
            type=QuestionType.CHOICE,
            instructions="선택",
            options=[str(index) for index in range(300)],
        )

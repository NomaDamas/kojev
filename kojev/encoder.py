"""ModernBERT span-pooling head for typed Korean decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, assert_never, override

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from kojev.schema import Example, QuestionType, confidence

if TYPE_CHECKING:
    from collections.abc import Sequence


_MARKERS: Final = ("[STATE]", "[Q]", "[OPT]")
_DEFAULT_BACKBONE: Final = "skt/A.X-Encoder-base"


class Tokenizer(Protocol):
    """Tokenizer operations required by span packing."""

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        """Register additional marker tokens."""
        ...

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text without adding model boundary tokens."""
        ...


class BackboneOutput(Protocol):
    """Contextual token states returned by an encoder."""

    last_hidden_state: Tensor


class BackboneConfig(Protocol):
    """Backbone configuration fields used by the scoring head."""

    @property
    def hidden_size(self) -> int:
        """Return the hidden width."""
        ...


class Backbone(Protocol):
    """Encoder operations required by the decision model."""

    @property
    def config(self) -> BackboneConfig:
        """Return the encoder configuration."""
        ...

    def __call__(self, *, input_ids: Tensor, attention_mask: Tensor) -> BackboneOutput:
        """Return contextual hidden states for packed token ids."""
        ...


if TYPE_CHECKING:

    def _load_pretrained(_model_name: str) -> tuple[Backbone, Tokenizer]: ...
else:

    def _load_pretrained(model_name: str) -> tuple[Backbone, Tokenizer]:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        config = ModernBertConfig.from_pretrained(model_name)
        backbone = ModernBertModel.from_pretrained(model_name, config=config)
        _ = backbone.resize_token_embeddings(len(tokenizer))
        return backbone, tokenizer


@dataclass(frozen=True, slots=True)
class EncodingError(ValueError):
    """Input cannot be represented by the configured sequence budget."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the structured failure reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class SpanBatch:
    """Padded sequences plus flattened question and option spans."""

    input_ids: Tensor
    attention_mask: Tensor
    question_spans: tuple[tuple[int, int, int], ...]
    option_spans: tuple[tuple[int, int, int], ...]
    question_groups: tuple[tuple[int, ...], ...]
    question_types: tuple[QuestionType, ...]
    gold_indices: tuple[int | None, ...]
    question_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """Flattened option scores and grouped probabilities."""

    logits: Tensor
    probabilities: Tensor
    groups: tuple[tuple[int, ...], ...]
    loss: Tensor | None


@dataclass(frozen=True, slots=True)
class DecisionAnswer:
    """One schema-valid typed-decision readout."""

    type: QuestionType
    probabilities: tuple[float, ...]
    confidence: float
    choice: int | None
    score: float | None
    noul: float | None


class SpanCollator:
    """Pack state and all questions while truncating state tokens first."""

    tokenizer: Tokenizer
    max_length: int
    marker_ids: tuple[int, int, int]
    pad_token_id: int

    def __init__(self, tokenizer: Tokenizer, max_length: int = 4096) -> None:
        """Register markers and configure the sequence budget."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        _ = tokenizer.add_special_tokens({"additional_special_tokens": list(_MARKERS)})
        marker_ids = [
            tokenizer.encode(token, add_special_tokens=False)[0] for token in _MARKERS
        ]
        self.marker_ids = (marker_ids[0], marker_ids[1], marker_ids[2])
        pad_token = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad_token if isinstance(pad_token, int) else 0

    def __call__(self, examples: Sequence[Example]) -> SpanBatch:
        """Create one packed sequence per example and preserve text-only spans."""
        sequences: list[list[int]] = []
        question_spans: list[tuple[int, int, int]] = []
        option_spans: list[tuple[int, int, int]] = []
        groups: list[tuple[int, ...]] = []
        kinds: list[QuestionType] = []
        golds: list[int | None] = []
        question_token_ids: list[int] = []
        for sequence_index, example in enumerate(examples):
            suffix = [self.marker_ids[0]]
            local_questions: list[tuple[int, int]] = []
            local_options: list[tuple[int, int]] = []
            for question in example.questions:
                suffix.append(self.marker_ids[1])
                question_start = len(suffix)
                suffix.extend(self._text_ids(question.instructions, "question"))
                question_end = len(suffix)
                option_indices: list[int] = []
                for option in question.options:
                    suffix.append(self.marker_ids[2])
                    option_start = len(suffix)
                    suffix.extend(self._text_ids(option, "option"))
                    option_end = len(suffix)
                    local_questions.append((question_start, question_end))
                    local_options.append((option_start, option_end))
                    option_indices.append(len(option_spans) + len(local_options) - 1)
                groups.append(tuple(option_indices))
                kinds.append(question.type)
                golds.append(question.gold)
            state_budget = self.max_length - len(suffix)
            if state_budget < 0:
                reason = "questions and options exceed max_length"
                raise EncodingError(reason)
            state_ids = self.tokenizer.encode(example.state, add_special_tokens=False)
            state_ids = state_ids[-state_budget:] if state_budget else []
            offset = len(state_ids)
            sequence = [self.marker_ids[0], *state_ids, *suffix[1:]]
            sequences.append(sequence)
            question_spans.extend(
                (sequence_index, offset + start, offset + end)
                for start, end in local_questions
            )
            option_spans.extend(
                (sequence_index, offset + start, offset + end)
                for start, end in local_options
            )
            for start, end in local_questions:
                question_token_ids.extend(sequence[offset + start : offset + end])
        width = max(map(len, sequences), default=1)
        input_ids = torch.full(
            (len(sequences), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence)
            attention_mask[index, : len(sequence)] = 1
        return SpanBatch(
            input_ids,
            attention_mask,
            tuple(question_spans),
            tuple(option_spans),
            tuple(groups),
            tuple(kinds),
            tuple(golds),
            tuple(question_token_ids),
        )

    def _text_ids(self, text: str, span_name: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            reason = f"{span_name} text produced no tokens"
            raise EncodingError(reason)
        return token_ids


class SpanScorer(nn.Module):
    """Typed three-layer GELU MLP producing one option logit."""

    input_layer: nn.Linear
    hidden_layer: nn.Linear
    output_layer: nn.Linear
    activation: nn.GELU

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Create the scalar scoring MLP."""
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.GELU()

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Return one scalar logit per option feature row."""
        hidden = self.activation.forward(self.input_layer.forward(features))
        hidden = self.activation.forward(self.hidden_layer.forward(hidden))
        return self.output_layer.forward(hidden)


class KoJevModel(nn.Module):
    """Encoder with question/option mean pooling and grouped softmax."""

    backbone: Backbone
    scorer: SpanScorer

    def __init__(self, backbone: Backbone, head_hidden_size: int = 1024) -> None:
        """Create the span scorer over a transformer backbone."""
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.scorer = SpanScorer(hidden_size * 3, head_hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = _DEFAULT_BACKBONE,
        head_hidden_size: int = 1024,
    ) -> tuple[KoJevModel, SpanCollator]:
        """Load A.X-Encoder, register markers, and resize token embeddings."""
        backbone, tokenizer = _load_pretrained(model_name)
        return cls(backbone, head_hidden_size), SpanCollator(tokenizer)

    @override
    def forward(self, batch: SpanBatch) -> EncoderOutput:
        """Score pooled text spans without reading marker hidden states."""
        backbone_output: BackboneOutput = self.backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        hidden: Tensor = backbone_output.last_hidden_state
        features: list[Tensor] = []
        for question_span, option_span in zip(
            batch.question_spans, batch.option_spans, strict=True
        ):
            question_mean = self._mean_span(hidden, question_span)
            option_mean = self._mean_span(hidden, option_span)
            features.append(
                torch.cat((question_mean, option_mean, question_mean * option_mean))
            )
        logits = self.scorer.forward(torch.stack(features)).squeeze(-1)
        probabilities = torch.zeros_like(logits)
        losses: list[Tensor] = []
        for group, gold in zip(batch.question_groups, batch.gold_indices, strict=True):
            indices = torch.tensor(group, device=logits.device)
            group_logits = logits[indices]
            group_probabilities = torch.softmax(group_logits, dim=0)
            probabilities[indices] = group_probabilities
            if gold is not None:
                target = torch.tensor([gold], device=logits.device)
                one_hot = (
                    functional.one_hot(target, len(group)).squeeze(0).to(logits.dtype)
                )
                losses.append(
                    functional.cross_entropy(group_logits.unsqueeze(0), target)
                    + torch.square(group_probabilities - one_hot).sum()
                )
        loss = torch.stack(losses).mean() if losses else None
        return EncoderOutput(logits, probabilities, batch.question_groups, loss)

    @staticmethod
    def _mean_span(hidden: Tensor, span: tuple[int, int, int]) -> Tensor:
        sequence_index, start, end = span
        return hidden[sequence_index, start:end].mean(dim=0)

    @torch.no_grad()
    def decide(
        self, example: Example, collator: SpanCollator
    ) -> tuple[DecisionAnswer, ...]:
        """Return typed answers for choice, score, and noul questions."""
        output = self.forward(collator([example]))
        answers: list[DecisionAnswer] = []
        for question, group in zip(example.questions, output.groups, strict=True):
            probabilities = tuple(float(output.probabilities[index]) for index in group)
            predicted = max(range(len(group)), key=probabilities.__getitem__)
            common = (question.type, probabilities, confidence(probabilities))
            match question.type:
                case QuestionType.CHOICE:
                    answers.append(DecisionAnswer(*common, predicted, None, None))
                case QuestionType.SCORE:
                    expected = sum(
                        index * probability
                        for index, probability in enumerate(probabilities)
                    )
                    answers.append(DecisionAnswer(*common, None, expected, None))
                case QuestionType.NOUL:
                    answers.append(
                        DecisionAnswer(*common, None, None, probabilities[1])
                    )
                case unreachable:
                    assert_never(unreachable)
        return tuple(answers)

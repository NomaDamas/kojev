"""ModernBERT span-pooling head for typed Korean decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, assert_never, override
from typing import cast as _cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional
from transformers import AutoConfig, AutoModel, AutoTokenizer

from kojev.schema import Example, QuestionType, confidence

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


_MARKERS: Final = ("[STATE]", "[Q]", "[OPT]")
_DEFAULT_BACKBONE: Final = "skt/A.X-Encoder-base"
_HEAD_WEIGHTS: Final = "head.safetensors"
_KOJEV_CONFIG: Final = "kojev_config.json"
_PLAN_MAX_LENGTH: Final = 4096
QUESTIONS_EXCEED_MAX_LENGTH: Final = "questions and options exceed max_length"


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    """Calibration and provenance recorded alongside saved weights."""

    temperature: float
    model_name: str
    seed: int
    train_path: str
    val_path: str


class Tokenizer(Protocol):
    """Tokenizer operations required by span packing."""

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        """Register additional marker tokens."""
        ...

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text without adding model boundary tokens."""
        ...


class SizedTokenizer(Tokenizer, Protocol):
    """A tokenizer that also reports its vocabulary size.

    Only the pretrained loader needs this: span packing itself never asks how
    large the vocabulary is, so the offline fixtures stay free of ``__len__``.
    """

    def __len__(self) -> int:
        """Return the vocabulary size including registered special tokens."""
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


class ResizableBackbone(Backbone, Protocol):
    """A backbone whose token embedding table can be widened.

    Only the pretrained loader needs this. Offline fixtures allocate an
    embedding table directly and never resize, so they stay free of this method.
    """

    def resize_token_embeddings(self, new_num_tokens: int) -> object:
        """Resize the token embedding table to hold exactly this many ids."""
        ...


if TYPE_CHECKING:

    def _load_pretrained(
        _model_name: str,
    ) -> tuple[ResizableBackbone, SizedTokenizer]: ...
else:

    def _load_pretrained(
        model_name: str,
    ) -> tuple[ResizableBackbone, SizedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        config = AutoConfig.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, config=config)
        return backbone, tokenizer


if TYPE_CHECKING:

    def _save_backbone(_backbone: object, _directory: Path) -> None: ...

    def _save_tokenizer(_tokenizer: object, _directory: Path) -> None: ...

    def _load_local(
        _directory: Path,
    ) -> tuple[ResizableBackbone, SizedTokenizer]: ...
else:

    def _save_backbone(backbone: object, directory: Path) -> None:
        backbone.save_pretrained(str(directory))

    def _save_tokenizer(tokenizer: object, directory: Path) -> None:
        tokenizer.save_pretrained(str(directory))

    def _load_local(directory: Path) -> tuple[ResizableBackbone, SizedTokenizer]:
        # local_files_only keeps a restored checkpoint independent of the Hub:
        # the tokenizer, its registered markers, and the config all come from
        # the saved directory.
        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), use_fast=True, local_files_only=True
        )
        config = AutoConfig.from_pretrained(str(directory), local_files_only=True)
        backbone = AutoModel.from_pretrained(
            str(directory), config=config, local_files_only=True
        )
        return backbone, tokenizer


@dataclass(frozen=True, slots=True)
class EncodingError(ValueError):
    """Input cannot be represented by the configured sequence budget."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the structured failure reason."""
        return self.reason

    @classmethod
    def questions_exceed_budget(cls) -> EncodingError:
        """Questions and options overflow the collator window even with no state."""
        return cls(reason=QUESTIONS_EXCEED_MAX_LENGTH)


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
    question_weights: tuple[float, ...]

    def to(self, device: torch.device) -> SpanBatch:
        """Move packed tensors onto the encoder device."""
        return SpanBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            question_spans=self.question_spans,
            option_spans=self.option_spans,
            question_groups=self.question_groups,
            question_types=self.question_types,
            gold_indices=self.gold_indices,
            question_token_ids=self.question_token_ids,
            question_weights=self.question_weights,
        )


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """Flattened option scores and grouped probabilities."""

    logits: Tensor
    probabilities: Tensor
    groups: tuple[tuple[int, ...], ...]
    loss: Tensor | None
    question_losses: tuple[Tensor, ...] = ()
    question_weights: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionAnswer:
    """One schema-valid typed-decision readout."""

    type: QuestionType
    probabilities: tuple[float, ...]
    confidence: float
    choice: int | None
    score: float | None
    noul: float | None


def _collator_max_length(backbone: Backbone) -> int:
    """Cap packing to the backbone's native window, never above the plan budget."""
    native = getattr(backbone.config, "max_position_embeddings", None)
    if isinstance(native, int) and native > 0:
        return min(_PLAN_MAX_LENGTH, native)
    return _PLAN_MAX_LENGTH


class SpanCollator:
    """Pack state and all questions while truncating state tokens first."""

    tokenizer: Tokenizer
    max_length: int
    marker_ids: tuple[int, int, int]
    pad_token_id: int
    distill_weight: float

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 4096,
        distill_weight: float = 1.0,
    ) -> None:
        """Register markers and configure the sequence budget."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.distill_weight = distill_weight
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
        question_weights: list[float] = []
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
                label_source = question.meta.get("label_source")
                question_weights.append(
                    self.distill_weight
                    if isinstance(label_source, str)
                    and label_source.startswith("teacher:")
                    else 1.0
                )
            state_budget = self.max_length - len(suffix)
            if state_budget < 0:
                raise EncodingError.questions_exceed_budget()
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
            tuple(question_weights),
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
    hidden_width: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Create the scalar scoring MLP."""
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.GELU()
        # Recorded so a checkpoint can rebuild an identically shaped head.
        self.hidden_width = hidden_size

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Return one scalar logit per option feature row."""
        # A bfloat16 checkpoint emits reduced-precision hidden states while this
        # head holds fp32 weights. Autocast hides the difference during training
        # but evaluation runs without it (gpu01 job 13633), so align explicitly.
        aligned = features.to(self.input_layer.weight.dtype)
        hidden = self.activation.forward(self.input_layer.forward(aligned))
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
        distill_weight: float = 1.0,
    ) -> tuple[KoJevModel, SpanCollator]:
        """Load A.X-Encoder, register markers, and resize token embeddings."""
        backbone, tokenizer = _load_pretrained(model_name)
        # SpanCollator registers the three marker tokens, which lengthens the
        # tokenizer. The embedding table must be sized AFTER that, otherwise the
        # markers receive ids past the end of the table and every batch raises
        # IndexError inside tok_embeddings (gpu01 job 13625).
        collator = SpanCollator(
            tokenizer,
            max_length=_collator_max_length(backbone),
            distill_weight=distill_weight,
        )
        _ = backbone.resize_token_embeddings(len(tokenizer))
        return cls(backbone, head_hidden_size), collator

    def save_checkpoint(
        self,
        directory: Path,
        collator: SpanCollator,
        provenance: CheckpointProvenance,
    ) -> None:
        """Write a self-contained checkpoint that reloads without the Hub.

        The layout matches the release bundle the plan specifies: the backbone in
        Hugging Face format, the tokenizer (so the registered span markers
        survive), the head as ``head.safetensors``, and ``kojev_config.json``
        carrying pooling, calibration, and provenance.
        """
        directory.mkdir(parents=True, exist_ok=True)
        _save_backbone(self.backbone, directory)
        _save_tokenizer(collator.tokenizer, directory)
        save_file(self.scorer.state_dict(), str(directory / _HEAD_WEIGHTS))
        metadata: dict[str, object] = {
            "pooling": "span_mean",
            "markers": list(_MARKERS),
            "max_length": collator.max_length,
            "distill_weight": collator.distill_weight,
            "head_hidden_size": self.scorer.hidden_width,
            "temperature": provenance.temperature,
            "model_name": provenance.model_name,
            "seed": provenance.seed,
            "train_path": provenance.train_path,
            "val_path": provenance.val_path,
        }
        _ = (directory / _KOJEV_CONFIG).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

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
        weights: list[float] = []
        for group, gold, weight in zip(
            batch.question_groups,
            batch.gold_indices,
            batch.question_weights,
            strict=True,
        ):
            indices = torch.tensor(group, device=logits.device)
            group_logits = logits[indices]
            group_probabilities = torch.softmax(group_logits, dim=0)
            # Autocast promotes softmax to fp32 on CUDA while the logits stay in
            # reduced precision, so the write back has to match the destination
            # dtype explicitly (gpu01 job 13630).
            probabilities[indices] = group_probabilities.to(probabilities.dtype)
            if gold is not None:
                target = torch.tensor([gold], device=logits.device)
                one_hot = (
                    functional.one_hot(target, len(group)).squeeze(0).to(logits.dtype)
                )
                losses.append(
                    functional.cross_entropy(group_logits.unsqueeze(0), target)
                    + torch.square(group_probabilities - one_hot).sum()
                )
                weights.append(weight)
        loss = torch.stack(losses).mean() if losses else None
        if losses and any(weight != 1.0 for weight in weights):
            weight_tensor = torch.tensor(weights, device=logits.device)
            loss = torch.sum(torch.stack(losses) * weight_tensor) / weight_tensor.sum()
        return EncoderOutput(
            logits,
            probabilities,
            batch.question_groups,
            loss,
            tuple(losses),
            tuple(weights),
        )

    @staticmethod
    def _mean_span(hidden: Tensor, span: tuple[int, int, int]) -> Tensor:
        """Mean-pool one contiguous span of a packed sequence."""
        sequence_index, start, end = span
        return hidden[sequence_index, start:end].mean(dim=0)

    @torch.no_grad()
    def decide(
        self, example: Example, collator: SpanCollator
    ) -> tuple[DecisionAnswer, ...]:
        """Return typed answers for choice, score, and noul questions."""
        device = next(self.parameters()).device
        output = self.forward(collator([example]).to(device))
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


def _checked(
    metadata: dict[str, object],
    key: str,
    kinds: tuple[type, ...],
    default: float,
) -> float | int:
    """Return a config value, raising when it is present with the wrong type.

    An ABSENT key takes its default so older checkpoints stay loadable. A key
    that is PRESENT with the wrong type is corruption and must fail loudly:
    silently substituting a default hides a tampered or truncated bundle, and
    the model then loads with settings nobody chose.
    """
    if key not in metadata:
        return default
    value = metadata[key]
    # bool is a subclass of int and is never a valid numeric config value here.
    if isinstance(value, bool) or not isinstance(value, kinds):
        reason = (
            f"{_KOJEV_CONFIG} field {key!r} must be "
            f"{' or '.join(kind.__name__ for kind in kinds)}, got "
            f"{type(value).__name__}"
        )
        raise EncodingError(reason)
    return _cast("float | int", value)


def load_checkpoint(
    directory: Path,
) -> tuple[KoJevModel, SpanCollator, dict[str, object]]:
    """Restore a checkpoint written by :meth:`KoJevModel.save_checkpoint`.

    The returned model reproduces the saved model's outputs: the head weights
    are loaded from ``head.safetensors`` rather than left randomly initialised,
    which is the failure this function's test pins down.
    """
    metadata_path = directory / _KOJEV_CONFIG
    if not metadata_path.is_file():
        reason = f"checkpoint is missing {_KOJEV_CONFIG}: {directory}"
        raise EncodingError(reason)
    raw = _cast("object", json.loads(metadata_path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        reason = f"{_KOJEV_CONFIG} must contain an object: {directory}"
        raise EncodingError(reason)
    entries = _cast("dict[object, object]", raw)
    metadata: dict[str, object] = {str(key): value for key, value in entries.items()}

    # Validate BEFORE loading the backbone: a tampered bundle must be rejected
    # without paying for a model load, and the error must name the bad field.
    max_length = int(_checked(metadata, "max_length", (int,), 4096))
    distill_weight = float(_checked(metadata, "distill_weight", (int, float), 1.0))
    head_hidden_size = int(_checked(metadata, "head_hidden_size", (int,), 1024))
    _ = _checked(metadata, "temperature", (int, float), 1.0)

    backbone, tokenizer = _load_local(directory)
    # The saved tokenizer already carries the markers, so registering them again
    # is a no-op and the marker ids stay inside the saved embedding table.
    collator = SpanCollator(
        tokenizer,
        max_length=max_length,
        distill_weight=distill_weight,
    )
    # No resize on load: the checkpoint was saved AFTER the markers were
    # registered and the table widened, so the saved config already records the
    # correct vocabulary size and AutoModel allocates it.
    model = KoJevModel(backbone, head_hidden_size)
    _ = model.scorer.load_state_dict(load_file(str(directory / _HEAD_WEIGHTS)))
    _ = model.eval()
    return model, collator, metadata

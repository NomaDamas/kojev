"""Span-pooling encoder behavior on a tiny deterministic CPU fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor, nn
from transformers import (
    DebertaV2Config,
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
)

from kojev.encoder import (
    CheckpointProvenance,
    KoJevModel,
    SpanCollator,
    Tokenizer,
    load_checkpoint,
)
from kojev.schema import Example, Question, QuestionType

if TYPE_CHECKING:
    from pathlib import Path

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


class ExactVocabTokenizer:
    """Tokenizer whose length grows when marker tokens are registered.


    This mirrors a real Hugging Face tokenizer: ``len(tokenizer)`` counts the
    base vocabulary plus any added special tokens, so registering markers makes
    it longer than the checkpoint's configured ``vocab_size``.
    """

    base_size: int

    def __init__(self, base_size: int) -> None:
        self.base_size = base_size
        self.special: dict[str, int] = {}

    def __len__(self) -> int:
        return self.base_size + len(self.special)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.special:
                self.special[token] = self.base_size + len(self.special)
                added += 1
        return added

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.special:
            return [self.special[text]]
        return [(hash(token) % self.base_size) for token in text.split()]


class ResizingBackbone(nn.Module):
    """Backbone whose embedding table is exactly as wide as it was resized to."""

    config: TinyConfig
    embedding: nn.Embedding

    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = TinyConfig(hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        self.embedding = nn.Embedding(new_num_tokens, self.config.hidden_size)
        return self.embedding

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> TinyOutput:
        del attention_mask
        return TinyOutput(self.embedding.forward(input_ids))


def test_from_pretrained_sizes_embeddings_after_registering_marker_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marker ids must index inside the embedding table.

    Regression for gpu01 job 13625, which died after 20 minutes with
    ``IndexError: index out of range in self`` inside ``tok_embeddings``. The
    loader resized the embedding table to ``len(tokenizer)`` BEFORE
    ``SpanCollator`` registered its three marker tokens, so the markers received
    ids just past the end of the table. Every sequence begins with a marker, so
    every batch raised. The tiny fixtures never caught this because they
    construct ``KoJevModel`` directly and hand it an oversized embedding table.
    """
    base_size = 64
    tokenizer = ExactVocabTokenizer(base_size)

    def _fake_load(model_name: str) -> tuple[ResizingBackbone, ExactVocabTokenizer]:
        del model_name
        # The checkpoint's table is exactly as wide as its own vocabulary,
        # which is the real A.X-Encoder-base situation (50000 rows, 50000 ids).
        return ResizingBackbone(len(tokenizer)), tokenizer

    monkeypatch.setattr("kojev.encoder._load_pretrained", _fake_load)

    model, collator = KoJevModel.from_pretrained("stub-backbone")
    batch = collator(_examples())

    backbone = model.backbone
    assert isinstance(backbone, ResizingBackbone)
    rows: int = backbone.embedding.num_embeddings
    assert int(batch.input_ids.max()) < rows, (
        f"collated ids reach {int(batch.input_ids.max())} but the embedding table "
        f"has only {rows} rows"
    )
    # Reproduce the cluster failure directly: this raised IndexError before the fix.
    _ = model.forward(batch)


def test_deberta_v2_pretrained_path_uses_auto_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deberta-v2-shaped checkpoint must use the generic loader path."""
    config = DebertaV2Config(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    tokenizer = ExactVocabTokenizer(config.vocab_size)
    backbone = ResizingBackbone(config.vocab_size, hidden_size=config.hidden_size)
    calls: list[str] = []

    def fake_config(name: str) -> DebertaV2Config:
        calls.append(f"config:{name}")
        return config

    def fake_model(name: str, *, config: DebertaV2Config) -> ResizingBackbone:
        calls.append(f"model:{name}:{config.model_type}")
        return backbone

    def fake_tokenizer(name: str, *, use_fast: bool) -> ExactVocabTokenizer:
        assert use_fast is True
        calls.append(f"tokenizer:{name}")
        return tokenizer

    monkeypatch.setattr("kojev.encoder.AutoConfig.from_pretrained", fake_config)
    monkeypatch.setattr("kojev.encoder.AutoModel.from_pretrained", fake_model)
    monkeypatch.setattr("kojev.encoder.AutoTokenizer.from_pretrained", fake_tokenizer)

    model, _ = KoJevModel.from_pretrained("synthetic-deberta")

    assert calls == [
        "tokenizer:synthetic-deberta",
        "config:synthetic-deberta",
        "model:synthetic-deberta:deberta-v2",
    ]
    assert model.backbone is backbone


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


def test_checkpoint_round_trip_is_offline_and_preserves_outputs(
    tmp_path: Path,
) -> None:
    """Saved backbone, tokenizer, and head must reload without the Hub."""
    unknown_marker = "[UNK]"
    padding_marker = "[PAD]"
    vocab = {unknown_marker: 0, padding_marker: 1}
    for token in ("배송", "빠르다", "평가", "나쁨", "좋음"):
        vocab[token] = len(vocab)
    tokenizer_backend = HFTokenizer(WordLevel(vocab, unk_token=unknown_marker))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token=unknown_marker,
        pad_token=padding_marker,
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
    # A real Hugging Face tokenizer provides every behaviour SpanCollator uses,
    # but its signatures are wider than the narrow Tokenizer protocol, so the
    # structural check has to be relaxed through `object`. The round-trip
    # assertions below are what actually prove the adaptation is sound.
    protocol_tokenizer = cast("Tokenizer", cast("object", tokenizer))
    collator = SpanCollator(protocol_tokenizer, max_length=64)
    _ = backbone.resize_token_embeddings(len(tokenizer))
    model = KoJevModel(backbone, head_hidden_size=16)
    example = _examples()[0]
    original = model.forward(collator([example]))
    checkpoint = tmp_path / "checkpoint"

    model.save_checkpoint(
        checkpoint,
        collator,
        CheckpointProvenance(
            temperature=1.25,
            model_name="synthetic/modernbert",
            seed=7,
            train_path="gold.jsonl",
            val_path="val.jsonl",
        ),
    )

    loaded, loaded_collator, metadata = load_checkpoint(checkpoint)
    reloaded = loaded.forward(loaded_collator([example]))

    torch.testing.assert_close(reloaded.logits, original.logits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        reloaded.probabilities, original.probabilities, rtol=1e-6, atol=1e-6
    )
    # get_input_embeddings() is the architecture-agnostic accessor: ModernBERT
    # names its table `tok_embeddings` while DeBERTa uses `word_embeddings`, and
    # this loader must work for both.
    reloaded_backbone = loaded.backbone
    assert isinstance(reloaded_backbone, ModernBertModel)
    embedding_rows: int = reloaded_backbone.get_input_embeddings().num_embeddings
    assert max(loaded_collator.marker_ids) < embedding_rows
    assert metadata["temperature"] == 1.25
    assert (checkpoint / "head.safetensors").is_file()
    assert (checkpoint / "tokenizer_config.json").is_file()
    assert json.loads((checkpoint / "kojev_config.json").read_text())["seed"] == 7


def test_forward_survives_softmax_promoting_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped softmax must survive a softmax that promotes dtype.

    Regression for gpu01 job 13630. Under CUDA autocast, `logits` are BFloat16
    but `torch.softmax` sits on the fp32 autocast list and returns Float, so
    writing the group probabilities back raised:

        RuntimeError: Index put requires the source and destination dtypes
        match, got BFloat16 for the destination and Float for the source

    CPU autocast does NOT promote softmax, so the plain autocast path cannot
    reproduce this. The CUDA policy is emulated directly instead, which keeps the
    test faithful to the invariant and independent of the host's accelerator.
    """
    real_softmax = torch.softmax

    def _promoting_softmax(value: Tensor, dim: int) -> Tensor:
        return real_softmax(value, dim=dim).float()

    monkeypatch.setattr(torch, "softmax", _promoting_softmax)

    model, collator = _tiny_model()
    batch = collator(_examples())

    # Autocast also makes the head emit reduced precision, which is the other
    # half of the mismatch: BFloat16 destination, Float source.
    real_scorer_forward = model.scorer.forward

    def _half_precision_forward(features: Tensor) -> Tensor:
        return real_scorer_forward(features).bfloat16()

    monkeypatch.setattr(model.scorer, "forward", _half_precision_forward)

    output = model.forward(batch)

    assert torch.isfinite(output.probabilities).all()
    grouped = output.probabilities.float()
    for group in output.groups:
        total = float(grouped[torch.tensor(group)].sum())
        assert total == pytest.approx(1.0, abs=1e-2)

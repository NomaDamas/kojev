"""Gold-data SFT loop with calibration, evaluation, and divergence reporting."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NotRequired, TypedDict, Unpack, cast, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from kojev.augment import augment_example
from kojev.encoder import (
    CheckpointProvenance,
    EncoderOutput,
    KoJevModel,
    SpanBatch,
    SpanCollator,
)
from kojev.schema import Example, QuestionType, read_jsonl

_DEFAULT_AUGMENTATION: Final = 0.7
_EVALS_PER_EPOCH: Final = 4
_ECE_BINS: Final = 15

Batch = SpanBatch | tuple[Tensor, ...]
Collate = Callable[[Sequence[Example]], Batch]
Forward = Callable[[nn.Module, Batch], EncoderOutput]


class TrainReport(TypedDict):
    """Persisted training report contract."""

    args: dict[str, str | int | float | bool]
    data_counts: dict[str, int]
    loss_curve: list[float]
    wall_time: float
    peak_memory: int
    metrics: dict[str, dict[str, float | int]]
    temperature: float
    diverged: bool
    checkpoint: NotRequired[str]


@dataclass(slots=True)
class DivergenceWatch:
    """Track a rolling loss window; mutation is the detector's purpose."""

    window: int = 200
    rise_fraction: float = 0.25
    _values: deque[float] = field(init=False)
    diverged: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Allocate the rolling window after validating configuration."""
        self._values = deque(maxlen=self.window)

    def observe(self, value: float) -> bool:
        """Record one loss and return the sticky divergence state."""
        if not math.isfinite(value):
            self.diverged = True
            return True
        if len(self._values) == self.window:
            baseline = sum(self._values) / self.window
            if baseline > 0.0 and value > baseline * (1.0 + self.rise_fraction):
                self.diverged = True
        self._values.append(value)
        return self.diverged


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Immutable configuration for one training run."""

    train_path: Path
    val_path: Path
    out_dir: Path
    epochs: int = 1
    seed: int = 0
    limit: int | None = None
    batch_size: int = 4
    augmentation_probability: float = _DEFAULT_AUGMENTATION
    backbone_lr: float = 2e-5
    head_lr: float = 1e-3
    bf16: bool = False
    model_name: str = "skt/A.X-Encoder-base"
    distill_weight: float = 0.5


class RunTrainingKwargs(TypedDict):
    """Typed keyword boundary for the public training helper."""

    train_path: Path
    val_path: Path
    out_dir: Path
    epochs: NotRequired[int]
    seed: NotRequired[int]
    limit: NotRequired[int | None]
    model: NotRequired[nn.Module | None]
    collate_fn: NotRequired[Collate | None]
    forward_fn: NotRequired[Forward | None]
    batch_size: NotRequired[int]
    augmentation_probability: NotRequired[float]
    backbone_lr: NotRequired[float]
    head_lr: NotRequired[float]
    bf16: NotRequired[bool]
    model_name: NotRequired[str]
    distill_path: NotRequired[Path | None]
    distill_weight: NotRequired[float]


@dataclass(frozen=True, slots=True)
class _Runtime:
    """Resolved model and data adapters for one run."""

    model: nn.Module
    collate: Collate
    forward: Forward


@dataclass(frozen=True, slots=True)
class _TrainingLoop:
    """Dependencies and data consumed by the minibatch loop."""

    config: TrainConfig
    runtime: _Runtime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    train_examples: list[Example]
    val_examples: list[Example]
    device: torch.device


@dataclass(frozen=True, slots=True)
class _ReportContext:
    """Inputs required to calibrate and persist a finished run."""

    config: TrainConfig
    runtime: _Runtime
    train_examples: list[Example]
    val_examples: list[Example]
    losses: list[float]
    watch: DivergenceWatch
    started: float


def training_loss(output: EncoderOutput) -> Tensor:
    """Return the encoder's grouped cross-entropy plus Brier loss."""
    if output.loss is None:
        msg = "training examples must contain gold answers"
        raise ValueError(msg)
    if output.question_losses:
        if all(weight == 1.0 for weight in output.question_weights):
            return torch.stack(output.question_losses).mean()
        weights = torch.tensor(
            output.question_weights, device=output.question_losses[0].device
        )
        return torch.sum(torch.stack(output.question_losses) * weights) / weights.sum()
    return output.loss


def _is_distill_question(question: object) -> bool:
    """Return whether a question carries teacher-label provenance."""
    meta = getattr(question, "meta", None)
    if not isinstance(meta, dict):
        return False
    metadata = cast("dict[str, object]", meta)
    label_source = metadata.get("label_source")
    return isinstance(label_source, str) and label_source.startswith("teacher:")


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    """Fit one positive temperature by minimizing validation NLL."""
    best_temperature = 1.0
    best_loss = math.inf
    for step in range(200):
        temperature = 0.05 + step * (9.95 / 199)
        loss = functional.cross_entropy(logits.float() / temperature, labels)
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_temperature = temperature
    return best_temperature


def _pad_logits(rows: Sequence[Tensor]) -> Tensor:
    width = max((row.numel() for row in rows), default=2)
    padded = torch.full((len(rows), width), float("-inf"))
    for index, row in enumerate(rows):
        padded[index, : row.numel()] = row.detach().float()
    return padded


def _metrics(
    logits: Tensor, labels: Tensor, temperature: float
) -> dict[str, float | int]:
    probabilities = torch.softmax(logits / temperature, dim=1)
    predictions = probabilities.argmax(dim=1)
    count = int(labels.numel())
    accuracy = float((predictions == labels).float().mean().item())
    targets = functional.one_hot(labels, probabilities.shape[1]).to(probabilities.dtype)
    brier = float(torch.square(probabilities - targets).sum(dim=1).mean().item())
    confidence = probabilities.max(dim=1).values
    correct = (predictions == labels).float()
    ece = 0.0
    for index in range(_ECE_BINS):
        lower = index / _ECE_BINS
        upper = (index + 1) / _ECE_BINS
        mask = (confidence >= lower) & (
            confidence <= upper if index == _ECE_BINS - 1 else confidence < upper
        )
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].mean().item())
                - float(confidence[mask].mean().item())
            )
    return {"count": count, "accuracy": accuracy, "brier": brier, "ece": ece}


def _collect(
    model: nn.Module,
    examples: Sequence[Example],
    collate: Collate,
    forward: Forward,
) -> tuple[list[Tensor], list[int], list[QuestionType], list[str]]:
    rows: list[Tensor] = []
    labels: list[int] = []
    kinds: list[QuestionType] = []
    sources: list[str] = []
    _ = model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), 4):
            chunk = examples[start : start + 4]
            output = forward(model, collate(chunk))
            question_index = 0
            for example in chunk:
                for question in example.questions:
                    group = output.groups[question_index]
                    question_index += 1
                    if question.gold is not None:
                        rows.append(output.logits[list(group)])
                        labels.append(question.gold)
                        kinds.append(question.type)
                        sources.append(example.source)
    return rows, labels, kinds, sources


def _evaluate(
    model: nn.Module,
    examples: Sequence[Example],
    collate: Collate,
    forward: Forward,
    temperature: float,
) -> tuple[dict[str, dict[str, float | int]], Tensor, Tensor]:
    rows, labels, kinds, sources = _collect(model, examples, collate, forward)
    padded = _pad_logits(rows)
    targets = torch.tensor(labels, dtype=torch.long)
    metrics = {"overall": _metrics(padded, targets, temperature)}
    slices: dict[str, list[int]] = defaultdict(list)
    for index, (kind, source) in enumerate(zip(kinds, sources, strict=True)):
        slices[f"kind:{kind.value}"].append(index)
        slices[f"source:{source}"].append(index)
    for name, indices in sorted(slices.items()):
        metrics[name] = _metrics(padded[indices], targets[indices], temperature)
    return metrics, padded, targets


def select_device() -> torch.device:
    """Return the accelerator to train on, preferring CUDA when present.

    gpu01 jobs 13626 and 13627 both exhausted their wall clocks because the
    training loop autocast on "cpu" and never moved anything to the allocated
    GPU. Device choice is therefore explicit and unit-tested rather than implied.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Move a collated batch's tensors onto the training device."""
    if device.type == "cpu" or not isinstance(batch, SpanBatch):
        return batch
    return SpanBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        question_spans=batch.question_spans,
        option_spans=batch.option_spans,
        question_groups=batch.question_groups,
        question_types=batch.question_types,
        gold_indices=batch.gold_indices,
        question_token_ids=batch.question_token_ids,
        question_weights=batch.question_weights,
    )


def _optimizer(
    model: nn.Module, backbone_lr: float, head_lr: float
) -> torch.optim.AdamW:
    if isinstance(model, KoJevModel):
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("backbone.")
        ]
        head_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("backbone.")
        ]
        return torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": backbone_lr},
                {"params": head_parameters, "lr": head_lr},
            ]
        )
    return torch.optim.AdamW(model.parameters(), lr=head_lr)


def _encoder_forward(model: nn.Module, batch: Batch) -> EncoderOutput:
    """Call the typed encoder path used by the default CLI."""
    if not isinstance(model, KoJevModel):
        msg = "default forward requires KoJevModel"
        raise TypeError(msg)
    if not isinstance(batch, SpanBatch):
        msg = "default forward requires SpanBatch"
        raise TypeError(msg)
    return model.forward(batch)


def _resolve_runtime(
    model: nn.Module | None,
    collate: Collate | None,
    forward: Forward | None,
    model_name: str,
    distill_weight: float,
) -> _Runtime:
    """Resolve the default pretrained model or preserve injected test adapters."""
    if model is None or collate is None or forward is None:
        loaded_model, loaded_collator = KoJevModel.from_pretrained(
            model_name, distill_weight=distill_weight
        )
        return _Runtime(loaded_model, loaded_collator, _encoder_forward)
    return _Runtime(model, collate, forward)


def _make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the six-percent warmup and cosine decay schedule."""
    warmup_steps = max(1, math.ceil(total_steps * 0.06))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _train_epochs(
    loop: _TrainingLoop,
) -> tuple[list[float], dict[str, dict[str, float | int]], DivergenceWatch]:
    """Run minibatches, periodic validation, clipping, and divergence checks."""
    config = loop.config
    runtime = loop.runtime
    augment_rng = random.Random(config.seed)  # noqa: S311
    shuffled = list(loop.train_examples)
    steps_per_epoch = max(1, math.ceil(len(loop.train_examples) / config.batch_size))
    eval_interval = max(1, math.ceil(steps_per_epoch / _EVALS_PER_EPOCH))
    watch = DivergenceWatch()
    losses: list[float] = []
    metrics: dict[str, dict[str, float | int]] = {}
    step = 0
    for epoch in range(config.epochs):
        random.Random(config.seed + epoch).shuffle(shuffled)  # noqa: S311
        _ = runtime.model.train()
        for start in range(0, len(shuffled), config.batch_size):
            rows = [
                augment_example(row, augment_rng, config.augmentation_probability)
                for row in shuffled[start : start + config.batch_size]
            ]
            batch = _to_device(runtime.collate(rows), loop.device)
            loop.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                loop.device.type, dtype=torch.bfloat16, enabled=config.bf16
            ):
                loss = training_loss(runtime.forward(runtime.model, batch))
            loss.backward()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
            _ = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0)
            loop.optimizer.step()
            loop.scheduler.step()
            step += 1
            loss_value = float(loss.detach().float().item())
            losses.append(loss_value)
            if step % eval_interval == 0:
                metrics, _, _ = _evaluate(
                    runtime.model,
                    loop.val_examples,
                    runtime.collate,
                    runtime.forward,
                    1.0,
                )
                _ = runtime.model.train()
            if watch.observe(loss_value):
                break
        if watch.diverged:
            break
    return losses, metrics, watch


def _persist_report(context: _ReportContext) -> TrainReport:
    """Calibrate validation output and persist the complete run report."""
    config = context.config
    runtime = context.runtime
    metrics, logits, labels = _evaluate(
        runtime.model, context.val_examples, runtime.collate, runtime.forward, 1.0
    )
    temperature = fit_temperature(logits, labels)
    metrics, _, _ = _evaluate(
        runtime.model,
        context.val_examples,
        runtime.collate,
        runtime.forward,
        temperature,
    )
    _, peak_memory = tracemalloc.get_traced_memory()
    report: TrainReport = {
        "args": {
            "epochs": config.epochs,
            "seed": config.seed,
            "limit": -1 if config.limit is None else config.limit,
            "batch_size": config.batch_size,
            "augmentation_probability": config.augmentation_probability,
            "backbone_lr": config.backbone_lr,
            "head_lr": config.head_lr,
            "bf16": config.bf16,
            "model": config.model_name,
            "distill_weight": config.distill_weight,
        },
        "data_counts": {
            "train": len(context.train_examples),
            "val": len(context.val_examples),
            "train_questions_distill": sum(
                1
                for example in context.train_examples
                for question in example.questions
                if _is_distill_question(question)
            ),
            "val_questions_distill": sum(
                1
                for example in context.val_examples
                for question in example.questions
                if _is_distill_question(question)
            ),
        },
        "loss_curve": context.losses,
        "wall_time": time.perf_counter() - context.started,
        "peak_memory": peak_memory,
        "metrics": metrics,
        "temperature": temperature,
        "diverged": context.watch.diverged,
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.out_dir / "checkpoint"
    saved = _save_checkpoint_if_supported(
        runtime, checkpoint_dir, temperature=temperature, config=config
    )
    if saved:
        report["checkpoint"] = str(checkpoint_dir)
    _ = (config.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _save_checkpoint_if_supported(
    runtime: _Runtime,
    directory: Path,
    *,
    temperature: float,
    config: TrainConfig,
) -> bool:
    """Persist a loadable checkpoint when the runtime is a real KoJev model.

    Downstream tasks (RLCD initialisation, evaluation, serving, and the release
    bundle) all need weights on disk, not just report.json. Injected test
    adapters and the offline tiny mode are not savable, so they are skipped
    rather than forced through a Hugging Face serialisation path.
    """
    model = runtime.model
    collator = runtime.collate
    if not isinstance(model, KoJevModel) or not isinstance(collator, SpanCollator):
        return False
    model.save_checkpoint(
        directory,
        collator,
        CheckpointProvenance(
            temperature=temperature,
            model_name=config.model_name,
            seed=config.seed,
            train_path=str(config.train_path),
            val_path=str(config.val_path),
        ),
    )
    return True


def run_training(**kwargs: Unpack[RunTrainingKwargs]) -> TrainReport:
    """Train on JSONL, evaluate each quarter epoch, calibrate, and report."""
    started = time.perf_counter()
    tracemalloc.start()
    config = TrainConfig(
        train_path=kwargs["train_path"],
        val_path=kwargs["val_path"],
        out_dir=kwargs["out_dir"],
        epochs=kwargs.get("epochs", 1),
        seed=kwargs.get("seed", 0),
        limit=kwargs.get("limit"),
        batch_size=kwargs.get("batch_size", 4),
        augmentation_probability=kwargs.get(
            "augmentation_probability", _DEFAULT_AUGMENTATION
        ),
        backbone_lr=kwargs.get("backbone_lr", 2e-5),
        head_lr=kwargs.get("head_lr", 1e-3),
        bf16=kwargs.get("bf16", False),
        model_name=kwargs.get("model_name", "skt/A.X-Encoder-base"),
        distill_weight=kwargs.get("distill_weight", 0.5),
    )
    _ = torch.manual_seed(config.seed)  # pyright: ignore[reportUnknownMemberType]
    train_examples = read_jsonl(config.train_path)
    val_examples = read_jsonl(config.val_path)
    distill_path = kwargs.get("distill_path")
    if distill_path is not None:
        train_examples.extend(read_jsonl(distill_path))
    if config.limit is not None:
        train_examples = train_examples[: config.limit]
        val_examples = val_examples[: config.limit]
    runtime = _resolve_runtime(
        kwargs.get("model"),
        kwargs.get("collate_fn"),
        kwargs.get("forward_fn"),
        config.model_name,
        config.distill_weight,
    )
    steps = max(1, math.ceil(len(train_examples) / config.batch_size) * config.epochs)
    optimizer = _optimizer(runtime.model, config.backbone_lr, config.head_lr)
    device = select_device()
    # Move the model ONCE, before the optimizer steps over its parameters.
    _ = runtime.model.to(device)
    loop = _TrainingLoop(
        config,
        runtime,
        optimizer,
        _make_scheduler(optimizer, steps),
        train_examples,
        val_examples,
        device,
    )
    losses, _, watch = _train_epochs(loop)
    report = _persist_report(
        _ReportContext(
            config, runtime, train_examples, val_examples, losses, watch, started
        )
    )
    tracemalloc.stop()
    return report


@dataclass(frozen=True, slots=True)
class _TinyConfig:
    hidden_size: int = 16


@dataclass(frozen=True, slots=True)
class _TinyOutput:
    last_hidden_state: Tensor


class _TinyBackbone(nn.Module):
    """Offline random backbone used only by the explicit tiny CLI mode."""

    config: _TinyConfig
    embedding: nn.Embedding

    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.embedding = nn.Embedding(4096, self.config.hidden_size)

    @override
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> _TinyOutput:
        del attention_mask
        hidden: Tensor = self.embedding.forward(input_ids)
        return _TinyOutput(hidden)


class _TinyTokenizer:
    """Offline whitespace tokenizer used only by the explicit tiny CLI mode."""

    pad_token_id: int = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        before = len(self.vocab)
        for token in payload["additional_special_tokens"]:
            _ = self.vocab.setdefault(token, len(self.vocab) + 1)
        return len(self.vocab) - before

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [
            self.vocab.setdefault(token, len(self.vocab) + 1) for token in text.split()
        ]


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Typed command-line arguments."""

    train: Path
    val: Path
    limit: int | None
    epochs: int
    seed: int
    out: Path
    batch_size: int
    augmentation_probability: float
    bf16: bool
    tiny: bool
    model: str
    distill: Path | None
    distill_weight: float


def _parse_args() -> CliArgs:
    """Parse CLI flags into an explicitly typed immutable value."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--train", type=Path, default=Path("data/gold/train.jsonl"))
    _ = parser.add_argument("--val", type=Path, default=Path("data/gold/val.jsonl"))
    _ = parser.add_argument("--limit", type=int)
    _ = parser.add_argument("--epochs", type=int, default=1)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--batch-size", type=int, default=4)
    _ = parser.add_argument("--augmentation-probability", type=float, default=0.7)
    _ = parser.add_argument("--bf16", action="store_true")
    _ = parser.add_argument("--tiny", action="store_true")
    _ = parser.add_argument("--model", default="skt/A.X-Encoder-base")
    _ = parser.add_argument("--distill", type=Path)
    _ = parser.add_argument("--distill-weight", type=float, default=0.5)
    namespace = parser.parse_args(sys.argv[1:])
    return CliArgs(
        train=Path(namespace.train),  # pyright: ignore[reportAny]
        val=Path(namespace.val),  # pyright: ignore[reportAny]
        limit=namespace.limit,  # pyright: ignore[reportAny]
        epochs=namespace.epochs,  # pyright: ignore[reportAny]
        seed=namespace.seed,  # pyright: ignore[reportAny]
        out=Path(namespace.out),  # pyright: ignore[reportAny]
        batch_size=namespace.batch_size,  # pyright: ignore[reportAny]
        augmentation_probability=namespace.augmentation_probability,  # pyright: ignore[reportAny]
        bf16=namespace.bf16,  # pyright: ignore[reportAny]
        tiny=namespace.tiny,  # pyright: ignore[reportAny]
        model=namespace.model,  # pyright: ignore[reportAny]
        distill=namespace.distill,  # pyright: ignore[reportAny]
        distill_weight=namespace.distill_weight,  # pyright: ignore[reportAny]
    )


def main() -> None:
    """Run the local training CLI."""
    args = _parse_args()
    model: nn.Module | None = None
    collator: SpanCollator | None = None
    forward: Forward | None = None
    if args.tiny:
        model = KoJevModel(_TinyBackbone())
        collator = SpanCollator(
            _TinyTokenizer(), max_length=128, distill_weight=args.distill_weight
        )
        forward = _encoder_forward
    report = run_training(
        train_path=args.train,
        val_path=args.val,
        out_dir=args.out,
        epochs=args.epochs,
        seed=args.seed,
        limit=args.limit,
        model=model,
        collate_fn=collator,
        forward_fn=forward,
        batch_size=args.batch_size,
        augmentation_probability=args.augmentation_probability,
        bf16=args.bf16,
        model_name=args.model,
        distill_path=args.distill,
        distill_weight=args.distill_weight,
    )
    _ = report
    print(json.dumps({"report": str(args.out / "report.json")}, ensure_ascii=False))  # noqa: T201


if __name__ == "__main__":
    main()

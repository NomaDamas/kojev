"""Evaluation harness: gold/OOD tables, latency, and contamination checks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Final, override

from kojev.bench import DecisionModel, Metrics, calibration_metrics
from kojev.schema import Example, read_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from kojev.schema import JsonValue

_P95: Final = 0.95
_MIN_FOR_P95: Final = 2


class EvaluationError(RuntimeError):
    """Evaluation could not run against the requested inputs."""

    __slots__: tuple[str, ...] = ("reason",)

    def __init__(self, reason: str) -> None:
        """Record the structured reason."""
        super().__init__(reason)
        self.reason: str = reason

    @override
    def __str__(self) -> str:
        """Return the structured reason."""
        return self.reason

    @classmethod
    def missing_checkpoint(cls, path: Path) -> EvaluationError:
        """Reject a checkpoint path that does not exist."""
        return cls(f"checkpoint does not exist: {path}")

    @classmethod
    def invalid_checkpoint(cls, path: Path, detail: str) -> EvaluationError:
        """Reject a checkpoint path that exists but is unusable."""
        return cls(f"checkpoint is not usable: {path}: {detail}")

    @classmethod
    def missing_split(cls, path: Path) -> EvaluationError:
        """Reject an evaluation split file that does not exist."""
        return cls(f"evaluation split does not exist: {path}")

    @classmethod
    def contaminated(cls, name: str, overlap: int, sample: str) -> EvaluationError:
        """Reject an evaluation split that shares states with training."""
        detail = f"{overlap} state(s) also appear in train, e.g. {sample!r}"
        return cls(f"contamination in {name}: {detail}")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Inputs for one evaluation run."""

    checkpoint: Path
    train: Path
    splits: tuple[tuple[str, Path], ...]
    out: Path


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Wall-clock latency over evaluated states."""

    count: int
    p50_ms: float
    p95_ms: float

    def as_payload(self) -> dict[str, JsonValue]:
        """Render for the JSON report."""
        return {"count": self.count, "p50_ms": self.p50_ms, "p95_ms": self.p95_ms}


def resolve_checkpoint(path: Path) -> Path:
    """Validate a checkpoint path before any evaluation work begins.

    A missing or unusable checkpoint must fail loudly: an evaluation report
    full of zeros is worse than no report at all.

    Raises:
        EvaluationError: If the path is absent or is not a readable directory
            or file.
    """
    if not path.exists():
        raise EvaluationError.missing_checkpoint(path)
    if path.is_dir() and not any(path.iterdir()):
        raise EvaluationError.invalid_checkpoint(path, "directory is empty")
    if path.is_file() and path.stat().st_size == 0:
        raise EvaluationError.invalid_checkpoint(path, "file is empty")
    return path


def _states(examples: Sequence[Example]) -> set[str]:
    """Collect the distinct state strings of a split."""
    return {example.state for example in examples}


def assert_no_train_contamination(
    name: str,
    evaluated: Sequence[Example],
    train_states: set[str],
) -> None:
    """Fail loudly when an evaluated split shares any state with training.

    Raises:
        EvaluationError: If any evaluated state also appears in train.
    """
    overlap = sorted(_states(evaluated) & train_states)
    if overlap:
        raise EvaluationError.contaminated(name, len(overlap), overlap[0])


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return a nearest-rank percentile over already-sorted-able values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) < _MIN_FOR_P95:
        return ordered[0]
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _summarise_latency(samples: Sequence[float]) -> LatencySummary:
    """Summarise per-state latency samples in milliseconds."""
    if not samples:
        return LatencySummary(count=0, p50_ms=0.0, p95_ms=0.0)
    return LatencySummary(
        count=len(samples),
        p50_ms=float(median(samples)),
        p95_ms=_percentile(samples, _P95),
    )


def _metric_payload(metrics: Metrics, count: int) -> dict[str, JsonValue]:
    """Render metrics for the JSON report."""
    return {
        "count": count,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "brier": metrics.brier,
        "ece_15": metrics.ece_15,
    }


def _score_split(
    examples: Sequence[Example],
    model: DecisionModel,
) -> tuple[dict[str, JsonValue], list[float]]:
    """Run a model over one split, returning tables and latency samples."""
    golds: list[int] = []
    probabilities: list[tuple[float, ...]] = []
    by_kind: dict[str, tuple[list[int], list[tuple[float, ...]]]] = {}
    by_source: dict[str, tuple[list[int], list[tuple[float, ...]]]] = {}
    samples: list[float] = []

    for example in examples:
        questions = tuple(example.questions)
        started = time.perf_counter()
        vectors = model.decide(example.state, questions)
        samples.append((time.perf_counter() - started) * 1000.0)
        for question, vector in zip(questions, vectors, strict=True):
            gold = question.gold
            if gold is None:
                continue
            golds.append(gold)
            probabilities.append(vector)
            kind_bucket = by_kind.setdefault(str(question.type), ([], []))
            kind_bucket[0].append(gold)
            kind_bucket[1].append(vector)
            source_bucket = by_source.setdefault(example.source, ([], []))
            source_bucket[0].append(gold)
            source_bucket[1].append(vector)

    tables: dict[str, JsonValue] = {
        "overall": _metric_payload(
            calibration_metrics(tuple(golds), tuple(probabilities)), len(golds)
        ),
        "kinds": {
            kind: _metric_payload(calibration_metrics(tuple(g), tuple(p)), len(g))
            for kind, (g, p) in sorted(by_kind.items())
        },
        "sources": {
            source: _metric_payload(calibration_metrics(tuple(g), tuple(p)), len(g))
            for source, (g, p) in sorted(by_source.items())
        },
    }
    return tables, samples


def evaluate(config: EvaluationConfig, model: DecisionModel) -> dict[str, JsonValue]:
    """Evaluate a model over every configured split and write a JSON report.

    Raises:
        EvaluationError: If the checkpoint is unusable, a split file is
            missing, or an evaluated split is contaminated by train states.
    """
    checkpoint = resolve_checkpoint(config.checkpoint)
    if not config.train.is_file():
        raise EvaluationError.missing_split(config.train)
    train_states = _states(read_jsonl(config.train))

    splits: dict[str, JsonValue] = {}
    for name, path in config.splits:
        if not path.is_file():
            raise EvaluationError.missing_split(path)
        examples = read_jsonl(path)
        assert_no_train_contamination(name, examples, train_states)
        tables, samples = _score_split(examples, model)
        splits[name] = {
            "states": len(examples),
            "tables": tables,
            "latency": _summarise_latency(samples).as_payload(),
        }

    report: dict[str, JsonValue] = {
        "checkpoint": str(checkpoint),
        "train_states": len(train_states),
        "splits": splits,
        "contamination": "clean",
    }
    config.out.parent.mkdir(parents=True, exist_ok=True)
    _ = config.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

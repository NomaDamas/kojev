"""Evaluation harness: gold/OOD tables, latency, and contamination checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Final, override

from kojev.bench import DecisionModel, Metrics, calibration_metrics
from kojev.schema import Example, read_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence

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


if TYPE_CHECKING:

    def _load_decision_model(_checkpoint: Path) -> DecisionModel: ...
else:
    from kojev.encoder import load_checkpoint

    def _load_decision_model(checkpoint: Path) -> DecisionModel:
        model, collator, _ = load_checkpoint(checkpoint)

        class _CheckpointModel:
            def decide(self, example: Example) -> object:
                return model.decide(example, collator)

        return _CheckpointModel()


class _EvalArgs(argparse.Namespace):
    """Typed mutable namespace populated by argparse."""

    # None rather than [] so the class attribute stays immutable while argparse's
    # append action still receives a list it may mutate.
    checkpoint: list[str] | None = None
    train: Path = Path("data/gold/train.jsonl")
    split: list[str] | None = None
    out: Path = Path("eval/report.json")
    results: Path = Path("eval/RESULTS.md")


def _parse_pairs(values: Sequence[str], flag: str) -> tuple[tuple[str, Path], ...]:
    """Parse repeated NAME=PATH options into ordered pairs."""
    pairs: list[tuple[str, Path]] = []
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or not raw:
            reason = f"{flag} expects NAME=PATH, got: {value}"
            raise EvaluationError(reason)
        pairs.append((name, Path(raw)))
    return tuple(pairs)


def _render_results(rows: Sequence[tuple[str, str, dict[str, JsonValue]]]) -> str:
    """Render one Markdown row per model and split, traceable to its report."""
    lines = [
        "# KoJev evaluation results",
        "",
        "| model | split | states | p50_ms | p95_ms | report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, report_path, payload in rows:
        splits = payload.get("splits")
        if not isinstance(splits, dict):
            continue
        for split_name, split_payload in splits.items():
            if not isinstance(split_payload, dict):
                continue
            latency = split_payload.get("latency")
            p50 = p95 = "n/a"
            if isinstance(latency, dict):
                p50 = str(latency.get("p50_ms", "n/a"))
                p95 = str(latency.get("p95_ms", "n/a"))
            states = str(split_payload.get("states", "n/a"))
            cells = (model_name, split_name, states, p50, p95, report_path)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> _EvalArgs:
    parser = argparse.ArgumentParser(
        prog="kojev.evaluate",
        description="Score named checkpoints over gold/OOD splits into RESULTS.md.",
    )
    _ = parser.add_argument("--checkpoint", action="append", required=True)
    _ = parser.add_argument("--train", type=Path, required=True)
    _ = parser.add_argument("--split", action="append", required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args(argv, namespace=_EvalArgs())


def main(argv: list[str] | None = None) -> int:
    """Evaluate every named checkpoint and emit RESULTS.md, or fail writing none.

    Every checkpoint is evaluated before anything is rendered, so a single
    unusable checkpoint cannot leave a partial results table on disk claiming to
    be a measurement.
    """
    args = _parse_args(argv)
    try:
        models = _parse_pairs(args.checkpoint or [], "--checkpoint")
        splits = _parse_pairs(args.split or [], "--split")
        rows: list[tuple[str, str, dict[str, JsonValue]]] = []
        for model_name, checkpoint in models:
            # Validate BEFORE loading: resolve_checkpoint raises the typed
            # EvaluationError, whereas a loader handed a missing directory would
            # raise an untyped error and escape this handler.
            _ = resolve_checkpoint(checkpoint)
            report_path = (
                args.out
                if len(models) == 1
                else args.out.with_name(
                    f"{args.out.stem}-{model_name}{args.out.suffix}"
                )
            )
            payload = evaluate(
                EvaluationConfig(
                    checkpoint=checkpoint,
                    train=args.train,
                    splits=splits,
                    out=report_path,
                ),
                _load_decision_model(checkpoint),
            )
            rows.append((model_name, str(report_path), payload))
    except EvaluationError as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1

    args.results.parent.mkdir(parents=True, exist_ok=True)
    _ = args.results.write_text(_render_results(rows), encoding="utf-8")
    print(json.dumps({"results": str(args.results), "models": len(rows)}))  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

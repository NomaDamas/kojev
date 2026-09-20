"""Summarise training run reports: median of healthy seeds, diverged flagged.

The plan's main-checkpoint rule is the median val-acc of non-diverged seed
runs. A diverged artefact with a spectacular accuracy must not win that
selection by being left in the table unflagged.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kojev.schema import JsonValue

_REPORT_NAME: Final = "report.json"


class ReportError(ValueError):
    """A runs directory or report.json failed structural validation.

    Plain exception subclass rather than a frozen dataclass: frozen dataclass
    exceptions reject ``__traceback__`` assignment.
    """

    def __init__(self, reason: str) -> None:
        """Record the structured reason."""
        super().__init__(reason)
        self.reason: str = reason

    @override
    def __str__(self) -> str:
        """Return the structured reason."""
        return self.reason

    @classmethod
    def missing_runs(cls, directory: Path) -> ReportError:
        """Reject a directory that contains no report.json files."""
        return cls(f"{directory}: no report.json files")

    @classmethod
    def unreadable(cls, path: Path) -> ReportError:
        """Reject a report that is not valid JSON."""
        return cls(f"{path}: report is not valid JSON")

    @classmethod
    def missing_field(cls, path: Path, field: str) -> ReportError:
        """Reject a report missing a field the median rule reads."""
        return cls(f"{path}: missing {field}")

    @classmethod
    def bad_field(cls, path: Path, field: str, value: object) -> ReportError:
        """Reject a report field present with the wrong type."""
        return cls(f"{path}: {field} has type {type(value).__name__}")

    @classmethod
    def no_healthy_runs(cls, directory: Path) -> ReportError:
        """Reject a median taken over a set that contains only diverged runs."""
        return cls(f"{directory}: no non-diverged runs to take a median over")

    @classmethod
    def missing_argument(cls, name: str) -> ReportError:
        """Reject a CLI namespace lacking an expected path argument."""
        return cls(f"missing path argument: {name}")


@dataclass(frozen=True, slots=True)
class RunRow:
    """One training run as the median rule sees it."""

    name: str
    seed: int
    model: str
    accuracy: float
    majority: float
    brier: float
    ece: float
    diverged: bool
    path: Path


def load_runs(directory: Path) -> tuple[RunRow, ...]:
    """Load every ``*/report.json`` under ``directory``, sorted by run name."""
    paths = sorted(directory.glob(f"*/{_REPORT_NAME}"))
    if not paths:
        raise ReportError.missing_runs(directory)
    return tuple(_load_row(path) for path in paths)


def median_accuracy(rows: Sequence[RunRow]) -> float:
    """Median overall accuracy of non-diverged runs.

    Diverged rows stay in the table so the failure is visible, but they are
    excluded from the number that selects the main checkpoint.
    """
    healthy = [row.accuracy for row in rows if not row.diverged]
    if not healthy:
        directory = rows[0].path.parent.parent if rows else Path()
        raise ReportError.no_healthy_runs(directory)
    return float(statistics.median(healthy))


def render_table(rows: Sequence[RunRow]) -> str:
    """Render a markdown table with diverged runs flagged in the last column."""
    header = "| run | seed | model | acc | majority | gap | brier | ece | status |"
    rule = "|---|---:|---|---:|---:|---:|---:|---:|---|"
    lines = [header, rule]
    for row in rows:
        status = "DIVERGED" if row.diverged else "ok"
        gap = row.accuracy - row.majority
        lines.append(
            "| "
            + " | ".join(
                (
                    row.name,
                    str(row.seed),
                    row.model,
                    f"{row.accuracy:.4f}",
                    f"{row.majority:.4f}",
                    f"{gap:+.4f}",
                    f"{row.brier:.4f}",
                    f"{row.ece:.4f}",
                    status,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _load_row(path: Path) -> RunRow:
    try:
        raw = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ReportError.unreadable(path) from error
    if not isinstance(raw, dict):
        raise ReportError.bad_field(path, "root", raw)
    args = raw.get("args")
    if not isinstance(args, dict):
        args = {}
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ReportError.missing_field(path, "metrics.overall.accuracy")
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise ReportError.missing_field(path, "metrics.overall.accuracy")
    diverged = raw.get("diverged")
    if not isinstance(diverged, bool):
        raise ReportError.bad_field(path, "diverged", diverged)
    return RunRow(
        name=path.parent.name,
        seed=_int_field(path, "args.seed", args.get("seed"), default=0),
        model=_str_field(path, "args.model", args.get("model"), default="unknown"),
        accuracy=_number_field(
            path, "metrics.overall.accuracy", overall.get("accuracy")
        ),
        majority=_number_field(
            path, "metrics.overall.majority", overall.get("majority"), default=0.0
        ),
        brier=_number_field(path, "metrics.overall.brier", overall.get("brier")),
        ece=_number_field(path, "metrics.overall.ece", overall.get("ece")),
        diverged=diverged,
        path=path,
    )


def _number_field(
    path: Path, field: str, value: object, default: float | None = None
) -> float:
    if value is None:
        if default is not None:
            return default
        raise ReportError.missing_field(path, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError.bad_field(path, field, value)
    return float(value)


def _int_field(path: Path, field: str, value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError.bad_field(path, field, value)
    return value


def _str_field(path: Path, field: str, value: object, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ReportError.bad_field(path, field, value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Print the run table and the median of non-diverged accuracies."""
    parser = argparse.ArgumentParser(prog="kojev.report")
    _ = parser.add_argument("--runs", type=Path, required=True)
    values = cast("dict[str, object]", vars(parser.parse_args(argv)))
    runs_dir = values.get("runs")
    if not isinstance(runs_dir, Path):
        missing = ReportError.missing_argument("runs")
        raise missing
    rows = load_runs(runs_dir)
    median = median_accuracy(rows)
    payload = {
        "median_accuracy": median,
        "healthy": sum(1 for row in rows if not row.diverged),
        "diverged": sum(1 for row in rows if row.diverged),
        "runs": len(rows),
    }
    _ = sys.stdout.write(render_table(rows) + "\n")
    _ = sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""Run-report table: median of healthy seeds, diverged runs flagged and excluded."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kojev.report import ReportError, load_runs, median_accuracy, render_table

if TYPE_CHECKING:
    from pathlib import Path


def _write_report(directory: Path, name: str, payload: dict[str, object]) -> Path:
    run_dir = directory / name
    run_dir.mkdir(parents=True)
    path = run_dir / "report.json"
    _ = path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payload(
    *,
    seed: int,
    accuracy: float,
    brier: float,
    ece: float,
    diverged: bool,
) -> dict[str, object]:
    return {
        "args": {"seed": seed, "model": "skt/A.X-Encoder-base", "epochs": 1},
        "diverged": diverged,
        "metrics": {
            "overall": {
                "count": 10,
                "accuracy": accuracy,
                "majority": 0.5,
                "brier": brier,
                "ece": ece,
            }
        },
        "temperature": 1.0,
    }


def test_median_uses_the_middle_healthy_accuracy(tmp_path: Path) -> None:
    """Three healthy seeds: the median is the middle accuracy, not the mean."""
    _ = _write_report(
        tmp_path,
        "seed0",
        _payload(seed=0, accuracy=0.70, brier=0.2, ece=0.1, diverged=False),
    )
    _ = _write_report(
        tmp_path,
        "seed1",
        _payload(seed=1, accuracy=0.80, brier=0.2, ece=0.1, diverged=False),
    )
    _ = _write_report(
        tmp_path,
        "seed2",
        _payload(seed=2, accuracy=0.90, brier=0.2, ece=0.1, diverged=False),
    )

    rows = load_runs(tmp_path)
    assert median_accuracy(rows) == pytest.approx(0.80)


def test_diverged_run_is_flagged_and_excluded_from_the_median(
    tmp_path: Path,
) -> None:
    """A diverged smoke artefact must not become the reported main checkpoint."""
    _ = _write_report(
        tmp_path,
        "healthy-a",
        _payload(seed=0, accuracy=0.71, brier=0.2, ece=0.1, diverged=False),
    )
    _ = _write_report(
        tmp_path,
        "healthy-b",
        _payload(seed=1, accuracy=0.73, brier=0.2, ece=0.1, diverged=False),
    )
    _ = _write_report(
        tmp_path,
        "sft-smoke-gpu4",
        _payload(seed=2, accuracy=0.99, brier=0.01, ece=0.0, diverged=True),
    )

    rows = load_runs(tmp_path)
    table = render_table(rows)
    assert "DIVERGED" in table
    assert "sft-smoke-gpu4" in table
    # 0.99 would be the median if the diverged row were kept.
    assert median_accuracy(rows) == pytest.approx(0.72)


def test_empty_runs_directory_raises_a_typed_error(tmp_path: Path) -> None:
    """A directory with no report.json files is an error, not a blank table."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReportError, match=r"no report\.json"):
        _ = load_runs(tmp_path / "empty")


def test_report_missing_overall_accuracy_raises_a_typed_error(tmp_path: Path) -> None:
    """A truncated report must fail loudly rather than render a blank accuracy."""
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    _ = (run_dir / "report.json").write_text(
        json.dumps({"args": {"seed": 0}, "diverged": False, "metrics": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ReportError, match="accuracy"):
        _ = load_runs(tmp_path)

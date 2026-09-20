"""Deterministic CPU tests for the RLCD objective and gate."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from kojev.rlcd import (
    GateDecision,
    RLCDConfig,
    RLCDInputs,
    RunningMeanBaseline,
    main,
    rlcd_gate,
    rlcd_loss,
)

if TYPE_CHECKING:
    from pathlib import Path


def _read_report(path: Path) -> dict[str, object]:
    """Read a gate report as typed data rather than Any."""
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(raw, dict)
    entries = cast("dict[object, object]", raw)
    return {str(key): value for key, value in entries.items()}


def test_loss_is_differentiable_for_grouped_distributions() -> None:
    """Given model logits, the RLCD loss backpropagates finite gradients."""
    model: nn.Linear = nn.Linear(3, 4, bias=False)
    features = torch.tensor([[1.0, -0.5, 0.25], [0.5, 0.75, -1.0]])
    logits: torch.Tensor = features @ model.weight.transpose(0, 1)
    positive: torch.Tensor = torch.softmax(logits, dim=1)
    negative: torch.Tensor = torch.softmax(logits * 0.5, dim=1)

    loss = rlcd_loss(
        RLCDInputs(
            positive,
            negative,
            groups=((0, 1), (2, 3)),
            gold_indices=(0, 1),
        ),
        RunningMeanBaseline(),
    )
    torch.autograd.backward((loss,))

    assert math.isfinite(float(loss.detach()))
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert not torch.all(model.weight.grad == 0)


def test_running_baseline_reduces_fixed_stream_variance() -> None:
    """Given a fixed reward stream, centering by its running mean reduces variance."""
    rewards = (1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
    baseline = RunningMeanBaseline()
    adjusted = tuple(baseline.update(reward) for reward in rewards)

    assert torch.tensor(adjusted).var(unbiased=False) < torch.tensor(rewards).var(
        unbiased=False
    )


def test_gate_goes_only_when_delta_clears_threshold() -> None:
    """Given named gate thresholds, clear, weak, and invalid deltas are classified."""
    config = RLCDConfig(gate_delta_threshold=0.05)

    assert rlcd_gate(0.051, config) is GateDecision.GO
    assert rlcd_gate(0.05, config) is GateDecision.GO
    assert rlcd_gate(0.049, config) is GateDecision.NO_GO
    assert rlcd_gate(float("nan"), config) is GateDecision.NO_GO
    assert rlcd_gate(float("inf"), config) is GateDecision.NO_GO


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (torch.tensor([[0.5, 0.5]]), torch.tensor([[0.5, 0.5]])),
        (torch.tensor([[1.0 - 1e-7, 1e-7]]), torch.tensor([[1e-7, 1.0 - 1e-7]])),
    ],
)
def test_loss_is_finite_on_uniform_and_near_deterministic_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> None:
    """Given valid edge distributions, the loss remains finite."""
    loss = rlcd_loss(
        RLCDInputs(positive, negative, groups=((0, 1),), gold_indices=(0,)),
        RunningMeanBaseline(),
    )

    assert math.isfinite(float(loss))


def _metrics_file(path: Path, before: float, after: float) -> Path:
    """Write a held-out metrics pair the CLI can read without a GPU."""
    _ = path.write_text(
        json.dumps({"baseline_metric": before, "candidate_metric": after}),
        encoding="utf-8",
    )
    return path


def test_cli_writes_gate_verdict_and_delta_into_report(tmp_path: Path) -> None:
    """The CLI must persist the gate verdict and the delta it was computed from.

    Plan todo 13 requires one Slurm RLCD run whose report.json carries a GO or
    NO-GO verdict. The verdict must come from the already-tested rlcd_gate, not
    from a reimplementation, so a delta above the default 0.01 threshold has to
    produce GO and the recorded delta has to match the inputs exactly.
    """
    metrics = _metrics_file(tmp_path / "metrics.json", before=0.700, after=0.725)
    out_dir = tmp_path / "run"

    exit_code = main(
        [
            "--metrics",
            str(metrics),
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    report = _read_report(out_dir / "report.json")
    assert report["verdict"] == GateDecision.GO.value
    # 0.725 - 0.700 computed in float: assert the recorded delta, not a rounding.
    assert report["held_out_metric_delta"] == pytest.approx(0.725 - 0.700, abs=1e-12)
    assert report["baseline_metric"] == pytest.approx(0.700)
    assert report["candidate_metric"] == pytest.approx(0.725)
    assert report["gate_delta_threshold"] == pytest.approx(0.01)


def test_cli_reports_no_go_when_delta_misses_threshold(tmp_path: Path) -> None:
    """A delta below the threshold must be recorded as NO-GO, not suppressed."""
    metrics = _metrics_file(tmp_path / "metrics.json", before=0.700, after=0.705)
    out_dir = tmp_path / "run"

    exit_code = main(["--metrics", str(metrics), "--out", str(out_dir)])

    assert exit_code == 0
    report = _read_report(out_dir / "report.json")
    assert report["verdict"] == GateDecision.NO_GO.value


def test_cli_rejects_a_missing_metrics_file_without_writing_a_report(
    tmp_path: Path,
) -> None:
    """A missing input must fail loudly and leave no partial report behind."""
    out_dir = tmp_path / "run"

    exit_code = main(
        ["--metrics", str(tmp_path / "absent.json"), "--out", str(out_dir)]
    )

    assert exit_code != 0
    assert not (out_dir / "report.json").exists()

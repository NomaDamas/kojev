"""Expected-utility RLCD objective and the four-clause GO/NO-GO gate."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from kojev.rlcd import (
    GateDecision,
    GateTable,
    RLCDConfig,
    RLCDInputs,
    half_epoch_steps,
    main,
    rlcd_gate,
    rlcd_loss,
    run_rlcd_loop,
)

if TYPE_CHECKING:
    from pathlib import Path


def _read_report(path: Path) -> dict[str, object]:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(raw, dict)
    entries = cast("dict[object, object]", raw)
    return {str(key): value for key, value in entries.items()}


def _go_table(**changes: float) -> GateTable:
    """A table that clears every KEEP clause, with named overrides for failures."""
    values: dict[str, float] = {
        "ece_baseline": 0.080,
        "ece_candidate": 0.070,
        "selective_acc_baseline": 0.700,
        "selective_acc_candidate": 0.700,
        "in_domain_acc_baseline": 0.670,
        "in_domain_acc_candidate": 0.669,
        "brier_baseline": 0.380,
        "brier_candidate": 0.381,
        "ood_acc_baseline": 0.550,
        "ood_acc_candidate": 0.545,
    }
    values.update(changes)
    return GateTable(**values)


def test_gate_keeps_when_all_four_clauses_hold() -> None:
    """ECE improved 1pt, in-domain drop 0.1pt, Brier 0.001, OOD drop 0.5pt."""
    assert rlcd_gate(_go_table()) is GateDecision.GO


def test_gate_keeps_when_selective_accuracy_improves_instead_of_ece() -> None:
    """The quality clause is a disjunction: selective-acc +2pt also qualifies."""
    table = _go_table(ece_candidate=0.080, selective_acc_candidate=0.721)
    assert rlcd_gate(table) is GateDecision.GO


def test_acc_drop_clause_produces_no_go() -> None:
    """Named failure: in-domain accuracy dropping more than 0.5pt is NO-GO.

    Even a large ECE win must not override the acc-drop clause.
    """
    table = _go_table(ece_candidate=0.020, in_domain_acc_candidate=0.660)
    assert rlcd_gate(table) is GateDecision.NO_GO


def test_brier_drop_clause_produces_no_go() -> None:
    table = _go_table(brier_candidate=0.390)
    assert rlcd_gate(table) is GateDecision.NO_GO


def test_ood_acc_drop_clause_produces_no_go() -> None:
    table = _go_table(ood_acc_candidate=0.530)
    assert rlcd_gate(table) is GateDecision.NO_GO


def test_quality_clause_requires_ece_or_selective_gain() -> None:
    """Neither ECE nor selective accuracy improved: NO-GO even if acc holds."""
    table = _go_table(ece_candidate=0.080, selective_acc_candidate=0.700)
    assert rlcd_gate(table) is GateDecision.NO_GO


def test_non_finite_metrics_are_no_go() -> None:
    table = _go_table(ece_candidate=float("nan"))
    assert rlcd_gate(table) is GateDecision.NO_GO


def test_expected_utility_loss_is_pathwise_differentiable() -> None:
    """Gradients must flow through p_gold, not through a detached REINFORCE reward.

    The plan forbids REINFORCE-on-gold: it is redundant with CE in expectation.
    A linear map from features to logits has to receive a finite, nonzero grad.
    """
    model: nn.Linear = nn.Linear(3, 2, bias=False)
    features = torch.tensor([[1.0, -0.5, 0.25], [0.5, 0.75, -1.0]])
    logits = features @ model.weight.transpose(0, 1)
    probabilities = torch.softmax(logits, dim=1)
    frozen = probabilities.detach()

    loss = rlcd_loss(
        RLCDInputs(
            probabilities=(probabilities[0], probabilities[1]),
            gold_indices=(0, 1),
            frozen_probabilities=(frozen[0], frozen[1]),
        ),
        RLCDConfig(lambda_cal=0.5, beta=0.1),
    )
    torch.autograd.backward((loss,))

    assert math.isfinite(float(loss.detach()))
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert not torch.all(model.weight.grad == 0)


def test_utility_term_is_minus_one_when_acting_on_a_certain_gold() -> None:
    """s≈1 and p_gold=1 ⇒ expected utility U_ok=+1 ⇒ L_util=-1 (no cal, no KL)."""
    peaked = torch.tensor([1.0 - 1e-7, 1e-7], requires_grad=True)
    loss = rlcd_loss(
        RLCDInputs(
            probabilities=(peaked,),
            gold_indices=(0,),
            frozen_probabilities=(peaked.detach(),),
        ),
        RLCDConfig(lambda_cal=0.0, beta=0.0),
    )
    # sigmoid((1-0.6)/0.05)=sigmoid(8)≈0.99966, so L_util is just shy of -U_ok.
    assert float(loss.detach()) == pytest.approx(-1.0, abs=2e-3)


def test_utility_term_is_plus_four_when_acting_on_a_certain_error() -> None:
    """s≈1 and p_gold=0 ⇒ expected utility U_err=-4 ⇒ L_util=+4."""
    peaked = torch.tensor([1e-7, 1.0 - 1e-7], requires_grad=True)
    loss = rlcd_loss(
        RLCDInputs(
            probabilities=(peaked,),
            gold_indices=(0,),
            frozen_probabilities=(peaked.detach(),),
        ),
        RLCDConfig(lambda_cal=0.0, beta=0.0),
    )
    assert float(loss.detach()) == pytest.approx(4.0, abs=2e-3)


def _metrics_file(path: Path, table: GateTable) -> Path:
    payload = {
        "baseline": {
            "ece": table.ece_baseline,
            "selective_acc": table.selective_acc_baseline,
            "in_domain_acc": table.in_domain_acc_baseline,
            "brier": table.brier_baseline,
            "ood_acc": table.ood_acc_baseline,
        },
        "candidate": {
            "ece": table.ece_candidate,
            "selective_acc": table.selective_acc_candidate,
            "in_domain_acc": table.in_domain_acc_candidate,
            "brier": table.brier_candidate,
            "ood_acc": table.ood_acc_candidate,
        },
    }
    _ = path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_writes_go_verdict_from_a_four_clause_table(tmp_path: Path) -> None:
    metrics = _metrics_file(tmp_path / "metrics.json", _go_table())
    out_dir = tmp_path / "run"
    exit_code = main(["gate", "--metrics", str(metrics), "--out", str(out_dir)])
    assert exit_code == 0
    report = _read_report(out_dir / "report.json")
    assert report["verdict"] == GateDecision.GO.value
    assert report["in_domain_acc_drop"] == pytest.approx(0.001)


def test_cli_reports_no_go_on_acc_drop_and_names_the_clause(tmp_path: Path) -> None:
    table = _go_table(ece_candidate=0.020, in_domain_acc_candidate=0.660)
    metrics = _metrics_file(tmp_path / "metrics.json", table)
    out_dir = tmp_path / "run"
    exit_code = main(["gate", "--metrics", str(metrics), "--out", str(out_dir)])
    assert exit_code == 0
    report = _read_report(out_dir / "report.json")
    assert report["verdict"] == GateDecision.NO_GO.value
    assert report["failed_clauses"] == ["in_domain_acc_drop"]


def test_cli_rejects_a_missing_metrics_file_without_writing_a_report(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    exit_code = main(
        ["gate", "--metrics", str(tmp_path / "absent.json"), "--out", str(out_dir)]
    )
    assert exit_code != 0
    assert not (out_dir / "report.json").exists()


def test_loop_stops_after_two_consecutive_brier_rises() -> None:
    """Early stop is two rises in a row, not two rises anywhere in the run."""
    briers = iter((0.30, 0.31, 0.32, 0.40, 0.50))
    steps: list[int] = []

    def train_step(step: int) -> None:
        steps.append(step)

    result = run_rlcd_loop(
        n_steps=10,
        eval_every=1,
        early_stop_rises=2,
        train_step=train_step,
        measure_val_brier=lambda: next(briers),
    )
    assert result["stopped_early"] is True
    assert result["steps"] == 3
    assert result["brier_history"] == [0.30, 0.31, 0.32]


def test_loop_resets_the_rise_counter_after_a_brier_drop() -> None:
    briers = iter((0.30, 0.31, 0.29, 0.30, 0.31))
    result = run_rlcd_loop(
        n_steps=10,
        eval_every=1,
        early_stop_rises=2,
        train_step=lambda _step: None,
        measure_val_brier=lambda: next(briers),
    )
    assert result["stopped_early"] is True
    assert result["steps"] == 5
    assert result["brier_history"] == [0.30, 0.31, 0.29, 0.30, 0.31]


def test_half_epoch_is_at_least_one_step_and_not_a_full_pass() -> None:
    """0.5 epoch over 8 batches is 4 steps; an empty loader is zero."""
    assert half_epoch_steps(n_batches=8) == 4
    assert half_epoch_steps(n_batches=1) == 1
    assert half_epoch_steps(n_batches=0) == 0


def test_train_cli_rejects_a_missing_checkpoint_without_writing_a_report(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    exit_code = main(
        [
            "train",
            "--checkpoint",
            str(tmp_path / "absent-ckpt"),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--val",
            str(tmp_path / "val.jsonl"),
            "--ood",
            str(tmp_path / "ood.jsonl"),
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code != 0
    assert not (out_dir / "report.json").exists()

"""Expected-decision-utility RLCD stage and the four-clause GO/NO-GO gate.

The model already emits an explicit distribution, so this stage optimises
expected decision utility pathwise (no sampling). REINFORCE-on-gold is
forbidden here: against a gold label it is redundant with cross-entropy in
expectation.

KEEP the RLCD checkpoint only when all four clauses hold on val (never test
or KoBEST): ECE improves by >=0.005 OR selective accuracy@0.6 improves by
>=2pt, AND in-domain acc drop <=0.5pt, AND Brier degradation <=0.005, AND
OOD acc drop <=1pt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import torch
from torch import Tensor

_DEFAULT_PROBABILITY_FLOOR: Final = 1e-8
_SOFT_BIN_TEMPERATURE: Final = 0.05


class GateDecision(StrEnum):
    """Closed outcomes for the held-out RLCD quality gate."""

    GO = "GO"
    NO_GO = "NO-GO"


@dataclass(frozen=True, slots=True)
class RLCDConfig:
    """Immutable configuration for the utility objective and KEEP clauses."""

    probability_floor: float = _DEFAULT_PROBABILITY_FLOOR
    tau: float = 0.6
    kappa: float = 0.05
    u_ok: float = 1.0
    u_err: float = -4.0
    u_esc: float = 0.0
    lambda_cal: float = 0.5
    beta: float = 0.1
    n_bins: int = 15
    ece_improve_min: float = 0.005
    selective_acc_improve_min: float = 0.02
    in_domain_acc_drop_max: float = 0.005
    brier_drop_max: float = 0.005
    ood_acc_drop_max: float = 0.01


_DEFAULT_CONFIG: Final = RLCDConfig()


@dataclass(frozen=True, slots=True)
class GateTable:
    """Paired SFT/RLCD val metrics the four KEEP clauses read."""

    ece_baseline: float
    ece_candidate: float
    selective_acc_baseline: float
    selective_acc_candidate: float
    in_domain_acc_baseline: float
    in_domain_acc_candidate: float
    brier_baseline: float
    brier_candidate: float
    ood_acc_baseline: float
    ood_acc_candidate: float


@dataclass(frozen=True, slots=True)
class RLCDInputs:
    """Per-question policy distributions, gold options, and frozen SFT anchors."""

    probabilities: tuple[Tensor, ...]
    gold_indices: tuple[int, ...]
    frozen_probabilities: tuple[Tensor, ...]


def gate_clause_failures(
    table: GateTable, config: RLCDConfig = _DEFAULT_CONFIG
) -> tuple[str, ...]:
    """Return the KEEP clauses that failed, in a stable order."""
    values = (
        table.ece_baseline,
        table.ece_candidate,
        table.selective_acc_baseline,
        table.selective_acc_candidate,
        table.in_domain_acc_baseline,
        table.in_domain_acc_candidate,
        table.brier_baseline,
        table.brier_candidate,
        table.ood_acc_baseline,
        table.ood_acc_candidate,
    )
    if not all(math.isfinite(value) for value in values):
        return ("non_finite",)
    ece_improve = table.ece_baseline - table.ece_candidate
    selective_improve = table.selective_acc_candidate - table.selective_acc_baseline
    in_domain_drop = table.in_domain_acc_baseline - table.in_domain_acc_candidate
    brier_drop = table.brier_candidate - table.brier_baseline
    ood_drop = table.ood_acc_baseline - table.ood_acc_candidate
    failed: list[str] = []
    quality = (
        ece_improve >= config.ece_improve_min
        or selective_improve >= config.selective_acc_improve_min
    )
    if not quality:
        failed.append("quality")
    if in_domain_drop > config.in_domain_acc_drop_max:
        failed.append("in_domain_acc_drop")
    if brier_drop > config.brier_drop_max:
        failed.append("brier_drop")
    if ood_drop > config.ood_acc_drop_max:
        failed.append("ood_acc_drop")
    return tuple(failed)


def rlcd_gate(table: GateTable, config: RLCDConfig = _DEFAULT_CONFIG) -> GateDecision:
    """Return GO only when every KEEP clause holds on the val table."""
    if gate_clause_failures(table, config):
        return GateDecision.NO_GO
    return GateDecision.GO


def rlcd_loss(inputs: RLCDInputs, config: RLCDConfig = _DEFAULT_CONFIG) -> Tensor:
    """Return mean L_util + lambda_cal * softECE + beta * mean KL(p || p_sft).

    ``s = sigmoid((max_prob - tau) / kappa)`` is the soft act/escalate gate.
    Expected utility is ``s * (p_gold * U_ok + (1-p_gold) * U_err) + (1-s) * U_esc``.
    Gradients flow through ``p_gold`` and ``max_prob``; nothing is detached into
    a REINFORCE baseline.
    """
    if len(inputs.probabilities) != len(inputs.gold_indices):
        msg = "probabilities and gold_indices must be the same length"
        raise ValueError(msg)
    if len(inputs.probabilities) != len(inputs.frozen_probabilities):
        msg = "probabilities and frozen_probabilities must be the same length"
        raise ValueError(msg)
    utilities: list[Tensor] = []
    kls: list[Tensor] = []
    confidences: list[Tensor] = []
    correct: list[Tensor] = []  # per-question 0/1, detached argmax
    for probabilities, gold_index, frozen in zip(
        inputs.probabilities,
        inputs.gold_indices,
        inputs.frozen_probabilities,
        strict=True,
    ):
        policy = probabilities.clamp_min(config.probability_floor)
        policy = policy / policy.sum()
        max_prob = policy.max()
        p_gold = policy[gold_index]
        gate = torch.sigmoid((max_prob - config.tau) / config.kappa)
        expected = (
            gate * (p_gold * config.u_ok + (1.0 - p_gold) * config.u_err)
            + (1.0 - gate) * config.u_esc
        )
        utilities.append(-expected)
        anchor = frozen.clamp_min(config.probability_floor)
        anchor = anchor / anchor.sum()
        kls.append((policy * (policy.log() - anchor.log())).sum())
        confidences.append(max_prob)
        predicted = torch.argmax(policy.detach())
        correct.append((predicted == gold_index).to(dtype=policy.dtype))
    util = torch.stack(utilities).mean()
    kl = torch.stack(kls).mean()
    calibration = _soft_ece(
        torch.stack(confidences),
        torch.stack(correct),
        n_bins=config.n_bins,
    )
    return util + config.lambda_cal * calibration + config.beta * kl


def _soft_ece(confidences: Tensor, correct: Tensor, n_bins: int) -> Tensor:
    """Karandikar-style soft-binned ECE; bin membership is a softmax kernel."""
    if confidences.numel() == 0:
        return confidences.new_zeros(())
    start = 1.0 / (2.0 * n_bins)
    centers = torch.linspace(
        start, 1.0 - start, n_bins, device=confidences.device, dtype=confidences.dtype
    )
    membership = torch.softmax(
        -(confidences.unsqueeze(1) - centers).abs() / _SOFT_BIN_TEMPERATURE,
        dim=1,
    )
    weights = membership.sum(dim=0)
    denom = weights.clamp_min(_DEFAULT_PROBABILITY_FLOOR)
    bin_acc = (membership * correct.unsqueeze(1)).sum(dim=0) / denom
    bin_conf = (membership * confidences.unsqueeze(1)).sum(dim=0) / denom
    mass = weights / weights.sum().clamp_min(_DEFAULT_PROBABILITY_FLOOR)
    return (mass * (bin_acc - bin_conf).abs()).sum()


class GateInputError(RuntimeError):
    """A gate run cannot proceed because its declared inputs are unusable."""

    @classmethod
    def missing_metrics(cls, path: Path) -> GateInputError:
        """Reject a held-out metrics path that does not exist."""
        return cls(f"held-out metrics file does not exist: {path}")

    @classmethod
    def malformed_metrics(cls, path: Path, detail: str) -> GateInputError:
        """Reject a metrics file that exists but cannot supply the four clauses."""
        return cls(f"held-out metrics file is unusable: {path}: {detail}")


class _GateArgs(argparse.Namespace):
    """Typed mutable namespace populated by argparse."""

    metrics: Path = Path("runs/rlcd/metrics.json")
    out: Path = Path("runs/rlcd")


def _parse_args(argv: list[str] | None) -> _GateArgs:
    parser = argparse.ArgumentParser(
        prog="kojev.rlcd",
        description="Score an RLCD stage against the four-clause GO/NO-GO gate.",
    )
    _ = parser.add_argument("--metrics", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv, namespace=_GateArgs())


_SIDE_KEYS: Final = (
    "ece",
    "selective_acc",
    "in_domain_acc",
    "brier",
    "ood_acc",
)


def _read_side(
    entries: dict[object, object], side: str, path: Path
) -> dict[str, float]:
    raw = entries.get(side)
    if not isinstance(raw, dict):
        raise GateInputError.malformed_metrics(path, f"{side} must be an object")
    side_entries = cast("dict[object, object]", raw)
    values: dict[str, float] = {}
    for key in _SIDE_KEYS:
        value = side_entries.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GateInputError.malformed_metrics(
                path, f"{side}.{key} must be a number"
            )
        values[key] = float(value)
    return values


def _read_gate_table(path: Path) -> GateTable:
    """Read the paired val metrics the KEEP clauses consume."""
    if not path.is_file():
        raise GateInputError.missing_metrics(path)
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise GateInputError.malformed_metrics(path, "expected a JSON object")
    entries = cast("dict[object, object]", raw)
    baseline = _read_side(entries, "baseline", path)
    candidate = _read_side(entries, "candidate", path)
    return GateTable(
        ece_baseline=baseline["ece"],
        ece_candidate=candidate["ece"],
        selective_acc_baseline=baseline["selective_acc"],
        selective_acc_candidate=candidate["selective_acc"],
        in_domain_acc_baseline=baseline["in_domain_acc"],
        in_domain_acc_candidate=candidate["in_domain_acc"],
        brier_baseline=baseline["brier"],
        brier_candidate=candidate["brier"],
        ood_acc_baseline=baseline["ood_acc"],
        ood_acc_candidate=candidate["ood_acc"],
    )


def _deltas(table: GateTable) -> dict[str, float]:
    return {
        "ece_improve": table.ece_baseline - table.ece_candidate,
        "selective_acc_improve": (
            table.selective_acc_candidate - table.selective_acc_baseline
        ),
        "in_domain_acc_drop": table.in_domain_acc_baseline
        - table.in_domain_acc_candidate,
        "brier_drop": table.brier_candidate - table.brier_baseline,
        "ood_acc_drop": table.ood_acc_baseline - table.ood_acc_candidate,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a GO/NO-GO report for one RLCD stage, or fail without writing one.

    The verdict comes from :func:`rlcd_gate` on the same :class:`GateTable` the
    unit tests pin, so the CLI cannot drift from the four KEEP clauses. Inputs
    are validated before the output directory is created.
    """
    args = _parse_args(argv)
    try:
        table = _read_gate_table(args.metrics)
    except GateInputError as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1

    verdict = rlcd_gate(table)
    failed = gate_clause_failures(table)
    report: dict[str, object] = {
        "verdict": verdict.value,
        "failed_clauses": list(failed),
        "metrics_path": str(args.metrics),
        **_deltas(table),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _ = (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict.value, "failed_clauses": list(failed)}))  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

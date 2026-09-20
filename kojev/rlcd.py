"""REINFORCE-style contrastive-distribution objective and evaluation gate."""

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
_DEFAULT_GATE_DELTA_THRESHOLD: Final = 0.01


class GateDecision(StrEnum):
    """Closed outcomes for the held-out RLCD quality gate."""

    GO = "GO"
    NO_GO = "NO-GO"


@dataclass(frozen=True, slots=True)
class RLCDConfig:
    """Immutable configuration for the local RLCD objective and gate."""

    probability_floor: float = _DEFAULT_PROBABILITY_FLOOR
    gate_delta_threshold: float = _DEFAULT_GATE_DELTA_THRESHOLD


_DEFAULT_CONFIG: Final = RLCDConfig()


@dataclass(slots=True)
class RunningMeanBaseline:
    """Maintain an online reward mean; mutation is the estimator's purpose."""

    _mean: float = 0.0
    _count: int = 0

    @property
    def value(self) -> float:
        """Return the mean of rewards observed before the next update."""
        return self._mean

    def update(self, reward: float) -> float:
        """Incorporate one reward and return its centered residual."""
        self._count += 1
        self._mean += (reward - self._mean) / self._count
        return reward - self._mean


@dataclass(frozen=True, slots=True)
class RLCDInputs:
    """Grouped positive/negative probabilities and gold option positions."""

    positive_probabilities: Tensor
    negative_probabilities: Tensor
    groups: tuple[tuple[int, ...], ...]
    gold_indices: tuple[int, ...]


def _gold_probability(probabilities: Tensor, row: int, option: int) -> Tensor:
    if probabilities.ndim == 1:
        return probabilities[option]
    return probabilities[row, option]


def rlcd_loss(
    inputs: RLCDInputs,
    baseline: RunningMeanBaseline,
    config: RLCDConfig = _DEFAULT_CONFIG,
) -> Tensor:
    """Return a baseline-centered policy loss over gold option log-ratios.

    The positive distribution supplies the policy log-probability. The detached
    gold log-ratio against the negative context is its contrastive reward, so
    gradients flow through the positive policy while the reward remains a
    REINFORCE learning signal rather than a pathwise shortcut.
    """
    policy_log_probabilities: list[Tensor] = []
    rewards: list[Tensor] = []
    for row, (group, gold_index) in enumerate(
        zip(inputs.groups, inputs.gold_indices, strict=True)
    ):
        option_index = group[gold_index]
        positive_gold = _gold_probability(
            inputs.positive_probabilities, row, option_index
        ).clamp_min(config.probability_floor)
        negative_gold = _gold_probability(
            inputs.negative_probabilities, row, option_index
        ).clamp_min(config.probability_floor)
        positive_log_probability = positive_gold.log()
        policy_log_probabilities.append(positive_log_probability)
        rewards.append((positive_log_probability - negative_gold.log()).detach())

    reward_tensor = torch.stack(rewards)
    prior_mean = baseline.value
    advantages = reward_tensor - prior_mean
    for reward in reward_tensor:
        _ = baseline.update(float(reward))
    policy_tensor = torch.stack(policy_log_probabilities)
    return -(advantages * policy_tensor).mean()


def rlcd_gate(held_out_metric_delta: float, config: RLCDConfig) -> GateDecision:
    """Return GO only for a finite held-out delta meeting the named threshold."""
    if not math.isfinite(held_out_metric_delta):
        return GateDecision.NO_GO
    if held_out_metric_delta >= config.gate_delta_threshold:
        return GateDecision.GO
    return GateDecision.NO_GO


class GateInputError(RuntimeError):
    """A gate run cannot proceed because its declared inputs are unusable."""

    @classmethod
    def missing_metrics(cls, path: Path) -> GateInputError:
        """Reject a held-out metrics path that does not exist."""
        return cls(f"held-out metrics file does not exist: {path}")

    @classmethod
    def malformed_metrics(cls, path: Path, detail: str) -> GateInputError:
        """Reject a metrics file that exists but cannot supply both metrics."""
        return cls(f"held-out metrics file is unusable: {path}: {detail}")


class _GateArgs(argparse.Namespace):
    """Typed mutable namespace populated by argparse."""

    metrics: Path = Path("runs/rlcd/metrics.json")
    out: Path = Path("runs/rlcd")
    gate_delta_threshold: float = _DEFAULT_GATE_DELTA_THRESHOLD


def _parse_args(argv: list[str] | None) -> _GateArgs:
    parser = argparse.ArgumentParser(
        prog="kojev.rlcd",
        description="Score an RLCD stage against the held-out GO/NO-GO gate.",
    )
    _ = parser.add_argument("--metrics", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument(
        "--gate-delta-threshold",
        type=float,
        default=_DEFAULT_GATE_DELTA_THRESHOLD,
    )
    return parser.parse_args(argv, namespace=_GateArgs())


def _read_metric_pair(path: Path) -> tuple[float, float]:
    """Read the baseline and candidate held-out metrics from disk."""
    if not path.is_file():
        raise GateInputError.missing_metrics(path)
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise GateInputError.malformed_metrics(path, "expected a JSON object")
    entries = cast("dict[object, object]", raw)
    values: list[float] = []
    for key in ("baseline_metric", "candidate_metric"):
        value = entries.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GateInputError.malformed_metrics(path, f"{key} must be a number")
        values.append(float(value))
    return values[0], values[1]


def main(argv: list[str] | None = None) -> int:
    """Write a GO/NO-GO report for one RLCD stage, or fail without writing one.

    The verdict comes from :func:`rlcd_gate`, the same function the unit tests
    pin, so the CLI cannot drift from the gate semantics. Inputs are validated
    before the output directory is created, which keeps a failed run from
    leaving a half-written report behind.
    """
    args = _parse_args(argv)
    try:
        baseline_metric, candidate_metric = _read_metric_pair(args.metrics)
    except GateInputError as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1

    delta = candidate_metric - baseline_metric
    config = RLCDConfig(gate_delta_threshold=args.gate_delta_threshold)
    verdict = rlcd_gate(delta, config)
    report: dict[str, object] = {
        "verdict": verdict.value,
        "baseline_metric": baseline_metric,
        "candidate_metric": candidate_metric,
        "held_out_metric_delta": delta,
        "gate_delta_threshold": config.gate_delta_threshold,
        "metrics_path": str(args.metrics),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _ = (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict.value, "delta": delta}))  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

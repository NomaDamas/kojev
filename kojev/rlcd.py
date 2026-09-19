"""REINFORCE-style contrastive-distribution objective and evaluation gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

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

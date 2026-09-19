"""Deterministic CPU tests for the RLCD objective and gate."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kojev.rlcd import (
    GateDecision,
    RLCDConfig,
    RLCDInputs,
    RunningMeanBaseline,
    rlcd_gate,
    rlcd_loss,
)


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

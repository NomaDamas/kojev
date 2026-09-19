"""Deterministic CPU tests for the local SFT training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
from pydantic import TypeAdapter
from torch import nn

from kojev.encoder import EncoderOutput
from kojev.schema import Example, Question, QuestionType, write_jsonl
from kojev.train import (
    Batch,
    DivergenceWatch,
    TrainReport,
    fit_temperature,
    run_training,
    training_loss,
)


def _example(split: str = "train") -> Example:
    return Example(
        state="배송이 빠르다",
        questions=[
            Question(
                type=QuestionType.CHOICE,
                instructions="평가는?",
                options=["나쁨", "좋음"],
                gold=1,
            )
        ],
        source="fixture",
        split=split,
    )


def test_training_loss_matches_grouped_cross_entropy_plus_brier() -> None:
    # Given one grouped question with known logits and gold index
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    probabilities = torch.softmax(logits, dim=0)
    expected = -torch.log(probabilities[1]) + torch.sum(
        (probabilities - torch.tensor([0.0, 1.0])) ** 2
    )
    output = EncoderOutput(logits, probabilities, ((0, 1),), expected)

    # When the encoder output is converted to the training loss
    loss = training_loss(output)

    # Then the loss is the encoder's grouped CE+Brier value
    assert loss.item() == pytest.approx(expected.item())


def test_divergence_watch_triggers_on_rise_and_nan() -> None:
    # Given a 200-step baseline followed by a >25% rise
    watch = DivergenceWatch(window=200, rise_fraction=0.25)
    for value in [1.0] * 200:
        _ = watch.observe(value)

    # When the next loss rises sharply, then a NaN is observed
    assert watch.observe(1.3) is True
    assert watch.diverged is True
    assert watch.observe(float("nan")) is True
    assert watch.diverged is True


def test_divergence_watch_does_not_trigger_on_healthy_decrease() -> None:
    # Given a steadily decreasing training stream
    watch = DivergenceWatch(window=200, rise_fraction=0.25)

    # When all observed values improve
    for value in [2.0 - index * 0.005 for index in range(220)]:
        assert watch.observe(value) is False

    # Then divergence remains clear
    assert watch.diverged is False


def test_fit_temperature_increases_temperature_for_overconfident_logits() -> None:
    # Given deliberately overconfident logits whose labels are often wrong
    logits = torch.tensor([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    labels = torch.tensor([1, 0, 1, 0])

    # When temperature is fitted on validation logits
    temperature = fit_temperature(logits, labels)

    # Then calibration softens the overconfident predictions
    assert temperature > 1.0


def test_report_contains_required_keys(tmp_path: Path) -> None:
    # Given tiny local JSONL train and validation data and a CPU linear model
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, [_example()] * 2)
    write_jsonl(val_path, [_example("val")])
    model = nn.Linear(1, 2)

    # When the injectable tiny training entry point runs
    report = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=tmp_path / "run",
        epochs=1,
        seed=4,
        limit=2,
        model=model,
        collate_fn=lambda examples: (torch.ones(len(examples), 1),),
        forward_fn=_linear_forward,
    )

    # Then the persisted report has the complete required top-level contract
    payload = TypeAdapter(TrainReport).validate_json(
        (tmp_path / "run" / "report.json").read_text()
    )
    assert {
        "args",
        "data_counts",
        "loss_curve",
        "wall_time",
        "peak_memory",
        "metrics",
        "temperature",
        "diverged",
    } <= payload.keys()
    assert report["diverged"] is False


def _linear_forward(model: nn.Module, batch: Batch) -> EncoderOutput:
    if not isinstance(model, nn.Linear):
        msg = "linear test requires nn.Linear"
        raise TypeError(msg)
    if not isinstance(batch, tuple):
        msg = "linear test requires tuple batch"
        raise TypeError(msg)
    linear: torch.Tensor = model.forward(batch[0])
    logits: torch.Tensor = linear.flatten()
    probabilities = torch.softmax(logits.reshape(-1, 2), dim=1).flatten()
    one_hot = torch.tensor([0.0, 1.0])
    loss = (
        -torch.log(probabilities[1]) + torch.square(probabilities[:2] - one_hot).sum()
    )
    groups = tuple((index, index + 1) for index in range(0, logits.numel(), 2))
    return EncoderOutput(logits, probabilities, groups, loss)

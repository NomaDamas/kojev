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
    select_device,
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


def test_distill_question_weighting_matches_hand_computed_combination() -> None:
    """Teacher questions count at 0.5 while gold questions count at 1.0."""
    gold_loss = torch.tensor(2.0)
    distill_loss = torch.tensor(6.0)
    output = EncoderOutput(
        logits=torch.zeros(4),
        probabilities=torch.full((4,), 0.5),
        groups=((0, 1), (2, 3)),
        loss=(gold_loss + distill_loss) / 2,
        question_losses=(gold_loss, distill_loss),
        question_weights=(1.0, 0.5),
    )

    loss = training_loss(output)
    expected = (2.0 * 1.0 + 6.0 * 0.5) / (1.0 + 0.5)

    assert loss.item() == pytest.approx(expected)


def test_gold_only_training_loss_is_unchanged() -> None:
    """All-gold loss keeps the historical arithmetic exactly."""
    first = torch.tensor(2.0)
    second = torch.tensor(6.0)
    output = EncoderOutput(
        logits=torch.zeros(4),
        probabilities=torch.full((4,), 0.5),
        groups=((0, 1), (2, 3)),
        loss=(first + second) / 2,
        question_losses=(first, second),
        question_weights=(1.0, 1.0),
    )

    assert training_loss(output).item() == pytest.approx(4.0)


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
        model_name="synthetic/backbone",
        distill_weight=0.5,
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
    assert report["args"]["model"] == "synthetic/backbone"
    assert report["args"]["distill_weight"] == 0.5
    assert report["data_counts"]["train_questions_distill"] == 0


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


def test_select_device_prefers_cuda_when_it_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training must run on the GPU the Slurm job allocated.

    Regression for gpu01 jobs 13626 and 13627. The training loop hardcoded
    `torch.autocast("cpu", ...)` and never moved the model or batch to an
    accelerator, so both jobs trained entirely on CPU and exhausted their wall
    clocks (30 minutes and 2 hours) on a 2,000-example smoke while holding an
    idle rtx6000. The smoke had already proved torch.cuda.is_available() is True
    on that node, so the GPU was allocated and then unused.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device().type == "cuda"


def test_select_device_falls_back_to_cpu_without_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline CPU runs must keep working unchanged."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device().type == "cpu"

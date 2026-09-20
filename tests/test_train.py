"""Deterministic CPU tests for the local SFT training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

import random
import sys

import pytest
import torch
from pydantic import TypeAdapter
from torch import nn

import kojev.train as train_module
from kojev.encoder import EncoderOutput, EncodingError
from kojev.schema import Example, Question, QuestionType, write_jsonl
from kojev.train import (
    Batch,
    DivergenceWatch,
    TrainReport,
    fit_temperature,
    model_device,
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
    """A SUSTAINED rise trips the watch; a NaN trips it immediately.

    This test previously asserted that one elevated sample (1.3 after a 1.0
    baseline) was divergence. That is what made the detector fire on ordinary
    batch noise in every real run. The plan specifies the running MEAN rising
    over 200 steps, so the rise now has to persist.
    """
    watch = DivergenceWatch(window=200, rise_fraction=0.25)
    for value in [1.0] * 200:
        _ = watch.observe(value)

    # A single spike is noise, not divergence.
    assert watch.observe(1.3) is False
    assert watch.diverged is False

    # A rise that persists across the window is divergence.
    for value in [1.3] * 200:
        _ = watch.observe(value)
    assert watch.diverged is True

    # NaN is always immediate, regardless of window state.
    fresh = DivergenceWatch(window=200, rise_fraction=0.25)
    assert fresh.observe(float("nan")) is True
    assert fresh.diverged is True


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


def test_training_drops_examples_that_exceed_the_collator_budget(
    tmp_path: Path,
) -> None:
    """kf-deberta-base 512-window jobs must skip overlong families, not crash."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    overflow = _example()
    overflow = overflow.model_copy(update={"state": "OVERFLOW"})
    write_jsonl(train_path, [_example(), overflow, _example()])
    write_jsonl(val_path, [_example("val")])

    def collate(examples: Sequence[Example]) -> tuple[torch.Tensor, ...]:
        for example in examples:
            if example.state == "OVERFLOW":
                raise EncodingError.questions_exceed_budget()
        return (torch.ones(len(examples), 1),)

    report = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=tmp_path / "run",
        epochs=1,
        seed=0,
        model=nn.Linear(1, 2),
        collate_fn=collate,
        forward_fn=_linear_forward,
    )
    assert report["data_counts"]["train"] == 2
    assert report["data_counts"]["train_dropped"] == 1
    assert report["data_counts"]["val_dropped"] == 0


def test_quarter_epoch_eval_writes_progress_json(tmp_path: Path) -> None:
    """Quarter-epoch eval must persist progress.json as a mid-run heartbeat."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, [_example()] * 2)
    write_jsonl(val_path, [_example("val")])
    out_dir = tmp_path / "run"
    _ = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=out_dir,
        epochs=1,
        seed=4,
        limit=2,
        model=nn.Linear(1, 2),
        collate_fn=lambda examples: (torch.ones(len(examples), 1),),
        forward_fn=_linear_forward,
    )
    payload = TypeAdapter(dict[str, object]).validate_json(
        (out_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert payload["step"] == 1
    assert "metrics" in payload
    assert "loss" in payload


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


def test_model_device_reports_where_the_parameters_live() -> None:
    """Evaluation derives its device from the model, so this must be exact."""
    model = nn.Linear(1, 2)

    assert model_device(model) == next(model.parameters()).device


def test_evaluation_moves_its_batches_onto_the_model_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval path must move batches, not only the training loop.

    Regression for gpu01 job 13632. The training loop moved its batches to the
    GPU but `_collect` collated fresh batches for validation and left them on
    CPU, so the first evaluation raised:

        RuntimeError: Expected all tensors to be on the same device, but got
        index is on cpu, different from other tensors on cuda:0

    Counting the moves proves the evaluation path participates: a run with more
    evaluation passes than optimizer steps cannot reach this count from the
    training loop alone.
    """
    real_to_device = train_module._to_device  # pyright: ignore[reportPrivateUsage]
    moves: list[int] = []

    def _counting_to_device(batch: Batch, device: torch.device) -> Batch:
        moves.append(1)
        return real_to_device(batch, device)

    monkeypatch.setattr(train_module, "_to_device", _counting_to_device)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, [_example()] * 2)
    write_jsonl(val_path, [_example("val")] * 2)
    _ = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=tmp_path / "run",
        epochs=1,
        seed=4,
        limit=2,
        model=nn.Linear(1, 2),
        collate_fn=lambda examples: (torch.ones(len(examples), 1),),
        forward_fn=_linear_forward,
        model_name="synthetic/backbone",
        distill_weight=0.5,
    )

    # One optimizer step over 2 examples at the default batch size, so any count
    # beyond that came from evaluation collating its own batches.
    assert len(moves) > 1, f"evaluation never moved a batch: {len(moves)} moves"


def test_report_metrics_carry_a_majority_baseline(tmp_path: Path) -> None:
    """Every metric group must publish the majority-class baseline.

    Todo 10's acceptance gate is `val_acc_all - majority_all >= 0.05`. Job
    13634 COMPLETED and produced a well-formed report whose metric groups were
    overall, three kind: groups and eight source: groups, none of which carried
    a majority entry, so the gate could not be evaluated from the report at all.
    A baseline is what makes an accuracy number mean anything, so it belongs
    beside every accuracy the report emits.
    """
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, [_example()] * 2)
    write_jsonl(val_path, [_example("val")] * 2)

    report = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=tmp_path / "run",
        epochs=1,
        seed=4,
        limit=2,
        model=nn.Linear(1, 2),
        collate_fn=lambda examples: (torch.ones(len(examples), 1),),
        forward_fn=_linear_forward,
        model_name="synthetic/backbone",
        distill_weight=0.5,
    )

    metrics = report["metrics"]
    assert metrics, "report published no metrics"
    for group, values in metrics.items():
        assert "majority" in values, f"{group} has no majority baseline"
        assert 0.0 <= float(values["majority"]) <= 1.0


def test_majority_baseline_matches_hand_computed_share() -> None:
    """The baseline must be the share of the most frequent gold label.

    Asserting only that a `majority` key exists would pass for any number, so
    the value is pinned against a hand-computed case: three zeros and one one
    give a majority share of 3/4.
    """
    logits = torch.zeros(4, 2)
    labels = torch.tensor([0, 0, 0, 1])

    values = train_module._metrics(logits, labels, 1.0)  # pyright: ignore[reportPrivateUsage]

    assert values["majority"] == pytest.approx(0.75)
    assert values["count"] == 4


def test_cli_exposes_learning_rate_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lr escalation path must be reachable from the command line.

    Job 13635 died in 11 seconds with

        train.py: error: unrecognized arguments: --backbone-lr 1e-5

    `backbone_lr` and `head_lr` were already TrainConfig fields and were already
    written into report.json, but argparse never exposed them. Todo 12's
    escalation rule is to retry a diverged seed at a lower learning rate, so
    without these flags that rule cannot be carried out at all.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kojev.train",
            "--out",
            "run",
            "--backbone-lr",
            "1e-5",
            "--head-lr",
            "5e-4",
        ],
    )

    args = train_module._parse_args()  # pyright: ignore[reportPrivateUsage]

    assert args.backbone_lr == pytest.approx(1e-5)
    assert args.head_lr == pytest.approx(5e-4)


def _source_example(source: str) -> Example:
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
        source=source,
        split="train",
    )


def test_limit_samples_across_sources_instead_of_taking_a_head_slice(
    tmp_path: Path,
) -> None:
    """--limit must not silently restrict the run to the first few sources.

    The gold corpus is written source by source, so slicing the head selects a
    pathological subset. Measured on the real corpus, `--limit 2000` covered
    only 7 of 12 sources, inflated lawcompany/KLAID from 7.7% to 33.2%, dropped
    KorQuAD (the largest source) entirely, and trained on a distribution
    different from the one it then evaluated on: wicho/kor_3i4k was absent from
    the training slice while making up 11.8% of the evaluation slice.

    That makes the smoke's accuracy gate unreachable for reasons unrelated to
    the model, which is what four hyperparameter configurations plateauing at
    the same ~0.031 margin was really reporting.
    """
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    # Twelve sources, grouped, exactly as the builder writes them.
    rows = [
        _source_example(f"source-{index:02d}") for index in range(12) for _ in range(50)
    ]
    write_jsonl(train_path, rows)
    write_jsonl(val_path, rows[:20])

    report = run_training(
        train_path=train_path,
        val_path=val_path,
        out_dir=tmp_path / "run",
        epochs=1,
        seed=0,
        limit=120,
        model=nn.Linear(1, 2),
        collate_fn=lambda examples: (torch.ones(len(examples), 1),),
        forward_fn=_linear_forward,
        model_name="synthetic/backbone",
        distill_weight=0.5,
    )

    # A head slice of 120 rows would cover 3 of 12 sources (50+50+20).
    selected = report["data_counts"]["train_sources"]
    assert selected >= 10, f"limit covered only {selected} of 12 sources"


def test_divergence_watch_ignores_noise_around_a_flat_mean() -> None:
    """A noisy but healthy loss must not be reported as divergence.

    Measured on gpu01 job 13647: first-200-step mean 1.3579, last-200-step mean
    1.3553, min 0.7177, max 2.2734, no NaN. That loss is flat to slightly
    decreasing, yet the run reported diverged=true.

    The detector compared a SINGLE batch loss against the running mean, so any
    one batch above mean*1.25 tripped it. With this spread that is near certain.
    The plan specifies the running MEAN rising over 200 steps, which is a
    comparison between means, not between a sample and a mean.

    The pre-existing tests used noiseless streams, which is why the fault was
    invisible to them.
    """
    rng = random.Random(0)  # noqa: S311
    watch = DivergenceWatch(window=200, rise_fraction=0.25)

    for _ in range(800):
        # Same spread as the real run, with a flat mean.
        value = 1.36 + rng.uniform(-0.65, 0.92)
        _ = watch.observe(value)

    assert watch.diverged is False, "healthy noisy loss was reported as diverged"


def test_divergence_watch_still_triggers_on_a_sustained_rise() -> None:
    """A real divergence signature must still be caught."""
    watch = DivergenceWatch(window=200, rise_fraction=0.25)

    for _ in range(200):
        _ = watch.observe(1.0)
    # The documented signature: loss climbs and stays high (1.5 -> 6.1).
    for _ in range(200):
        _ = watch.observe(6.1)

    assert watch.diverged is True, "a sustained rise was not caught"

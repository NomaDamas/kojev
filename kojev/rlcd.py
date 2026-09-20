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
import copy
import json
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch
from torch import Tensor, nn

from kojev.encoder import EncoderOutput, SpanBatch, SpanCollator, load_checkpoint
from kojev.schema import read_jsonl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from kojev.schema import Example

_DEFAULT_PROBABILITY_FLOOR: Final = 1e-8
_SOFT_BIN_TEMPERATURE: Final = 0.05
_SELECTIVE_CONFIDENCE: Final = 0.6


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


def half_epoch_steps(n_batches: int) -> int:
    """Return the step budget for a 0.5-epoch pass.

    An empty loader is zero steps. A single-batch loader still gets one step so
    a tiny fixture can actually run the objective once.
    """
    if n_batches <= 0:
        return 0
    return max(1, math.ceil(n_batches * 0.5))


def run_rlcd_loop(
    *,
    n_steps: int,
    eval_every: int,
    early_stop_rises: int,
    train_step: Callable[[int], None],
    measure_val_brier: Callable[[], float],
) -> dict[str, object]:
    """Run train_step at most n_steps times; stop after two consecutive Brier rises."""
    last: float | None = None
    rises = 0
    history: list[float] = []
    stopped_early = False
    steps_run = 0
    for step in range(1, n_steps + 1):
        train_step(step)
        steps_run = step
        if eval_every <= 0 or step % eval_every != 0:
            continue
        brier = measure_val_brier()
        history.append(brier)
        if last is not None and brier > last:
            rises += 1
        else:
            rises = 0
        last = brier
        if rises >= early_stop_rises:
            stopped_early = True
            break
    return {
        "steps": steps_run,
        "stopped_early": stopped_early,
        "brier_history": history,
    }


def _move_batch(batch: SpanBatch, device: torch.device) -> SpanBatch:
    if device.type == "cpu":
        return batch
    return SpanBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        question_spans=batch.question_spans,
        option_spans=batch.option_spans,
        question_groups=batch.question_groups,
        question_types=batch.question_types,
        gold_indices=batch.gold_indices,
        question_token_ids=batch.question_token_ids,
        question_weights=batch.question_weights,
    )


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def _labeled_distributions(
    model: nn.Module,
    collator: SpanCollator,
    examples: Sequence[Example],
    batch_size: int,
) -> tuple[list[Tensor], list[int]]:
    device = _model_device(model)
    _ = model.eval()
    probabilities: list[Tensor] = []
    golds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = _move_batch(collator(chunk), device)
            output = cast("EncoderOutput", model.forward(batch))
            for group, gold in zip(output.groups, batch.gold_indices, strict=True):
                if gold is None:
                    continue
                indices = list(group)
                probabilities.append(output.probabilities[indices].detach().cpu())
                golds.append(gold)
    return probabilities, golds


def _distribution_metrics(
    probabilities: Sequence[Tensor], gold_indices: Sequence[int]
) -> dict[str, float]:
    if not probabilities:
        return {
            "accuracy": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "selective_acc": 0.0,
            "count": 0.0,
        }
    hits: list[float] = []
    confidences: list[float] = []
    briers: list[Tensor] = []
    for dist, gold in zip(probabilities, gold_indices, strict=True):
        predicted = int(dist.argmax().item())
        hits.append(1.0 if predicted == gold else 0.0)
        confidences.append(float(dist.max().item()))
        target = torch.zeros_like(dist)
        target[gold] = 1.0
        briers.append(torch.square(dist - target).sum())
    conf = torch.tensor(confidences)
    correct = torch.tensor(hits)
    ece = 0.0
    bins = 15
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (conf >= lower) & (conf <= upper if index == bins - 1 else conf < upper)
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].mean().item()) - float(conf[mask].mean().item())
            )
    selective_mask = conf >= _SELECTIVE_CONFIDENCE
    selective = (
        float(correct[selective_mask].mean().item()) if selective_mask.any() else 0.0
    )
    return {
        "accuracy": float(correct.mean().item()),
        "brier": float(torch.stack(briers).mean().item()),
        "ece": ece,
        "selective_acc": selective,
        "count": float(len(hits)),
    }


def _split_metrics(
    model: nn.Module,
    collator: SpanCollator,
    examples: Sequence[Example],
    batch_size: int,
) -> dict[str, float]:
    probabilities, golds = _labeled_distributions(model, collator, examples, batch_size)
    return _distribution_metrics(probabilities, golds)


@dataclass(frozen=True, slots=True)
class RLCDRun:
    """Data and schedule for one 0.5-epoch utility fine-tune."""

    train_examples: Sequence[Example]
    val_examples: Sequence[Example]
    ood_examples: Sequence[Example]
    out_dir: Path
    lr: float = 1e-5
    batch_size: int = 4
    eval_every: int = 200
    max_steps: int | None = None
    objective: RLCDConfig = _DEFAULT_CONFIG


def run_rlcd_training(
    model: nn.Module, collator: SpanCollator, run: RLCDRun
) -> dict[str, object]:
    """Fine-tune from an SFT checkpoint for at most half an epoch.

    Writes ``report.json`` with the four-clause KEEP verdict against the frozen
    SFT snapshot taken at the start of the run.
    """
    frozen = copy.deepcopy(model)
    _ = frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    baseline_val = _split_metrics(frozen, collator, run.val_examples, run.batch_size)
    baseline_ood = _split_metrics(frozen, collator, run.ood_examples, run.batch_size)
    n_batches = (
        math.ceil(len(run.train_examples) / run.batch_size) if run.train_examples else 0
    )
    n_steps = (
        run.max_steps if run.max_steps is not None else half_epoch_steps(n_batches)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=run.lr)

    def train_batch(chunk: Sequence[Example]) -> None:
        device = _model_device(model)
        _ = model.train()
        _ = frozen.eval()
        batch = _move_batch(collator(list(chunk)), device)
        output = cast("EncoderOutput", model.forward(batch))
        with torch.no_grad():
            frozen_output = cast("EncoderOutput", frozen.forward(batch))
        policy: list[Tensor] = []
        anchors: list[Tensor] = []
        golds: list[int] = []
        for group, gold in zip(output.groups, batch.gold_indices, strict=True):
            if gold is None:
                continue
            indices = list(group)
            policy.append(output.probabilities[indices])
            anchors.append(frozen_output.probabilities[indices].detach())
            golds.append(gold)
        if not policy:
            return
        loss = rlcd_loss(
            RLCDInputs(tuple(policy), tuple(golds), tuple(anchors)), run.objective
        )
        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

    def train_step(step: int) -> None:
        start = ((step - 1) * run.batch_size) % max(len(run.train_examples), 1)
        chunk = run.train_examples[start : start + run.batch_size]
        if not chunk:
            return
        train_batch(chunk)

    def measure() -> float:
        return _split_metrics(model, collator, run.val_examples, run.batch_size)[
            "brier"
        ]

    loop = run_rlcd_loop(
        n_steps=n_steps,
        eval_every=run.eval_every,
        early_stop_rises=2,
        train_step=train_step,
        measure_val_brier=measure,
    )
    candidate_val = _split_metrics(model, collator, run.val_examples, run.batch_size)
    candidate_ood = _split_metrics(model, collator, run.ood_examples, run.batch_size)
    table = GateTable(
        ece_baseline=baseline_val["ece"],
        ece_candidate=candidate_val["ece"],
        selective_acc_baseline=baseline_val["selective_acc"],
        selective_acc_candidate=candidate_val["selective_acc"],
        in_domain_acc_baseline=baseline_val["accuracy"],
        in_domain_acc_candidate=candidate_val["accuracy"],
        brier_baseline=baseline_val["brier"],
        brier_candidate=candidate_val["brier"],
        ood_acc_baseline=baseline_ood["accuracy"],
        ood_acc_candidate=candidate_ood["accuracy"],
    )
    verdict = rlcd_gate(table)
    report: dict[str, object] = {
        "verdict": verdict.value,
        "failed_clauses": list(gate_clause_failures(table)),
        "stopped_early": loop["stopped_early"],
        "steps": loop["steps"],
        "brier_history": loop["brier_history"],
        "baseline_val": baseline_val,
        "candidate_val": candidate_val,
        "baseline_ood": baseline_ood,
        "candidate_ood": candidate_ood,
        **_deltas(table),
    }
    run.out_dir.mkdir(parents=True, exist_ok=True)
    _ = (run.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kojev.rlcd",
        description="RLCD expected-utility stage and four-clause GO/NO-GO gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate", help="score a metrics table against KEEP")
    _ = gate.add_argument("--metrics", type=Path, required=True)
    _ = gate.add_argument("--out", type=Path, required=True)
    train = sub.add_parser(
        "train", help="0.5-epoch utility fine-tune from an SFT checkpoint"
    )
    _ = train.add_argument("--checkpoint", type=Path, required=True)
    _ = train.add_argument("--train", type=Path, required=True)
    _ = train.add_argument("--val", type=Path, required=True)
    _ = train.add_argument("--ood", type=Path, required=True)
    _ = train.add_argument("--distill", type=Path)
    _ = train.add_argument("--out", type=Path, required=True)
    _ = train.add_argument("--batch-size", type=int, default=4)
    _ = train.add_argument("--eval-every", type=int, default=200)
    _ = train.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args(argv)


def _run_gate(metrics: Path, out: Path) -> int:
    try:
        table = _read_gate_table(metrics)
    except GateInputError as error:
        print(str(error), file=sys.stderr)  # noqa: T201
        return 1
    verdict = rlcd_gate(table)
    failed = gate_clause_failures(table)
    report: dict[str, object] = {
        "verdict": verdict.value,
        "failed_clauses": list(failed),
        "metrics_path": str(metrics),
        **_deltas(table),
    }
    out.mkdir(parents=True, exist_ok=True)
    _ = (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict.value, "failed_clauses": list(failed)}))  # noqa: T201
    return 0


def _run_train(values: dict[str, object]) -> int:
    checkpoint = values.get("checkpoint")
    train_path = values.get("train")
    val_path = values.get("val")
    ood_path = values.get("ood")
    out = values.get("out")
    distill = values.get("distill")
    batch_size = values.get("batch_size")
    eval_every = values.get("eval_every")
    lr = values.get("lr")
    if not (
        isinstance(checkpoint, Path)
        and isinstance(train_path, Path)
        and isinstance(val_path, Path)
        and isinstance(ood_path, Path)
        and isinstance(out, Path)
        and isinstance(batch_size, int)
        and isinstance(eval_every, int)
        and isinstance(lr, float)
    ):
        return 1
    if not checkpoint.exists():
        print(f"checkpoint does not exist: {checkpoint}", file=sys.stderr)  # noqa: T201
        return 1
    model, collator, _metadata = load_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_examples = read_jsonl(train_path)
    if isinstance(distill, Path):
        train_examples.extend(read_jsonl(distill))
    report = run_rlcd_training(
        model,
        collator,
        RLCDRun(
            train_examples=train_examples,
            val_examples=read_jsonl(val_path),
            ood_examples=read_jsonl(ood_path),
            out_dir=out,
            lr=lr,
            batch_size=batch_size,
            eval_every=eval_every,
        ),
    )
    print(json.dumps({"verdict": report["verdict"], "steps": report["steps"]}))  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch gate scoring or the 0.5-epoch utility fine-tune."""
    args = _parse_args(argv)
    values = cast("dict[str, object]", vars(args))
    command = values.get("command")
    if command == "gate":
        metrics = values.get("metrics")
        out = values.get("out")
        if not isinstance(metrics, Path) or not isinstance(out, Path):
            return 1
        return _run_gate(metrics, out)
    if command == "train":
        return _run_train(values)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

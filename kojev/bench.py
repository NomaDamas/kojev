"""Zero-shot KoBEST and KLUE validation benchmark harness."""

# ruff: noqa: E501, EM101, EM102, S311, S603, TRY003

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict, cast, override

from pydantic import TypeAdapter

from kojev.schema import Example, JsonValue, Question, QuestionType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type Row = dict[str, JsonValue]
type Probabilities = tuple[float, ...]

KOBEST_CONFIGS: Final = ("boolq", "copa", "wic", "hellaswag", "sentineg")
KLUE_TASKS: Final = ("ynat", "nli", "sts")
TASK_COLUMNS: Final = KOBEST_CONFIGS + KLUE_TASKS
_BINARY: Final = ["아니오", "예"]
_NLI: Final = ["함의", "중립", "모순"]
_STS: Final = ["전혀 다름", "다름", "약간 다름", "비슷함", "거의 같음", "완전히 같음"]
_YNAT: Final = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
_ROW_ADAPTER: TypeAdapter[Row] = TypeAdapter(Row)


class DecisionModel(Protocol):
    """A model capable of answering typed questions."""

    def decide(
        self,
        state: str,
        questions: tuple[Question, ...],
    ) -> tuple[Probabilities, ...]:
        """Return one probability vector per question."""
        ...


class MetricPayload(TypedDict):
    """JSON payload for one evaluated slice."""

    count: int
    accuracy: float
    macro_f1: float
    brier: float
    ece_15: float


class ReportPayload(TypedDict):
    """JSON payload for one benchmark run."""

    tasks: dict[str, MetricPayload]
    kinds: dict[str, MetricPayload]
    total_count: int


class BenchmarkError(RuntimeError):
    """Malformed benchmark data or invocation."""

    reason: str
    __slots__: tuple[str, ...] = ("reason",)

    def __init__(self, reason: str) -> None:
        """Initialize with a structured reason string."""
        super().__init__(reason)
        self.reason = reason

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One typed benchmark question."""

    task: str
    kind: QuestionType
    example_id: str
    state: str
    question: Question


@dataclass(frozen=True, slots=True)
class Metrics:
    """Classification and calibration metrics."""

    accuracy: float
    macro_f1: float
    brier: float
    ece_15: float


@dataclass(frozen=True, slots=True)
class RandomDecisionModel:
    """Deterministic random-probability baseline."""

    seed: int = 0

    def decide(
        self,
        state: str,
        questions: tuple[Question, ...],
    ) -> tuple[Probabilities, ...]:
        """Return deterministic random probability vectors."""
        generator = random.Random(f"{self.seed}:{state}")
        outputs: list[Probabilities] = []
        for question in questions:
            values = tuple(generator.random() + 0.01 for _ in question.options)
            total = sum(values)
            outputs.append(tuple(value / total for value in values))
        return tuple(outputs)


def map_kobest_row(config: str, row: Row) -> Example:
    """Map a KoBEST row to one typed question."""
    example_id = _identifier(row)
    match config:
        case "boolq":
            state = f"지문: {_text_any(row, ('paragraph', 'passage', 'context'))}\n질문: {_text(row, 'question')}"
            question = Question(
                type=QuestionType.NOUL,
                instructions="질문이 지문에 의해 참인가?",
                options=_BINARY,
                gold=_integer(row, "label"),
                meta={"task": config, "id": example_id},
            )
        case "copa":
            state = _text(row, "premise")
            relation = _text(row, "question")
            options = [
                _text_any(row, ("alternative_1", "choice1")),
                _text_any(row, ("alternative_2", "choice2")),
            ]
            question = Question(
                type=QuestionType.CHOICE,
                instructions=f"문장의 {relation}에 알맞은 것은?",
                options=options,
                gold=_integer(row, "label"),
                meta={"task": config, "id": example_id},
            )
        case "wic":
            state = (
                f"단어: {_text(row, 'word')}\n"
                f"문장 1: {_text_any(row, ('context_1', 'sentence1'))}\n"
                f"문장 2: {_text_any(row, ('context_2', 'sentence2'))}"
            )
            question = Question(
                type=QuestionType.NOUL,
                instructions="두 문장에서 같은 의미로 쓰였는가?",
                options=_BINARY,
                gold=_integer(row, "label"),
                meta={"task": config, "id": example_id},
            )
        case "hellaswag":
            state = _text_any(row, ("context", "ctx", "ctx_a"))
            question = Question(
                type=QuestionType.CHOICE,
                instructions="이어질 가장 자연스러운 문장은?",
                options=_texts_any(
                    row,
                    (
                        "ending",
                        "endings",
                        "ending_1",
                        "ending_2",
                        "ending_3",
                        "ending_4",
                    ),
                ),
                gold=_integer(row, "label"),
                meta={"task": config, "id": example_id},
            )
        case "sentineg":
            state = _text(row, "sentence")
            question = Question(
                type=QuestionType.NOUL,
                instructions="이 문장은 긍정적인가?",
                options=_BINARY,
                gold=_integer(row, "label"),
                meta={"task": config, "id": example_id},
            )
        case _:
            raise BenchmarkError(f"unsupported KoBEST config: {config}")
    return Example(
        state=state, questions=[question], source="skt/kobest_v1", split="test"
    )


def map_klue_row(task: str, row: Row) -> Example:
    """Map a KLUE validation row to one typed question."""
    example_id = _identifier(row)
    match task:
        case "ynat":
            state = _text(row, "title")
            question = Question(
                type=QuestionType.CHOICE,
                instructions="기사의 주제는?",
                options=_YNAT,
                gold=_integer(row, "label"),
                meta={"task": task, "id": example_id},
            )
        case "nli":
            state = f"전제: {_text(row, 'premise')}\n가설: {_text(row, 'hypothesis')}"
            question = Question(
                type=QuestionType.CHOICE,
                instructions="가설과 전제의 관계는?",
                options=_NLI,
                gold=_integer(row, "label"),
                meta={"task": task, "id": example_id},
            )
        case "sts":
            state = (
                f"문장 1: {_text(row, 'sentence1')}\n문장 2: {_text(row, 'sentence2')}"
            )
            labels = _row(row, "labels")
            question = Question(
                type=QuestionType.SCORE,
                instructions="두 문장의 의미 유사도는?",
                options=_STS,
                gold=round(_number(labels, "real-label")),
                meta={"task": task, "id": example_id},
            )
        case _:
            raise BenchmarkError(f"unsupported KLUE task: {task}")
    return Example(
        state=state, questions=[question], source="klue/klue", split="validation"
    )


def calibration_metrics(
    gold: tuple[int, ...], probabilities: tuple[Probabilities, ...]
) -> Metrics:
    """Compute accuracy, macro-F1, multiclass Brier, and 15-bin ECE."""
    if not gold or len(gold) != len(probabilities):
        raise BenchmarkError("gold and probability rows must have equal nonzero length")
    predictions = tuple(
        max(range(len(row)), key=row.__getitem__) for row in probabilities
    )
    labels = sorted(set(gold) | set(predictions))
    accuracy = sum(a == b for a, b in zip(gold, predictions, strict=True)) / len(gold)
    f1: list[float] = []
    for label in labels:
        tp = sum(a == label == b for a, b in zip(gold, predictions, strict=True))
        fp = sum(
            a != label and b == label for a, b in zip(gold, predictions, strict=True)
        )
        fn = sum(
            a == label and b != label for a, b in zip(gold, predictions, strict=True)
        )
        f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    brier = sum(
        sum((p - int(i == target)) ** 2 for i, p in enumerate(row))
        for target, row in zip(gold, probabilities, strict=True)
    ) / len(gold)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(15)]
    for target, predicted, row in zip(gold, predictions, probabilities, strict=True):
        confidence = max(row)
        bins[min(14, int(confidence * 15))].append((confidence, target == predicted))
    ece = sum(
        len(bucket)
        / len(gold)
        * abs(
            sum(ok for _, ok in bucket) / len(bucket)
            - sum(conf for conf, _ in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return Metrics(accuracy, sum(f1) / len(f1), brier, ece)


def metric_report(
    items: tuple[BenchmarkItem, ...], model: DecisionModel
) -> ReportPayload:
    """Evaluate all task and question-kind slices."""
    tasks = {
        name: _slice(tuple(item for item in items if item.task == name), model)
        for name in sorted({item.task for item in items})
    }
    kinds = {
        kind: _slice(tuple(item for item in items if item.kind.value == kind), model)
        for kind in sorted({item.kind.value for item in items})
    }
    return ReportPayload(tasks=tasks, kinds=kinds, total_count=len(items))


def render_task_grid(
    models: Sequence[tuple[str, Mapping[str, MetricPayload]]],
    metric: str = "accuracy",
) -> str:
    """Render a wide table: one row per model, eight task columns of one metric."""
    header = "| model | " + " | ".join(TASK_COLUMNS) + " |"
    rule = "| --- | " + " | ".join(["---:"] * len(TASK_COLUMNS)) + " |"
    lines = [header, rule]
    for name, tasks in models:
        cells = [name]
        for task in TASK_COLUMNS:
            payload = tasks.get(task)
            if payload is None:
                raise BenchmarkError(
                    f"grid is missing task {task!r} for model {name!r}"
                )
            value = payload.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BenchmarkError(
                    f"grid is missing {metric!r} for {name!r} task {task!r}"
                )
            cells.append(f"{float(value):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_metric_tables(
    models: Sequence[tuple[str, Mapping[str, MetricPayload]]],
) -> str:
    """Render acc, macro-F1, Brier, and ECE-15 grids the RESULTS.md contract names."""
    sections = ["# KoJev benchmark", ""]
    for key, title in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("brier", "Brier"),
        ("ece_15", "ECE-15"),
    ):
        sections.append(f"## {title}")
        sections.append("")
        sections.append(render_task_grid(models, metric=key).rstrip())
        sections.append("")
    return "\n".join(sections) + "\n"


def assert_no_kobest_contamination(
    training_manifest: Path, kobest_ids: set[str]
) -> None:
    """Reject malformed manifests and any KoBEST ID overlap."""
    with training_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row: Row = _ROW_ADAPTER.validate_json(line)
            except ValueError as error:
                raise BenchmarkError(
                    f"malformed manifest line {line_number}: {error}"
                ) from error
            candidate = row.get("id", row.get("idx"))
            if isinstance(candidate, (str, int)) and str(candidate) in kobest_ids:
                raise AssertionError(
                    f"contamination: KoBEST id {candidate!r} at line {line_number}"
                )


def load_benchmark_items(
    limit: int, training_manifest: Path | None = None
) -> tuple[BenchmarkItem, ...]:
    """Load a bounded, balanced smoke slice through Hugging Face datasets."""
    task_count = len(KOBEST_CONFIGS) + len(KLUE_TASKS)
    if limit < task_count:
        raise BenchmarkError(reason="limit must cover all eight benchmark tasks")
    per_task = max(1, limit // task_count)
    items: list[BenchmarkItem] = []
    for task in KOBEST_CONFIGS:
        items.extend(_load_task("skt/kobest_v1", task, "test", per_task))
    if training_manifest is not None:
        assert_no_kobest_contamination(
            training_manifest, {item.example_id for item in items}
        )
    for task in KLUE_TASKS:
        items.extend(_load_task("klue/klue", task, "validation", per_task))
    return tuple(items[:limit])


def _render_named_grids(grids: list[str], results_path: Path | None) -> str:
    """Load NAME=PATH JSON reports and render the eight-task accuracy grid."""
    models: list[tuple[str, dict[str, MetricPayload]]] = []
    for spec in grids:
        name, separator, raw = spec.partition("=")
        if not separator or not name or not raw:
            raise BenchmarkError(reason=f"--grid expects NAME=PATH, got: {spec}")
        path = Path(raw)
        if not path.is_file():
            raise BenchmarkError(reason=f"grid report does not exist: {path}")
        loaded = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(loaded, dict):
            raise BenchmarkError(reason=f"grid report missing tasks object: {path}")
        entries = cast("dict[str, object]", loaded)
        raw_tasks = entries.get("tasks")
        if not isinstance(raw_tasks, dict):
            raise BenchmarkError(reason=f"grid report missing tasks object: {path}")
        tasks = cast("dict[str, MetricPayload]", raw_tasks)
        models.append((name, tasks))
    table = render_metric_tables(models)
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        _ = results_path.write_text(table, encoding="utf-8")
    return table


class _Cli(TypedDict):
    """Parsed argv for the bench CLI."""

    model: str
    limit: str
    output: str
    manifest: Path | None
    grids: list[str]
    results: Path | None


def _parse_cli(argv: list[str]) -> _Cli:
    """Parse bench flags without argparse."""
    values: _Cli = {
        "model": "random",
        "limit": "200",
        "output": "benchmark-report.json",
        "manifest": None,
        "grids": [],
        "results": None,
    }
    arguments = iter(argv)
    for argument in arguments:
        if argument == "--model":
            values["model"] = next(arguments)
        elif argument == "--limit":
            values["limit"] = next(arguments)
        elif argument == "--output":
            values["output"] = next(arguments)
        elif argument == "--training-manifest":
            values["manifest"] = Path(next(arguments))
        elif argument == "--grid":
            values["grids"].append(next(arguments))
        elif argument == "--results":
            values["results"] = Path(next(arguments))
        else:
            raise BenchmarkError(reason=f"unknown argument: {argument}")
    return values


def main() -> None:
    """Run the deterministic random baseline and write its JSON report."""
    if "--help" in sys.argv:
        usage = (
            "usage: python -m kojev.bench [--model random] [--limit N] "
            "[--output PATH] [--training-manifest PATH] "
            "[--grid NAME=PATH ...] [--results PATH]\n"
        )
        _ = sys.stdout.write(usage)
        return
    parsed = _parse_cli(sys.argv[1:])
    if parsed["grids"]:
        _ = sys.stdout.write(_render_named_grids(parsed["grids"], parsed["results"]))
        return
    values = {
        "model": parsed["model"],
        "limit": parsed["limit"],
        "output": parsed["output"],
    }
    training_manifest = parsed["manifest"]
    if values["model"] != "random":
        raise BenchmarkError(reason="only --model random is implemented")
    limit = int(values["limit"])
    output = Path(values["output"])
    report = metric_report(
        load_benchmark_items(limit, training_manifest), RandomDecisionModel()
    )
    _ = output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_task(
    dataset: str, task: str, split: str, limit: int
) -> tuple[BenchmarkItem, ...]:
    code = "import json,sys; from datasets import load_dataset; d=load_dataset(sys.argv[1],sys.argv[2],split=sys.argv[3]); [print(json.dumps(d[i],ensure_ascii=False)) for i in range(min(int(sys.argv[4]),len(d)))]"
    result = subprocess.run(
        [sys.executable, "-c", code, dataset, task, split, str(limit)],
        check=True,
        text=True,
        capture_output=True,
        timeout=300,
    )
    mapper = map_kobest_row if dataset == "skt/kobest_v1" else map_klue_row
    output: list[BenchmarkItem] = []
    for index, line in enumerate(result.stdout.splitlines()):
        row: Row = _ROW_ADAPTER.validate_json(line)
        if not any(name in row for name in ("idx", "id", "guid")):
            row["idx"] = f"{task}-{index}"
        example = mapper(task, row)
        question = example.questions[0]
        output.append(
            BenchmarkItem(
                task, question.type, _identifier(row), example.state, question
            )
        )
    return tuple(output)


def _slice(items: tuple[BenchmarkItem, ...], model: DecisionModel) -> MetricPayload:
    gold = tuple(_gold(item.question) for item in items)
    probabilities = tuple(
        model.decide(item.state, (item.question,))[0] for item in items
    )
    metrics = calibration_metrics(gold, probabilities)
    return MetricPayload(
        count=len(items),
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        brier=metrics.brier,
        ece_15=metrics.ece_15,
    )


def _identifier(row: Row) -> str:
    for name in ("idx", "id", "guid"):
        value = row.get(name)
        if isinstance(value, (str, int)):
            return str(value)
    raise BenchmarkError(reason="row has no idx, id, or guid")


def _text(row: Row, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise BenchmarkError(reason=f"{name} must be text")
    return value


def _text_any(row: Row, names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str):
            return value
    raise BenchmarkError(reason=f"none of {names} is text")


def _texts_any(row: Row, names: tuple[str, ...]) -> list[str]:
    value: JsonValue | None = None
    for name in names:
        candidate = row.get(name)
        if isinstance(candidate, list):
            value = candidate
            break
    if value is None and all(name in row for name in names[-4:]):
        value = [row[name] for name in names[-4:]]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BenchmarkError(reason=f"none of {names} is a text list")
    return [item for item in value if isinstance(item, str)]


def _row(row: Row, name: str) -> Row:
    value = row.get(name)
    if not isinstance(value, dict):
        raise BenchmarkError(reason=f"{name} must be an object")
    return _ROW_ADAPTER.validate_python(value)


def _integer(row: Row, name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(reason=f"{name} must be an integer")
    return value


def _number(row: Row, name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(reason=f"{name} must be numeric")
    return float(value)


def _gold(question: Question) -> int:
    if question.gold is None:
        raise BenchmarkError(reason="benchmark question has no gold")
    return question.gold


if __name__ == "__main__":
    main()

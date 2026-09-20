"""Evaluation harness contracts: tables, contamination, latency, failures."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

import pytest

from kojev.evaluate import (
    EvaluationConfig,
    EvaluationError,
    assert_no_train_contamination,
    evaluate,
    load_open_jev,
    main,
    resolve_checkpoint,
    wrap_kojev_model,
    wrap_open_jev,
)
from kojev.schema import Example, Question, QuestionType, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _FixedModel:
    """Returns a scripted probability vector per question index."""

    def __init__(self, vectors: Sequence[tuple[float, ...]]) -> None:
        self._vectors: Sequence[tuple[float, ...]] = vectors
        self.calls: int = 0

    def decide(
        self,
        state: str,
        questions: tuple[Question, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Return one scripted vector per question."""
        _ = state
        start = self.calls
        self.calls += len(questions)
        return tuple(self._vectors[start + offset] for offset in range(len(questions)))


def _noul(instructions: str, gold: int) -> Question:
    return Question(
        type=QuestionType.NOUL,
        instructions=instructions,
        options=["아니오", "예"],
        gold=gold,
        meta={},
    )


def _example(state: str, gold: int, split: str = "test") -> Example:
    return Example(
        state=state,
        questions=[_noul("긍정이다.", gold)],
        source="e9t/nsmc",
        split=split,
    )


def _checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _ = (ckpt / "config.json").write_text("{}", encoding="utf-8")
    return ckpt


def test_missing_checkpoint_raises_typed_error_naming_the_path(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "no-such-checkpoint"

    with pytest.raises(EvaluationError, match="checkpoint does not exist") as caught:
        _ = resolve_checkpoint(absent)

    assert type(caught.value) is EvaluationError
    assert str(absent) in str(caught.value)


def test_empty_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty-ckpt"
    empty.mkdir()

    with pytest.raises(EvaluationError, match="directory is empty"):
        _ = resolve_checkpoint(empty)


def test_contamination_check_fails_when_a_train_state_is_injected() -> None:
    shared = "공유된 리뷰"
    evaluated = [_example(shared, 1), _example("고유한 리뷰", 0)]

    with pytest.raises(EvaluationError, match="contamination in gold_test") as caught:
        assert_no_train_contamination("gold_test", evaluated, {shared})

    assert shared in str(caught.value)


def test_contamination_check_passes_on_disjoint_states() -> None:
    evaluated = [_example("평가 전용", 1)]

    assert_no_train_contamination("gold_test", evaluated, {"학습 전용"})


def test_report_metrics_are_arithmetically_correct_on_a_hand_fixture(
    tmp_path: Path,
) -> None:
    # Three questions. Two are answered correctly with probability 0.75 on the
    # gold option and one incorrectly with probability 0.25 on gold, so accuracy
    # is two thirds. Each correct vector contributes a squared error of 0.125 and
    # the incorrect one contributes 1.125, so the mean Brier score is the sum of
    # 0.125, 0.125 and 1.125 divided by three, which is 0.4583333 recurring.
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example("학습 상태", 1, split="train")])
    write_jsonl(
        test,
        [_example("평가 A", 1), _example("평가 B", 1), _example("평가 C", 1)],
    )
    model = _FixedModel([(0.25, 0.75), (0.25, 0.75), (0.75, 0.25)])
    out = tmp_path / "report.json"

    report = evaluate(
        EvaluationConfig(
            checkpoint=_checkpoint(tmp_path),
            train=train,
            splits=(("gold_test", test),),
            out=out,
        ),
        model,
    )

    splits = report["splits"]
    assert isinstance(splits, dict)
    gold_test = splits["gold_test"]
    assert isinstance(gold_test, dict)
    tables = gold_test["tables"]
    assert isinstance(tables, dict)
    overall = tables["overall"]
    assert isinstance(overall, dict)
    assert overall["count"] == 3
    assert isinstance(overall["accuracy"], float)
    assert math.isclose(overall["accuracy"], 2 / 3, rel_tol=1e-9)
    assert isinstance(overall["brier"], float)
    assert math.isclose(overall["brier"], 0.4583333333333333, rel_tol=1e-9)


def test_report_contains_per_kind_and_per_source_tables_and_latency(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example("학습 상태", 1, split="train")])
    write_jsonl(test, [_example("평가 A", 1), _example("평가 B", 0)])
    out = tmp_path / "report.json"

    report = evaluate(
        EvaluationConfig(
            checkpoint=_checkpoint(tmp_path),
            train=train,
            splits=(("gold_test", test),),
            out=out,
        ),
        _FixedModel([(0.1, 0.9), (0.8, 0.2)]),
    )

    splits = report["splits"]
    assert isinstance(splits, dict)
    gold_test = splits["gold_test"]
    assert isinstance(gold_test, dict)
    tables = gold_test["tables"]
    assert isinstance(tables, dict)
    assert set(tables) == {"overall", "kinds", "sources"}
    kinds = tables["kinds"]
    sources = tables["sources"]
    assert isinstance(kinds, dict)
    assert isinstance(sources, dict)
    assert "noul" in kinds
    assert "e9t/nsmc" in sources

    latency = gold_test["latency"]
    assert isinstance(latency, dict)
    assert latency["count"] == 2
    for key in ("p50_ms", "p95_ms"):
        value = latency[key]
        assert isinstance(value, float)
        assert math.isfinite(value)

    written = cast("dict[str, object]", json.loads(out.read_text(encoding="utf-8")))
    assert written["contamination"] == "clean"


def test_missing_split_file_raises_typed_error(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [_example("학습 상태", 1, split="train")])

    with pytest.raises(EvaluationError, match="evaluation split does not exist"):
        _ = evaluate(
            EvaluationConfig(
                checkpoint=_checkpoint(tmp_path),
                train=train,
                splits=(("gold_test", tmp_path / "absent.jsonl"),),
                out=tmp_path / "report.json",
            ),
            _FixedModel([(0.5, 0.5)]),
        )


def test_evaluate_refuses_a_contaminated_split_end_to_end(tmp_path: Path) -> None:
    shared = "양쪽에 존재하는 상태"
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example(shared, 1, split="train")])
    write_jsonl(test, [_example(shared, 1)])
    out = tmp_path / "report.json"

    with pytest.raises(EvaluationError, match="contamination in gold_test"):
        _ = evaluate(
            EvaluationConfig(
                checkpoint=_checkpoint(tmp_path),
                train=train,
                splits=(("gold_test", test),),
                out=out,
            ),
            _FixedModel([(0.5, 0.5)]),
        )

    assert not out.exists()


def test_cli_rejects_a_missing_checkpoint_and_writes_no_results(
    tmp_path: Path,
) -> None:
    """Todo 14 names this failure: a bad checkpoint path must exit nonzero.

    It must also leave no partial RESULTS.md behind, because a half-written
    results table is worse than no table: it looks like a real measurement.
    """
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example("훈련 상태", 1, split="train")])
    write_jsonl(test, [_example("평가 상태", 1)])
    results = tmp_path / "RESULTS.md"
    absent = tmp_path / "no-such-checkpoint"

    exit_code = main(
        [
            "--checkpoint",
            f"main={absent}",
            "--train",
            str(train),
            "--split",
            f"gold_test={test}",
            "--out",
            str(tmp_path / "report.json"),
            "--results",
            str(results),
        ]
    )

    assert exit_code != 0
    assert not results.exists()


def test_cli_emits_a_results_table_for_each_named_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESULTS.md must carry one row per named model, traceable to its report."""
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example("훈련 상태", 1, split="train")])
    write_jsonl(test, [_example("평가 상태", 1)])
    checkpoint = _checkpoint(tmp_path)
    results = tmp_path / "RESULTS.md"

    def _fake_load(_checkpoint: Path) -> _FixedModel:
        return _FixedModel([(0.25, 0.75)])

    monkeypatch.setattr("kojev.evaluate._load_decision_model", _fake_load)

    exit_code = main(
        [
            "--checkpoint",
            f"main={checkpoint}",
            "--train",
            str(train),
            "--split",
            f"gold_test={test}",
            "--out",
            str(tmp_path / "report.json"),
            "--results",
            str(results),
        ]
    )

    assert exit_code == 0
    rendered = results.read_text(encoding="utf-8")
    assert "| model |" in rendered
    assert "main" in rendered
    assert "gold_test" in rendered


class _FakeKojev:
    """Mimics KoJevModel.decide(example, collator)."""

    def __init__(self) -> None:
        self.seen_state: str | None = None

    def decide(self, example: Example, collator: object) -> tuple[object, ...]:
        _ = collator
        self.seen_state = example.state
        return tuple(
            type("Answer", (), {"probabilities": (0.25, 0.75)})()
            for _question in example.questions
        )


def test_wrap_kojev_model_matches_the_decision_model_contract() -> None:
    """evaluate() calls decide(state, questions), not decide(example).

    The production loader used to forward KoJevModel.decide(example, collator)
    as if it were a DecisionModel. That only passed because CLI tests
    monkeypatched the loader.
    """
    inner = _FakeKojev()
    wrapped = wrap_kojev_model(inner, object())
    question = _noul("긍정이다.", 1)
    vectors = wrapped.decide("배가 고프다", (question,))
    assert inner.seen_state == "배가 고프다"
    assert vectors == ((0.25, 0.75),)


class _FakeOpenJev:
    def decide(self, state: str, questions: object) -> list[dict[str, object]]:
        _ = state
        rows: list[dict[str, object]] = []
        payload = cast("list[dict[str, object]]", questions)
        for item in payload:
            options = cast("list[str]", item["options"])
            mass = 1.0 / len(options)
            rows.append({"probabilities": dict.fromkeys(options, mass)})
        return rows


def test_wrap_open_jev_emits_option_ordered_probability_tuples() -> None:
    """The public baseline returns a dict keyed by option text."""
    wrapped = wrap_open_jev(_FakeOpenJev())
    question = Question(
        type=QuestionType.CHOICE,
        instructions="주제",
        options=["수수료", "환불", "기타"],
        gold=1,
        meta={},
    )
    vectors = wrapped.decide("charged twice", (question,))
    assert len(vectors) == 1
    assert len(vectors[0]) == 3
    assert vectors[0][0] == pytest.approx(1 / 3)
    assert vectors[0][1] == pytest.approx(1 / 3)
    assert vectors[0][2] == pytest.approx(1 / 3)


def test_load_open_jev_raises_typed_error_when_the_package_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ImportError(name)

    monkeypatch.setattr("kojev.evaluate.importlib.import_module", _missing)

    with pytest.raises(EvaluationError, match="open-jev"):
        _ = load_open_jev("com-kotobalabs/open-jev-deberta-v3-large")


def test_cli_rejects_missing_open_jev_package_without_writing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [_example("훈련 상태", 1, split="train")])
    write_jsonl(test, [_example("평가 상태", 1)])
    results = tmp_path / "RESULTS.md"

    def _missing(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ImportError(name)

    monkeypatch.setattr("kojev.evaluate.importlib.import_module", _missing)
    exit_code = main(
        [
            "--checkpoint",
            "english=com-kotobalabs/open-jev-deberta-v3-large",
            "--train",
            str(train),
            "--split",
            f"gold_test={test}",
            "--out",
            str(tmp_path / "report.json"),
            "--results",
            str(results),
        ]
    )
    assert exit_code != 0
    assert not results.exists()

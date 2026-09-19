"""Build deterministic rule-based OOD questions from GOLD test states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, override

from kojev.schema import Example, Question, QuestionType, read_jsonl, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_NSMC_LEVELS: Final = ["불만족", "만족"]
_NSMC_REVERSED_LEVELS: Final = ["만족", "불만족"]
_STS_REVERSED_LEVELS: Final = [
    "완전히 같음",
    "매우 비슷함",
    "비슷함",
    "다소 다름",
    "매우 다름",
    "전혀 다름",
]
_YNAT_COARSE: Final = ["사회", "경제", "문화", "기타"]
_YNAT_COARSE_BY_LABEL: Final = {
    "IT과학": "기타",
    "경제": "경제",
    "사회": "사회",
    "생활문화": "문화",
    "세계": "기타",
    "스포츠": "문화",
    "정치": "사회",
}
_YNAT_LABELS: Final = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
_NLI_CONTRADICTION_LABEL: Final = 2
_SUPPORTED: Final = {
    "e9t/nsmc",
    "klue/klue:ynat",
    "klue/klue:nli",
    "klue/klue:sts",
    "KorQuAD/squad_kor_v1",
    "smilegate-ai/kor_unsmile",
}


class FamilySummary(TypedDict):
    """Majority-class statistics for one OOD instruction family."""

    questions: int
    majority_gold: int
    majority_baseline: float


@dataclass(frozen=True, slots=True)
class InstructionOverlapError(AssertionError):
    """Raised when an OOD instruction is present in GOLD train."""

    overlap: list[str]

    @override
    def __str__(self) -> str:
        return f"OOD instructions overlap GOLD train: {self.overlap}"


class CliArgs(argparse.Namespace):
    """Typed command-line arguments."""

    gold: Path = Path("data/gold")
    out: Path = Path("data/ood")


def _question(
    kind: QuestionType, instructions: str, options: list[str], gold: int
) -> Question:
    return Question(type=kind, instructions=instructions, options=options, gold=gold)


def reversed_monotone_gold(gold: int, level_count: int) -> int:
    """Return the index for a reversed ordered scale."""
    return level_count - 1 - gold


def _nsmc_questions(example: Example) -> list[Question]:
    gold = example.questions[0].gold
    if gold is None:
        return []
    return [
        _question(
            QuestionType.NOUL, "이 리뷰는 긍정적이지 않다.", ["아니오", "예"], 1 - gold
        ),
        _question(QuestionType.SCORE, "리뷰의 만족도를 고르시오.", _NSMC_LEVELS, gold),
        _question(
            QuestionType.SCORE,
            "리뷰의 만족도를 반대 방향 척도로 고르시오.",
            _NSMC_REVERSED_LEVELS,
            reversed_monotone_gold(gold, len(_NSMC_REVERSED_LEVELS)),
        ),
    ]


def _ynat_questions(example: Example) -> list[Question]:
    gold = example.questions[0].gold
    if gold is None or not 0 <= gold < len(_YNAT_LABELS):
        return []
    coarse = _YNAT_COARSE_BY_LABEL[_YNAT_LABELS[gold]]
    return [
        _question(
            QuestionType.CHOICE,
            "기사의 큰 주제를 고르시오.",
            _YNAT_COARSE,
            _YNAT_COARSE.index(coarse),
        ),
        _question(
            QuestionType.NOUL,
            "경제 관련 기사다.",
            ["아니오", "예"],
            int(coarse == "경제"),
        ),
    ]


def _nli_questions(example: Example) -> list[Question]:
    gold = example.questions[0].gold
    if gold is None:
        return []
    contradiction = int(gold == _NLI_CONTRADICTION_LABEL)
    return [
        _question(
            QuestionType.NOUL,
            "전제와 가설은 모순된다.",
            ["아니오", "예"],
            contradiction,
        ),
        _question(
            QuestionType.NOUL,
            "전제와 가설은 모순되지 않는다.",
            ["아니오", "예"],
            1 - contradiction,
        ),
    ]


def _sts_questions(example: Example) -> list[Question]:
    gold = example.questions[0].gold
    if gold is None:
        return []
    return [
        _question(
            QuestionType.SCORE,
            "두 문장의 의미 유사도를 반대 방향 척도로 고르시오.",
            _STS_REVERSED_LEVELS,
            5 - gold,
        )
    ]


def _korquad_questions(example: Example) -> list[Question]:
    gold = example.questions[0].gold
    if gold is None:
        return []
    return [
        _question(
            QuestionType.NOUL,
            "이 지문은 질문과 무관하다.",
            ["아니오", "예"],
            1 - gold,
        )
    ]


def _unsmile_questions(example: Example) -> list[Question]:
    clean_gold = next(
        (
            question.gold
            for question in example.questions
            if question.instructions == "이 문장은 clean에 해당한다."
        ),
        None,
    )
    if clean_gold is None:
        return []
    return [
        _question(
            QuestionType.CHOICE,
            "혐오 여부를 고르시오.",
            ["혐오 없음", "혐오 있음"],
            0 if clean_gold == 1 else 1,
        )
    ]


_QUESTION_BUILDERS: Final[dict[str, Callable[[Example], list[Question]]]] = {
    "e9t/nsmc": _nsmc_questions,
    "klue/klue:ynat": _ynat_questions,
    "klue/klue:nli": _nli_questions,
    "klue/klue:sts": _sts_questions,
    "KorQuAD/squad_kor_v1": _korquad_questions,
    "smilegate-ai/kor_unsmile": _unsmile_questions,
}


def build_ood_examples(gold_dir: Path) -> list[Example]:
    """Build OOD examples from the GOLD test split only."""
    test_path = gold_dir / "test.jsonl"
    examples: list[Example] = []
    for source_example in read_jsonl(test_path):
        if source_example.split != "test" or source_example.source not in _SUPPORTED:
            continue
        questions = _QUESTION_BUILDERS[source_example.source](source_example)
        if questions:
            examples.append(
                Example(
                    state=source_example.state,
                    questions=questions,
                    source=source_example.source,
                    split="test",
                )
            )
    return sorted(examples, key=lambda example: example.model_dump_json())


def _instructions(examples: Iterable[Example]) -> set[str]:
    return {
        question.instructions for example in examples for question in example.questions
    }


def assert_no_train_instruction_overlap(train_path: Path, ood_path: Path) -> None:
    """Fail if any OOD instruction occurs in the GOLD train split."""
    train_instructions = _instructions(read_jsonl(train_path))
    ood_instructions = _instructions(read_jsonl(ood_path))
    overlap = sorted(train_instructions & ood_instructions)
    if overlap:
        raise InstructionOverlapError(overlap)


def _summary(examples: Iterable[Example]) -> dict[str, FamilySummary]:
    values: dict[str, list[int]] = defaultdict(list)
    for example in examples:
        for question in example.questions:
            if question.gold is not None:
                values[question.instructions].append(question.gold)
    result: dict[str, FamilySummary] = {}
    for family, golds in sorted(values.items()):
        counts = Counter(golds)
        majority_gold, majority_count = counts.most_common(1)[0]
        result[family] = {
            "questions": len(golds),
            "majority_gold": majority_gold,
            "majority_baseline": majority_count / len(golds),
        }
    return result


def build_ood(
    gold_dir: Path, out_dir: Path
) -> tuple[list[Example], dict[str, FamilySummary]]:
    """Write OOD JSONL and family-level majority baselines."""
    examples = build_ood_examples(gold_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ood_path = out_dir / "test.jsonl"
    write_jsonl(ood_path, examples)
    assert_no_train_instruction_overlap(gold_dir / "train.jsonl", ood_path)
    summary = _summary(examples)
    summary_payload = {
        "states": len(examples),
        "questions": sum(len(example.questions) for example in examples),
        "families": sorted({example.source for example in examples}),
        "majority_class_baseline": summary,
    }
    _ = (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return examples, summary


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rule-based Korean OOD questions."
    )
    _ = parser.add_argument("--gold", type=Path, default=Path("data/gold"))
    _ = parser.add_argument("--out", type=Path, default=Path("data/ood"))
    args = parser.parse_args(namespace=CliArgs())
    examples, _ = build_ood(args.gold, args.out)
    _ = sys.stdout.write(
        json.dumps(
            {
                "states": len(examples),
                "questions": sum(len(example.questions) for example in examples),
                "families": sorted({example.source for example in examples}),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    _ = sys.exit(_main())

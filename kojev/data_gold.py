"""Build the Korean human-labeled gold corpus from Hugging Face datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final, TypedDict

from datasets import load_dataset
from pydantic import ValidationError

from kojev.schema import (
    Example,
    JsonValue,
    Question,
    QuestionType,
    SchemaError,
    write_jsonl,
)

RowValue = (
    str | int | float | bool | list[str] | list[int] | Mapping[str, str | int | float]
)
Row = Mapping[str, RowValue]

_CHOICE_POLARITY: Final = ["부정", "긍정"]
_NLI_OPTIONS: Final = ["함의", "중립", "모순"]
_STS_OPTIONS: Final = [
    "전혀 다름",
    "매우 다름",
    "다소 다름",
    "비슷함",
    "매우 비슷함",
    "완전히 같음",
]
_UNSMILE_CATEGORIES: Final = [
    "여성/가족 혐오",
    "남성 혐오",
    "성소수자 혐오",
    "인종/국적 혐오",
    "연령 혐오",
    "지역 혐오",
    "종교 혐오",
    "기타 혐오",
    "악플/욕설",
    "clean",
]
_UNSMILE_KEYS: Final = {
    "여성/가족 혐오": "여성/가족",
    "남성 혐오": "남성",
    "성소수자 혐오": "성소수자",
    "인종/국적 혐오": "인종/국적",
    "연령 혐오": "연령",
    "지역 혐오": "지역",
    "종교 혐오": "종교",
    "기타 혐오": "기타 혐오",
    "악플/욕설": "악플/욕설",
    "clean": "clean",
}
_KOTE_EMOTIONS: Final = [
    "기쁨",
    "즐거움",
    "슬픔",
    "분노",
    "불안",
    "사랑",
    "감사",
    "중립",
]
_KOTE_COARSE: Final = ["기쁨", "슬픔", "분노", "중립"]
_KOTE_LABELS: Final = [
    "불평/불만",
    "환영/호의",
    "감동/감탄",
    "지긋지긋",
    "고마움",
    "슬픔",
    "화남/분노",
    "존경",
    "기대감",
    "우쭐댐/무시함",
    "안타까움/실망",
    "비장함",
    "의심/불신",
    "뿌듯함",
    "편안/쾌적",
    "신기함/관심",
    "아껴주는",
    "부끄러움",
    "공포/무서움",
    "절망",
    "한심함",
    "역겨움/징그러움",
    "짜증",
    "어이없음",
    "없음",
    "패배/자기혐오",
    "귀찮음",
    "힘듦/지침",
    "즐거움/신남",
    "깨달음",
    "죄책감",
    "증오/혐오",
    "흐뭇함(귀여움/예쁨)",
    "당황/난처",
    "경악",
    "부담/안_내킴",
    "서러움",
    "재미없음",
    "불쌍함/연민",
    "놀람",
    "행복",
    "불안/걱정",
    "기쁨",
    "안심/신뢰",
]
_SPLIT_CAPS: Final = {"train": 8000, "val": 500, "test": 1000}
_SPLIT_SEED: Final = 0
_YNAT_POLITICS_LABEL: Final = 6


class CliArgs(argparse.Namespace):
    """Typed command-line arguments."""

    out: Path = Path("data/gold")
    validate: Path | None = None


class SourceConfig(TypedDict):
    """Configuration for one Hugging Face source and its source splits."""

    source: str
    config: str | None
    splits: tuple[str, ...]


SOURCES: Final[tuple[SourceConfig, ...]] = (
    {"source": "e9t/nsmc", "config": None, "splits": ("train", "test")},
    {"source": "klue/klue", "config": "ynat", "splits": ("train", "validation")},
    {"source": "klue/klue", "config": "nli", "splits": ("train", "validation")},
    {"source": "klue/klue", "config": "sts", "splits": ("train", "validation")},
    {"source": "kakaobrain/kor_nli", "config": "multi_nli", "splits": ("train",)},
    {"source": "kakaobrain/kor_nli", "config": "snli", "splits": ("train",)},
    {
        "source": "smilegate-ai/kor_unsmile",
        "config": None,
        "splits": ("train", "valid"),
    },
    {
        "source": "jeanlee/kmhas_korean_hate_speech",
        "config": None,
        "splits": ("train",),
    },
    {"source": "searle-j/kote", "config": None, "splits": ("train", "validation")},
    {"source": "wicho/kor_3i4k", "config": None, "splits": ("train",)},
    {"source": "lawcompany/KLAID", "config": None, "splits": ("train",)},
    {"source": "KorQuAD/squad_kor_v1", "config": None, "splits": ("train",)},
)


def _text(row: Row, key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        msg = f"{key} must be text"
        raise TypeError(msg)
    return value


def _integer(row: Row, key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise TypeError(msg)
    return value


def _question(
    kind: QuestionType,
    instructions: str,
    options: list[str],
    gold: int,
    **meta: JsonValue,
) -> Question:
    return Question(
        type=kind,
        instructions=instructions,
        options=options,
        gold=gold,
        meta=dict(meta),
    )


def _example(state: str, questions: list[Question], source: str, split: str) -> Example:
    return Example(state=state, questions=questions, source=source, split=split)


def map_nsmc_row(row: Row, *, split: str) -> Example:
    """Map an NSMC review to polarity choice and positive sentiment noul."""
    gold = _integer(row, "label")
    state = _text(row, "document")
    return _example(
        state,
        [
            _question(
                QuestionType.CHOICE, "리뷰의 극성을 고르시오.", _CHOICE_POLARITY, gold
            ),
            _question(
                QuestionType.NOUL, "이 리뷰는 긍정적이다.", ["아니오", "예"], gold
            ),
        ],
        "e9t/nsmc",
        split,
    )


def map_ynat_row(row: Row, *, split: str) -> Example:
    """Map a KLUE YNAT headline to topic and political-news questions."""
    options = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
    gold = _integer(row, "label")
    return _example(
        _text(row, "title"),
        [
            _question(QuestionType.CHOICE, "기사의 주제를 고르시오.", options, gold),
            _question(
                QuestionType.NOUL,
                "정치 기사이다.",
                ["아니오", "예"],
                int(gold == _YNAT_POLITICS_LABEL),
            ),
        ],
        "klue/klue:ynat",
        split,
    )


def map_nli_row(row: Row, *, split: str, source: str = "klue/klue:nli") -> Example:
    """Map a three-way natural-language-inference row."""
    state = f"전제: {_text(row, 'premise')}\n가설: {_text(row, 'hypothesis')}"
    gold = _integer(row, "label")
    return _example(
        state,
        [
            _question(
                QuestionType.CHOICE,
                "전제와 가설의 관계를 고르시오.",
                _NLI_OPTIONS,
                gold,
            ),
            _question(
                QuestionType.NOUL,
                "가설은 전제에 의해 함의된다.",
                ["아니오", "예"],
                int(gold == 0),
            ),
        ],
        source,
        split,
    )


def map_sts_row(row: Row, *, split: str) -> Example:
    """Map KLUE STS similarity labels to ordered score and noul questions."""
    labels = row["labels"]
    if not isinstance(labels, Mapping):
        msg = "labels must be a mapping"
        raise TypeError(msg)
    real = labels["real-label"]
    binary = labels["binary-label"]
    if (
        not isinstance(real, (int, float))
        or isinstance(real, bool)
        or not isinstance(binary, int)
    ):
        msg = "STS labels have an invalid shape"
        raise TypeError(msg)
    return _example(
        f"{_text(row, 'sentence1')}\n{_text(row, 'sentence2')}",
        [
            _question(
                QuestionType.SCORE,
                "두 문장의 의미 유사도를 고르시오.",
                _STS_OPTIONS,
                round(real),
            ),
            _question(
                QuestionType.NOUL,
                "두 문장은 사실상 같은 의미다.",
                ["아니오", "예"],
                binary,
            ),
        ],
        "klue/klue:sts",
        split,
    )


def build_korquad_pair(
    row: Row,
    *,
    split: str,
    negative_question: str | None = None,
) -> Example:
    """Build a KorQuAD answerability noul, flipping gold for a negative question."""
    question = _text(row, "question")
    # The asked question is the substituted one for a negative pair; reusing the
    # row's own question made every negative a duplicate of its positive.
    asked = question if negative_question is None else negative_question
    # Answerable only when no substitution happened, or the substitute happens
    # to coincide with this row's own question.
    gold = int(negative_question is None or question == negative_question)
    state = f"지문: {_text(row, 'context')}\n질문: {asked}"
    return _example(
        state,
        [
            _question(
                QuestionType.NOUL,
                "이 지문으로 질문에 답할 수 있다.",
                ["아니오", "예"],
                gold,
                question_id=_text(row, "id"),
            )
        ],
        "KorQuAD/squad_kor_v1",
        split,
    )


def map_unsmile_row(row: Row, *, split: str) -> Example:
    """Map active UnSmile categories to nouls and a single-label choice."""
    state = _text(row, "문장")
    active = [
        category
        for category in _UNSMILE_CATEGORIES
        if (
            _UNSMILE_KEYS[category] in row
            and _integer(row, _UNSMILE_KEYS[category]) == 1
        )
        or (category in row and _integer(row, category) == 1)
    ]
    questions = [
        _question(
            QuestionType.NOUL,
            f"이 문장은 {category}에 해당한다.",
            ["아니오", "예"],
            1,
        )
        for category in active
    ]
    hate_active = [category for category in active if category != "clean"]
    if not questions:
        questions.append(
            _question(
                QuestionType.NOUL,
                "이 문장은 혐오 표현이 아니다.",
                ["아니오", "예"],
                1,
            )
        )
    if len(hate_active) == 1:
        questions.append(
            _question(
                QuestionType.CHOICE,
                "주된 혐오 유형을 고르시오.",
                _UNSMILE_CATEGORIES[:-1],
                _UNSMILE_CATEGORIES[:-1].index(hate_active[0]),
            )
        )
    return _example(state, questions, "smilegate-ai/kor_unsmile", split)


def map_kmhas_row(row: Row, *, split: str) -> Example:
    """Map the nine K-MHaS multi-label columns to noul questions."""
    labels = row["label"] if "label" in row else row["labels"]
    if not isinstance(labels, list) or any(
        isinstance(label, bool) or not isinstance(label, int) for label in labels
    ):
        msg = "K-MHaS labels must be integer list"
        raise TypeError(msg)
    integer_labels = [int(index in labels) for index in range(9)]
    categories = [f"혐오 유형 {index + 1}" for index in range(9)]
    return _example(
        _text(row, "text"),
        [
            _question(
                QuestionType.NOUL,
                f"이 문장은 {category}에 해당한다.",
                ["아니오", "예"],
                label,
            )
            for category, label in zip(categories, integer_labels, strict=True)
        ],
        "jeanlee/kmhas_korean_hate_speech",
        split,
    )


def map_kote_row(row: Row, *, split: str) -> Example:
    """Map KOTE multi-label emotions to curated nouls and a coarse choice."""
    labels = row["labels"]
    if not isinstance(labels, list):
        msg = "KOTE labels must be a list"
        raise TypeError(msg)
    label_names = [
        label if isinstance(label, str) else _KOTE_LABELS[label] for label in labels
    ]
    questions = [
        _question(
            QuestionType.NOUL,
            f"이 문장은 {emotion} 감정이다.",
            ["아니오", "예"],
            int(emotion in label_names),
        )
        for emotion in _KOTE_EMOTIONS
    ]
    coarse = "중립"
    if label_names:
        first = label_names[0]
        coarse = (
            "기쁨"
            if first in {"기쁨", "즐거움/신남", "행복", "안심/신뢰"}
            else "슬픔"
            if first in {"슬픔", "불안/걱정", "절망"}
            else "분노"
            if first in {"화남/분노", "증오/혐오", "짜증"}
            else "중립"
        )
    questions.append(
        _question(
            QuestionType.CHOICE,
            "가장 가까운 감정 범주를 고르시오.",
            _KOTE_COARSE,
            _KOTE_COARSE.index(coarse),
        )
    )
    return _example(_text(row, "text"), questions, "searle-j/kote", split)


Mapper = Callable[[Row, str], Example]
_SIMPLE_MAPPERS: dict[str, Mapper] = {
    "e9t/nsmc": lambda row, split: map_nsmc_row(row, split=split),
    "klue/klue:ynat": lambda row, split: map_ynat_row(row, split=split),
    "klue/klue:nli": lambda row, split: map_nli_row(row, split=split),
    "klue/klue:sts": lambda row, split: map_sts_row(row, split=split),
    "KorQuAD/squad_kor_v1": lambda row, split: build_korquad_pair(row, split=split),
    "smilegate-ai/kor_unsmile": lambda row, split: map_unsmile_row(row, split=split),
    "jeanlee/kmhas_korean_hate_speech": lambda row, split: map_kmhas_row(
        row, split=split
    ),
    "searle-j/kote": lambda row, split: map_kote_row(row, split=split),
}


def _is_blank_state_error(error: ValidationError) -> bool:
    """Report whether a validation failure is the blank-state invariant."""
    return SchemaError.blank_state().reason in str(error)


def _korquad_examples(
    row: Row, split: str, negative_question: str | None
) -> list[Example]:
    """Build one KorQuAD pair, dropping rows whose upstream text is blank."""
    try:
        return [
            build_korquad_pair(row, split=split, negative_question=negative_question)
        ]
    except ValidationError as error:
        if _is_blank_state_error(error):
            return []
        raise


def _mapped_examples(source: str, row: Row, split: str) -> list[Example]:
    """Map one row, dropping rows whose upstream text is blank."""
    try:
        return _row_to_examples(source, row, split)
    except ValidationError as error:
        if _is_blank_state_error(error):
            return []
        raise


def _row_to_examples(source: str, row: Row, split: str) -> list[Example]:
    """Dispatch one trusted dataset row to its deterministic mapper."""
    mapper = _SIMPLE_MAPPERS.get(source)
    if mapper is not None:
        return [mapper(row, split)]
    if source.startswith("kakaobrain/kor_nli:"):
        return [map_nli_row(row, split=split, source=source)]
    if source == "wicho/kor_3i4k":
        options = ["예약", "문의", "취소", "변경", "결제", "불만", "기타"]
        gold = _integer(row, "label")
        text = _text(row, "text")
        return [
            _example(
                text,
                [
                    _question(
                        QuestionType.CHOICE, "문장의 의도를 고르시오.", options, gold
                    ),
                    _question(
                        QuestionType.NOUL,
                        "질문이다.",
                        ["아니오", "예"],
                        int(text.endswith("?")),
                    ),
                ],
                source,
                split,
            )
        ]
    if source in {"lawcompany/KLAID", "lawcompany/KLAID:ljp"}:
        gold = _integer(row, "laws_service_id")
        options = [f"법조 서비스 {index}" for index in range(1, 178)]
        laws = _text(row, "laws_service")
        return [
            _example(
                _text(row, "fact"),
                [
                    _question(
                        QuestionType.CHOICE, "적용되는 법조를 고르시오.", options, gold
                    ),
                    _question(
                        QuestionType.NOUL,
                        "형법 조항이 적용된다.",
                        ["아니오", "예"],
                        int(laws.startswith("형법")),
                    ),
                ],
                source,
                split,
            )
        ]
    msg = f"unsupported gold source: {source}"
    raise ValueError(msg)


def _source_name(config: SourceConfig) -> str:
    return (
        config["source"]
        if config["config"] is None
        else f"{config['source']}:{config['config']}"
    )


def _iter_source_rows(config: SourceConfig, split: str) -> Iterator[Row]:
    kwargs = {} if config["config"] is None else {"name": config["config"]}
    dataset = load_dataset(config["source"], split=split, **kwargs)
    yield from dataset


def _shuffle_key(row: Row) -> str:
    """Order rows deterministically under the split seed, stable across runtimes."""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(f"{_SPLIT_SEED}:{payload}".encode())
    return digest.hexdigest()


def _source_split_rows(config: SourceConfig) -> dict[str, list[Row]]:
    """Pool a source's rows and carve deterministic per-source train/val/test.

    Every source contributes to all three splits: the gold test set must span
    sources rather than only those datasets that ship an upstream test split.
    """
    rows: list[Row] = []
    for hf_split in config["splits"]:
        rows.extend(_iter_source_rows(config, hf_split))
    rows.sort(key=_shuffle_key)
    test_end = _SPLIT_CAPS["test"]
    val_end = test_end + _SPLIT_CAPS["val"]
    train_end = val_end + _SPLIT_CAPS["train"]
    return {
        "test": rows[:test_end],
        "val": rows[test_end:val_end],
        "train": rows[val_end:train_end],
    }


def _enforce_split_state_isolation(
    buckets: dict[str, list[Example]],
    per_source_mapped: dict[str, list[Example]],
    summary: dict[str, dict[str, int | str]],
) -> None:
    """Guarantee a state string never appears in more than one split.

    Upstream corpora contain duplicate texts, so row-level carving can place the
    same state on both sides of the train/test boundary and silently contaminate
    held-out evaluation. Held-out splits win: test keeps every state it holds,
    val yields to test, and train yields to both.
    """
    reserved: set[str] = set()
    kept_states: dict[str, set[str]] = {}
    for split in ("test", "val", "train"):
        kept = [example for example in buckets[split] if example.state not in reserved]
        buckets[split] = kept
        kept_states[split] = {example.state for example in kept}
        reserved |= kept_states[split]
    for key, mapped in per_source_mapped.items():
        split = key.rsplit(":", 1)[1]
        allowed = kept_states[split]
        survivors = [example for example in mapped if example.state in allowed]
        entry = summary[key]
        removed = len(mapped) - len(survivors)
        entry["states"] = len({example.state for example in survivors})
        entry["questions"] = sum(len(x.questions) for x in survivors)
        entry["dropped_cross_split_state"] = removed


def build_gold(out: Path) -> dict[str, dict[str, int | str]]:
    """Download, map, cap, and write the deterministic gold corpus."""
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    per_source_mapped: dict[str, list[Example]] = {}
    summary: dict[str, dict[str, int | str]] = {}
    for config in SOURCES:
        source_name = _source_name(config)
        for split, selected in _source_split_rows(config).items():
            dropped = 0
            if source_name == "KorQuAD/squad_kor_v1":
                mapped: list[Example] = []
                for index, row in enumerate(selected):
                    positive = _korquad_examples(row, split, None)
                    dropped += 1 - len(positive)
                    mapped.extend(positive)
                    negative = next(
                        (
                            candidate
                            for candidate in selected[index + 1 :]
                            if candidate.get("title") != row.get("title")
                        ),
                        None,
                    )
                    if negative is not None:
                        mapped.extend(
                            _korquad_examples(row, split, _text(negative, "question"))
                        )
            else:
                mapped = []
                for row in selected:
                    examples = _mapped_examples(source_name, row, split)
                    if not examples:
                        dropped += 1
                    mapped.extend(examples)
            buckets[split].extend(mapped)
            per_source_mapped[f"{source_name}:{split}"] = mapped
            summary[f"{source_name}:{split}"] = {
                "states": len(selected) - dropped,
                "questions": sum(len(x.questions) for x in mapped),
                "dropped_blank_state": dropped,
            }
    _enforce_split_state_isolation(buckets, per_source_mapped, summary)
    for split, examples in buckets.items():
        ordered = sorted(
            examples,
            key=lambda example: example.model_dump_json(),
        )
        write_jsonl(out / f"{split}.jsonl", ordered)
    _ = (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_jsonl(path: Path) -> list[str]:
    """Return deterministic line-numbered errors from a schema JSONL file."""
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                _ = Example.model_validate_json(line)
            except json.JSONDecodeError:
                errors.append(f"{path}:{line_number}: invalid JSON")
            except ValidationError as error:
                first_error = error.errors()[0]
                if first_error["type"] == "json_invalid":
                    errors.append(f"{path}:{line_number}: invalid JSON")
                else:
                    errors.append(f"{path}:{line_number}: {first_error['msg']}")
    return errors


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the Korean gold corpus."
    )
    _ = parser.add_argument("--out", type=Path, default=Path("data/gold"))
    _ = parser.add_argument("--validate", type=Path)
    args = parser.parse_args(namespace=CliArgs())
    if args.validate is not None:
        paths = sorted(args.validate.glob("*.jsonl"))
        errors = [error for path in paths for error in validate_jsonl(path)]
        for error in errors:
            _ = sys.stdout.write(f"{error}\n")
        return int(bool(errors))
    _ = build_gold(args.out)
    return 0


if __name__ == "__main__":
    _ = sys.exit(_main())

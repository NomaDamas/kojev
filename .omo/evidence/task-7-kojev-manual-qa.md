# Manual QA Matrix — KoJev Todo 7 Augmentation

## Surface evidence

| Scenario | Criterion reference | Surface | Exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|---|
| S1 | choice shuffle/remap | Python data-shaped driver | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run python - <<'PY' ... PY` with 1,000 seeded choice questions and `shuffle_options`; assert original gold option equals transformed gold option | PASS | `A1`, `A2` |
| S2 | >=8 Korean templates and final-consonant josa | Python data-shaped driver | Same driver invocation; 128 seeded calls each for `감정` and `주제`; assert 은/는 and 이/가 outputs plus `PARAPHRASE_TEMPLATES`/`INSTRUCTION_TEMPLATES` coverage through pytest | PASS | `A1`, `A2` |
| S3 | distractor drop never gold | Python data-shaped driver | Same driver invocation; 1,000 seeded choice questions; assert original gold option remains in dropped options; deliberate broken fixture must raise assertion | PASS | `A1`, `A2` |
| S4 | ordered score synonyms | Python data-shaped driver | Same driver invocation; 1,000 seeded score questions; assert gold index unchanged and option order unchanged by `swap_score_synonyms` | PASS | `A1`, `A2` |
| S5 | exact noul negation flip | Python data-shaped driver | Same driver invocation; assert `이 리뷰는 긍정적이다` becomes `이 리뷰는 긍정적이 아니다` and gold `1` becomes `0` | PASS | `A1`, `A2` |
| S6 | integrated augmentation and regression suite | Python test surface | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run pytest -q` | PASS | `A1`, `A3` |
| S7 | static quality gates | Python project tooling | `uv run ruff check kojev/augment.py tests/test_augment.py && uv run basedpyright kojev/augment.py tests/test_augment.py` | PASS | `A1`, `A4` |

## Adversarial cases

| Scenario | Criterion reference | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| ADV1 | choice shuffle/remap | repeated deterministic seeds | Same seeded invocation preserves the semantic gold option, regardless of permutation | PASS | `A2` |
| ADV2 | distractor drop never gold | negative regression probe | A candidate with the actual gold option removed must fail the invariant assertion; product must never produce that state | PASS | `A2`, `A3` |
| ADV3 | final-consonant josa | Korean vowel/consonant boundary | `감정` selects 은/이 and `주제` selects 는/가 across generated templates | PASS | `A2` |
| ADV4 | exact noul negation flip | already binary gold at both labels | Negation changes gold exactly from 0 to 1 or 1 to 0 and preserves the exact expected Korean wording | PASS | `A2`, `A3` |
| ADV5 | score synonyms | unknown/non-score labels | Unmapped options remain in their original positions and gold index is unchanged | PASS | `A1`, `A3` |
| ADV6 | schema boundary | invalid option cardinality | Schema rejects invalid choice input before augmentation; existing regression test asserts `ValueError` | PASS | `A3` |
| ADV7 | browser UI / HTTP / desktop GUI | not applicable | This change is a pure Python data-transform module with no HTTP, browser, terminal UI, or desktop GUI surface | NOT_APPLICABLE — no such surface exists | `A1` |

## Artifact references

| ID | Kind | Description | Path |
|---|---|---|---|
| A1 | evidence log | RED baseline, test/lint/type results, LOC, and driver summary | `/Volumes/SSD1/1/KoJev-todo-7/.omo/evidence/task-7-kojev.txt` |
| A2 | runtime transcript | Real deterministic 1000-example driver output: `PROPERTY_1000=1000/1000`, josa, noul, and negative probe | `/Volumes/SSD1/1/KoJev-todo-7/.omo/evidence/task-7-kojev.txt` |
| A3 | test transcript | `uv run pytest -q` output: 24 passed | terminal output captured in A1 | `/Volumes/SSD1/1/KoJev-todo-7/.omo/evidence/task-7-kojev.txt` |
| A4 | static-check transcript | Ruff and basedpyright success output | terminal output captured in A1 | `/Volumes/SSD1/1/KoJev-todo-7/.omo/evidence/task-7-kojev.txt` |

# Todo 3 Corrective Manual QA

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Corrective static quality gates | Ruff + basedpyright | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/ruff check . && .venv/bin/basedpyright` | PASS - Ruff clean; basedpyright 0 errors, 0 warnings | A1 |
| S2 | Regression safety | Pytest | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/pytest tests/test_data_gold.py tests/test_schema.py -q && .venv/bin/pytest -q` | PASS - target and full suites each pass; 13 tests | A1 |
| S3 | Build acceptance: 12 configured sources and emitted corpus | Real CLI + cached Hugging Face datasets | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --out /tmp/kojev-gold-corrective` | PASS - exit 0; 12 sources; 96,000 train states; 103,999 train JSONL examples; 2,500 val; 1,000 test | A1 |
| S4 | Clean JSONL validation | Real CLI validation | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --validate /tmp/kojev-gold-corrective` | PASS - exit 0 with no diagnostics | A1 |
| S5 | Forbidden-source and split exclusions | Generated summary inspection | `uv run python - <<'PY' ... summary.json ... PY` after S3 | PASS - `kobest_entries=[]`; KorQuAD appears only as `KorQuAD/squad_kor_v1:train` | A1 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | Validation failure localization | Malformed JSON | Replacing JSONL line 2 with `not json` returns nonzero and names exact file and line | PASS - exit 1; `...train.jsonl:2: invalid JSON` | A1 |
| A2 | Typed non-empty invariant | Schema-invalid empty questions | Replacing JSONL line 2 `questions` with `[]` returns nonzero and reports the invariant | PASS - exit 1; line 2 reports `examples require at least one question (minimum 1)` | A1 |
| A3 | Builder robustness under real source data | Empty-label source row | Rows that map to no source-specific questions receive a deterministic fallback question; build completes | PASS - real 12-source build exit 0 | A1 |
| A4 | Contamination exclusion | Forbidden-source exclusion | KoBEST must not occur in summary or outputs | PASS - measured `kobest_entries=[]` | A1 |
| A5 | Evaluation split exclusion | Split contamination | KorQuAD must be train-only | PASS - exact summary entry is `KorQuAD/squad_kor_v1:train` | A1 |
| A6 | Evidence integrity | Count mismatch | Evidence must distinguish train states from emitted JSONL examples | PASS - evidence records 96,000 train states and 103,999 train JSONL lines | A1 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | text | Corrective static checks, target/full tests, real build counts, clean validation, malformed JSON failure, empty-question failure, exclusions, and cleanup record | `.omo/evidence/task-3-corrective-qa.txt` |

## cleanup

Temporary build, corruption, and transcript files under `/tmp/kojev-gold-corrective*` were removed after evidence capture. The repository evidence file remains.

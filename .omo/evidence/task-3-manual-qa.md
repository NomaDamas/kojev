# Todo 3 Manual QA Matrix

Date: 2026-09-19
Worktree: `/Volumes/SSD1/1/KoJev-todo3`
Branch: `feat/data-gold`
HEAD before cleanup: `6947a0fb5422eee694c1ba49d8095a38d63772a9`
origin/feat/data-gold before cleanup: `6947a0fb5422eee694c1ba49d8095a38d63772a9`

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Real deterministic build A | Python CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --out /tmp/kojev-todo3-qa-a` | PASS for execution only: exit 0 and all output files generated from cached Hugging Face sources; acceptance blocked by blank-state scan | QA1 |
| S2 | Real deterministic build B | Python CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --out /tmp/kojev-todo3-qa-b` | PASS for execution only: exit 0 and all output files generated; acceptance blocked by the same blank-state scan | QA1 |
| S3 | Byte-identical repeated outputs | Shell artifact comparison | `sha256sum ...; cmp -s /tmp/kojev-todo3-qa-a/<file> /tmp/kojev-todo3-qa-b/<file>` for `summary.json`, `train.jsonl`, `val.jsonl`, `test.jsonl` | PASS: all four pairs have identical SHA-256 and `cmp` exit 0 | QA1 |
| S4 | Exact emitted counts | Shell CLI | `wc -l /tmp/kojev-todo3-qa-a/{train,val,test}.jsonl /tmp/kojev-todo3-qa-b/{train,val,test}.jsonl` | PASS: both builds are train 103,999, val 2,500, test 1,000 | QA1 |
| S5 | Clean validation | Python CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --validate /tmp/kojev-todo3-qa-a` and same for `qa-b` | FAIL: both exit 0 despite 8 blank `state` values in each corpus; validation does not enforce non-blank state | QA1 |
| S6 | Target and full regression suites | Pytest CLI | `uv run pytest tests/test_data_gold.py tests/test_schema.py -q`; `uv run pytest -q` | PASS: 13 passed in each run | QA1 |
| S7 | Static quality gates | Ruff and basedpyright CLI | `uv run ruff check .`; `uv run basedpyright` | PASS: Ruff clean; basedpyright 0 errors, 0 warnings, 0 notes | QA1 |
| S8 | Source exclusions | Parsed generated summary | Python summary inspection in QA1 transcript | PASS: `kobest_entries=[]`; KorQuAD entry is only `KorQuAD/squad_kor_v1:train` | QA1 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | Blank-state correction | Real source blank-state scan | No emitted example has a non-string or whitespace-only `state`; exact count must be zero | FAIL: both builds have exactly 8 blank NSMC states: train lines 2310-2314 and test lines 5-7; all are `state: ""` | QA1 |
| A2 | Malformed JSON corruption | Malformed JSON / line localization | Validation exits nonzero and identifies exact file and line | PASS: `/tmp/kojev-todo3-corrupt-json/train.jsonl:2: invalid JSON`, exit 1 | QA1 |
| A3 | Empty-question corruption | Schema invariant | Validation exits nonzero and reports minimum-question invariant | PASS: `/tmp/kojev-todo3-corrupt-empty/train.jsonl:2: Value error, examples require at least one question (minimum 1)`, exit 1 | QA1 |
| A4 | Determinism | Repeated real build | Outputs are byte-identical across two runs | PASS: all four output files byte-identical | QA1 |
| A5 | Forbidden-source exclusion | Contamination exclusion | KoBEST is absent from summary/output corpus | PASS: no KoBEST summary entries | QA1 |
| A6 | Evaluation split exclusion | Split contamination | KorQuAD is train-only | PASS: only `KorQuAD/squad_kor_v1:train` summary entry | QA1 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| QA1 | terminal transcript | Two real cached-source builds, hashes, exact line counts, parsed blank-state details, validation exits, source exclusions, corruption failures, pytest, Ruff, and basedpyright | `.omo/evidence/task-3-blank-state-builds.log` |
| QA2 | prior external audit | Preserved independent adversarial audit artifact; not modified or deleted | `.omo/evidence/task-3-adversarial-audit.txt` |
| QA3 | prior external audit matrix | Preserved independent adversarial manual QA artifact; not modified or deleted | `.omo/evidence/task-3-adversarial-manual-qa.md` |
| QA4 | committed corrective evidence | Existing corrective evidence reviewed against fresh runtime evidence; it does not prove the blank-state criterion | `.omo/evidence/task-3-corrective-qa.txt` |

## Cleanup

- Removed only task-generated `.debug-journal.md` and fresh temporary build/corruption directories under `/tmp/kojev-todo3-*` after evidence capture.
- Preserved external audit files `task-3-adversarial-audit.txt` and `task-3-adversarial-manual-qa.md`.
- No product source or tests were modified by this QA run.
- Push/SHA status: blocked. No corrective commit was created or pushed because the blank-state acceptance criterion failed; local and remote remain at `6947a0fb5422eee694c1ba49d8095a38d63772a9`.

## Verdict

**FAIL / BLOCKED.** The current pushed commit is reproducible and deterministic, and all non-blank-state gates pass, but it still emits exactly 8 blank NSMC states and its validation accepts them. No success claim or corrective push is justified.

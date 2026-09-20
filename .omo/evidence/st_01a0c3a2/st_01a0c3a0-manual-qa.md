# KoJev todo 6 manual QA

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| qa-6-01 | model acceptance: target tests, grouped probabilities, overfit | CPU pytest in preserved worktree `/Volumes/SSD1/1/KoJev-task6` | `cd /Volumes/SSD1/1/KoJev-task6 && .venv/bin/pytest tests/test_encoder.py -q` | PASS | `art-task6-evidence` |
| qa-6-02 | repository regression safety | CPU pytest full suite | `cd /Volumes/SSD1/1/KoJev-task6 && .venv/bin/pytest -q` | PASS | `art-task6-evidence` |
| qa-6-03 | lint/format gate | Ruff CLI | `cd /Volumes/SSD1/1/KoJev-task6 && .venv/bin/ruff check . && .venv/bin/ruff format --check .` | PASS | `art-task6-evidence` |
| qa-6-04 | required static type gate | basedpyright CLI | `cd /Volumes/SSD1/1/KoJev-task6 && .venv/bin/basedpyright kojev tests` | PASS | `art-task6-evidence` |
| qa-6-05 | decide() mixed choice/score/noul schema-valid output | real minimal Python driver | `cd /Volumes/SSD1/1/KoJev-task6 && .venv/bin/python - <<'PY' ... model.decide(example, collator) ... PY` | PASS | `art-task6-evidence` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| adv-6-01 | schema guard: choice <=255 options | oversized choice (300 options) | construction rejects before collation/model execution with a 255-cardinality validation error | PASS | `art-task6-evidence` |
| adv-6-02 | no marker hidden-state head | marker leakage | pooled question/option spans contain no marker token ids; logits still have one scalar per option | PASS | `art-task6-evidence` |
| adv-6-03 | grouped softmax | multiple question groups | every question group sums to 1 independently, not across the flattened batch | PASS | `art-task6-evidence` |
| adv-6-04 | loss mechanics | tiny random ModernBERT CPU overfit | 16-example fixture reaches loss <0.01 within 200 optimizer steps | PASS | `art-task6-evidence` |
| adv-6-05 | no network/downloads in unit tests | download/network dependency | unit suite uses TinyTokenizer and a local ModernBertConfig; no model download is triggered | PASS | `art-task6-evidence` |
| adv-6-06 | static typing regression | transformer boundary/protocol and Torch module typing | basedpyright reports 0 errors, 0 warnings, 0 notes | PASS | `art-task6-evidence` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `art-task6-evidence` | consolidated transcript | Non-empty transcript containing prior RED, corrected static green, target/full pytest, Ruff, formatting, real mixed-kind driver, 300-option rejection, grouped softmax, CPU overfit, and cleanup/status evidence | `/Volumes/SSD1/1/KoJev-task6/.omo/evidence/task-6-kojev.txt` |
| `art-task6-matrix` | QA matrix | This manual QA matrix | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c3a2/st_01a0c3a0-manual-qa.md` |
| `art-task6-commit` | git receipt | Commit and push receipt for exact SHA `ccdfbbbfc664ac43c21e0ae5329f044e432fcbe4` | `/Volumes/SSD1/1/KoJev-task6/.git` |

## verdict

PASS. The 12 basedpyright errors were corrected without suppressions or weakened tests. All requested runtime, adversarial, lint, formatting, and static checks passed. Commit `ccdfbbbfc664ac43c21e0ae5329f044e432fcbe4` was pushed to `origin/feat/task-6-span-pooling`.

# Task 3 Manual QA Matrix

## surfaceEvidence

| Scenario | Criterion reference | Surface | Exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|---|
| S1 | Build acceptance: >=9 sources, >=40k train states, >=90k train questions; no KoBEST; KorQuAD train-only | CLI + real Hugging Face datasets | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --out /tmp/kojev-gold-qa` | PASS - real build exit 0; summary measured 12 sources, 96,000 train states, 302,085 train questions; no KoBEST entries; KorQuAD only `:train` | A1 |
| S2 | Build acceptance: emitted train/val/test JSONL and summary | Generated JSONL artifacts | `wc -l /tmp/kojev-gold-qa/*.jsonl` after S1 | PASS - train 96,000 lines, val 2,500 lines, test 1,000 lines; summary present and counted | A1 |
| S3 | QA happy path: `--validate` reparses every line with zero errors | CLI validation | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --validate /tmp/kojev-gold-qa` | PASS - exit 0 with no diagnostics | A1 |
| S4 | Target mapper/schema tests | Python test runner | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run pytest tests/test_data_gold.py tests/test_schema.py -q` | PASS - 12 passed | A1 |

## adversarialCases

| Scenario | Criterion reference | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| A1 | QA failure path: corrupt one JSONL line and prove line-numbered failure | Malformed JSON / line localization | Validation exits nonzero and identifies the corrupted file and exact line | PASS - `/tmp/kojev-gold-corrupt/train.jsonl:2: invalid JSON`, exit 1 | A1 |
| A2 | Must NOT have: KoBEST training contamination | Forbidden-source exclusion | Generated summary contains no KoBEST source entry | PASS - measured `kobest_entries=[]` | A1 |
| A3 | Must NOT have: KorQuAD dev exclusion | Split contamination | KorQuAD appears only as `KorQuAD/squad_kor_v1:train` in summary | PASS - measured exact entry list | A1 |
| A4 | Static quality gate | Static analysis | Ruff and basedpyright should pass if implementation is clean | FAIL - ruff and basedpyright both report findings; details recorded in A1, not suppressed | A1 |

## artifactRefs

| ID | Kind | Description | Path |
|---|---|---|---|
| A1 | text | Complete command transcript, measured real counts, clean validation result, corruption result, test result, static-check failures, and cleanup record | `.omo/evidence/task-3-kojev.txt` |

## Cleanup

The temporary clean corpus, corrupted copy, build log, and debug journal were removed after evidence capture. The required evidence files remain in `.omo/evidence/`.

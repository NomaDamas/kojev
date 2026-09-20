# Todo 4 Corrective Commit `af5551e` — Manual QA

## AdversarialVerify

```yaml
verdict: confirmed
safe_to_land: true
scope: todo 4 only
repository: /Volumes/SSD1/1/KoJev-todo4
branch: feat/todo-4-kobest-klue-bench
commit: af5551e9e3076b0b254e96765c4b3252a5808a0a
remote_sha: af5551e9e3076b0b254e96765c4b3252a5808a0a
base: a96c1707eda45317194af1005109f700d181a9b6
reason: >-
  The remote branch and worktree are at the claimed corrective commit. The
  targeted tests, Ruff, and basedpyright pass; the real random CLI produces a
  parsed 200-item report with all eight task names, positive counts, and exact
  metric keys; executable overlap and malformed training-manifest probes both
  fail before report creation; and cleanup leaves no benchmark report or
  QA-owned benchmark/test/validator process.
```

## Surface evidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | remote SHA, clean worktree, containment | Git repository | `git status --short --branch`; `git rev-parse HEAD`; `git ls-remote origin refs/heads/feat/todo-4-kobest-klue-bench`; `git diff --name-status a96c1707eda45317194af1005109f700d181a9b6 af5551e9e3076b0b254e96765c4b3252a5808a0a` | PASS: HEAD and remote both `af5551e...`; clean; corrective diff is evidence, `kojev/bench.py`, and `tests/test_bench.py` only | A1 |
| S2 | targeted benchmark regression suite | pytest | `uv run pytest tests/test_bench.py -q` | PASS: 9 passed | A2 |
| S3 | static quality | Ruff | `uv run ruff check kojev tests` | PASS: All checks passed | A3 |
| S4 | static type quality | basedpyright | `uv run basedpyright kojev tests` | PASS: 0 errors, 0 warnings, 0 notes | A4 |
| S5 | real runtime gate and report shape | CLI | `uv run python -m kojev.bench --model random --limit 200 --output /Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/benchmark-report-af5551e.json` | PASS: exit 0; parsed report has total 200, exactly eight tasks, count 25 each, positive counts, and exact keys `count`, `accuracy`, `macro_f1`, `brier`, `ece_15` | A5, A6 |
| S6 | default executable smoke path | CLI | `uv run python -m kojev.bench --model random --limit 200` | PASS: exit 0; parsed all eight tasks and kind counts choice=100, noul=75, score=25; generated report removed afterward | A7 |
| S7 | overlap training-manifest guard | CLI | `uv run python -m kojev.bench --model random --limit 8 --training-manifest /Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/overlap-af5551e.jsonl --output .../overlap-should-not-exist.json` | PASS: exit 1 with `AssertionError: contamination: KoBEST id 'boolq-0'`; report absent | A8 |
| S8 | malformed training-manifest guard | CLI | `uv run python -m kojev.bench --model random --limit 8 --training-manifest /Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/malformed-af5551e.jsonl --output .../malformed-should-not-exist.json` | PASS: exit 1 with `BenchmarkError: malformed manifest line 1`; report absent | A9 |
| S9 | no data/gold writes and process cleanup | filesystem/process | `find . -maxdepth 1 -name benchmark-report.json`; `pgrep -fal 'kojev\\.bench|python.*tests/test_bench.py|ruff check kojev tests|basedpyright kojev tests'` after cleanup | PASS: generated report absent; no QA-owned benchmark/test/ruff/basedpyright process remains; no `data/gold` path was created | A10 |

## Adversarial cases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| AC1 | runtime training-manifest guard | contamination / overlap | An actual KoBEST ID in the manifest must reject before report generation | PASS: CLI exit 1 with `contamination` assertion and no report | A8 |
| AC2 | runtime training-manifest guard | malformed input | Invalid JSONL must reject with line-specific `BenchmarkError` before report generation | PASS: CLI exit 1 with `malformed manifest line 1` and no report | A9 |
| AC3 | benchmark output contract | misleading success output | Exit 0 is insufficient; parsed output must prove all eight tasks, positive counts, and exact metric keys | PASS: report parse proves all conditions | A6, A7 |
| AC4 | deterministic random baseline | stale state / repeatability | Same state and seed should produce the same probabilities; targeted tests must pass | PASS: 9-test suite passes including repeatability regression | A2 |
| AC5 | filesystem safety | dirty/generated state | Benchmark may write its requested report, but cleanup must remove generated report and preserve repository cleanliness | PASS: final cleanup transcript shows no root report and clean worktree | A1, A10 |
| AC6 | long-running/process cleanup | leaked process | Real benchmark and validators must not leave QA-owned product or validator processes | PASS: final process probe has no matching QA-owned process | A10 |
| AC7 | static regression | code-quality drift | Ruff and basedpyright must remain clean on product and tests | PASS | A3, A4 |
| AC8 | cancellation/resume | not applicable | Not applicable: this change is a bounded one-shot CLI and has no resume/checkpoint contract | NOT_APPLICABLE — no resumable job/session is implemented |
| AC9 | repeated interruptions | not applicable | Not applicable: this change does not persist an interrupted benchmark session | NOT_APPLICABLE — no persistent interrupted-session state exists |

## Artifact references

| id | kind | description | path |
|---|---|---|---|
| A1 | command transcript | HEAD, remote branch SHA, clean worktree, and corrective commit boundary | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-final-cleanup.txt` |
| A2 | test transcript | Required targeted pytest run, 9 passed | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-pytest.txt` |
| A3 | lint transcript | Required Ruff run | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-ruff.txt` |
| A4 | type-check transcript | Required basedpyright run | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-basedpyright.txt` |
| A5 | CLI transcript | Exact real random-model invocation and exit status | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-cli-run.txt` |
| A6 | parsed data artifact | Parsed 200-item report: eight task names, counts, metric keys, kind counts | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-report-parse.txt` |
| A7 | CLI transcript | Default output-path smoke run with parsed contract and report cleanup | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-exact-cli.txt` |
| A8 | negative CLI transcript | Executable overlap-manifest rejection and no-report assertion | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-overlap-cli.txt` |
| A9 | negative CLI transcript | Executable malformed-manifest rejection and no-report assertion | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-malformed-cli.txt` |
| A10 | cleanup transcript | Final generated-file and process checks | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/af5551e-final-cleanup.txt` |

All PASS rows point to non-empty artifacts. The QA matrix is read-only with respect to `/Volumes/SSD1/1/KoJev-todo4`; only evidence files were written under `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/`.

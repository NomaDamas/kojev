# Manual QA Matrix — todo 14 evaluation harness

## Verdict

**FAIL / blocked.** The assigned implementation surface is absent from the requested worktree. No faithful runtime scenario can pass because `/Volumes/SSD1/1/KoJev-todo14/kojev/evaluate.py` and `/Volumes/SSD1/1/KoJev-todo14/tests/test_evaluate.py` do not exist. I did not infer behavior from source or mark skipped scenarios as passing.

The worktree was created as requested at `/Volumes/SSD1/1/KoJev-todo14` on branch `feat/todo-14-eval`, but remains a clean base checkout at commit `6cb0618` tracking `origin/main`. No implementation commit or pushed SHA was available to verify.

## Surface evidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| QA-14-01 | Required evaluation runner/report tables and metric keys | Python test surface | `cd /Volumes/SSD1/1/KoJev-todo14 && uv run pytest tests/test_evaluate.py -q` | **FAIL** — pytest reports `ERROR: file or directory not found: tests/test_evaluate.py`; no tests ran | `art-worktree-prerequisite`, `art-target-pytest` |
| QA-14-02 | Required missing/invalid checkpoint typed error | Python test surface | Intended invocation: `cd /Volumes/SSD1/1/KoJev-todo14 && uv run pytest tests/test_evaluate.py -q` | **FAIL / blocked** — target test module is absent, so the typed `EvaluationError` behavior cannot be exercised | `art-worktree-prerequisite`, `art-target-pytest` |
| QA-14-03 | Contamination failure on overlapping state and pass on clean input | Python test surface | Intended invocation: `cd /Volumes/SSD1/1/KoJev-todo14 && uv run pytest tests/test_evaluate.py -q` | **FAIL / blocked** — target test module and evaluator are absent; no runtime evidence | `art-worktree-prerequisite`, `art-target-pytest` |
| QA-14-04 | Exact hand-computable accuracy/Brier metrics | Python test surface | Intended invocation: `cd /Volumes/SSD1/1/KoJev-todo14 && uv run pytest tests/test_evaluate.py -q` | **FAIL / blocked** — no evaluator or target tests exist in the assigned worktree | `art-worktree-prerequisite`, `art-target-pytest` |
| QA-14-05 | Real latency fields and tiny-random-model CLI report | CLI surface | Intended invocation unavailable because no evaluator CLI entry point exists; prerequisite check: `test -f /Volumes/SSD1/1/KoJev-todo14/kojev/evaluate.py` | **FAIL / blocked** — evaluator file is missing, so no real CLI invocation can be made | `art-worktree-prerequisite` |
| QA-14-06 | Required quality gates on the implementation | Static/test surfaces | `cd /Volumes/SSD1/1/KoJev-todo14 && uv run ruff check kojev tests`; `cd /Volumes/SSD1/1/KoJev-todo14 && uv run basedpyright kojev tests`; `cd /Volumes/SSD1/1/KoJev-todo14 && uv run pytest -q` | **PASS for the clean base only, not evidence of todo 14** — ruff passed, basedpyright passed, and the pre-existing suite passed `70 passed in 29.81s`; these do not verify absent todo-14 behavior | `art-ruff-check`, `art-basedpyright`, `art-full-pytest` |

## Adversarial cases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-14-01 | Missing checkpoint must fail loudly with typed error naming path | invalid-input / missing prerequisite | A nonexistent or invalid checkpoint path raises the typed evaluation error and includes the path; it must not become `FileNotFoundError`, silent skip, or zero report | **FAIL / blocked** — no evaluator implementation or target tests are present | `art-worktree-prerequisite`, `art-target-pytest` |
| ADV-14-02 | Contamination check must reject overlap | data-integrity | Injecting a state string shared with training causes a loud contamination failure; clean evaluated data passes | **FAIL / blocked** — no evaluator implementation or target tests are present | `art-worktree-prerequisite`, `art-target-pytest` |
| ADV-14-03 | Metrics must be truthful | arithmetic / oracle | A hand fixture produces exact expected accuracy and Brier values, with per-kind and per-source slices | **FAIL / blocked** — no evaluator implementation or target tests are present | `art-worktree-prerequisite`, `art-target-pytest` |
| ADV-14-04 | Latency must be measured, not fabricated | runtime-observation | A real model call produces finite latency fields including p50/p95; no fixed-duration assertion is used | **FAIL / blocked** — no evaluator implementation or CLI entry point is present | `art-worktree-prerequisite` |
| ADV-14-05 | No-network tiny-random execution | environment / dependency | The local synthetic CLI run completes without network downloads and emits report keys/sample table | **FAIL / blocked** — no evaluator CLI exists in the assigned worktree | `art-worktree-prerequisite` |

## Artifact references

| id | kind | description | path |
|---|---|---|---|
| `art-worktree-prerequisite` | command transcript | Clean assigned worktree, branch/commit, and direct existence checks showing both required files are missing | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/worktree-prerequisite.txt` |
| `art-target-pytest` | command transcript | Exact target invocation and output: `file or directory not found: tests/test_evaluate.py`; no tests ran | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/target-pytest.txt` |
| `art-ruff-check` | command transcript | Exact ruff invocation; clean base result `All checks passed!` | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/ruff-check.txt` |
| `art-basedpyright` | command transcript | Exact basedpyright invocation; clean base result `0 errors, 0 warnings, 0 notes` | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/basedpyright.txt` |
| `art-full-pytest` | command transcript | Exact full-suite invocation; clean base result `70 passed in 29.81s` | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/full-pytest.txt` |
| `art-cleanup` | command transcript | Post-run cleanup receipt showing no uncommitted files in the assigned worktree after removing the temporary `.venv` | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c437/cleanup.txt` |

## Blocker

The parent implementation is required before manual QA can execute the requested real scenarios. Missing prerequisites are the production file, target test file, a runnable evaluator CLI invocation, and the implementation commit/pushed SHA. The clean-base gates above are explicitly not a todo-14 verification.

# Manual QA Matrix — KoJev todo 9

Verdict: **FAIL**. The dedicated worktree implementation is not releasable: the targeted test suite has a behavioral failure, Ruff and basedpyright fail, and all three real requests fail before append-only accounting because the actual OpenRouter top-5 logprobs response shape is not accepted.

Worktree: `/Volumes/SSD1/worktrees/kojev-task-9-teacher`
Branch: `feat/task-9-openrouter-teacher`

## surfaceEvidence

| Scenario | Criterion reference | Surface | Exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|---|
| S1 | mocked exact accounting, parser, cap/restart, bounded retry | Python test surface | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest tests/test_teacher.py -q` | **FAIL** — 3 passed, 1 failed; retry test observed `{1: 0}` instead of expected `{1: 0, 2: 1}` | A1 |
| S2 | full mocked/project regression suite | Python test surface | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest -q` | **FAIL** — 1 failed, 12 passed | A1 |
| S3 | required lint gate | Ruff CLI | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run ruff check kojev/teacher.py tests/test_teacher.py` | **FAIL** — 13 errors | A1 |
| S4 | required type-check gate | basedpyright CLI | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run basedpyright kojev/teacher.py tests/test_teacher.py` | **FAIL** — 4 errors, 2 warnings | A1 |
| S5 | exactly three real requests when API is available; no secret logging; total under $0.01 | OpenRouter HTTPS API through async client | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && LEDGER=<temp> uv run python <three-request-script>`; script issued requests 1, 2, and 3 with `include_logprobs=True` and a temporary ledger | **FAIL** — all 3 requests reached the provider but raised Pydantic `ValidationError` before accounting; ledger had 0 lines. No secret was logged. | A2 |

## adversarialCases

| Scenario | Criterion reference | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| A1 | external text must not override line parser | prompt-injection / tolerant line parser | Ignore prose/injection and parse only constrained indexed answer lines | **PASS** — targeted parser test passed; non-empty test output artifact | A1 |
| A2 | restart-safe refusal at >=$95 with `BUDGET_REACHED` | cap bypass / restart | Existing usage at cap prevents HTTP call and appends durable marker | **PASS** — targeted cap/restart test passed; non-empty test output artifact | A1 |
| A3 | exact provider-reported pricing and append-only JSONL | accounting integrity | Compute exact token-price cost and append one usage record | **PASS (mocked only)** — accounting test passed; real run could not reach append because logprobs parsing failed | A1, A2 |
| A4 | 429/5xx bounded backoff and adaptive concurrency | transient failure / concurrency saturation | Retry bounded 429/5xx responses, avoid duplicate charges, and adapt concurrency within 32..64 | **FAIL** — retry test failed; no passing evidence for adaptive 32->64 behavior; implementation inspection is not a pass substitute | A1 |
| A5 | optional logprobs top-5 | provider schema adversarial shape | Accept the provider's actual top-5 logprobs structure when enabled | **FAIL** — all three real requests returned actual nested top_logprobs entries and parsing raised `ValidationError`; artifact contains redacted error evidence | A2 |
| A6 | gold-labeled families / fake probe / model restriction | scope guard | No calls on gold-labeled families or fake probe; only pinned model | **not_applicable** — this QA invocation used only synthetic non-gold prompts and the pinned client model; it did not trigger those prohibited dataset paths | A2 |
| A7 | secret handling | credential leakage | API key must not appear in output or artifacts | **PASS** — real-request artifact contains no key; output was redacted and only exception summaries, usage-independent fields, and ledger line count were retained | A2 |

## artifactRefs

| ID | Kind | Description | Path |
|---|---|---|---|
| A1 | command transcript | Dedicated-worktree pytest, Ruff, basedpyright, and LOC measurements | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c3a2/task-9-command-results.txt` |
| A2 | real API transcript | Exactly three real OpenRouter request attempts, redacted output, exit status, and temporary ledger line count | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-real-three-requests.txt` |

## Scope / repository state

The dedicated worktree remained uncommitted and unpushed during QA. Observed status was `?? kojev/teacher.py`, `?? tests/test_teacher.py`, and `?? .omo/evidence/task-9-real-three-requests.txt`; no commit SHA exists to report. No product changes were made by this QA executor.

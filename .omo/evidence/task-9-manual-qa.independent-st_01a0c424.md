# Task 9 independent manual QA

Commit under review: `f4a26a1c0ea08799576b763e842555dccd98cce4`
Worktree: `/Volumes/SSD1/worktrees/kojev-task-9-teacher`
Branch: `feat/task-9-openrouter-teacher`

## Verdict

- **result:** needs-fix
- **safe_to_land:** false
- **reason:** the public exhausted-retry path is not cleanly observable. Four local 503 responses reach the retry limit, but `TeacherRequestError` is replaced by `TypeError: super(type, obj): obj must be an instance or subtype of type` during traceback assignment. The existing retry-green case only covers recovery before exhaustion.
- **request IDs / token-usage evidence gap:** evidence-only, not the plan-blocking failure. The live transcript does not expose per-request IDs or token counts, and the activity endpoint returned HTTP 403 because the credential is not a management key. The implementation has typed `id` and `usage` fields and the local provider-shaped runtime check confirms they are retained in the `TeacherResult` and ledger. No further live request was made.

## `manualQa`

### `surfaceEvidence`

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| QA-LOCAL-001 | Commit and remote containment | Git repository | `git -C /Volumes/SSD1/worktrees/kojev-task-9-teacher rev-parse HEAD`; `git ... ls-remote origin refs/heads/feat/task-9-openrouter-teacher`; `git ... status --porcelain=v1`; `git ... diff --exit-code f4a26a1... --` | PASS: local and remote both equal target SHA; clean status; no diff from target | `a-git-containment` |
| QA-GATE-001 | Full pytest | Local Python test runner | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest` | PASS: 14 passed in 0.09s | `a-pytest` |
| QA-GATE-002 | Ruff | Local static lint | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run ruff check kojev tests` | PASS: All checks passed | `a-ruff` |
| QA-GATE-003 | basedpyright | Local static type checker | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run basedpyright` | PASS: 0 errors, 0 warnings, 0 notes | `a-basedpyright` |
| QA-API-001 | Public API surface | Python import/introspection | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... inspect.signature(...) ... PY` | PASS: public exports include pinned constants, errors, client, result, and parser; signatures are typed and callable | `a-public-api` |
| QA-RUNTIME-001 | Typed nested OpenRouter top-5 parsing, exact pricing, append-only ledger, request ID and usage retention | Python async `TeacherClient` with `httpx.MockTransport` and provider-shaped nested payload | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... payload choices[0].logprobs.content[*].top_logprobs ... PY` | PASS: answers `{1: 0, 2: 1}`, nested top token `B`, model pinned, request ID retained, usage `100/20/120`, exact cost `2.08e-05`, one ledger line | `a-independent-runtime` |
| QA-RUNTIME-002 | Retry behavior and answer preservation | Python async `TeacherClient` with MockTransport returning 429, 503, then 200 | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest tests/test_teacher.py::test_retries_429_and_5xx_with_bounded_attempts` | PASS: 3 attempts, answers preserved, one ledger charge | `a-green-target`, `a-independent-runtime` |
| QA-RUNTIME-003 | Retry exhaustion is bounded and typed | Python async `TeacherClient` with MockTransport returning 503 for every request | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... await client.ask(...) ... PY` | FAIL: attempts are bounded at 4, but public call raises `TypeError`, not `TeacherRequestError` | `a-independent-exhausted` |
| QA-RUNTIME-004 | Restart cap at $95 and durable marker | Python async `TeacherClient` with preloaded $95 ledger and MockTransport | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... ledger cost_usd=95.0 ... PY` | PASS in prior artifact: `BudgetReached`, zero provider calls, marker present | `a-parser-budget` |
| QA-LIVE-001 | Three successful real requests, pinned model, nested parse, ledger total below cap | OpenRouter HTTPS completion API through `TeacherClient` | Prior exact invocation recorded in evidence: `LEDGER=$(mktemp -t kojev-task9-ledger.XXXXXX) && LEDGER="$LEDGER" uv run python <captured live script>` | PASS per captured live transcript: 3 successes, model `qwen/qwen3-vl-8b-instruct`, 3 ledger records, total `0.000029757` USD, temporary ledger removed; not rerun to avoid cost | `a-live-three` |
| QA-ACTIVITY-001 | Provider request IDs for prior live calls | OpenRouter activity HTTP API | `curl -i -sS https://openrouter.ai/api/v1/activity -H 'Authorization: Bearer $OPENROUTER_API_KEY'` (secret redacted in artifact) | BLOCKED: HTTP 403, management key required; no live completion call made | `a-activity-403` |
| QA-CLEANUP-001 | Temp/proc cleanup and patch containment | Worktree and process inspection | `ps aux | grep -E 'uv run|pytest|teacher|openrouter' | grep -v grep`; `find ... -name '*ledger*'`; `git status --short` | PASS for reviewed worktree: no generated ledger/temp file remains; no task-9 process; clean status | `a-cleanup` |

### `adversarialCases`

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-001 | Parser injection safety | Prompt/data injection in prose | Ignore prose instructions and parse only constrained indexed answer lines | PASS: `{1: 2, 2: 1}` | `a-parser-budget` |
| ADV-002 | Retry resilience | 429 then 503 then success | Retry within bounded attempts and preserve all requested answers without duplicate ledger charges | PASS: 3 attempts and one ledger line | `a-adversarial-retry`, `a-independent-runtime` |
| ADV-003 | Retry exhaustion | Persistent 503 responses | Stop after `MAX_RETRIES` and raise the typed `TeacherRequestError` through `ask()` | FAIL: stopped after 4 attempts but surfaced `TypeError` during exception traceback handling | `a-independent-exhausted` |
| ADV-004 | Nested provider schema | `top_logprobs` is list of objects, not scalar/list[int] | Typed boundary parser accepts provider shape and preserves candidates | PASS | `a-green-nested`, `a-independent-runtime` |
| ADV-005 | Exact pricing | Explicit prompt/completion token counts | Cost equals `(prompt*0.117 + completion*0.455)/1e6` without rounding | PASS | `a-independent-runtime`, `a-live-three` |
| ADV-006 | Append-only accounting | Successful request followed by ledger inspection | Add exactly one usage line with model, request ID, usage, cost, and logprobs | PASS in local runtime; live transcript confirms 3 lines | `a-independent-runtime`, `a-live-three` |
| ADV-007 | Restart cap | Existing ledger total exactly $95 | Make zero provider calls and append durable budget marker | PASS | `a-parser-budget` |
| ADV-008 | Concurrency/backoff bounds | Error-triggered concurrency adjustment | Keep configured concurrency bounded at 32 initial / 64 max and retries bounded | PASS for constants and recovery path; exhaustion defect remains separate | `a-public-api`, `a-independent-runtime` |
| ADV-009 | Gold-family call surface | Teacher request model selection | Use only pinned Qwen teacher model; no gold-family model call | PASS: request body and source use only `qwen/qwen3-vl-8b-instruct` | `a-independent-runtime`, `a-live-three` |
| ADV-010 | Secret handling | Runtime evidence and request capture | Do not emit API key or secret-bearing headers | PASS in prior live capture: `secret_logged=false`; independent run used local sentinel only | `a-live-three` |

## `artifactRefs`

| id | kind | description | path |
|---|---|---|---|
| `a-git-containment` | terminal transcript | Local/remote SHA equality, clean status, and zero diff at target | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-git-containment.txt` |
| `a-pytest` | terminal transcript | Independent full pytest run | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-pytest.txt` |
| `a-ruff` | terminal transcript | Independent Ruff run | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-ruff.txt` |
| `a-basedpyright` | terminal transcript | Independent basedpyright run | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-basedpyright.txt` |
| `a-public-api` | terminal transcript | Public export and signature inspection | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-public-api.txt` |
| `a-independent-runtime` | terminal transcript | Local provider-shaped runtime checks for nested parsing, pricing, usage, ledger, retry recovery, cap, and parser | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-runtime.txt` |
| `a-independent-exhausted` | terminal transcript | Persistent-503 reproduction showing 4 attempts then wrong `TypeError` | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-exhausted-retry.txt` |
| `a-green-target` | terminal transcript | Existing focused green teacher tests | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-green-target-pytest.txt` |
| `a-green-nested` | terminal transcript | Existing nested parser and ledger evidence | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-green-nested-logprobs.txt` |
| `a-adversarial-retry` | terminal transcript | Existing retry runtime evidence | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-adversarial-retry.txt` |
| `a-parser-budget` | terminal transcript | Existing injection parser and $95 restart refusal evidence | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-adversarial-parser-budget.txt` |
| `a-live-three` | terminal transcript | Existing three-request live success transcript; no secrets | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-live-three-green.txt` |
| `a-activity-403` | HTTP transcript | Existing activity endpoint 403 response | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-openrouter-activity.txt` |
| `a-cleanup` | terminal transcript | Final process/temp/worktree cleanup check | `/Volumes/SSD1/1/KoJev/.omo/evidence/task-9-independent-cleanup.txt` |

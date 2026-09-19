# Task 9 manual QA

## `manualQa`

### `surfaceEvidence`

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| QA-RED-001 | Retry answer preservation | Python async client with MockTransport | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest tests/test_teacher.py` before fix | FAIL (RED captured: `{1: 0}` instead of `{1: 0, 2: 1}`) | `a-red-target`, `a-red-full` |
| QA-GREEN-001 | Retry answer preservation | Python async client with MockTransport | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest` | PASS: 14 passed | `a-green-full` |
| QA-GREEN-002 | Nested OpenRouter logprobs parsing and usage append | Python async client with MockTransport and provider-shaped nested payload | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest tests/test_teacher.py` | PASS: 5 passed; nested token candidate parsed and ledger record appended | `a-green-target`, `a-green-nested` |
| QA-LIVE-001 | Exactly three fresh real OpenRouter requests; pinned model; nested parse; usage ledger; total < $0.01 | OpenRouter HTTPS API through `TeacherClient` | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && LEDGER=$(mktemp -t kojev-task9-ledger.XXXXXX) && LEDGER="$LEDGER" uv run python <captured live script>` | PASS: exactly 3 successful requests, each nested logprobs parsed, 3 ledger records, total `0.000029757` USD, temporary ledger removed | `a-live-three` |
| QA-GATE-001 | Full pytest | Local Python test runner | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest` | PASS: 14 passed in 0.08s | `a-green-full` |
| QA-GATE-002 | Ruff | Local static lint | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run ruff check kojev tests` | PASS: All checks passed | `a-green-ruff` |
| QA-GATE-003 | basedpyright | Local static type checker | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run basedpyright` | PASS: 0 errors, 0 warnings, 0 notes | `a-green-pyright` |
| QA-BLOCK-001 | Provider request IDs for the three already-completed calls | OpenRouter authenticated activity API | `curl -i -sS https://openrouter.ai/api/v1/activity -H "Authorization: Bearer $OPENROUTER_API_KEY"` | BLOCKED: HTTP 403; endpoint requires a management key. No fourth request made. | `a-activity-403` |

### `adversarialCases`

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-001 | Injection-safe parser | Prompt/data injection in prose around answer lines | Ignore external instructions and parse only constrained indexed answer lines | PASS: parsed `{1: 2, 2: 1}` from injected input | `a-parser-budget` |
| ADV-002 | Retry resilience | HTTP 429 then 503 then success | Retry within bounded attempts and preserve all requested answers without duplicate charges | PASS: 3 attempts; full suite green | `a-adversarial-retry`, `a-green-full` |
| ADV-003 | Restart-safe budget refusal | Ledger already at exactly $95 | Return `BudgetReached`, append durable marker, make zero provider calls | PASS: `BudgetReached`, provider calls `0`, marker present | `a-parser-budget` |
| ADV-004 | Nested provider schema | `choices[0].logprobs.content[*].top_logprobs` is list of objects | Parse nested token candidates and append usage | PASS in focused test and all three live calls | `a-green-nested`, `a-live-three` |
| ADV-005 | Cost cap | Three fresh live calls | Keep cumulative recorded cost below $0.01 | PASS: `0.000029757` USD | `a-live-three` |
| ADV-006 | Secret handling | Live request evidence output | Do not print API key or secret-bearing headers | PASS: `secret_logged=false`; capture contains no key | `a-live-three` |
| ADV-007 | Gold-family routing | Teacher request model selection | Use only pinned Qwen teacher model; do not call gold-family models | PASS: live summary model `qwen/qwen3-vl-8b-instruct`; no gold-family call in client | `a-live-three`, `a-green-full` |
| ADV-008 | Concurrency/backoff bounds | Retry/backoff configuration | Keep concurrency constants bounded at 32 initial and 64 maximum; retries bounded | PASS: constants and bounded retry test verified | `a-green-full`, `a-green-ruff`, `a-green-pyright` |

## `artifactRefs`

| id | kind | description | path |
|---|---|---|---|
| `a-red-target` | terminal transcript | Pre-fix focused pytest showing retry answer loss | `.omo/evidence/task-9-qa-red-target-pytest.txt` |
| `a-red-full` | terminal transcript | Pre-fix full pytest: 12 passed, 1 failed | `.omo/evidence/task-9-qa-red-full-pytest.txt` |
| `a-red-ruff` | terminal transcript | Pre-fix Ruff errors | `.omo/evidence/task-9-qa-red-ruff.txt` |
| `a-red-pyright` | terminal transcript | Pre-fix basedpyright errors | `.omo/evidence/task-9-qa-red-basedpyright.txt` |
| `a-red-nested` | terminal transcript | Pre-fix nested logprobs validation failure and empty ledger | `.omo/evidence/task-9-qa-red-nested-logprobs.txt` |
| `a-green-target` | terminal transcript | Post-fix focused pytest: 5 passed | `.omo/evidence/task-9-qa-green-target-pytest.txt` |
| `a-green-nested` | terminal transcript | Post-fix nested parser and ledger runtime proof | `.omo/evidence/task-9-qa-green-nested-logprobs.txt` |
| `a-green-full` | terminal transcript | Final full pytest: 14 passed | `.omo/evidence/task-9-qa-green-full-pytest.txt` |
| `a-green-ruff` | terminal transcript | Final Ruff clean output | `.omo/evidence/task-9-qa-green-final-ruff.txt` |
| `a-green-pyright` | terminal transcript | Final basedpyright clean output | `.omo/evidence/task-9-qa-green-final-basedpyright.txt` |
| `a-live-three` | terminal transcript | Exactly three fresh successful real requests, nested parse, usage append, total cost | `.omo/evidence/task-9-qa-live-three-green.txt` |
| `a-parser-budget` | terminal transcript | Injection parser and $95 restart-safe refusal | `.omo/evidence/task-9-qa-adversarial-parser-budget.txt` |
| `a-adversarial-retry` | terminal transcript | Retry runtime scenario | `.omo/evidence/task-9-qa-adversarial-retry.txt` |
| `a-activity-403` | HTTP transcript | Activity API request-ID retrieval blocked by key permissions | `.omo/evidence/task-9-qa-openrouter-activity.txt` |

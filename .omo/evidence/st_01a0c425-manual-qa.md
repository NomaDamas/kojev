# Manual QA — corrective commit 9b679603762d52a96c63841a79e24f8e48b9663d

Verdict: **AdversarialVerify confirmed**

safe_to_land: **yes**

Scope: read-only verification of `/Volumes/SSD1/worktrees/kojev-task-9-teacher` at commit `9b679603762d52a96c63841a79e24f8e48b9663d`; no product files, refs, or target worktree files were edited. No real provider requests were made during this verification.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-QA-001 | Local/remote SHA, clean status, containment versus `f4a26a1c0ea08799576b763e842555dccd98cce4` | Git repository and origin ref | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && git rev-parse HEAD && git ls-remote origin refs/heads/feat/task-9-openrouter-teacher && git status --porcelain=v1 --branch && git merge-base --is-ancestor f4a26a1c0ea08799576b763e842555dccd98cce4 HEAD && git diff --check f4a26a1c0ea08799576b763e842555dccd98cce4..HEAD && git diff --name-status f4a26a1c0ea08799576b763e842555dccd98cce4..HEAD` | PASS: local and remote both `9b679603762d52a96c63841a79e24f8e48b9663d`; worktree clean; parent is ancestor; diff check clean; changed paths are evidence plus `kojev/teacher_types.py` and `tests/test_teacher.py` only | `ev-git` |
| ADV-QA-002 | Full pytest | Python test runner | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest` | PASS: 15 passed | `ev-pytest` |
| ADV-QA-003 | Target teacher pytest | Python target test module | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run pytest tests/test_teacher.py` | PASS: 6 passed | `ev-pytest` |
| ADV-QA-004 | Ruff | Static lint | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run ruff check kojev tests` | PASS: `All checks passed!` | `ev-static` |
| ADV-QA-005 | basedpyright | Static type checking | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run basedpyright` | PASS: `0 errors, 0 warnings, 0 notes` | `ev-static` |
| ADV-QA-006 | Persistent 503 typed exhaustion | Python async `TeacherClient` with `httpx.MockTransport` returning 503 on every request | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... anyio.run(main) ... PY` using `retry_delays=(0.0, 0.0, 0.0, 0.0)` | PASS: catchable `TeacherRequestError`, `status_code=503`, exact message `OpenRouter request exhausted retries: HTTP 503`, exactly 4 attempts, `bare_type_error=False` | `ev-503` |
| ADV-QA-007 | Nested top-5 logprobs typed parsing, exact pricing, request id/usage retention, append-only ledger | Python async `TeacherClient` with provider-shaped `httpx.MockTransport` and `TemporaryDirectory` ledger | `cd /Volumes/SSD1/worktrees/kojev-task-9-teacher && uv run python - <<'PY' ... provider-shaped response ... PY` | PASS: model pinned; `top_logprobs=5`; nested token `B` parsed; request id `req-qa`; usage `{prompt_tokens:100, completion_tokens:20, total_tokens:120}` retained; cost `2.08e-05` equals independent formula; ledger grew 1 to 2 lines without replacing first record | `ev-runtime` |
| ADV-QA-008 | `$95` restart refusal marker | Python async `TeacherClient` with pre-seeded ledger at `BUDGET_CAP_USD` and forbidden MockTransport | Included in same inline runtime invocation as `ADV-QA-007` | PASS: `BudgetReached`, durable `BUDGET_REACHED` marker, provider calls `0` | `ev-runtime` |
| ADV-QA-009 | Parser injection safety | Pure parser surface | Included in same inline runtime invocation as `ADV-QA-007` with prose injection plus `A1: 2` and `Q2 = option 1` | PASS: parsed exactly `{1: 2, 2: 1}` | `ev-runtime` |
| ADV-QA-010 | Pinned qwen model, concurrency/backoff constants, no gold-family call surface | Local Python client and request body through MockTransport | Included in same inline runtime invocation as `ADV-QA-007`; static grep audit also run over `kojev` and `tests` | PASS: model `qwen/qwen3-vl-8b-instruct`; bounds `32,64,4`; request body model matches; `gold_family_model=False` | `ev-runtime`, `ev-hygiene` |
| ADV-QA-011 | Preserved prior live three-request evidence and no secret exposure | Existing evidence files in caller evidence directory | `test -s` on prior artifacts; grep secret-like patterns across `.omo/evidence/task-9-*`; inspect prior live summary | PASS: prior artifact is non-empty and records exactly 3 successful requests, 3 ledger records, pinned model, total cost `2.9757000000000003e-05`, `secret_logged=false`; secret audit found no secret-like literals | `ev-hygiene`, `prior-live` |
| ADV-QA-012 | Provider request-ID evidence limitation | OpenRouter activity endpoint, prior recorded invocation | Prior artifact records `curl -i -sS https://openrouter.ai/api/v1/activity -H "Authorization: Bearer $OPENROUTER_API_KEY"` | NOT CONFIRMED: HTTP/2 403, `Only management keys can fetch activity for an account`; no fourth request was made. This does not invalidate the preserved three-request result, but request IDs cannot be independently recovered from activity with the available key | `prior-activity` |
| ADV-QA-013 | Temporary/process cleanup | Local temp dirs and target-worktree process/status audit | `ps ax ...` targeted audit plus `git status --porcelain=v1 --branch`; all custom runtime scenarios used `TemporaryDirectory` | PASS: temporary ledgers removed on context exit; target worktree clean; no lingering task-9 QA process remained after commands completed. Unrelated pre-existing process was not touched | `ev-hygiene` |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-001 | Persistent 5xx exhaustion | Typed exception / traceback mutation regression | All four bounded attempts against 503 raise normally as `TeacherRequestError`, retaining status and exact message; no frozen-dataclass `TypeError` | PASS | `ev-503`, `prior-red` |
| ADV-002 | Nested provider schema | Shape mismatch: `choices[0].logprobs.content[*].top_logprobs` is list of objects | Boundary parser accepts typed nested candidates and preserves the selected token | PASS | `ev-runtime`, `prior-live` |
| ADV-003 | Pricing | Numeric exactness / rounding drift | Cost equals `(prompt_tokens*0.117 + completion_tokens*0.455)/1_000_000` | PASS | `ev-runtime` |
| ADV-004 | Ledger | Append-only persistence | Each successful request appends a record and retains prior records, including request id, usage, model, and cost | PASS | `ev-runtime`, `prior-live` |
| ADV-005 | Budget cap | Restart at exact `$95` boundary | Refuse before provider call and append durable `BUDGET_REACHED` marker | PASS | `ev-runtime` |
| ADV-006 | Parser injection | External instructions embedded in response prose | Ignore prose and parse only constrained indexed answer lines | PASS | `ev-runtime` |
| ADV-007 | Routing | Gold-family contamination / model drift | Teacher call surface uses only pinned Qwen model and does not call gold-family models | PASS | `ev-runtime`, `ev-hygiene` |
| ADV-008 | Retry/concurrency | Unbounded retries or concurrency drift | Retry count remains 4; initial/max concurrency remain 32/64; retry behavior is bounded | PASS | `ev-runtime`, `ev-pytest` |
| ADV-009 | Secret handling | Credential leakage in QA/live artifacts | No API key or secret-bearing literal appears in preserved evidence; live summary explicitly records `secret_logged=false` | PASS | `ev-hygiene`, `prior-live` |
| ADV-010 | Network safety | Accidental real provider request during verification | All fresh scenarios use `httpx.MockTransport`; no real provider request is issued | PASS | `ev-503`, `ev-runtime` |
| ADV-011 | Request metadata | Dropped request ID or usage fields | Parsed result and ledger retain provider request ID and all usage counters | PASS | `ev-runtime`, `prior-live` |
| ADV-012 | Provider activity lookup | Insufficient key permissions | Activity lookup may be unavailable; record explicit HTTP 403 rather than infer request IDs | PASS as limitation handling; request IDs remain unverified | `prior-activity`, `ev-hygiene` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `ev-git` | terminal transcript | Fresh local/remote SHA, clean status, parent ancestry, diff check, and changed paths | `.omo/evidence/task-9-corrective-git.txt` |
| `ev-pytest` | terminal transcript | Fresh full pytest and target `tests/test_teacher.py` runs | `.omo/evidence/task-9-corrective-pytest.txt` |
| `ev-static` | terminal transcript | Fresh Ruff and basedpyright runs | `.omo/evidence/task-9-corrective-static.txt` |
| `ev-503` | terminal transcript | Fresh all-503 MockTransport runtime proof: typed exception, exact message/status, four attempts, no TypeError | `.omo/evidence/task-9-corrective-503-runtime.txt` |
| `ev-runtime` | terminal transcript | Fresh no-network runtime proof for nested parsing, pricing, ledger, budget marker, parser, bounds, model, and gold-family surface | `.omo/evidence/task-9-corrective-runtime.txt` |
| `ev-hygiene` | terminal transcript | Fresh preserved-artifact, secret-scan, worktree-clean, and process cleanup audit | `.omo/evidence/task-9-corrective-hygiene.txt` |
| `prior-live` | terminal transcript | Preserved prior live three-request evidence with nested parsing, three ledger records, pinned model, cost, and `secret_logged=false` | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-live-three-green.txt` |
| `prior-red` | terminal transcript | Prior failing all-503 evidence showing the old frozen-dataclass TypeError | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-red-frozen-dataclass-exception.txt` |
| `prior-activity` | HTTP transcript | Prior activity endpoint request-ID lookup blocked by HTTP 403 management-key requirement | `/Volumes/SSD1/worktrees/kojev-task-9-teacher/.omo/evidence/task-9-qa-openrouter-activity.txt` |

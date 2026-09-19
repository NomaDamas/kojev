# Task 11 manual QA

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| wave-scale | bounded scheduling and target | CLI/tmux | `uv run python -m kojev.label --korquad-only --limit 33334 --output data/distill/train.jsonl --ledger data/distill/ledger.jsonl`, then `--limit 33350` | PASS: 100020 questions, $2.945570641 | `task11-evidence` |
| wave-kill-restart | resume/dedupe | CLI/tmux | kill `--limit 500` job after 15s, restart same command | PASS: 497 ledger lines, 497 unique IDs, 0 duplicates, 497 outputs | `task11-evidence` |
| test-gates | regression safety | pytest/ruff/basedpyright | five required gate commands | PASS | `task11-evidence` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| timeout-retry | non-fatal timeout | one request timeout | retry in wave, continue overall run | PASS: deterministic test | `task11-evidence` |
| bounded-wave | queue control | five candidates, wave size two | peak active tasks never exceeds two | PASS: deterministic test | `task11-evidence` |
| request-dedupe | restart safety | rerun completed states/provider IDs | no duplicate request IDs | PASS: real 497/497 audit | `task11-evidence` |
| budget-cap | cost boundary | scale target | stop below $20 | PASS: $2.945570641 | `task11-evidence` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `task11-evidence` | text | Full gates, live scale arithmetic, wave kill/restart integers, cleanup and final counts | `.omo/evidence/task-11-kojev.txt` |

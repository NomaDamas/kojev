# Manual QA Matrix — todo 16 release bundle

## Status

**BLOCKED / NOT CERTIFIED.** The checked-out branch `feat/todo-16-release` is at `6cb0618` and contains neither `kojev/release.py` nor `tests/test_release.py`. The required implementation and tests are absent, so no release-bundle scenario can be honestly marked PASS. This is a local QA blocker, not evidence that the requested behavior works.

The plan explicitly scopes this increment to synthetic/fixture packaging machinery. Populating README report numbers from todo 14 evaluation tables and real trained checkpoints is a separate follow-up increment and was not attempted.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| QA-RED-01 | TDD RED proof | pytest collection | `cd /Volumes/SSD1/1/KoJev-todo16 && uv run pytest tests/test_release.py -q` | BLOCKED — collection fails because `kojev.release` does not exist | `red-proof` |
| QA-CLI-01 | build CLI | local CLI (`python -m kojev.release build`) | Synthetic-tree invocation recorded in `cli-blocker` | BLOCKED — `/Volumes/SSD1/1/KoJev-todo16/.venv/bin/python: No module named kojev.release` (exit 1) | `cli-blocker` |
| QA-RT-01 | ROUND-TRIP | local CLI (`python -m kojev.release build/restore`) | Not runnable: `kojev.release` is absent and no production CLI exists | BLOCKED — missing prerequisite | `cli-blocker` |
| QA-CONFIG-01 | TAMPERED CONFIG typed failure | local filesystem loader | Not runnable: `load_bundle` is absent | BLOCKED — missing prerequisite | `red-proof` |
| QA-HASH-01 | TAMPERED FILE manifest rejection | local filesystem loader | Not runnable: `load_bundle` is absent | BLOCKED — missing prerequisite | `red-proof` |
| QA-MISSING-01 | MISSING FILE rejection | local filesystem loader | Not runnable: `load_bundle` is absent | BLOCKED — missing prerequisite | `red-proof` |
| QA-BUDGET-01 | BUDGET AUDIT | local ledger parser | Not runnable: `audit_budget` is absent | BLOCKED — missing prerequisite | `red-proof` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV-01 | typed validation | tampered structured config | Raise typed `BundleError` naming the config problem; never bare `KeyError`/`ValueError`/assertion | BLOCKED — loader absent | `red-proof` |
| ADV-02 | manifest verification | modified payload bytes | Reject and name the offending relative path | BLOCKED — loader absent | `red-proof` |
| ADV-03 | manifest verification | deleted manifest-listed file | Reject as typed failure and name the missing path | BLOCKED — loader absent | `red-proof` |
| ADV-04 | budget cap | ledger total over USD 100 | Return an over-cap audit result / flag; do not silently pass | BLOCKED — audit helper absent | `red-proof` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `red-proof` | terminal transcript | Required RED invocation and exact `ModuleNotFoundError: No module named 'kojev.release'` output | `.omo/evidence/st_01a0c435-red-proof.txt` |
| `cli-blocker` | terminal transcript | Exact synthetic-tree CLI invocation; module resolution failure and exit code 1 | `.omo/evidence/st_01a0c435-cli-blocker.txt` |

## Cleanup receipt

- Provisional QA test and production files were removed.
- Synthetic CLI source tree and temporary directory were removed.
- Worktree status after cleanup: clean relative to `origin/main` (`git status --short --branch` reported only `## feat/todo-16-release...origin/main`).
- No bundle, restore directory, network publication, commit, or push was performed by this QA executor.

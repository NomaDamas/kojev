# Todo 1 manual QA

Date: 2026-09-19
Repository: `/Volumes/SSD1/1/KoJev`
Claim under test: pushed commit `d1e9fcf50efc5035db83618717f9cefc9abdd42b` on `origin/feat/schema`, based on bootstrap `7dfb5c9c660b37dfa98804f20de4b928d0cdbf78` on `origin/main`.

## Surface evidence

| Scenario | Criterion | Surface and exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|
| S1 | Git state, pushed commit, scope containment | `git status --short --branch && git branch -vv && git remote -v && git log --oneline --decorate -12`; `git show --stat d1e9fcf...`; `git ls-remote origin refs/heads/main refs/heads/feat/schema`; `git diff --name-status 7dfb5c9..d1e9fcf...`; `git diff --name-status d1e9fcf..HEAD` | PASS for claimed commits and refs. Local `HEAD`, `origin/feat/schema`, and remote all equal `d1e9fcf`; `main` and `origin/main` equal `7dfb5c9`. Feature commit contains only evidence, package, schema, manifest, tests, and lockfile. No diff exists after `HEAD`. | A1, A2 |
| S2 | `uv run pytest tests/test_schema.py -q` | Exact command: `uv run pytest tests/test_schema.py -q` | PASS: `...... [100%]`, `6 passed in 0.03s`. | A3 |
| S3 | Ruff and basedpyright if available | Exact command: `uv run ruff check kojev tests && uv run ruff format --check kojev tests && uv run basedpyright kojev tests` | PASS: `All checks passed!`; `3 files already formatted`; `0 errors, 0 warnings, 0 notes`. | A4 |
| S4 | Real JSONL round-trip, mixed question kinds | Writer process: `uv run python - <path>` constructs `Example` with choice/score/noul and calls `write_jsonl`; independent reader process: `uv run python - <path>` calls `read_jsonl` and asserts state/source/split/types/options/gold/meta. | PASS: reader printed `fresh_process_equal_fields=True` and the parsed example. Assertions checked values, not exit status alone. | A5 |
| S5 | Malformed score cardinalities and choice 256 | Exact command: `uv run python -` constructing score with 1, score with 11, and choice with 256 options; catches `ValueError`, asserts kind and count appear in the message. | PASS: all three printed `type=ValidationError; is_value_error=True; ... message_contains_kind_and_count=True`. | A6 |
| S6 | Evidence claim comparison | Read `.omo/evidence/task-1-kojev.txt`; compared RED/GREEN, static checks, round-trip, malformed probes, adversarial notes, and cleanup receipt against A1-A8. | PASS with one stale/non-material claim: evidence says QA left “only untracked product files (kojev/, tests/, pyproject.toml, uv.lock, .omo/)”; current status has only untracked `.omo/plans/kojev.md`. Product files are tracked and clean. The claimed behavior/results remain reproduced. | A7, A1 |
| S7 | Cleanup | Exact command: `test ! -e .omo/tmp-task1-qa && echo cleanup_tmp_task1=absent; test ! -e /tmp/kojev-task1-qa.log && echo cleanup_log=absent; pgrep -af 'kojev|uvicorn' || true` | PASS for cleanup paths: both absent. No KoJev/uvicorn process was listed. The final `pgrep` emitted a numeric PID from the shell command itself only, not a matching process. | A8 |

## Adversarial cases

| Scenario | Criterion | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| A-M1 | malformed input | malformed_input | Illegal score cardinalities 1 and 11 and choice cardinality 256 must raise typed validation errors, with kind/count evidence. | PASS | A6 |
| A-M2 | JSONL persistence | stale_state | A fresh process must reload the just-written JSONL and preserve semantic fields. | PASS | A5 |
| A-M3 | QA output integrity | misleading_success_output | Assertions must inspect parsed values and exception type/message, not only process exit code or printed prose. | PASS | A5, A6 |
| A-M4 | repository hygiene | dirty_worktree | No hidden product changes may exist outside claimed commits; unrelated dirt must be reported. | PASS with caveat | A1, A2. Current unrelated dirt is `.omo/plans/kojev.md`, untracked and outside product commit. |
| A-M5 | repeatability | flaky_tests | Test and probes must run deterministically without sleeps, polling, or timing luck. | PASS | A3, A5, A6. No sleeps/polling used; pytest completed in 0.03s. |
| A-M6 | external/untrusted prompt text | prompt_injection | Not applicable: todo 1 is local schema/JSONL code and accepts structured fields only; no prompt or external instruction source is exercised. | NOT_APPLICABLE | A7 |
| A-M7 | interruption/resume | cancel_resume | Not applicable: todo 1 has no resumable job, network call, or long-running stateful operation. | NOT_APPLICABLE | A7 |
| A-M8 | command duration | hung_long_commands | Not applicable to product behavior: no long-running command is part of todo 1; bounded local test/static/probe commands completed. | NOT_APPLICABLE | A3, A5, A8 |
| A-M9 | repeated interruption | repeated_interruptions | Not applicable: no daemon, worker, or interruptible workflow is introduced by todo 1. | NOT_APPLICABLE | A7, A8 |

## Plan and integration conclusion

Todo 1 acceptance in `.omo/plans/kojev.md` is: `uv run pytest tests/test_schema.py -q` green, JSONL round-trip, and invalid inputs raise. All three are independently reproduced. The QA scenario's failure case is also reproduced with score cardinality 1.

Direct-main integration is **not required before todo 1 can close**. The plan explicitly assigns todo 1 to `feat/schema`, requires its commit to be pushed, and keeps `origin/main` as the clean integration base. `origin/main` remains exactly bootstrap commit `7dfb5c9c660b37dfa98804f20de4b928d0cdbf78`; no acceptance criterion says to merge this todo into main. Integration may be needed later to assemble the full plan, but it is not a prerequisite for this todo's close.

## Artifact references

- **A1** — command transcript — Git status/branches/remotes: this file, Surface S1.
- **A2** — command transcript — commit stats, remote refs, and containment diff: this file, Surface S1.
- **A3** — command transcript — pytest result: this file, Surface S2.
- **A4** — command transcript — ruff/format/basedpyright result: this file, Surface S3.
- **A5** — command transcript — two-process JSONL round-trip and asserted fields: this file, Surface S4.
- **A6** — command transcript — malformed score 1/11 and choice 256 typed-error probes: this file, Surface S5.
- **A7** — source artifact — `.omo/evidence/task-1-kojev.txt` and `.omo/plans/kojev.md` claim/acceptance comparison.
- **A8** — command transcript — cleanup path and process check: this file, Surface S7.

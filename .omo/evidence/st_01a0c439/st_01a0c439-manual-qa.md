# manualQa — af14ec7a885068cf3bfab875bdb98bca44735523

Verdict: **confirmed**
Safe to land: **true**
Measured resume transport-call count: **0** (both the seeded original-defect probe and the existing-ledger rerun)
Provider calls: **0 paid calls; all transport probes were injected `httpx.MockTransport`**.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | 1 | `LabelRunner.run` with seeded completed ledger and injected transport | `python -m pytest tests/test_label.py::test_seeded_request_id_skips_transport -q` plus fresh MockTransport probe with seeded `already-issued` record | PASS — fresh probe measured `transport_calls=0`, `new_labeled_count=0`, no output | A1 |
| S2 | 2 | Existing preserved `data/distill` output/ledger resume | Python `LabelRunner` invocation loading `/Volumes/SSD1/1/KoJev-todo11/data/distill/train.jsonl` and running against its ledger with transport that raises if called | PASS — `candidate_count=33340`, `transport_calls=0`, `new_labeled_count=0` | A1 |
| S3 | 3 | `_candidate_id` direct runtime calls | Python probe computed one ID three times on equivalent state/question families | PASS — same SHA-256 on all three calls: `19854adf...c89a9` | A1 |
| S4 | 4 | `kojev/label.py` scheduling path | `git show af14ec7:kojev/label.py` and line-path inspection; `_issued_records` and candidate filtering precede wave creation / `_run_wave` / `_label_with_retries` | PASS — filter is before transport invocation | A1 |
| S5 | 5 | Disposable mutation of pre-wave filters | Removed both identity predicates in detached worktree, then `python -m pytest tests/test_label.py::test_seeded_request_id_skips_transport -q` | PASS — regression test failed with `assert 1 == 0`, exit 1; proves filter is causal | A1 |
| S6 | 6 | Preserved output and ledger data | Streaming Python JSON recount over `/Volumes/SSD1/1/KoJev-todo11/data/distill/{train,ledger}.jsonl` | PASS — 100020 questions; 37557 ledger/usage lines; 37557 unique IDs; 0 duplicates; `$2.9455706409999483`; 0 malformed lines | A1 |
| S7 | 7 | Synthetic labeled output metadata | MockTransport returned provider ID `provider-123`; inspect output JSON and question metadata | PASS — deterministic candidate ID and provider request ID both persisted and distinct; top-level request ID is provider ID | A1 |
| S8 | 8 | Git source and static worker path | `git diff --exit-code af14ec7^ af14ec7 -- kojev/teacher.py data/gold`; inspect `teacher.py` and run tests | PASS — teacher unchanged; bounded waves, persistence, retry behavior retained | A1 |
| S9 | 9 | Gold filesystem/diff surface | `git diff --name-status af14ec7^ af14ec7 -- data/gold`; inspect branch tree | PASS — no `data/gold` writes or tree changes | A1 |
| S10 | 10 | Commit patch containment | `git diff-tree --no-commit-id --name-status -r af14ec7` | PASS — product/test changes only `kojev/label.py`, `tests/test_label.py`; evidence files are task-11 evidence only | A1 |
| S11 | 11 | Five validation gates | Targeted pytest; full pytest; Ruff check; Ruff format check; basedpyright with complete dependency environment | PASS — 7 passed; 77 passed; Ruff clean; 25 files formatted; basedpyright 0 errors/warnings/notes | A1 |
| S12 | 12 | Process and temp filesystem cleanup | `git worktree prune`; `find /tmp -maxdepth 1 -type d -name 'kojev-qa-*'`; process scan | PASS — disposable worktree/temp dirs removed; no KoJev/pytest/basedpyright process remained | A1 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | 1/5 | seeded completed candidate / pre-wave bypass | Existing candidate must not invoke injected transport; bypassing the filter must make the regression test fail | PASS — normal `0` calls; mutated code made test observe `1` call and fail | A1 |
| A2 | 2/6 | full existing-ledger replay / silent data rewrite | Replay must issue no requests, emit no new records, and preserve counts/cost/data | PASS — 33340 candidates, 0 calls, 0 new labels; recount remains 100020 / 37557 / 0 / `$2.9455706409999483` | A1 |
| A3 | 3 | nondeterministic resume identity | Repeated same state/family must yield identical ID, not per-attempt randomness | PASS — three equal SHA-256 IDs | A1 |
| A4 | 4 | post-response deduplication masquerading as pre-filter | Filter must execute before `_run_wave` and `_label_with_retries`, not after provider response | PASS — code order plus mutation failure establish pre-call filtering | A1 |
| A5 | 7 | provider/candidate identity conflation | Candidate identity must remain deterministic while provider ID remains response metadata and differs | PASS — synthetic `bf7b6748...` candidate vs `provider-123` provider ID; both persisted | A1 |
| A6 | 8/9/10 | scope creep / collateral mutation | Teacher and gold paths must be unchanged; patch must stay within allowed paths | PASS — teacher/gold diff clean; commit names only allowed product/test/evidence paths | A1 |
| A7 | 11/12 | verification hygiene / paid call / leaked temp state | Gates must be executed once; no paid provider call; no test/temp process remains | PASS — all five gates green in complete env; injected transports only; cleanup verified | A1 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | evidence bundle | Fresh detached-commit runtime probes, full data recount, gate output summary, code-order inspection, mutation regression result, metadata probe, containment and cleanup observations | `.omo/evidence/st_01a0c439/af14ec7-qa-evidence.md` |

## Gate note

An initial basedpyright run with the smaller `/Volumes/SSD1/1/KoJev/.venv` reported unrelated missing `torch`/`transformers` and training-surface errors. This was not accepted as a verdict: the complete dependency environment at `/Volumes/SSD1/1/KoJev-todo11/.venv` was then used, with `PYTHONPATH` set for the detached worktree, and produced `0 errors, 0 warnings, 0 notes`.

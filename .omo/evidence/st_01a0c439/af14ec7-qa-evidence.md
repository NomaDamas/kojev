# Adversarial verification evidence: af14ec7a885068cf3bfab875bdb98bca44735523

Date: 2026-09-20
Branch under test: `feat/todo-11-teacher-labels`
Commit under test: `af14ec7a885068cf3bfab875bdb98bca44735523`
Execution location: disposable detached worktree `/tmp/kojev-qa-kOQZNJ` (removed after verification).
Provider safety: all transport probes used `httpx.MockTransport`; no paid API call was made.

## Exact invocations and observed results

1. Targeted tests: `cd /tmp/kojev-qa-kOQZNJ && /Volumes/SSD1/1/KoJev/.venv/bin/python -m pytest tests/test_label.py -q`
   Result: `7 passed in 1.05s`.
2. Full tests: `cd /tmp/kojev-qa-kOQZNJ && /Volumes/SSD1/1/KoJev/.venv/bin/python -m pytest -q`
   Result: `77 passed in 3.87s`.
3. Ruff: `/Volumes/SSD1/1/KoJev/.venv/bin/ruff check kojev tests`; result `All checks passed!`.
4. Format: `/Volumes/SSD1/1/KoJev/.venv/bin/ruff format --check kojev tests`; result `25 files already formatted`.
5. Changed-surface basedpyright: `/Volumes/SSD1/1/KoJev-todo11/.venv/bin/basedpyright kojev/label.py tests/test_label.py`; result `0 errors, 0 warnings, 0 notes`.
6. Full basedpyright with the complete dependency environment: `PYTHONPATH=/Volumes/SSD1/1/KoJev-todo11/.venv/lib/python3.12/site-packages /Volumes/SSD1/1/KoJev-todo11/.venv/bin/basedpyright kojev tests`; result `0 errors, 0 warnings, 0 notes`.
7. Fresh injected-transport probe: a seeded usage record with request/candidate ID `already-issued` and a transport that increments a counter. Result: `new_labeled_count=0`, `transport_calls=0`, `output_exists=false`.
8. Fresh existing-ledger probe: loaded all preserved output examples from `/Volumes/SSD1/1/KoJev-todo11/data/distill/train.jsonl` and reran `LabelRunner` against its ledger with a transport that raises if called. Result: `candidate_count=33340`, `new_labeled_count=0`, `transport_calls=0`.
9. Determinism probe: computed `_candidate_id` three times for equivalent state/question families. Same SHA-256 each time: `19854adf1baa894470307a7800e2486faed0559ce9213906e41f3135ea4c89a9`; equality `true`.
10. Mutation probe: in the disposable worktree, removed both pre-wave identity predicates (`candidate_ids` and `issued`) and ran `tests/test_label.py::test_seeded_request_id_skips_transport -q`. Result: expected failure, `assert calls == 0`, observed `assert 1 == 0`; exit 1. The source was restored before teardown.
11. Synthetic metadata probe with MockTransport response ID `provider-123`: output had top-level provider `request_id=provider-123`, deterministic `candidate_id=bf7b67489c462faad62b3fdeb43f64c77ce5eae402b5a4f9408bdb063a0780a`, and question metadata with `request_id` equal to candidate ID and `provider_request_id=provider-123`; candidate/provider IDs were distinct.
12. Preserved-data recount from `/Volumes/SSD1/1/KoJev-todo11/data/distill`: `questions=100020`, `ledger_lines=37557`, `usage_lines=37557`, `unique_request_ids=37557`, `duplicate_request_ids=0`, `total_cost_usd=2.9455706409999483`, `output_ids_missing_from_ledger=0`, malformed JSON `0`.
13. Preserved data hashes observed after the no-write resume probe: `train.jsonl` SHA-256 `9e41acb60492ee43e767e88c97f2ef346454eff7bf2b426eb2a67b1b83f61ec7`; `ledger.jsonl` SHA-256 `05466eaf1d1ba41248a88309234a1d31538cb2153653b7330656e05b62f492b2`.
14. Containment: `git diff-tree --name-status -r af14ec7` reported only `.omo/evidence/task-11-kojev.txt`, `kojev/label.py`, `tests/test_label.py`; `git diff --exit-code af14ec7^ af14ec7 -- kojev/teacher.py data/gold` returned 0. The branch worktree had no `data/gold` tree and no status changes in `kojev/teacher.py` or `data/gold`.
15. Cleanup: disposable worktree removed, `git worktree prune` completed, no `/tmp/kojev-qa-*` directories remained, and no KoJev/pytest/basedpyright process remained. Existing unrelated tail/ssh shell jobs were not touched.

## Code-path observations

`LabelRunner.run` calls `_issued_records` first (lines 352-353), then builds `candidates` (lines 355-364) with `_goldless`, completed-state, candidate-ID, and issued-ID filters. Only after that does it enter bounded waves and call `_run_wave` (lines 366-369). `_run_wave` invokes `_label_with_retries`, so the identity filter is before any provider transport call. `_candidate_id` uses an explicit request ID from question metadata when present; otherwise it hashes canonical JSON containing state and the question family with SHA-256.

`_label_with_retries` retains the deterministic candidate ID in question metadata as `request_id`, while retaining the provider response ID as `provider_request_id`; `LabeledRecord.request_id` and top-level output `request_id` remain the provider ID, and output `candidate_id` carries the deterministic ID for newly generated records.

`kojev/teacher.py` is unchanged from the parent commit. Its bounded concurrency, per-request ledger persistence, and timeout/retry behavior remain present. No source or ref was edited in the base repository.

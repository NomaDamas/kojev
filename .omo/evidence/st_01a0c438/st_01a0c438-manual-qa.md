# Manual QA — todo 11 adversarial verification

Commit under test: `5c288155e9cdbab24295a21108e8b80a13147419` (`feat/todo-11-teacher-labels`)

Verdict: **needs-fix**
Safe to land: **false**

The run artifacts are real and the scale/count claims are confirmed, but the resume claim is not: the runner does not use ledger-issued request IDs to skip a candidate before making a new provider call. It only deduplicates a returned ID after the call. A local injected-transport probe seeded ledger ID `already-issued`; `LabelRunner.run` still made `transport_calls=1` and emitted a new record. This violates criterion 11 and can spend again after restart when output state is absent or mismatched.

Measured independently from `/Volumes/SSD1/1/KoJev-todo11/data/distill`:

- emitted output lines: **33,340**
- emitted questions: **100,020**
- ledger lines: **37,557**
- usage records: **37,557**
- unique ledger request IDs: **37,557**
- duplicate ledger request IDs: **0**
- recomputed ledger total: **$2.9455706409999483** (difference from claim: `-5.20e-14`)

The output request IDs are all present in the ledger (`0` missing); the ledger has 4,217 additional request IDs from provider calls whose results were not emitted, consistent with fallback/partial calls rather than fabrication. That join does not prove restart skipping, and the adversarial probe disproves the stronger claim.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| data-counts | 1, 2, 3 | preserved run JSONL | `/Volumes/SSD1/1/KoJev-todo11/data/distill/{train.jsonl,ledger.jsonl}`; one-pass independent Python `json.loads` streaming count/set/sum | PASS — 100,020 questions; 37,557 ledger lines; 37,557 unique IDs; 0 duplicates; $2.9455706409999483 | `data-audit` |
| schema-labels | 4, 5 | preserved output JSONL + `kojev.schema.Example` | same streaming audit, validating each output after removing only top-level run metadata (`label_source`, `request_id`, `cost_usd`) | PASS — 0 JSON errors, 0 blank states, 0 empty/invalid questions, 0 schema failures, 0 wrong top-level or per-question label sources | `data-audit` |
| provider-provenance | 6 | output/ledger join | streaming audit of output request IDs against ledger usage IDs; inspect ledger fields and distributions | PASS — every output ID joins a usage record; all 37,557 records are model `qwen/qwen3-vl-8b-instruct`, have prompt/completion/total token fields and logprobs; 598 distinct token/cost tuples; answer distributions are non-constant | `data-audit`, `teacher-code` |
| gold-containment | 7 | commit diff + worker checkout data tree | `git diff --name-status 5c288155^ 5c288155`; inspect `/Volumes/SSD1/1/KoJev-todo11/data/gold` and output `source` values | PASS — no `data/gold` diff; worker checkout has no gold output; all 33,340 emitted rows are `KorQuAD/squad_kor_v1:goldless` | `gold-surface`, `commit-diff`, `data-audit` |
| patch-containment | 8 | Git commit tree | `git diff --name-status 5c288155^ 5c288155` | PASS — exactly `.omo/evidence/task-11-kojev-manual-qa.md`, `.omo/evidence/task-11-kojev.txt`, `kojev/label.py`, `tests/test_label.py`; `kojev/teacher.py` unchanged | `commit-diff` |
| five-gates | 9 | branch checkout | `uv run pytest tests/test_label.py -q`; `uv run pytest -q`; `uv run ruff check kojev tests`; `uv run ruff format --check kojev tests`; `uv run basedpyright kojev tests` in `/Volumes/SSD1/1/KoJev-todo11` | PASS — 6 passed; 76 passed; ruff clean; format clean; basedpyright 0/0/0 | `gates-worker-checkout` |
| wave-resume-code | 10, 11 | committed `kojev/label.py` | inspect `WAVE_SIZE`, `run`, `_run_wave`, `_label_with_retries`, `_issued_records`; regression tests via `uv run pytest tests/test_label.py -q` | **FAIL for criterion 11** — explicit waves are awaited and timeout/skips behave correctly, but `issued` is only checked after `_label_with_retries` returns. The runner filters completed output states, not candidate request IDs from the ledger. | `scheduling-code`, `resume-ledger-only-probe`, `gates-worker-checkout` |
| cleanup | 12 | OS process/temp state | `ps -axo ...` plus removal/check of disposable `/private/tmp/kojev-todo11-verify.*` and probe files | PASS — no verifier processes/temp dirs remain; one pre-existing unrelated `tail -n 0 -F /tmp/kojev-deliver.log` was observed during the run and was not created by this verification | `processes`, `cleanup` |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| duplicate-request-ids | 2, 11 | duplicate identity / restart | repeated provider IDs must not be emitted twice | PASS for artifact uniqueness — ledger arithmetic found 0 duplicate IDs; output IDs are unique; code checks `issued` before append | `data-audit`, `scheduling-code` |
| ledger-output-mismatch | 6 | orphaned output / fabricated row | every emitted row must reference a real ledger usage ID | PASS — output IDs missing from ledger: 0 | `data-audit` |
| constant-usage-placeholder | 6 | placeholder accounting | usage and cost must vary with provider response fields | PASS — 598 distinct token/cost tuples across 37,557 usage records; all records contain nonzero token fields and logprobs | `data-audit`, `teacher-code` |
| constant-answer-placeholder | 6 | templated answer fabrication | labels must not be one fixed answer pattern | PASS — choice answers span all 8 values; noul and score answers span their legal alternatives; output rows carry varying provider IDs | `data-audit` |
| schema-corruption | 5 | malformed/blank data | reject or avoid blank state, empty questions, invalid schema | PASS — 0 malformed JSON, 0 blank states, 0 empty questions, 0 `Example` validation failures | `data-audit` |
| gold-contamination | 7 | dataset-gold leakage | existing gold families must not receive teacher labels | PASS — emitted source is exclusively `KorQuAD/...:goldless`; no `data/gold` change in commit or worker output tree | `gold-surface`, `data-audit` |
| timeout-in-wave | 10 | transient timeout | timeout retries within the current wave and does not abort the run | PASS — deterministic regression test included in commit and passed; `_label_with_retries` catches `TimeoutError` for two attempts | `gates-worker-checkout`, `scheduling-code` |
| skipped-item | 10 | exhausted/skip path | a skipped item must not abort subsequent work | PASS by code path — `record is None` returns from item task; task group continues and wave append proceeds; six label tests and full suite pass | `scheduling-code`, `gates-worker-checkout` |
| unbounded-scheduling | 10 | concurrency pressure | next wave must not be created until current wave completes | PASS — `records = await self._run_wave(...)` precedes next loop iteration; bounded-wave regression test passed with peak 2 for wave size 2 | `scheduling-code`, `gates-worker-checkout` |
| ledger-issued-restart-skip | 11 | resume / duplicate spend | a request ID already present in ledger must be skipped before a new provider call | **FAIL** — seeded ledger ID `already-issued` still resulted in `transport_calls=1`, `new_labeled_count=1`, `ledger_lines=2`, `output_lines=1` | `resume-ledger-only-probe`, `scheduling-code` |
| live-paid-provider-call | task constraint | paid API invocation | verification must not issue paid calls | PASS — no labeling CLI was run; only preserved artifacts, local tests, and local parsing were used | `processes`, `gates-worker-checkout` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `data-audit` | json | Independent one-pass counts, schema validation results, ID join, cost sum, usage/cost/answer variability | `.omo/evidence/st_01a0c438/data-audit.json` |
| `commit-diff` | text | Exact commit patch file list and unchanged teacher/gold checks | `.omo/evidence/st_01a0c438/commit-diff.txt` |
| `gold-surface` | text | Worker checkout `data/gold` and data-tree inspection | `.omo/evidence/st_01a0c438/gold-surface.txt` |
| `teacher-code` | text | Provider ledger implementation inspection showing response ID, usage-derived cost, and token fields | `.omo/evidence/st_01a0c438/teacher-code.txt` |
| `scheduling-code` | text | `label.py` scheduling/resume line evidence | `.omo/evidence/st_01a0c438/scheduling-resume-code.txt` |
| `resume-ledger-only-probe` | json | Paid-free injected-transport probe proving ledger-issued ID is not skipped before a provider call | `.omo/evidence/st_01a0c438/resume-ledger-only-probe.json` |
| `gates-worker-checkout` | text | Exact five gate invocations and output on the branch checkout | `.omo/evidence/st_01a0c438/gates-worker-checkout.txt` |
| `processes` | text | Process scan after QA | `.omo/evidence/st_01a0c438/processes.txt` |
| `cleanup` | text | Disposable verifier cleanup checks | `.omo/evidence/st_01a0c438/cleanup.txt` |

Notes:

- The committed worker evidence's original live scale invocation was not repeated because this verification was explicitly forbidden from making paid API calls.
- A disposable export gate run initially lacked the exported tree's dependency environment and produced dependency-resolution basedpyright errors; this is not a product failure. The required five gates were then rerun in the real branch checkout with `uv run`, producing the claimed clean results, and those outputs are the authoritative gate artifact.

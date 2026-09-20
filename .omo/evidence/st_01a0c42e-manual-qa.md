# Adversarial manual QA — todo 15 commit `01fde17d6f3836d828dad4ee189d51bfe5871abc`

Repository: `/Volumes/SSD1/1/KoJev` (read-only verification). The commit was archived to `/tmp/kojev-qa-01fde17` and tested there; no source files, refs, or real worktree code were edited.

## Verdict

**confirmed** — **safe_to_land: true**.

The target commit exists at `refs/remotes/origin/feat/todo-15-serve`, and the commit diff is contained exactly to `kojev/serve.py`, `tests/test_serve.py`, `tests/fixtures/req.json`, and `.omo/evidence/task-15-kojev.txt`. The trained todo-12 checkpoint is absent as expected; `KOJEV_CKPT=tiny-random` was used as the permitted stand-in.

## Manual QA matrix

### `surfaceEvidence`

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | 3 happy HTTP request | Real HTTP, uvicorn bound to loopback | `PYTHONPATH=/tmp/kojev-qa-01fde17 KOJEV_CKPT=tiny-random /Volumes/SSD1/worktrees/kojev-todo-15/.venv/bin/uvicorn kojev.serve:app --host 127.0.0.1 --port 18931`; then `curl -sS -i -X POST http://127.0.0.1:18931/v1/systemone -H 'Content-Type: application/json' -d @tests/fixtures/req.json` | PASS; observed `HTTP/1.1 200 OK`, answer keys `choice`, `score`, `noul`; each had typed field, float confidence, and probability sum within float precision of 1 | A7, A8, A9 |
| S2 | 3 300-option rejection | Real HTTP curl | `curl -sS -i -X POST http://127.0.0.1:18931/v1/systemone ... -d '<JSON choice with options 0..299>'` | PASS; observed `HTTP/1.1 422 Unprocessable Entity`, `detail.error_type=schema_error`, non-empty `detail.message` | A10, A11 |
| S3 | 3 missing-type rejection | Real HTTP curl | `curl -sS -i -X POST http://127.0.0.1:18931/v1/systemone ... -d '{"model":"tiny-random","state":"x","questions":{"broken":{"instructions":"q","options":["a","b"]}}}'` | PASS; observed `HTTP/1.1 422 Unprocessable Entity`, `detail.error_type=validation_error`, non-empty `detail.message` | A12, A11 |
| S4 | 5 gate target | Archived commit CLI | `pytest tests/test_serve.py -q` | PASS; `4 passed in 2.59s` | A1 |
| S5 | 5 full suite | Archived commit CLI | `pytest -q` | PASS; `60 passed in 3.04s` | A2 |
| S6 | 5 lint/format/type gates | Archived commit CLI | `ruff check kojev tests`; `ruff format --check kojev tests`; `basedpyright kojev tests` | PASS; all checks passed, 19 files formatted, 0 errors/0 warnings/0 notes | A3, A4, A5 |
| S7 | 7 determinism | Archived commit CLI | Three sequential invocations of `pytest tests/test_serve.py -q` | PASS; all three reported `4 passed` with no failures | A6 |
| S8 | 6 criteria mapping and round-trip kinds | Direct typed boundary exercise plus S1 | `python - <<'PY' ... WireQuestion(... criteria=[...]); WireQuestion(... criteria={...}); ...` | PASS; list and dict both yielded ordered schema options; choice, score, noul options mapped and S1 round-tripped all three answer fields | A13, A7, A8 |
| S9 | 5 scope/security | OpenAPI HTTP surface and source inspection | `curl -sS http://127.0.0.1:18931/openapi.json`; `grep -nE 'KOJEV_CKPT|uvicorn.run|host=|/v1/systemone|generate|auth|security' kojev/serve.py tests/test_serve.py` | PASS; only `/v1/systemone` path; no security schemes; source uses `KOJEV_CKPT` and `host="127.0.0.1"`; no auth or generation endpoint | A14, A15 |

### `adversarialCases`

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A-S1 | 3 | cardinality boundary | Choice with 300 options is rejected with structured 422 | PASS | A10, A11 |
| A-S2 | 3 | missing discriminator | Question without `type` is rejected with structured 422 containing both error fields | PASS | A12, A11 |
| A-S3 | 6 | alternate wire representation | Criteria list and criteria dict both become schema option lists without losing order | PASS | A13 |
| A-S4 | 6 | mixed tagged variants | Choice, score, and noul all produce their corresponding typed answer field in one request | PASS | A7, A8 |
| A-S5 | 7 | nondeterminism/repeatability | Target test suite remains green on three separate runs without sleeps or network downloads | PASS | A6 |
| A-S6 | 5 | endpoint exposure | OpenAPI exposes only the intended route and no auth or generation surface | PASS | A14, A15 |
| A-S7 | 4 | teardown/process leak | QA-owned server must terminate, `kill -0` must fail, and QA-owned port must be unbound | PASS | A16, A17 |
| A-S8 | 4 | pre-existing process contamination | No unrelated process may be falsely attributed to this run | not_applicable — a pre-existing worker was observed before QA on PID 57914/port 8930 and was not touched | A18 |

## Gate and containment evidence

- `origin/feat/todo-15-serve` resolves exactly to `01fde17d6f3836d828dad4ee189d51bfe5871abc`.
- Commit diff names exactly the four requested paths.
- No `pyproject.toml` diff exists in the target commit.
- No new suppressions were added by the target commit; the only suppression scan hits are pre-existing lines in `kojev/augment.py` and `tests/test_data_gold.py`.
- No test or gate used sleeps or network downloads.

## Teardown receipt

QA-owned server: PID `89526`, port `127.0.0.1:18931`.

Observed receipt: `kill -0 89526: exit 1 (expected nonzero)`; `ps -p 89526` empty; `lsof -nP -iTCP:18931 -sTCP:LISTEN` empty. Uvicorn log ends with `Finished server process [89526]`.

Pre-existing unrelated process observed before this run: PID `57914`, parent `57912`, listening on `127.0.0.1:8930` from `/Volumes/SSD1/worktrees/kojev-todo-15`; it remains because killing it would modify another worker's runtime and is outside this read-only verification.

## `artifactRefs`

| id | kind | description | path |
|---|---|---|---|
| A1 | test-log | Targeted serving suite | `.omo/evidence/qa-todo15-01fde17/target-pytest.txt` |
| A2 | test-log | Full pytest suite | `.omo/evidence/qa-todo15-01fde17/full-pytest.txt` |
| A3 | lint-log | Ruff check | `.omo/evidence/qa-todo15-01fde17/ruff-check.txt` |
| A4 | format-log | Ruff format check | `.omo/evidence/qa-todo15-01fde17/ruff-format.txt` |
| A5 | type-log | Basedpyright | `.omo/evidence/qa-todo15-01fde17/basedpyright.txt` |
| A6 | test-log | Three target-suite runs | `.omo/evidence/qa-todo15-01fde17/target-repeat-3x.txt` |
| A7 | http-transcript | Happy `curl -i` status and body | `.omo/evidence/qa-todo15-01fde17/curl-happy.txt` |
| A8 | http-analysis | Parsed answer fields and probability sums | `.omo/evidence/qa-todo15-01fde17/response-validation.txt` |
| A9 | server-log | Uvicorn request log for happy and error requests | `.omo/evidence/qa-todo15-01fde17/uvicorn-log.txt` |
| A10 | http-transcript | 300-option `curl -i` status and body | `.omo/evidence/qa-todo15-01fde17/curl-300-option.txt` |
| A11 | http-analysis | Structured error detail validation | `.omo/evidence/qa-todo15-01fde17/response-validation.txt` |
| A12 | http-transcript | Missing-type `curl -i` status and body | `.omo/evidence/qa-todo15-01fde17/curl-missing-type.txt` |
| A13 | cli-output | Criteria list/dict and three-kind mapping output | `.omo/evidence/qa-todo15-01fde17/criteria-mapping.txt` |
| A14 | http-analysis | OpenAPI route/security inspection | `.omo/evidence/qa-todo15-01fde17/openapi-surface.txt` |
| A15 | source-inspection | Bind/env/endpoint/suppression scan | `.omo/evidence/qa-todo15-01fde17/static-scope.txt` |
| A16 | teardown-log | PID and port teardown receipt | `.omo/evidence/qa-todo15-01fde17/teardown.txt` |
| A17 | server-log | Uvicorn graceful shutdown | `.omo/evidence/qa-todo15-01fde17/uvicorn-log.txt` |
| A18 | process-observation | Pre-run process inventory showing unrelated worker | `.omo/evidence/qa-todo15-01fde17/preexisting-process.txt` |

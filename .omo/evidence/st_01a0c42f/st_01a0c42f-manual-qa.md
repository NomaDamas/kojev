# Manual QA — todo 11 teacher labeling

## Verdict

**FAIL — blocked before implementation/runtime verification.** The inspected checkout is `/Volumes/SSD1/1/KoJev` on `main` at `6cb06184084bb4e3ddbbe1e2faa552bc22817450`. The requested branch `feat/todo-11-teacher-labels` is not present in the local worktrees or `origin`. The task-11 implementation, tests, evidence, ledger, and output files are absent. The required target test invocation fails with `ERROR: file or directory not found: tests/test_label.py` and exit code 4.

The API key is present in the environment, but no paid run was started: the implementation and required safety checks are absent, so starting a paid job would not be a valid or safe verification of the requested change.

## `surfaceEvidence`

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Requested branch/commit provenance | Git CLI | `git worktree list --porcelain; git ls-remote origin refs/heads/feat/todo-11-teacher-labels; git log --all --oneline --decorate -- kojev/label.py tests/test_label.py .omo/evidence/task-11-kojev.txt` | FAIL — live checkout is `main` at `6cb06184084bb4e3ddbbe1e2faa552bc22817450`; requested branch has no `ls-remote` result; no task-11 paths appear in history. | A1 |
| S2 | Required implementation/test presence | Filesystem CLI | `for p in kojev/label.py tests/test_label.py .omo/evidence/task-11-kojev.txt data/distill/ledger.jsonl data/distill/train.jsonl; do test -e "$p" ...; done` | FAIL — all five required artifacts are missing. | A2 |
| S3 | Required focused gate | Pytest CLI | `uv run pytest tests/test_label.py -q` | FAIL — `ERROR: file or directory not found: tests/test_label.py`; `no tests ran in 0.00s`; exit code 4. | A3 |
| S4 | Paid-run prerequisite observation | Environment CLI | `test -n "${OPENROUTER_API_KEY:-}"` | PASS as an environment observation only — API key is present; this does not establish that a labeling runner or budget projection exists. | A2 |

## `adversarialCases`

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | Hard budget <= $20 and pilot projection | budget-boundary / paid execution | A small pilot is run, explicit arithmetic is recorded, and production aborts if projected spend exceeds $20. | FAIL — no runner, pilot, projection, or task evidence exists; paid execution was not attempted. | A2 |
| A2 | Reuse landed teacher client and append-only ledger | integration / ledger safety | The real runner uses `kojev.teacher.TeacherClient`, writes append-only `data/distill/ledger.jsonl`, and stops on `BUDGET_REACHED`. | FAIL — no task-11 runner or ledger exists to exercise. | A2 |
| A3 | Gold-less family construction | data provenance / contamination | Fresh Korean states without dataset gold receive teacher labels; families with dataset gold receive none. | FAIL — no `label.py`, input build, or output corpus exists to inspect. | A2 |
| A4 | Label metadata contract | output schema | Every emitted question carries `label_source: "teacher:qwen3-vl-8b-instruct"`. | FAIL — `data/distill/train.jsonl` is missing, so no emitted question can be checked. | A2 |
| A5 | Resume-from-ledger | interruption/restart / duplicate request IDs | Kill mid-flight, restart, and prove request-id set size equals ledger line count with no duplicates. | FAIL — no runnable labeling job or ledger exists; interruption/restart proof cannot be produced. | A2 |
| A6 | Target >=100k within budget | scale/cost boundary | Real run reaches at least 100k questions without exceeding $20, or stops with exact arithmetic shortfall. | FAIL — no pilot or production run exists, so counts and spend are unavailable. | A2 |
| A7 | Output isolation | filesystem contamination | Output is written to `data/distill/train.jsonl` and never to `data/gold`. | FAIL — required output and runner are absent; no write surface was exercised. | A2 |
| A8 | Full required gates | regression/static quality | Focused pytest, full pytest, Ruff, format, and basedpyright all pass without suppression/config weakening. | FAIL — focused gate already fails due to missing test file; remaining task-specific gates cannot be meaningfully run. | A3 |

## `artifactRefs`

| id | kind | description | path |
|---|---|---|---|
| A1 | git-transcript | Worktree inventory, remote branch lookup, task-11 path history, and live status | `.omo/evidence/st_01a0c42f/git-provenance.txt` |
| A2 | prerequisite-transcript | Presence checks for implementation/tests/evidence/ledger/output and API-key observation | `.omo/evidence/st_01a0c42f/artifact-prerequisites.txt` |
| A3 | test-transcript | Exact required focused pytest invocation and failure output | `.omo/evidence/st_01a0c42f/target-gate.txt` |

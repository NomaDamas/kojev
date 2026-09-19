# Todo 3 AdversarialVerify Manual QA

Audited commit `001c8c46043d8a9083a6ed6d6ff8c8a86a95fc5d` in `/Volumes/SSD1/1/KoJev-todo3`, branch `feat/data-gold`, read-only apart from this untracked audit artifact. Local `HEAD` and `origin/feat/data-gold` both resolved to the target SHA before the audit. Full command output is in `task-3-adversarial-audit.txt`.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Repository provenance and clean containment | Git CLI | `git -C /Volumes/SSD1/1/KoJev-todo3 status --short --branch; git rev-parse HEAD; git rev-parse origin/feat/data-gold; git diff --check HEAD^ HEAD; git diff-tree --no-commit-id --name-only -r HEAD` | PASS - local and remote branch SHA are both the target; commit patch has five contained paths and no whitespace errors. The only post-checkout dirty paths are this audit's untracked artifacts. | A1, A2 |
| S2 | Target tests rerun | Pytest CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run pytest tests/test_data_gold.py tests/test_schema.py -q` | PASS - `12 passed in 0.03s`. | A2 |
| S3 | Full test regression check | Pytest CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run pytest -q` | PASS - `12 passed in 0.03s`. | A2 |
| S4 | Real corpus build | CLI + installed Hugging Face datasets/cache | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --out /tmp/kojev-gold-qa-audit` | PASS for execution and real-source use - exit 0; loader used installed `datasets` and existing Hugging Face cache fallback. This is not fresh-online provenance proof. | A2, A3 |
| S5 | Corpus counts and output files | Generated JSONL + summary | `uv run python - <<'PY' ... summary.json and count each *.jsonl ... PY` after S4 | NEEDS-FIX - summary reports 12 unique sources, 96,000 train states, and 302,085 train questions; generated files contain `train.jsonl=103,999`, `val.jsonl=2,500`, `test.jsonl=1,000`. The committed evidence's claim that train has 96,000 JSONL lines is false; it conflates states with emitted examples. | A2, A3 |
| S6 | Clean validation | CLI validation | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run python -m kojev.data_gold --validate /tmp/kojev-gold-qa-audit` | PASS - exit 0 and no diagnostics. | A2 |
| S7 | Forbidden-source and split exclusions | Summary inspection | `print([k for k in summary if 'kobest' in k.lower()]); print([k for k in summary if 'korquad' in k.lower()])` | PASS - `kobest_entries=[]`; KorQuAD is only `KorQuAD/squad_kor_v1:train`. | A2 |
| S8 | Static quality gates | Ruff and basedpyright CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && uv run ruff check kojev/data_gold.py tests/test_data_gold.py; uv run basedpyright kojev/data_gold.py tests/test_data_gold.py` | FAIL - Ruff reports 50 findings; basedpyright reports 27 errors and 6 warnings. The committed evidence honestly records these failures. | A2, A4 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | Malformed JSON corruption proof | Malformed JSON / line localization | Validation exits nonzero and identifies exact file and line. | PASS - `/tmp/kojev-gold-corrupt-audit/train.jsonl:2: invalid JSON`, exit 1. | A2 |
| A2 | Schema-invalid corruption proof | Schema validation boundary | Syntactically valid JSON with an invalid empty `questions` collection should be rejected. | FAIL - replacing line 2 with `{"state":"x","questions":[],"source":"fixture","split":"train"}` exits 0. `Example` permits an empty question list, so the claimed proof is incomplete. | A2, A5 |
| A3 | Forbidden-source exclusion | Contamination exclusion | KoBEST must not appear in generated summary or outputs. | PASS - summary measured `kobest_entries=[]`. | A2 |
| A4 | Evaluation split exclusion | Split contamination | KorQuAD must be train-only with no dev/validation entry. | PASS - exact summary entry is `KorQuAD/squad_kor_v1:train`. | A2 |
| A5 | Real-source availability | Network/cache availability | Build should use actual configured sources; cache fallback must be observable if remote access is unavailable. | PASS with environment qualification - build completed from actual cached datasets and logs record Hugging Face cached-version fallback. No fresh-online provenance can be claimed here. | A2 |
| A6 | Static failure classification | Quality-gate failure | Diagnostics are reported without suppression and classified. | PASS as reporting; underlying gate FAIL. Direct defects: unused helpers/import, line length, complexity, exception-style, print, and random-security lint findings. Missing dependency/configuration: absent `datasets` typing stubs and downstream unknown types. No independent environment-only static failure observed. | A2, A4 |
| A7 | Evidence honesty | Evidence integrity | Committed evidence matches independently observed counts and outcomes. | FAIL - tests/static/malformed-JSON outcomes are honestly recorded, but the claimed `train=96,000` JSONL line count is contradicted by the generated artifact (`103,999`). | A1, A2, A3 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | committed evidence | Author-provided task 3 transcript and claimed measurements | `/Volumes/SSD1/1/KoJev-todo3/.omo/evidence/task-3-kojev.txt` |
| A2 | audit transcript | Independent provenance, tests, build, counts, validation, corruption, Ruff, and basedpyright output | `/Volumes/SSD1/1/KoJev-todo3/.omo/evidence/task-3-adversarial-audit.txt` |
| A3 | generated corpus | Independent real-build output used for counts and source/split inspection | `/tmp/kojev-gold-qa-audit/` |
| A4 | static diagnostics | Complete Ruff and basedpyright output in A2 | `/Volumes/SSD1/1/KoJev-todo3/.omo/evidence/task-3-adversarial-audit.txt` |
| A5 | schema-invalid fixture | Audit-generated copy with line 2 replaced by a syntactically valid empty-question example | `/tmp/kojev-schema-corrupt-audit/train.jsonl` |

## AdversarialVerify verdict

**NEEDS-FIX; do not safely land as-is.** The commit is reproducible at the pinned local/remote SHA, target/full tests pass, real cached Hugging Face sources build successfully, and KoBEST/KorQuAD exclusions are evidenced. Landing is blocked by two concrete issues: (1) committed corpus line-count evidence is inaccurate (`103,999` emitted train examples versus claimed `96,000` lines), and (2) validation accepts a schema-invalid empty-question example, so corruption/schema proof is incomplete and the product boundary likely needs a minimum-question invariant. Static checks also fail: Ruff's direct changed-file findings should be corrected; basedpyright's `datasets` stub/type-coverage failures require dependency or checker configuration, while unused-helper findings are direct defects. No evidence supports classifying these static failures as environment-only.

Recommended corrective scope: correct and rerun evidence/counting; add a schema/validation test and invariant for non-empty `Example.questions` if empty examples are not intended; remove or use the two unused helpers and unused pytest import; address Ruff findings in changed files; configure or provide typing coverage for `datasets` and rerun basedpyright. Do not suppress diagnostics or broaden into unrelated cleanup.

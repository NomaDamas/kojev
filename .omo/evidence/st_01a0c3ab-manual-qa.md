# Manual QA Matrix — KoJev Todo 7 Adversarial Verification

Verdict: **needs-fix**
Safe to land: **false**
Reason: the implementation and required quality gates pass, but the cleanup/no-temp-artifact criterion is not satisfied because the worker left `/tmp/kojev-todo-7-driver.txt` behind. The requested commit is otherwise confirmed at the expected local and remote SHA with a clean repository.

## surfaceEvidence

| Scenario | Criterion reference | Surface | Exact invocation | Verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | commit identity and containment | Git repository | `git -C /Volumes/SSD1/1/KoJev-todo-7 status --short --branch; git ... rev-parse HEAD; git ... rev-parse refs/remotes/origin/feat/todo-7-augment; git ... diff-tree --no-commit-id --name-status -r 774e2f80...` | PASS | `A1` |
| S2 | full regression suite | Python test surface | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run pytest -q` | PASS | `A1` |
| S3 | augmentation target tests | Python test surface | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run pytest -q tests/test_augment.py` | PASS | `A1` |
| S4 | lint | Python project tooling | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run ruff check .` | PASS | `A1` |
| S5 | type checking | Python project tooling | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run basedpyright` | PASS | `A1` |
| S6 | independent 1000-example semantics and required option/template/noul checks | Python data-shaped API | `cd /Volumes/SSD1/1/KoJev-todo-7 && uv run python - <<'PY' ... PY` | PASS | `A1` |
| S7 | malformed boundary inputs | Python data-shaped API | Fresh driver constructs invalid choice/noul/score `Question.model_validate` payloads and invalid probabilities `(-.01, 1.01, NaN, +Inf, -Inf)` | PASS | `A1` |
| S8 | cleanup and no leftover task artifacts/processes | OS/filesystem/process surface | `git diff --exit-code; ps -axo ...; test ! -e /tmp/kojev-todo-7-driver.txt; find /Volumes/SSD1/1/KoJev-todo-7 ...` | FAIL | `A1` |

## adversarialCases

| Scenario | Criterion reference | Adversarial class | Expected behavior | Verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV1 | gold preservation | independent semantic identity | A transformed choice's gold must point to the original option identity, not merely a rederived output index | PASS | `A1` |
| ADV2 | option remap | repeated deterministic seeds | 1,000 independently seeded shuffles preserve the original gold option | PASS | `A1` |
| ADV3 | distractor never gold | negative dropped-gold invariant | A candidate that removes the original gold must fail the semantic invariant | PASS | `A1` |
| ADV4 | >=8 templates | template coverage | Each `QuestionType` exposes at least eight fixed instruction templates; fresh outputs reach at least eight distinct templates | PASS | `A1` |
| ADV5 | josa behavior | Korean final-consonant boundary | `감정` produces 은/이 forms and `주제` produces 는/가 forms across fresh seeded outputs | PASS | `A1` |
| ADV6 | score order | synonym remap | Synonyms remain in the same rank bucket and gold index remains unchanged across 1,000 cases | PASS | `A1` |
| ADV7 | noul exact flip | binary label adversary | Exact `이 리뷰는 긍정적이다` -> `이 리뷰는 긍정적이 아니다`, with gold 0 <-> 1 | PASS | `A1` |
| ADV8 | repeatability | stale seed / global RNG state | Equal seeds produce equal results; unrelated prior RNG consumption does not change output | PASS | `A1` |
| ADV9 | misleading pass/rederived gold | duplicate-label adversary | A value-only/rederived-gold check must be demonstrably distinguishable from original indexed identity | PASS | `A1` |
| ADV10 | malformed inputs | boundary rejection | Invalid option cardinality, out-of-range gold, and invalid probability values are rejected | PASS | `A1` |
| ADV11 | cleanup/no artifacts | stale temp artifact | No task-specific temporary files should remain after execution | FAIL — `/tmp/kojev-todo-7-driver.txt` exists from the worker run | `A1` |
| ADV12 | HTTP/browser/desktop GUI | applicability | Not applicable: this change is a pure Python data-transform module and exposes no HTTP, browser, or desktop GUI surface | NOT_APPLICABLE — one-line reason | `A1` |

## artifactRefs

| ID | kind | description | path |
|---|---|---|---|
| A1 | runtime transcript | Git truth, required test/lint/type results, fresh independent driver output, malformed/adversarial probes, and cleanup/process observations | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c3ab-runtime.txt` |
| A2 | QA matrix | This manual QA matrix and final adversarial verdict | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c3ab-manual-qa.md` |

## Final AdversarialVerify

- Verdict: **needs-fix**
- safe_to_land: **false**
- Confirmed facts: local HEAD and `origin/feat/todo-7-augment` both equal `774e2f80ff317b4cd5cba8dd6834e8486f64d814`; worktree is clean; commit containment is exactly the four expected added paths; `24` full-suite tests and `15` target tests pass; Ruff and basedpyright pass; fresh independent 1,000-example and adversarial driver passes all semantic checks.
- Blocking finding: `/tmp/kojev-todo-7-driver.txt` remains after the worker run. The implementation itself did not fail any tested semantic or quality criterion.

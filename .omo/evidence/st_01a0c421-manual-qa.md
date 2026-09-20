# Todo 3 AdversarialVerify Manual QA

Date: 2026-09-19  
Target worktree: `/Volumes/SSD1/1/KoJev-todo3`  
Branch: `feat/data-gold`  
Claimed corrective commit: `6947a0fb5422eee694c1ba49d8095a38d63772a9`

## Verdict

**AdversarialVerify: needs-fix**  
**safe_to_land: false**

The claimed commit is present locally and remotely at the same SHA. Scoped Ruff and basedpyright, target pytest, full pytest, the real cached 12-source build, clean validation, malformed-JSON localization, empty-question rejection, source/split exclusions, source/count floors, and deterministic rebuild all pass. However, direct inspection of the emitted corpus found 8 schema-valid examples with blank `state` values (5 train and 3 test, all from `e9t/nsmc`). The commit enforces non-empty questions but not non-empty state text. This is a concrete data-validity gap, so the commit is not safe to land without an explicit decision/fix for blank source text.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Commit provenance and patch containment | Git CLI | `git -C /Volumes/SSD1/1/KoJev-todo3 rev-parse HEAD; git -C /Volumes/SSD1/1/KoJev-todo3 ls-remote origin refs/heads/feat/data-gold; git -C /Volumes/SSD1/1/KoJev-todo3 diff-tree --no-commit-id --name-only -r 6947a0fb5422eee694c1ba49d8095a38d63772a9` | PASS - local HEAD and remote branch both equal `6947a0fb5422eee694c1ba49d8095a38d63772a9`; patch is limited to 7 expected paths. | R1 |
| S2 | Scoped static quality gates | Ruff and basedpyright CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/ruff check kojev/data_gold.py kojev/schema.py tests/test_data_gold.py && .venv/bin/basedpyright kojev/data_gold.py kojev/schema.py tests/test_data_gold.py` | PASS - Ruff clean; basedpyright reports 0 errors, 0 warnings, 0 notes. | R1 |
| S3 | Target regression suite | Pytest CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/pytest tests/test_data_gold.py tests/test_schema.py -q` | PASS - 13 passed in 0.47s. | R1 |
| S4 | Full regression suite | Pytest CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/pytest -q` | PASS - 13 passed in 0.47s. | R1 |
| S5 | Real configured-source build | Python CLI with cached Hugging Face datasets | `cd /Volumes/SSD1/1/KoJev-todo3 && HF_DATASETS_OFFLINE=1 .venv/bin/python -m kojev.data_gold --out /tmp/kojev-todo3-6947a0f-run1` | PASS with environment qualification - exit 0; all 12 configured real datasets loaded from local Hugging Face cache, and no synthetic/substitute rows were used. | R1 |
| S6 | Output counts and floors | Python parsed summary/JSONL output | `.venv/bin/python` script over `/tmp/kojev-todo3-6947a0f-run1/summary.json` and each JSONL file | PASS - configured sources=12; selected states train/val/test=96000/2500/1000; emitted lines=103999/2500/1000; all 12 sources occur in train, 5 in val, 1 in test; all emitted examples have >=1 question and valid gold indexes. | R1 |
| S7 | Clean output validation | Python CLI | `cd /Volumes/SSD1/1/KoJev-todo3 && .venv/bin/python -m kojev.data_gold --validate /tmp/kojev-todo3-6947a0f-run1` | PASS - exit 0 with no diagnostics. | R1 |
| S8 | Deterministic rebuild | Python CLI + byte comparison | Run the same build to `/tmp/kojev-todo3-6947a0f-run2`, then `cmp -s` train/val/test/summary files | PASS - second build exit 0; all four output files byte-identical to run 1. | R1 |
| S9 | Emitted-state validity | Python parsed corpus probe | `Example.model_validate_json` over all emitted lines, plus `state.strip()==''` scan | FAIL / needs-fix - all rows are schema-valid, but 8 rows have blank state: `train.jsonl:2310-2314` and `test.jsonl:5-7`, all source `e9t/nsmc`. | R1 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | Empty-question rejection | Schema-boundary invalid data | A syntactically valid line with `questions: []` must fail validation with file and line number. | PASS - exit 1 and `/tmp/kojev-todo3-6947a0f-empty/train.jsonl:2: Value error, examples require at least one question (minimum 1)`. | R1 |
| A2 | Malformed JSON line localization | Malformed JSON | A malformed line must fail nonzero and identify exact file and line. | PASS - exit 1 and `/tmp/kojev-todo3-6947a0f-malformed/train.jsonl:2: invalid JSON`. | R1 |
| A3 | Forbidden-source exclusion | Contamination / forbidden dataset | KoBEST must be absent from summary and emitted source values. | PASS - `kobest_keys=[]`; no KoBEST source appears in parsed outputs. | R1 |
| A4 | Evaluation split exclusion | Train-only source split | KorQuAD must appear only as train, not val/dev/test. | PASS - exact summary key is `KorQuAD/squad_kor_v1:train`; no other KorQuAD split key. | R1 |
| A5 | Synthetic substitution | Source provenance / fallback integrity | Build must use configured real sources, not generated substitute rows. | PASS with qualification - loader output explicitly identifies each Hugging Face cached dataset/config; no synthetic-row path exists in the build output. This proves cached real-source execution, not fresh-online provenance. | R1 |
| A6 | Source count and count floors | Coverage/count integrity | Build must use 12 configured sources and meet the observed selected-state and emitted-line floors. | PASS - 12 configured sources; train/val/test selected states 96000/2500/1000; emitted lines 103999/2500/1000. | R1 |
| A7 | Evidence count semantics | State-vs-emitted-line mismatch | Evidence must distinguish selected source states from emitted examples, including KorQuAD expansion. | PASS - independent measurement records train selected states=96000 and train emitted lines=103999; delta=7999 is explained by KorQuAD negative-pair expansion. | R1 |
| A8 | Data validity gap | Blank source state | Every emitted example should have meaningful non-empty state text. | FAIL - 8 examples have blank `state`; current schema only enforces non-empty `questions`, and the builder does not reject/skip blank source text. | R1 |
| A9 | Rebuild determinism | Repeatability | Rebuilding from the same cached inputs must produce identical artifacts. | PASS - train, val, test, and summary are byte-identical across two runs. | R1 |

## Artifact classification and cleanup decision

- `.debug-journal.md` in the target worktree is **task-generated cleanup debris**: it is untracked, and its own artifact list says it was intended for removal. It remains present. **Decision: remove in the corrective cleanup owner/task; this audit did not delete it because the target was read-only.**
- `.omo/evidence/task-3-adversarial-audit.txt` and `.omo/evidence/task-3-adversarial-manual-qa.md` are **preserved external provenance** from the preceding adversarial audit. **Decision: retain; do not delete.**
- `.omo/evidence/task-3-corrective-qa.txt` and `.omo/evidence/task-3-manual-qa.md` are committed corrective evidence and are retained.
- Temporary `/tmp/kojev-todo3-6947a0f-*` build/corruption directories and logs were created outside the target worktree for this read-only audit. They are not part of the target patch; the evidence records their paths and results.

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| R1 | runtime transcript | Independent provenance, static checks, target/full pytest, offline real 12-source builds, counts, validation, malformed/empty corruption probes, deterministic rebuild, source/split checks, and blank-state validity probe. | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c421-runtime.txt` |
| R2 | QA matrix | This manual QA matrix and final AdversarialVerify verdict. | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c421-manual-qa.md` |
| R3 | claimed corrective evidence | Worker/commit-provided static, test, build, count, validation, and exclusion claims. | `/Volumes/SSD1/1/KoJev-todo3/.omo/evidence/task-3-corrective-qa.txt` |
| R4 | prior adversarial provenance | Earlier adversarial audit and its pre-correction findings. | `/Volumes/SSD1/1/KoJev-todo3/.omo/evidence/task-3-adversarial-audit.txt` |

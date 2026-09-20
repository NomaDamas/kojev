# AdversarialVerify / Manual QA — todo 8

Verdict: **needs-fix**
Safe to land: **false**

The implementation passes its focused and repository tests, emits 10,256 questions across 6 families from the fresh local corpus build, and correctly rejects an injected instruction collision. It is not safe to land because the independent state-origin audit found 22 emitted OOD rows whose states also occur in GOLD train. The acceptance explicitly makes any train/val state overlap a blocking failure.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| refs | local HEAD/origin/patch containment; origin/main splitter ancestry | git CLI | `git status --short --branch; git rev-parse HEAD origin/feat/todo-8 origin/main; git merge-base --is-ancestor 9d7670e HEAD; git diff --name-only origin/main...HEAD` | PASS: HEAD and origin/feat/todo-8 are `aa6a21cf0ac4971e01a7c7c91034a2d4ccf5dd9b`; `9d7670e` is an ancestor; patch files are only `.omo/evidence/task-8-kojev.txt`, `kojev/data_ood.py`, `tests/test_data_ood.py`; worktree clean | `git-refs` |
| focused-tests | acceptance: flip/reversal/train-overlap tests | pytest CLI | `uv run pytest tests/test_data_ood.py -q` | PASS: 4 passed | `tests-lint` |
| full-tests | requested full suite | pytest CLI | `uv run pytest -q` | PASS: 60 passed | `tests-lint` |
| lint | requested Ruff lint | CLI | `uv run ruff check kojev tests` | PASS: All checks passed | `tests-lint` |
| types | requested basedpyright | CLI | `uv run basedpyright kojev tests` | PASS: 0 errors, 0 warnings, 0 notes | `tests-lint` |
| format | requested repository format check | CLI | `uv run ruff format --check kojev tests` | FAIL outside todo-8 scope: `kojev/teacher.py` and `kojev/teacher_types.py` would be reformatted; checking the same files from `origin/main` passes, confirming the stated rebase condition | `format-blocker` |
| gold-build | independent gold corpus prerequisite | CLI | `uv run python -m kojev.data_gold --out /tmp/kojev-qa-task8-gold` | PASS: exit 0; cached datasets loaded and corpus files emitted | `gold-build` |
| ood-build | >=2500 questions and >=6 families; emitted files | CLI | `uv run python -m kojev.data_ood --gold /tmp/kojev-qa-task8-gold --out /tmp/kojev-qa-task8-ood` | PASS: exit 0; independent parse counts 6,258 OOD states, 10,256 questions, 6 families | `ood-build`, `independent-audit` |
| schema-and-content | no blank state, no teacher labels, no synthetic data, schema validity | Python CLI audit | `uv run python /tmp/qa_task8_verify.py` | PASS: `blank_states=0`, `teacher_label_fields=[]`, `synthetic_sources=[]`, `schema_validation=PASS 6258` | `independent-audit` |
| majority | majority baseline per family/instruction and arithmetic | Python CLI audit | `uv run python /tmp/qa_task8_verify.py` | PASS: every emitted instruction's independently computed count, majority gold, and fraction matched `summary.json`; `majority_all_ok=True` | `independent-audit` |
| cleanup | no worker-created temp build dirs/processes | shell CLI | `rm -rf /tmp/kojev-qa-task8-gold /tmp/kojev-qa-task8-ood /tmp/kojev-qa-task8-adversarial*; ps/find audit` | PASS for QA-created temp corpus dirs; no todo-8/data_gold/data_ood worker process remains; unrelated pre-existing local services were not touched | `independent-audit`, `cleanup` |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| split-leakage | GOLD TEST states only; critical leakage check | exact state intersection against GOLD train and val | OOD states must have zero intersection with train and val; any overlap blocks | FAIL: 22 emitted OOD rows overlap GOLD train; val overlap is 0; all 6,258 OOD rows have a `(source,state)` origin in GOLD test, but duplicate state text crosses the train/test boundary | `leakage-rules` |
| source-origin | GOLD TEST states only | source/state origin audit | every OOD `(source,state)` must occur in gold test | PASS: `ood_rows_without_test_origin=0`; 5,252 unique OOD states | `leakage-rules` |
| instruction-overlap | every OOD instruction absent from train; injected failure | adversarial train-instruction injection | injected collision must raise and name the instruction | PASS: `InstructionOverlapError: OOD instructions overlap GOLD train: ['침범 질문']`, exit 1; fresh corpus intersection is empty | `independent-audit` |
| nsmc-negation | nsmc deterministic flip | rule-gold correctness | negated noul gold equals `1 - original` | PASS on fresh spot-check: original gold 1 -> negated gold 0 | `leakage-rules` |
| nsmc-reversed | binary reversed-direction monotone probe | reversed option order + gold | options must be `['만족', '불만족']` and gold must be flipped | PASS on fresh spot-check: original gold 1 -> reversed options and gold 0 | `leakage-rules` |
| sts-reversal | STS deterministic reversal | score direction inversion | options must be reversed and gold must equal `5 - original` | PASS on fresh spot-check: original gold 4 -> reversed options and gold 1 | `leakage-rules` |
| korquad-flip | KorQuAD deterministic flip | noul polarity inversion | gold must equal `1 - original` | PASS on fresh spot-check: original gold 1 -> irrelevant-context gold 0 | `leakage-rules` |
| train-state-duplication | critical leakage; duplicate raw state | adversarial repeated state text | repeated state in train must still be rejected as leakage | FAIL: examples include duplicate raw states such as `.` / `굿` / `Very good!` and KorQuAD context+question strings in train; this is the blocking finding, not an inference | `leakage-rules` |
| schema-boundary | all emitted examples pass `kojev.schema` | malformed/blank-state class | parser must reject invalid output; emitted output must have nonblank states and valid question shapes | PASS: `read_jsonl` parsed all 6,258 emitted examples; blank count 0 | `independent-audit` |
| teacher-contamination | no teacher labels | metadata/source contamination | no teacher label metadata or synthetic source should be emitted | PASS: no teacher-bearing metadata and no synthetic sources | `independent-audit` |

## Blocking findings

1. **Train-state leakage: 22 OOD rows overlap GOLD train by exact `state` text.** Fresh audit output lists the overlapping states and sources. This violates the explicit critical requirement that any train or val state overlap is blocking. The likely mechanism is duplicate state text across source splits, but the verdict does not rely on that hypothesis: the exact set intersection is observed.
2. Repository-wide format check remains red on `teacher.py` and `teacher_types.py`. This is the specifically allowed pre-existing mainline formatting condition: the same files extracted from `origin/main` pass `ruff format --check`, while this branch's `origin/main` ref is `4a02c3ad...`; rebase onto the fixed mainline is required separately from the todo-8 leakage fix.

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `git-refs` | terminal transcript | HEAD/origin identity, ancestry, patch containment, clean status | `st_01a0c429/git-refs-and-containment.log` |
| `tests-lint` | terminal transcript | focused pytest, full pytest, Ruff lint, basedpyright | `st_01a0c429/tests-lint-types.log` |
| `format-blocker` | terminal transcript | repository format failure and origin/main control pass | `st_01a0c429/format-rebase-blocker.log` |
| `gold-build` | terminal transcript | fresh gold corpus build invocation and output | `st_01a0c429/gold-build.log` |
| `ood-build` | terminal transcript | fresh OOD build invocation and CLI output | `st_01a0c429/ood-build.log` |
| `independent-audit` | terminal transcript | independent counts, family set, instruction overlap, schema/teacher/synthetic checks, baseline arithmetic, injected failure | `st_01a0c429/independent-corpus-audit.log` |
| `leakage-rules` | terminal transcript | exact train/val/test state intersections and rule spot-checks | `st_01a0c429/leakage-and-rule-spotchecks.log` |
| `cleanup` | terminal transcript | QA cleanup and process/temp-dir check | `st_01a0c429/cleanup.log` |

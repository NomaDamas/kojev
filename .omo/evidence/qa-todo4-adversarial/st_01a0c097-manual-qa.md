# Todo 4 Adversarial Manual QA

## AdversarialVerify

```yaml
verdict: needs-fix
safe_to_land: false
scope: todo 4 only
repository: /Volumes/SSD1/1/KoJev-todo4
branch: feat/todo-4-kobest-klue-bench
commit: a96c1707eda45317194af1005109f700d181a9b6
base: d1e9fcf50efc5035db83618717f9cefc9abdd42b
reason: >-
  The direct contamination helper rejects overlap and malformed manifests, but the
  benchmark runtime never accepts or invokes a training-manifest check. The todo-4
  requirement says the harness asserts this at runtime, so the implementation is
  not safe to land until that gate is wired into the executable benchmark path.
```

## Surface evidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | commit/remote SHA, base, containment, clean worktree | Git repository | `git -C /Volumes/SSD1/1/KoJev-todo4 status --short --branch`; `git rev-parse HEAD`; `git show -s --format='%H%n%P%n%D%n%s' HEAD`; `git ls-remote origin refs/heads/feat/todo-4-kobest-klue-bench refs/heads/main`; `git diff --name-status d1e9fcf50efc5035db83618717f9cefc9abdd42b..a96c1707eda45317194af1005109f700d181a9b6`; `git ls-files 'data/gold/**'` | PASS | A1 |
| S2 | all five KoBEST mappings | Python mapping surface | `uv run pytest tests/test_bench.py -q` | PASS (7 passed) | A2 |
| S3 | all three KLUE mappings, typed schema output | Python mapping surface | `uv run pytest tests/test_bench.py -q`; source inspection of `map_klue_row()` and `Question`/`Example` construction | PASS | A2, A3 |
| S4 | metrics: accuracy, macro-F1, Brier, 15-bin ECE | Python metrics/report surface | `uv run pytest tests/test_bench.py -q`; live report parse after `uv run python -m kojev.bench --model random --limit 200` | PASS; every task has exact keys `count, accuracy, macro_f1, brier, ece_15` | A2, A4 |
| S5 | `decide` protocol | Python model surface | `uv run pytest tests/test_bench.py -q`; source inspection of `DecisionModel` protocol and `RandomDecisionModel.decide()` | PASS | A2, A3 |
| S6 | random behavior/repeatability | Python model + CLI surface | direct `RandomDecisionModel(seed=19).decide(...)` twice on same state; three fresh `uv run pytest tests/test_bench.py -q` runs | PASS; exact repeatability and 7/7 on all three runs | A5 |
| S7 | executable smoke report: all eight names, positive counts, metrics | CLI surface | `uv run python -m kojev.bench --model random --limit 200`; parse `benchmark-report.json` | PASS; total 200; all eight task names; 25 each; all counts > 0; kind counts choice=100, noul=75, score=25 | A4 |
| S8 | CLI repeatability and no data/gold writes | CLI/filesystem surface | run CLI twice; compare `sha256sum` of `benchmark-report.json`; snapshot `data/gold`; remove only generated report | PASS; identical SHA-256; `data/gold` absent before/after; report cleaned | A4, A6 |
| S9 | contamination gate is present and rejects overlap/malformed input | direct Python API surface | temporary JSONL manifests; `assert_no_kobest_contamination(manifest, {'kobest-42'})` and malformed manifest call | PASS for direct helper behavior | A5 |
| S10 | contamination gate is enforced at runtime | executable benchmark surface | source/runtime inspection of `main()` and `load_benchmark_items()`; `grep -R 'assert_no_kobest_contamination('` | FAIL: helper is only referenced by tests; `main()` has no training-manifest input and never invokes it | A7 |
| S11 | malformed input has meaningful failure | CLI and mapper surface | `uv run python -m kojev.bench --model unsupported --limit 8`; `--limit nope`; `--limit`; unsupported mapper probe | PASS for nonzero failure and meaningful messages; invalid numeric/missing argument expose raw `ValueError`/`StopIteration` | A5, A8 |
| S12 | static quality gates | repository tooling surface | `uv run ruff check kojev tests`; `uv run basedpyright kojev tests` | PASS; no ruff findings; 0 basedpyright errors/warnings/notes | A9, A10 |

## Adversarial cases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| AC1 | malformed_input | malformed JSONL manifest | Reject with a line-specific `BenchmarkError`, not success | PASS | A5 |
| AC2 | malformed_input | unsupported task/row shape | Reject with `BenchmarkError` naming unsupported task or malformed field | PASS | A5 |
| AC3 | prompt_injection | benchmark text contains instruction-like payload | Treat text as data; do not execute or reinterpret it | PASS | A5 |
| AC4 | stale state/repeatability | same state/question and seed repeated | Return exactly identical random probabilities | PASS | A5 |
| AC5 | stale state/repeatability | full CLI rerun with cached data | Produce byte-identical report | PASS | A4 |
| AC6 | dirty worktree | generated report remains after run | Generated artifact must be identified and removable without product diff | PASS | A6 |
| AC7 | misleading success output | CLI exits 0 but report is incomplete | Parsed report must prove all eight tasks, positive counts, exact metric keys | PASS | A4 |
| AC8 | contamination | overlapping KoBEST ID in manifest | Reject with contamination assertion | PASS for helper; runtime enforcement FAIL | A5, A7 |
| AC9 | contamination | malformed training manifest | Reject with meaningful malformed-manifest error | PASS for helper; runtime enforcement FAIL | A5, A7 |
| AC10 | flaky tests | repeated targeted suite | Same suite must pass repeatedly without sleeps/polling | PASS; 3 consecutive runs, 7 passed each | A5 |
| AC11 | long-running/process cleanup | benchmark and validators leave live product processes | No `kojev.bench`, pytest, ruff, or basedpyright command from this QA remains; generated report removed | PASS; no product QA processes remained; existing language-server processes were not owned by this QA | A6 |
| AC12 | cancel/resume | resumable long-running benchmark protocol | Not applicable: todo 4 CLI is a bounded one-shot benchmark and has no resume/checkpoint contract | NOT_APPLICABLE | A7 |
| AC13 | repeated interruptions | interruption recovery | Not applicable: no persistent job/session or resumable state is part of todo 4 | NOT_APPLICABLE | A7 |

## Artifact references

| id | kind | description | path |
|---|---|---|---|
| A1 | command transcript | Git identity, remote SHA, base parent, commit containment, changed-file list, and absence of tracked `data/gold` | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/git-truth.txt` |
| A2 | test transcript | Required targeted suite: `7 passed` | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/pytest-target.txt` |
| A3 | source inspection | `kojev/bench.py` and `kojev/schema.py` implementation read during QA | `/Volumes/SSD1/1/KoJev-todo4/kojev/bench.py` and `/Volumes/SSD1/1/KoJev-todo4/kojev/schema.py` |
| A4 | parsed CLI artifact | Live 200-item report parse: all eight names, 25 counts each, exact metric keys, total 200, kind aggregates, SHA-256 | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/report-parse.txt` |
| A5 | adversarial transcript | Contamination, malformed manifest, prompt-injection-as-data, repeatability, unsupported-task probes | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/adversarial-probes.txt` |
| A6 | cleanup transcript | Final clean todo-4 worktree, absent generated report/data-gold, no QA-owned benchmark/test processes | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/cleanup.txt` |
| A7 | source/reference evidence | Runtime call-site inspection showing helper is defined and tested but not invoked by `main()`/`load_benchmark_items()`; plan requirement states runtime manifest assertion | `/Volumes/SSD1/1/KoJev-todo4/kojev/bench.py` lines 292-310, 313-352; `/Volumes/SSD1/1/KoJev/.omo/plans/kojev.md` todo 4 acceptance/guardrail text |
| A8 | malformed CLI transcript | Nonzero failures for unsupported model, nonnumeric limit, and missing limit argument | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/malformed-cli.txt` |
| A9 | lint transcript | Ruff clean | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/ruff.txt` |
| A10 | type-check transcript | Basedpyright clean: 0 errors, warnings, notes | `/Volumes/SSD1/1/KoJev/.omo/evidence/qa-todo4-adversarial/basedpyright.txt` |

## Cleanup conclusion

The verifier-owned `benchmark-report.json` and temporary manifest/probe files were removed. The todo-4 worktree is clean and still points at the pinned commit. No benchmark, pytest, ruff, or basedpyright process owned by this QA remains. Existing basedpyright language-server processes were observed but were not spawned by this QA and were left untouched. No product file, Git ref, plan, or committed/generated product artifact was modified.

## Safe to land

`false`. Do not land this commit as complete todo 4 until the executable benchmark path receives a training-manifest input/configuration and invokes the KoBEST overlap/malformed-manifest gate before evaluating or writing the report. The direct helper and its tests are not sufficient evidence for the required runtime guard.

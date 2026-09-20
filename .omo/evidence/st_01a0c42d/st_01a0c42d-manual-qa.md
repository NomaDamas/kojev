# AdversarialVerify — KoJev todo 10 local CPU refactor

Commit under test: `3c4ba43b36964acd54ed75e1d3963c3326453e00`
Branch: `origin/feat/todo-10-train`
Base ancestry: `168e7576000b32a7b2aabd016711a6bd5a5a40c2` is an ancestor.
Mode: read-only verification in a disposable `git archive` under `/tmp`; the live checkout remained on `main` and was not edited.

## Verdict

**needs-fix** — **safe_to_land: false**.

The requested commit is otherwise healthy: exact branch identity and ancestry are confirmed, patch containment is correct, all five gates are clean, tests are byte-identical to `ac89e868`, requested behavior is observed independently, both mutation tests fail when behavior is broken, and the real tiny CPU CLI produces the complete report contract. However, criterion 3 explicitly requires remaining suppressions to be only S311, T201, and targeted Torch-boundary pyright ignores. The target still contains ten `# pyright: ignore[reportAny]` suppressions at `kojev/train.py:529-538` on argparse namespace field reads. They are pre-existing from `ac89e868`, but they remain in the commit and are not targeted Torch-boundary ignores. This is the sole blocking finding. The deliberately out-of-scope `gpu01` Slurm run was not required.

## manualQa matrix

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| branch-identity-ancestry | requested branch commit and base ancestry | Git refs | `git show-ref --verify refs/remotes/origin/feat/todo-10-train`; `git merge-base --is-ancestor 168e7576000b32a7b2aabd016711a6bd5a5a40c2 3c4ba43b36964acd54ed75e1d3963c3326453e00` | PASS; remote branch resolves to the requested SHA; ancestor exit 0 | `ev-gates` |
| patch-containment | criterion 1 | Git tree | `git diff-tree --no-commit-id --name-only -r 3c4ba43b^ 3c4ba43b` | PASS; only `.omo/evidence/task-10-kojev.txt`, `kojev/train.py`, and `tests/test_train.py` | `ev-gates` |
| target-unit-suite | criterion 2 | pytest | `archived/.venv/bin/pytest tests/test_train.py -q` | PASS; 5 passed | `ev-gates` |
| full-regression-suite | criterion 2 | pytest | `archived/.venv/bin/pytest -q` | PASS; 66 passed | `ev-gates` |
| ruff-lint | criterion 2 | CLI | `archived/.venv/bin/ruff check kojev tests` | PASS; all checks passed | `ev-gates` |
| ruff-format | criterion 2 | CLI | `archived/.venv/bin/ruff format --check kojev tests` | PASS; 21 files already formatted | `ev-gates` |
| basedpyright | criterion 2 | CLI | `archived/.venv/bin/basedpyright kojev tests` | PASS; 0 errors, 0 warnings, 0 notes | `ev-gates` |
| local-cpu-e2e | criterion 7 | CLI | `archived/.venv/bin/python -m kojev.train --tiny --train /tmp/kojev-task10-qa-data/train.jsonl --val /tmp/kojev-task10-qa-data/val.jsonl --limit 2 --epochs 1 --seed 17 --batch-size 1 --augmentation-probability 0 --out /tmp/kojev-task10-qa-out` | PASS; exit 0 and stdout points to `report.json` | `ev-cli`, `ev-report` |
| report-contract | criterion 7 | JSON data | `json.loads(Path('/tmp/kojev-task10-qa-out/report.json').read_text())` and inspect top-level/metric keys | PASS; top-level keys are `args`, `data_counts`, `loss_curve`, `wall_time`, `peak_memory`, `metrics`, `temperature`, `diverged`; metrics include `overall`, `kind:choice`, `source:qa-fixture`, with `brier` and `ece` | `ev-report` |
| deterministic-repeat | criterion 8 | pytest | three sequential invocations of `archived/.venv/bin/pytest tests/test_train.py -q` | PASS; 5 passed on each run; no sleeps or network | `ev-repeat` |
| cleanup | criterion 8 / cleanup | OS process and filesystem | `ps ax -o pid=,command= | grep -E 'pytest|kojev.train|python.*train'`; existence checks for archive and `/tmp/kojev-task10-qa-*` | PASS; no leftover training/test process and all disposable dirs removed | `ev-repeat` |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| suppression-complexity | criterion 3 | suppression audit | `# noqa: C901`, `PLR0913`, and `PLR0915` must be absent from the target; no config weakening | PASS; absent from `kojev/train.py`; no `pyproject.toml` diff; S311 and T201 are the only target `noqa` codes | `ev-suppressions` |
| suppression-allowlist | criterion 3 | suppression allowlist | remaining suppressions must be only S311, T201, and targeted Torch-boundary pyright ignores | **FAIL/BLOCKING**; `kojev/train.py:529-538` has ten `# pyright: ignore[reportAny]` comments on argparse namespace field reads, which are not Torch-boundary ignores | `ev-suppressions` |
| suppression-scope | criterion 3 | inherited suppression boundary | unrelated inherited suppressions must not be mistaken for refactor additions | PASS for provenance; the argparse ignores predate this refactor, and pre-existing `kojev/augment.py` `TC003` and test pyright ignores are unchanged from origin/main. Provenance does not satisfy the explicit allowlist. | `ev-suppressions` |
| tests-unchanged | criterion 4 | regression-test weakening | no removed, loosened, or vacuous assertion relative to `ac89e868` | PASS; SHA-256 of `tests/test_train.py` at both commits is `cc5cd23ba239e1291d3b062c0fb5a16a409329c742dbd67e717f3544ef912096` | `ev-gates` |
| divergence-rise | criterion 5 | rising loss stream | 200-value baseline followed by >25% rise sets sticky divergence true | PASS; target test passes and implementation observes `1.3` after 200 `1.0` values as true | `ev-gates` |
| divergence-nan | criterion 5 | non-finite loss | NaN sets sticky divergence true | PASS; target test passes | `ev-gates` |
| divergence-healthy | criterion 5 | healthy decreasing stream | steadily decreasing values remain non-diverged | PASS; target test passes with `diverged is False` | `ev-gates` |
| temperature-overconfidence | criterion 5 | overconfident wrong logits | fitted temperature must exceed 1.0 | PASS; target test passes for logits of magnitude 8 with reversed labels | `ev-gates` |
| optimizer-schedule-loss | criterion 5 | training contract | parameter groups are `2e-5` and `1e-3`; warmup is 6%; schedule decays cosine; clip is 1.0; loss is CE+Brier | PASS; direct inspection reports `[2e-05, 0.001]`, reaches full LR at step 6/100, decays to near zero, and confirms clip/loss implementation | `ev-contract` |
| mutation-divergence | criterion 6 | behavior mutation | breaking both divergence branches must make the divergence test fail | PASS; mutated test fails at `assert watch.observe(1.3) is True`, exit 1 | `ev-mut-div` |
| mutation-temperature | criterion 6 | behavior mutation | forcing `fit_temperature` to return `1.0` must make calibration test fail | PASS; mutated test fails at `assert temperature > 1.0`, exit 1 | `ev-mut-temp` |
| gpu01-slurm | scope boundary | unavailable GPU/Slurm prerequisite | local CPU verification must not be blocked by absent gpu01 smoke | NOT_APPLICABLE; explicitly out of scope for this local CPU half | `ev-gates` |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `ev-gates` | text transcript | Commit identity/ancestry, patch containment, test-file hash comparison, and all five clean gates | `.omo/evidence/st_01a0c42d/all-gates-and-containment.txt` |
| `ev-cli` | CLI transcript | Real tiny CPU training invocation and exit output | `.omo/evidence/st_01a0c42d/cpu-cli-transcript.txt` |
| `ev-report` | JSON inspection transcript | Required report keys, per-kind/source metrics, Brier and ECE fields | `.omo/evidence/st_01a0c42d/cpu-cli-report-inspection.txt` |
| `ev-repeat` | pytest/process transcript | Three target-suite runs and final no-process cleanup observation | `.omo/evidence/st_01a0c42d/target-suite-3x.txt` |
| `ev-suppressions` | text audit | Target suppression audit, explicit allowlist failure, and unchanged inherited suppression context | `.omo/evidence/st_01a0c42d/suppression-audit.txt` |
| `ev-contract` | runtime inspection | Optimizer learning rates, 6% warmup/cosine schedule, clip marker, and CE+Brier marker | `.omo/evidence/st_01a0c42d/training-contract-inspection.txt` |
| `ev-mut-div` | pytest mutation transcript | Divergence behavior test fails after both divergence branches are disabled | `.omo/evidence/st_01a0c42d/mutation-divergence.txt` |
| `ev-mut-temp` | pytest mutation transcript | Temperature behavior test fails after fitting is forced to return 1.0 | `.omo/evidence/st_01a0c42d/mutation-temperature.txt` |
| `ev-env` | setup transcript | Disposable archive environment installed from the locked project | `.omo/evidence/st_01a0c42d/environment-setup.txt` |

## Notes

- The live checkout had pre-existing unrelated untracked evidence directories; none were modified or removed.
- The disposable archive and runtime data were removed after verification.
- No product files, refs, branches, or commits were edited.

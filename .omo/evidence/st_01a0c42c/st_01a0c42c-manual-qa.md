# AdversarialVerify — todo 10 local CPU training

Verdict: **needs-fix**
Safe to land: **false**

Blocking finding: the requested repository identity is not true in the live worktree. Observed `HEAD=168e7576000b32a7b2aabd016711a6bd5a5a40c2` on `main`; `origin/feat/todo-10-train=ac89e86853dd0faad529b7dac0d1e726fb137dda`; `origin/main=168e7576000b32a7b2aabd016711a6bd5a5a40c2`, not the required `d9c7cb4e9e720d25850e8b1868072e40ecd9d952`. The target commit itself has parent `d9c7cb4e9e720d25850e8b1868072e40ecd9d952` and was tested from a clean archive.

## manualQa matrix

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| refs-target-identity | 1 | Git refs | `git rev-parse HEAD; git rev-parse origin/feat/todo-10-train; git rev-parse origin/main; git show -s --format='%H %P %D' ac89e868` | FAIL: live HEAD/origin/main do not match requested values; target commit parent is the requested d9c7cb4 | `ev-ref-audit` |
| patch-containment | 1 | Git diff | `git diff --name-only d9c7cb4e9e720d25850e8b1868072e40ecd9d952 ac89e86853dd0faad529b7dac0d1e726fb137dda` | PASS: exactly `.omo/evidence/task-10-kojev.txt`, `kojev/train.py`, `tests/test_train.py` | `ev-ref-audit` |
| target-train-tests | 2 | pytest | `uv run pytest tests/test_train.py -q` in clean archive of `ac89e868` | PASS: `5 passed in 12.51s` | `ev-gates` |
| target-full-tests | 2 | pytest | `uv run pytest -q` in clean archive of `ac89e868` | PASS: `62 passed in 12.99s` | `ev-gates` |
| target-ruff-check | 2 | Ruff | `uv run ruff check kojev tests` in clean archive | PASS: `All checks passed!` | `ev-gates` |
| target-ruff-format | 2 | Ruff formatter | `uv run ruff format --check kojev tests` in clean archive | PASS: `19 files already formatted` | `ev-gates` |
| target-basedpyright | 2 | basedpyright | `uv run basedpyright kojev tests` in clean archive | PASS: `0 errors, 0 warnings, 0 notes` | `ev-gates` |
| suppression-audit | 3 | Git diff/source | `git diff ... | grep -nE '# *(type: ignore|pyright: ignore|noqa|ruff: noqa)|pyproject|pyright|ruff'`; source grep on committed `kojev/train.py` | PASS for no config change and no blanket `# type: ignore`/`# ruff: noqa`; targeted suppressions are present: C901/PLR0913/PLR0915, S311, T201, and specific pyright rules including torch/argparse boundaries. These are not blanket config loosening, but the worker claim of no suppressions is false literally. | `ev-suppressions` |
| deterministic-repeat | 9 | pytest | `uv run pytest tests/test_train.py -q` three times in clean archive | PASS: each run `5 passed`, exit 0; observed times 1.67s, 1.69s, 1.70s; no sleeps/network in changed tests | `ev-repeat` |
| cpu-cli-e2e | 7 | CLI + JSON | `uv run python -m kojev.train --tiny --train /tmp/.../train.jsonl --val /tmp/.../val.jsonl --limit 2 --epochs 1 --seed 7 --batch-size 2 --augmentation-probability 0 --out /tmp/.../out` | PASS: exit 0; report has args, data_counts, loss_curve, wall_time, peak_memory, metrics, temperature, diverged; metrics include `overall`, `kind:choice`, `source:fixture`; overall includes accuracy, brier, ece | `ev-e2e` |
| optimizer-plan | 8 | Source + runtime | Inspect `_optimizer`, scheduler, training loop and run `_optimizer(KoJevModel(_TinyBackbone()), 2e-5, 1e-3)` | PASS: two groups with lrs `[2e-05, 0.001]`; warmup is `ceil(total_steps * 0.06)`; post-warmup factor is cosine; clip norm is `1.0`; encoder loss is grouped CE plus squared Brier term | `ev-plan` |
| process-cleanup | 10 | OS process table | `pgrep -af 'kojev\.train|python.*train'; find /tmp -maxdepth 1 -type d -name 'kojev-task10-*'` | PASS: no training process remained; temporary QA dirs removed after each scenario | `ev-cleanup` |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| divergence-nan | 5 | synthetic NaN loss | `diverged` becomes true | PASS: direct call returned `True` and sticky state true | `ev-behavior` |
| divergence-rise | 5 | synthetic rising loss | after 200-value baseline, >25% rise sets `diverged` true | PASS: direct call returned `True` | `ev-behavior` |
| divergence-healthy | 5 | synthetic decreasing loss | healthy decrease remains non-diverged | PASS: direct call returned false throughout | `ev-behavior` |
| temperature-overconfidence | 6 | deliberately overconfident/wrong logits | fitted temperature > 1 | PASS: direct fit returned `10.0` | `ev-behavior` |
| divergence-mutation | 4 | disposable mutation of rising-loss branch | named divergence test must fail | PASS: `test_divergence_watch_triggers_on_rise_and_nan` failed at `assert watch.observe(1.3) is True` (`1 failed, 4 deselected`) | `ev-mutation` |
| temperature-mutation | 4 | disposable mutation returning temperature 1.0 | named temperature test must fail | PASS: `test_fit_temperature_increases_temperature_for_overconfident_logits` failed at `assert temperature > 1.0` (`1 failed, 4 deselected`) | `ev-mutation` |
| gpu01-slurm | scope note | unavailable separate GPU/Slurm increment | not required for local CPU half | NOT_APPLICABLE: explicitly excluded by task scope | `ev-ref-audit` |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| `ev-ref-audit` | text | Git identity, ancestry, containment, and scope observations | `.omo/evidence/st_01a0c42c/st_01a0c42c-ref-audit.txt` |
| `ev-gates` | text | All five required gates from clean target archive | `.omo/evidence/st_01a0c42c/st_01a0c42c-gates.txt` |
| `ev-suppressions` | text | Suppression/config audit | `.omo/evidence/st_01a0c42c/st_01a0c42c-suppressions.txt` |
| `ev-repeat` | text | Three target-suite runs | `.omo/evidence/st_01a0c42c/st_01a0c42c-repeat.txt` |
| `ev-behavior` | text | Direct divergence, temperature, and optimizer probes | `.omo/evidence/st_01a0c42c/st_01a0c42c-behavior.txt` |
| `ev-mutation` | text | Mutation-test failure transcripts | `.omo/evidence/st_01a0c42c/st_01a0c42c-mutation.txt` |
| `ev-e2e` | text | Real tiny CPU CLI invocation and parsed report evidence | `.omo/evidence/st_01a0c42c/st_01a0c42c-e2e.txt` |
| `ev-plan` | text | Source/runtime plan conformance evidence | `.omo/evidence/st_01a0c42c/st_01a0c42c-plan.txt` |
| `ev-cleanup` | text | Process and temporary-directory cleanup receipt | `.omo/evidence/st_01a0c42c/st_01a0c42c-cleanup.txt` |

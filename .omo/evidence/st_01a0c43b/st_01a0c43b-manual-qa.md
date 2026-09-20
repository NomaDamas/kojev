# AdversarialVerify / manualQa

- Target: `feat/infra` at `ea23798c69d2c25e849ba701ce4626381250a4aa`
- Mode: read-only verification in disposable clone `/tmp/kojev-st_01a0c43b.tmwO8a`
- Verdict: **confirmed**
- safe_to_land: **true**

## Measured gate outputs

Surface: disposable clone of the exact target commit. These are the exact invocations run from the clone root:

```text
$ uv run ruff format --check .
Using CPython 3.12.9
Creating virtual environment at: .venv
   Building kojev @ file:///private/tmp/kojev-st_01a0c43b.tmwO8a
      Built kojev @ file:///private/tmp/kojev-st_01a0c43b.tmwO8a
Installed 71 packages in 511ms
5 files already formatted
exit_code=0

$ uv run pytest -q
................                                                         [100%]
16 passed in 0.29s
exit_code=0

$ uv run ruff check .
All checks passed!
exit_code=0
```

## `surfaceEvidence` matrix

| Scenario | Criterion reference | Surface | Exact invocation | Verdict | Artifact refs |
|---|---|---|---|---|---|
| S1 | Local gate figure | Fresh disposable clone | `uv run ruff format --check .` | PASS; measured exactly `5 files already formatted`, exit 0 | A1, A4 |
| S2 | Local gates | Fresh disposable clone | `uv run pytest -q` | PASS; measured `16 passed in 0.29s`, exit 0 | A2 |
| S3 | Local gates | Fresh disposable clone | `uv run ruff check .` | PASS; measured `All checks passed!`, exit 0 | A3 |
| S4 | Commit provenance | Git history in exact clone | `git show --name-only` for `2076e20`, `8121903`, `6acd32b`, `06d5aa0`, `ee930e7`, `0153ab4` | PASS; every per-commit attribution matches exact changed paths; no mis-attribution | A5 |
| S5 | Job-state wording | Evidence file vs independent history artifact | Read full `.omo/evidence/task-2-kojev.txt`; compare table with prior captured `sacct` history | PASS; table has four FAILED (`13597`, `13599`, `13601`, `13621`), one CANCELLED (`13603`), one TIMEOUT (`13620`), and `13622 COMPLETED`; `13620 TIMEOUT 30:21` matches captured history | A6, A7 |
| S6 | Overclaim sweep | Full tracked evidence file | `nl -ba .omo/evidence/task-2-kojev.txt` plus cross-check against gate, history, static, and prior QA artifacts | PASS; no remaining unsupported statement or non-reproducible figure found | A4, A6, A7, A8 |
| S7 | Preserved-work qualification | Commit provenance prose | Read lines 101-105 of evidence file and compare with commit history | PASS; explicitly says template commits predated CUDA proof and were preserved work, never a completion claim | A4, A5 |
| S8 | Forbidden paths and node references | Exact branch source tree | `git grep -n -E 'gpu02|/data1([^0-9]|$)|/data([^0-9]|$)' HEAD` and script path audit | PASS; no `gpu02`; scripts contain only `/data2/...` operational paths and no writes to `/`, `/data`, or `/data1` | A8 |

## `adversarialCases` matrix

| Scenario | Criterion reference | Adversarial class | Expected behavior | Verdict | Artifact refs |
|---|---|---|---|---|---|
| ADV1 | Evidence accuracy | Formatter figure mismatch | Fresh checkout must reproduce the stated formatter count exactly | PASS; evidence says 5 and fresh run measured 5 | A1, A4 |
| ADV2 | Evidence accuracy | Gate-output mismatch | Fresh checkout must reproduce pytest and Ruff claims | PASS; measured 16 passed and All checks passed! exactly | A2, A3, A4 |
| ADV3 | Provenance | Mis-attributed files | Each listed commit must own exactly the claimed files | PASS; all six commit claims match `git show --name-only` | A5 |
| ADV4 | Job history | Collapsed or false terminal classes | Per-job table must preserve exact terminal classes | PASS; four FAILED, one CANCELLED, one TIMEOUT; independent history agrees | A6, A7 |
| ADV5 | Overclaim | Unsupported prose or stale figure | Every number and factual claim must have artifact support or fresh reproduction | PASS; no remaining unsupported claim found; prior cluster facts are backed by prior captured artifacts | A4, A6, A7, A8 |
| ADV6 | Completion semantics | Preserved work misrepresented as completion | Evidence must distinguish committed template work from CUDA completion proof | PASS; explicit preserved-work/non-completion wording present | A4, A5 |
| ADV7 | Path containment | Forbidden filesystem destination | No branch script may write to `/`, `/data`, or `/data1` | PASS; source audit found no forbidden destinations; all operational paths are under `/data2` | A8 |
| ADV8 | Node containment | Forbidden node reference | No branch artifact may reference `gpu02` | PASS; exact-tree grep returned no matches | A8 |

## Evidence notes

- The branch ref in the live repository resolves to the requested commit. The disposable clone is clean and checked out at that commit; live-HEAD identity was not used as a criterion.
- No cluster probes, SSH, or remote commands were run in this verification, as instructed.
- The prior independently captured history records job `13620` as top-level `TIMEOUT`, elapsed `00:30:21`, and the evidence table's `TIMEOUT 30:21` matches it. The batch child being `CANCELLED` does not change the top-level job state claimed in the table.
- No remaining unsupported claim was identified. Therefore no verbatim unsupported-claim quote applies.

## `artifactRefs`

| ID | Kind | Description | Path |
|---|---|---|---|
| A1 | command transcript | Fresh-clone `ruff format --check .` output, including exact `5 files already formatted` and exit code 0 | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/ruff-format-check.txt` |
| A2 | command transcript | Fresh-clone `pytest -q` output, including `16 passed in 0.29s` and exit code 0 | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/pytest-q.txt` |
| A3 | command transcript | Fresh-clone `ruff check .` output, including `All checks passed!` and exit code 0 | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/ruff-check.txt` |
| A4 | source artifact | Full target evidence file, read in the disposable clone and line-number audited | `/tmp/kojev-st_01a0c43b.tmwO8a/.omo/evidence/task-2-kojev.txt` |
| A5 | git transcript | Exact `git show --name-only` results for all six attributed commits | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/audit.txt` |
| A6 | prior command transcript | Independently captured `sacct` history for jobs 13597, 13599, 13601, 13603, 13620, and 13621 | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43a/cluster.txt` |
| A7 | prior QA artifact | Prior verification's cross-check recording exact terminal states and prior gate evidence | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43a/st_01a0c43a-manual-qa.md` |
| A8 | source audit transcript | Exact-tree scans for `gpu02`, forbidden path literals, and script destination paths | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/audit.txt` |
| A9 | clone identity | Disposable clone commit and clean status | `/Volumes/SSD1/1/KoJev/.omo/evidence/st_01a0c43b/clone-identity.txt` |

## Final AdversarialVerify

**confirmed; safe_to_land=true.** Measured gates are formatter `5 files already formatted`, pytest `16 passed`, and Ruff `All checks passed!`, all with exit code 0. Provenance is accurate, terminal-state wording is accurate, preserved-work wording is explicit, and the forbidden-path / `gpu02` sweep is clean. No remaining unsupported claim was found.

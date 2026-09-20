# AdversarialVerify / manualQa — commit 0153ab47681a9590c4bfedef92f80c01217c2ffc

## Verdict

**needs-fix**

**safe_to_land: false**

The real cluster smoke proof and scheduler rejection are confirmed, and the disposable local gates pass. The verification is not fully confirmable because the committed evidence overclaims the formatter output (`7 files already formatted`, measured `5 files already formatted`) and the commit itself contains only an evidence-file change, not the claimed templates/tests. The discrepancy must be corrected or explained before treating this commit as safe to land.

### Requested measured sacct line for 13622

```text
13622        kojev-smo+ interacti+  COMPLETED      0:0   00:00:03           gpu01 billing=8+
```

The batch child also measured:

```text
13622.batch       batch             COMPLETED      0:0   00:00:03           gpu01 cpu=8,gre+
```

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | 1: sacct COMPLETED/0:0/3s/gpu01 | Slurm accounting over SSH | `ssh -o BatchMode=yes -o ConnectTimeout=20 gpu01 'sacct -j 13622 --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocTRES --noheader'` | PASS | A1 |
| S2 | 2: stdout exactly `True` | Job stdout file over SSH | `ssh -o BatchMode=yes gpu01 'cat /data2/jeffrey/kojev/runs/kojev-smoke-13622.out'` | PASS | A1 |
| S3 | 3: shared venv target, torch/CUDA/device count | Remote interpreter | `ssh -o BatchMode=yes gpu01 'readlink -f /data2/jeffrey/kojev/venv-shared; /data2/jeffrey/kojev/venv-shared/bin/python -c "import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available()); print(\"device_count\", torch.cuda.device_count())"'` | PASS | A1 |
| S4 | 4: wrong-GRES rejection | Slurm submission over SSH | `ssh -o BatchMode=yes gpu01 'sbatch --nodelist=gpu01 --gres=gpu:h100:1 /data2/jeffrey/kojev/repo/scripts/slurm/smoke.sbatch'` | PASS | A2 |
| S5 | 5: no writes to `/`, `/data`, `/data1` in scripts | Disposable commit-tree static audit | `git archive 0153ab4...  tar -x; for f in scripts/slurm/*.sbatch scripts/sync.sh; do audit; done` | PASS | A3 |
| S6 | 6: gpu01-only rtx6000 and no gpu02 | Disposable commit-tree source audit | `grep -RInE 'gpu0[12]|gres|nodelist|partition' scripts tests` plus full-tree `gpu02` scan | PASS | A3 |
| S7 | 7: local pytest | Disposable checkout | `uv run pytest -q` | PASS | A4 |
| S8 | 7: local ruff check | Disposable checkout | `uv run ruff check .` | PASS | A4 |
| S9 | 7: local ruff format | Disposable checkout | `uv run ruff format --check .` | FAIL (claim discrepancy) | A4, A5 |
| S10 | 8: non-tautology mutation | Disposable checkout, mutate hardlink invariant | `sed replacement of expected UV_LINK_MODE string; uv run pytest -q tests/test_sync_slurm.py` | PASS | A6 |
| S11 | 9: patch containment | Git object inspection | `git diff-tree --no-commit-id --name-status -r 0153ab4...` | PASS with scope concern | A7 |
| S12 | 10: history spot-check | Slurm accounting over same batched SSH session | `ssh -o BatchMode=yes gpu01 'sacct -j 13597,13599,13601,13603,13620,13621 --format=JobIDRaw,State,ExitCode,Elapsed,NodeList --noheader'` | PASS | A1 |
| S13 | 11: evidence overclaim audit | Commit artifact vs measured artifacts | `git show 0153ab4:.omo/evidence/task-2-kojev.txt` compared with A1-A7 | FAIL | A5, A7 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV1 | 4 | invalid GPU type / scheduler rejection | `sbatch` rejects immediately with exact `Requested node configuration is not available`, non-zero status, and no job is submitted | PASS | A2 |
| ADV2 | 5 | forbidden destination audit | No script writes to `/`, `/data`, or `/data1`; `/data2` is allowed | PASS | A3 |
| ADV3 | 6 | forbidden node reference | No `gpu02` reference; templates request `gpu01`-compatible `gpu:rtx6000:1` | PASS | A3 |
| ADV4 | 8 | regression mutation of a required template invariant | The corresponding test must fail, not pass tautologically | PASS | A6 |
| ADV5 | 10 | fabricated failure history | Spot-checked IDs must have the claimed terminal classes | PASS | A1 |
| ADV6 | 11 | evidence artifact mismatch | Evidence prose must match measured outputs exactly | FAIL | A5 |
| ADV7 | 9 | patch containment | Every changed path must be listed and justified; unrelated product changes must not be hidden | PASS for changed-path accounting; concern noted | A7 |

## Exact evidence and discrepancies

### Cluster measurements

- 13622: `COMPLETED`, `0:0`, `00:00:03`, `gpu01`.
- `/data2/jeffrey/kojev/venv-shared -> /data2/jeffrey/kojev/runs/venv-prewarm2`.
- Measured `torch 2.14.0+cu130`, `cuda_available True`, `device_count 3`.
- Job stdout `/data2/jeffrey/kojev/runs/kojev-smoke-13622.out` contained exactly `True`.
- Real wrong-GRES probe returned exactly:

```text
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

and `wrong_gres_exit=1`.

- Historical spot-checks measured:
  - 13597 `FAILED 2:0`
  - 13599 `FAILED 1:0`
  - 13601 `FAILED 1:0`
  - 13603 `CANCELLED by 2105`, batch `CANCELLED 0:15`
  - 13620 `TIMEOUT 0:0`, batch `CANCELLED 0:15`
  - 13621 `FAILED 1:0`

The claimed history is substantively real. The evidence's phrase “Six jobs failed before 13622” is imprecise because two of those six were `CANCELLED` and `TIMEOUT`, not Slurm `FAILED` at the top-level record.

### Local measurements

```text
16 passed in 0.30s
All checks passed!
5 files already formatted
```

The committed evidence says:

```text
uv run ruff format --check . => 7 files already formatted
```

This is a direct artifact discrepancy. It is the reason the overall verdict is `needs-fix`, despite the formatter command itself exiting successfully.

### Mutation proof

The mutated test was `tests/test_sync_slurm.py::test_templates_hardlink_from_the_shared_cache_instead_of_copying`. It failed with one assertion failure and `1 failed, 9 passed`; the original checkout was restored after the probe.

### Patch containment

`git diff-tree` reports exactly one changed path:

```text
M	.omo/evidence/task-2-kojev.txt
```

That path is justified as a task evidence record. No scripts or tests are changed by this commit. Therefore the commit does not itself contain the claimed template implementation; those claims describe the commit tree/base state rather than a change introduced by this commit. This is a scope/provenance concern, not a silent product failure.

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | ssh-transcript | Batched live `sacct`, job stdout, shared venv/Python measurements, and six history spot-checks | `.omo/evidence/st_01a0c43a/cluster.txt` |
| A2 | ssh-transcript | Real wrong-GRES submission and exact scheduler error | `.omo/evidence/st_01a0c43a/cluster-second.txt` |
| A3 | static-audit | Commit-tree script forbidden-write audit, GPU template audit, and no-gpu02 scan | `.omo/evidence/st_01a0c43a/static.txt` |
| A4 | command-output | Disposable checkout pytest, ruff check, and ruff format output | `.omo/evidence/st_01a0c43a/pytest.txt`, `.omo/evidence/st_01a0c43a/ruff-check.txt`, `.omo/evidence/st_01a0c43a/ruff-format.txt` |
| A5 | source-artifact | Committed evidence file reviewed for exact claims and compared to measured output | `.omo/evidence/st_01a0c43a/evidence-commit.txt` |
| A6 | mutation-test-output | Deliberately broken hardlink invariant and named failing test | `.omo/evidence/st_01a0c43a/mutation.txt` |
| A7 | git-inspection | Commit changed-path list for containment/provenance review | `.omo/evidence/st_01a0c43a/paths.txt` |

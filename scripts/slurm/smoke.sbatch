#!/usr/bin/env bash
#SBATCH --job-name=kojev-smoke
#SBATCH --partition=interactive
#SBATCH --gres=gpu:rtx6000:1
# 2h, not 30m: the plan's own smoke command (--limit 2000, 1 epoch) evaluates the
# full 6,499-example validation split at every 0.25 epoch, which does not fit a
# 30-minute cap on this hardware (job 13626 was still training at 25 minutes).
# The interactive partition permits up to 12h, so this stays well inside policy.
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data2/jeffrey/kojev/runs/kojev-smoke-%j.out
#SBATCH --error=/data2/jeffrey/kojev/runs/kojev-smoke-%j.err

set -euo pipefail

export HF_HOME=/data2/jeffrey/kojev/hf-cache
# Shared, warm package cache. A per-job cache makes every job re-download the
# multi-GB CUDA wheel set, which is what exhausted the wall clock on jobs
# 13599/13601/13603. The venv below stays job-isolated to avoid the .venv race.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data2/jeffrey/kojev/uv-cache}"
export UV_PROJECT_ENVIRONMENT=/data2/jeffrey/kojev/runs/venv-${SLURM_JOB_ID}
# hardlink, not copy: the shared cache and the per-job venv live on the same
# /data2 mount, so hardlinking installs torch (~10k files, ~2GB) without
# re-copying it per job. Removing a job venv only drops link counts; the
# shared cache entries survive.
export UV_LINK_MODE=hardlink

cd /data2/jeffrey/kojev/repo
# Prefer a PREBUILT shared environment. Installing torch (~10k files) into a
# per-job venv costs more than the smoke wall clock on this NFS mount, which is
# what exhausted job 13620. The shared venv is read-only to jobs, so the
# original 13597 .venv race cannot recur.
SHARED_VENV=/data2/jeffrey/kojev/venv-shared
if [ -x "${SHARED_VENV}/bin/python" ]; then
  export UV_PROJECT_ENVIRONMENT="${SHARED_VENV}"
  export VIRTUAL_ENV="${SHARED_VENV}"
  export PATH="${SHARED_VENV}/bin:${PATH}"
else
  # --no-dev: the cluster trains, it does not lint. basedpyright ships ~20k tiny
  # typeshed stub files and pulls nodejs-wheel-binaries; unpacking those over
  # NFS dominated the job wall clock for no runtime benefit.
  uv venv --allow-existing "$UV_PROJECT_ENVIRONMENT"
  uv sync --no-dev
fi
# The plan documents `sbatch smoke.sbatch -- python -m kojev.train ...`; sbatch
# forwards that leading `--` as $1, so drop it before dispatching.
if [ "${1:-}" = "--" ]; then
  shift
fi
if [ "$#" -eq 0 ]; then
  set -- python -c "import torch; print(torch.cuda.is_available())"
fi
# With a prebuilt shared environment, run its interpreter directly: uv is not
# needed to launch and must never mutate the shared venv. Fall back to
# `uv run --no-sync` only on the per-job path.
if [ -x "${SHARED_VENV}/bin/python" ]; then
  exec "$@"
else
  exec uv run --no-sync "$@"
fi

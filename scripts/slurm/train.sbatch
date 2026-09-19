#!/usr/bin/env bash
#SBATCH --job-name=kojev-train
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/data2/jeffrey/kojev/runs/kojev-train-%j.out
#SBATCH --error=/data2/jeffrey/kojev/runs/kojev-train-%j.err

set -euo pipefail

export HF_HOME=/data2/jeffrey/kojev/hf-cache
# Shared, warm package cache; see smoke.sbatch. The venv stays job-isolated.
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
if [ "$#" -eq 0 ]; then
  echo "usage: sbatch train.sbatch -- <uv-run-args>" >&2
  exit 1
fi
# With a prebuilt shared environment, run its interpreter directly: uv is not
# needed to launch and must never mutate the shared venv. Fall back to
# `uv run --no-sync` only on the per-job path.
if [ -x "${SHARED_VENV}/bin/python" ]; then
  exec "$@"
else
  exec uv run --no-sync "$@"
fi

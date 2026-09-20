#!/usr/bin/env bash
#SBATCH --job-name=kojev-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data2/jeffrey/kojev/runs/kojev-eval-%j.out
#SBATCH --error=/data2/jeffrey/kojev/runs/kojev-eval-%j.err

set -euo pipefail

export HF_HOME=/data2/jeffrey/kojev/hf-cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data2/jeffrey/kojev/uv-cache}"
export UV_LINK_MODE=hardlink

cd /data2/jeffrey/kojev/repo
SHARED_VENV=/data2/jeffrey/kojev/venv-shared
if [ -x "${SHARED_VENV}/bin/python" ]; then
  export UV_PROJECT_ENVIRONMENT="${SHARED_VENV}"
  export VIRTUAL_ENV="${SHARED_VENV}"
  export PATH="${SHARED_VENV}/bin:${PATH}"
else
  export UV_PROJECT_ENVIRONMENT=/data2/jeffrey/kojev/runs/venv-${SLURM_JOB_ID}
  uv venv --allow-existing "$UV_PROJECT_ENVIRONMENT"
  uv sync --no-dev
fi
if [ "${1:-}" = "--" ]; then
  shift
fi
if [ "$#" -eq 0 ]; then
  echo "usage: sbatch eval.sbatch -- python -m kojev.evaluate ..." >&2
  exit 1
fi
if [ -x "${SHARED_VENV}/bin/python" ]; then
  exec "$@"
else
  exec uv run --no-sync "$@"
fi

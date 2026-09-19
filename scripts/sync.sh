#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh gpu01 "mkdir -p /data2/jeffrey/kojev/{repo,data,runs,hf-cache}"

rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude data \
  "${ROOT}/" gpu01:/data2/jeffrey/kojev/repo/

#!/bin/bash
# CLI for the sqlite-backed experiment queue (src/qnas_queue).
#
# Usage:
#   scripts/qnas-queue.sh add --mode pipeline --config configs/config_files_cifar/01_deterministic_13-8-4.yml \
#       --experiment-path experiment_cifar10_progressive/exp7 --extra "-d cifar10 -M -T -X" \
#       --gpu-ids 0                     # optional: pin this job to GPU(s) via CUDA_VISIBLE_DEVICES
#   scripts/qnas-queue.sh list
#   scripts/qnas-queue.sh remove 3       # or: cancel 3
#   scripts/qnas-queue.sh start --workers 2   # concurrency; default 1, or $QNAS_QUEUE_WORKERS
#   scripts/qnas-queue.sh status
#   scripts/qnas-queue.sh logs -f
#   scripts/qnas-queue.sh stop
#
# Run `scripts/qnas-queue.sh <command> --help` for per-command options.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"
cd "${PROJECT_DIR}"

export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec uv run python -m qnas_queue.cli "$@"

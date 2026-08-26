#!/bin/bash
# Streamlit dashboard for the qnas_torch experiment queue: live queue state,
# per-job log tailing, and iterative evolution/retrain progress charts.
#
# Usage:
#   scripts/run_dashboard.sh [-- <extra streamlit args>]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"
cd "${PROJECT_DIR}"

export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec uv run streamlit run "${SRC_DIR}/qnas_dashboard/app.py" "$@"

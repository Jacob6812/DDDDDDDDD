#!/usr/bin/env bash
# FinAgent baseline × (wA, wD), half-year warm-up, full offline data path.
#
# Warm-up runs 20 symbols concurrently; only the very first symbol per process
# runs under a small init gate. Per-bar workers are uncapped. Warm-up state is
# cached to disk (storage/cache/external_strategy_adapters/finagent_warmup/...),
# so re-runs and the second window skip the 6-month training loop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/storage/logs/external_agents_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[FATAL] env file not found: ${ENV_FILE}" >&2; exit 1
fi
load_env_file "${ENV_FILE}" --force
init_run_env

# Symbol-level thread pool for per-bar LLM fanout; warm-up init fans out at
# this width too.
export EXTERNAL_STRATEGY_PARALLEL_WORKERS="${EXTERNAL_STRATEGY_PARALLEL_WORKERS:-100}"
# FinAgent-specific override (defaults to EXTERNAL_STRATEGY_PARALLEL_WORKERS).
export FINAGENT_PARALLEL_WORKERS="${FINAGENT_PARALLEL_WORKERS:-100}"

# Polygon retry (should not fire because precache ran, but keep robust).
export POLYGON_CLIENT_MAX_ATTEMPTS="${POLYGON_CLIENT_MAX_ATTEMPTS:-100}"
export POLYGON_CLIENT_RETRY_INTERVAL_SEC="${POLYGON_CLIENT_RETRY_INTERVAL_SEC:-1.0}"

# Half-year warm-up config (training_years=0.5).
CFG="${ROOT}/backtest/stockbench/config_darwintrade.yaml"

declare -a WINDOWS=(
    "wA:2025-03-03:2025-06-30"
    "wD:2025-12-01:2026-03-31"
)

for winspec in "${WINDOWS[@]}"; do
    IFS=":" read -r winlabel wstart wend <<< "$winspec"
    run_id="ext_${winlabel}_finagent_ls"
    launch "${winlabel}_finagent" \
        "${PYTHON_BIN}" -m backtest.stockbench.cli \
            --cfg "${CFG}" \
            --strategy finagent \
            --start "${wstart}" --end "${wend}" \
            --run-id "${run_id}" \
            --data-mode offline_only \
            --resume
done

wait_all
exit $?

#!/usr/bin/env bash
# Parallel launcher for a set of stockbench ablations over one window.
#
# Which ablations run is controlled by $ABLATIONS (space-separated names from
# the ABL_CFG map in _common.sh). Runs are namespaced by $OUT_NS so multiple
# env files (e.g. .env vs .envmm) can run concurrently without clobbering.
#
# Env overrides:
#   ABLATIONS   (default "baseline no-strategic no-tactical no-memory")
#               use "${FACTORIAL[*]}" for the full 2^3 factorial.
#   ENV_FILE    (default .env)
#   OUT_NS      output namespace prefix for run-ids + log dir (default empty)
#   START_DATE  (default 2025-03-03)
#   END_DATE    (default 2025-06-30)
#   DATA_MODE   (default auto)  passed to --data-mode
#   PYTHON      python interpreter (default: first on PATH)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

OUT_NS="${OUT_NS:-}"
START_DATE="${START_DATE:-2025-03-03}"
END_DATE="${END_DATE:-2025-06-30}"
DATA_MODE="${DATA_MODE:-auto}"
ABLATIONS="${ABLATIONS:-baseline no-strategic no-tactical no-memory}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_NS="${OUT_NS:+${OUT_NS}_}"
LOG_DIR="${ROOT}/storage/logs/ablations_${LOG_NS}${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
[[ -f "${ENV_FILE}" ]] && load_env_file "${ENV_FILE}"
init_run_env

RID_PREFIX="${OUT_NS:+${OUT_NS}_}"
for abl in ${ABLATIONS}; do
    cfg="${ABL_CFG[${abl}]:-}"
    if [[ -z "${cfg}" ]]; then
        echo "[ERR] unknown ablation '${abl}'. Valid: ${!ABL_CFG[*]}" >&2
        exit 2
    fi
    run_id="${RID_PREFIX}sb_${abl}_${TIMESTAMP}"
    launch "stockbench_${abl}" \
        "${PYTHON_BIN}" -m backtest.stockbench.cli \
            --cfg "${cfg}" \
            --start "${START_DATE}" --end "${END_DATE}" \
            --run-id "${run_id}" \
            --data-mode "${DATA_MODE}" \
            --resume
done

wait_all
exit $?

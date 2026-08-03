#!/usr/bin/env bash
# Experiment-matrix launcher.
#
# Runs the reproducibility matrix: the full 2^3 evolution factorial plus the
# four design-alternative variants, over one or more windows and repeats. Each
# run is one (experiment, config, window, repeat) tuple with a unique run-id so
# nothing overwrites anything, and LLM sampling varies per repeat via LLM_SEED.
#
# Experiments:
#   factorial   full 2^3 factorial over analyst/tactical/strategic evolution
#   shadow      rollback judged against a replayed pre-patch policy
#   signed-ic   negative-IC roles used contrarian instead of dropped
#   linear-conf confidence enters sizing linearly instead of quadratically
#   rule-regime deterministic trend rule replaces the LLM regime classifier
#
# Env overrides:
#   EXP         experiments to include (default: all five)
#   REPEATS     repeat indices to launch (default "1 2 3")
#   WINDOWS     windows to run (default "w1 w2")
#   SEQUENTIAL  1 = finish each repeat before starting the next; 0 = all at
#               once, fully parallel (default 0)
#   RESUME_TS   reuse this run-id timestamp so --resume continues existing
#               report dirs (default: fresh timestamp)
#   ENV_FILE    (default .env)
#   DATA_MODE   passed to --data-mode (default auto)
#   PYTHON      python interpreter (default: first on PATH)
#   DRYRUN      1 = list the runs without launching
#
# Examples:
#   bash scripts/run_experiments.sh                          # full matrix, r1-3
#   EXP=factorial REPEATS=1 bash scripts/run_experiments.sh  # factorial only
#   SEQUENTIAL=1 bash scripts/run_experiments.sh             # chain r1 -> r2 -> r3
#   RESUME_TS=20250721_031223 REPEATS=2 bash scripts/run_experiments.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EXP="${EXP:-factorial shadow signed-ic linear-conf rule-regime}"
REPEATS="${REPEATS:-1 2 3}"
WINDOWS="${WINDOWS:-w1 w2}"
SEQUENTIAL="${SEQUENTIAL:-0}"
RESUME_TS="${RESUME_TS:-}"
DATA_MODE="${DATA_MODE:-auto}"
DRYRUN="${DRYRUN:-0}"

ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
[[ -f "${ENV_FILE}" ]] && load_env_file "${ENV_FILE}"
init_run_env

# Per-repeat seeds so replicates differ in LLM sampling.
REPEAT_SEEDS=(101 202 303 404 505 606)

# Window bounds.
declare -A WIN_START=( [w1]=2025-03-03 [w2]=2025-12-01 )
declare -A WIN_END=(   [w1]=2025-06-30 [w2]=2026-03-31 )

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/storage/logs/exp_${TS}"
mkdir -p "${LOG_DIR}"

# Extra per-experiment env toggles (baseline cfg unless running the factorial).
declare -A EXP_ENV=(
    [shadow]="DARWIN_ROLLBACK_REFERENCE=pre_patch_policy"
    [signed-ic]="DARWIN_IC_SIGN_MODE=signed"
    [linear-conf]="DARWIN_CONFIDENCE_SQUARED=0"
    [rule-regime]="DARWIN_REGIME_RULE_ONLY=1"
)

# launch_exp <label> <cfg> <win> <seed> [EXTRA_ENV...]
# Builds the run-id (reusing RESUME_TS when set so --resume finds the dir) and
# launches one backtest with the given per-run env overrides.
launch_exp() {
    local label="$1" cfg="$2" win="$3" seed="$4"; shift 4
    local rid_ts="${RESUME_TS:-${TS}}"
    local run_id="${label}_${rid_ts}"
    local logf="${LOG_DIR}/${label}.log"
    if [[ "${DRYRUN}" == "1" ]]; then
        echo "[DRY] ${label}  seed=${seed}  extra=[$*]  cfg=$(basename "${cfg}")  ${win}"
        return
    fi
    ( env "$@" LLM_SEED="${seed}" \
        "${PYTHON_BIN}" -m backtest.stockbench.cli \
            --cfg "${cfg}" \
            --start "${WIN_START[$win]}" --end "${WIN_END[$win]}" \
            --run-id "${run_id}" --data-mode "${DATA_MODE}" --resume \
        >"${logf}" 2>&1 ) &
    PIDS+=($!); LABELS+=("${label}")
    echo "[LAUNCH] ${label}  pid=$! seed=${seed}"
}

# Launch every (experiment, window) tuple for a single repeat.
launch_repeat() {
    local r="$1" seed="${REPEAT_SEEDS[$(( $1 - 1 ))]}" exp win
    for exp in ${EXP}; do
        for win in ${WINDOWS}; do
            case "${exp}" in
                factorial)
                    local abl
                    for abl in "${FACTORIAL[@]}"; do
                        launch_exp "factorial_${win}_${abl}_r${r}" "${ABL_CFG[$abl]}" "${win}" "${seed}"
                    done ;;
                shadow|signed-ic|linear-conf|rule-regime)
                    launch_exp "${exp}_${win}_r${r}" "${BASE_CFG}" "${win}" "${seed}" ${EXP_ENV[$exp]} ;;
                *)
                    echo "[ERR] unknown experiment '${exp}'. Valid: factorial shadow signed-ic linear-conf rule-regime" >&2
                    exit 2 ;;
            esac
        done
    done
}

if [[ "${SEQUENTIAL}" == "1" ]]; then
    # Chain repeats: finish each before starting the next.
    FAIL=0
    for r in ${REPEATS}; do
        echo "###### repeat r${r} starting at $(date) ######"
        PIDS=(); LABELS=(); LOGFILES=()
        launch_repeat "${r}"
        [[ "${DRYRUN}" == "1" ]] && continue
        wait_all || FAIL=$((FAIL + $?))
        echo "###### repeat r${r} finished at $(date) ######"
    done
    [[ "${DRYRUN}" == "1" ]] && { echo "(dry run)"; exit 0; }
    echo "ALL REPEATS DONE. ${FAIL} failures. logs under ${ROOT}/storage/logs/"
    exit ${FAIL}
else
    # All repeats at once, fully parallel.
    for r in ${REPEATS}; do
        launch_repeat "${r}"
    done
    [[ "${DRYRUN}" == "1" ]] && { echo "(dry run — nothing launched)"; exit 0; }
    printf '%s\n' "${LABELS[@]}" > "${LOG_DIR}/_manifest.txt"
    wait_all
    exit $?
fi

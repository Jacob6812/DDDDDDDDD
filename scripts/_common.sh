# Shared helpers for the stockbench run scripts. Source this file:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/_common.sh"
# It sets ROOT, resolves PYTHON_BIN, exports thread/hash env, and provides
# load_env_file, launch, and wait_all.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ── resolve python ──────────────────────────────────────────────────────────
# Honours $PYTHON if set, otherwise picks the first interpreter on PATH.
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python python.exe python3 py.exe py; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

# ── env-file loading ──────────────────────────────────────────────────────────
trim_text() { local t="$1"; t="${t#"${t%%[![:space:]]*}"}"; t="${t%"${t##*[![:space:]]}"}"; printf '%s' "${t}"; }

# load_env_file <path> [--force]
# Parses KEY=VALUE lines (optionally `export KEY=VALUE`, quoted values), then
# exports them. Without --force, a var already set by the caller is preserved,
# so a wrapper can override LLM_SEED / DARWIN_* per run.
load_env_file() {
    local env_path="$1" force="${2:-}"
    local line trimmed key value
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"; trimmed="$(trim_text "${line}")"
        [[ -z "${trimmed}" || "${trimmed}" == \#* ]] && continue
        [[ "${trimmed}" == export\ * ]] && trimmed="${trimmed#export }"
        [[ "${trimmed}" != *=* ]] && continue
        key="$(trim_text "${trimmed%%=*}")"; value="$(trim_text "${trimmed#*=}")"
        [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [[ ${#value} -ge 2 ]]; then
            if   [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then value="${value:1:${#value}-2}"
            elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then value="${value:1:${#value}-2}"; fi
        fi
        if [[ "${force}" == "--force" ]]; then
            export "${key}=${value}"
        else
            [[ -z "${!key+x}" ]] && export "${key}=${value}"
        fi
    done < "$env_path"
}

# ── deterministic single-threaded numeric env ─────────────────────────────────
init_run_env() {
    export PYTHONPATH="${ROOT}"
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

# ── process tracking + launch/wait ────────────────────────────────────────────
declare -a PIDS=() LABELS=() LOGFILES=()

# launch <label> <command...>
# Runs the command in the background, redirecting output to ${LOG_DIR}/<label>.log,
# and records its pid/label/logfile for wait_all. LOG_DIR must be set.
launch() {
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label}.log"
    echo "[LAUNCH] ${label}"
    "$@" >"${logfile}" 2>&1 &
    PIDS+=($!); LABELS+=("${label}"); LOGFILES+=("${logfile}")
}

# wait_all — wait for every launched pid, report per-run status, return #failures.
wait_all() {
    local failed=0 i
    echo ""
    echo "  ${#PIDS[@]} processes launched — waiting..."
    echo "  Logs: ${LOG_DIR}"
    echo ""
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[OK]   ${LABELS[$i]}"
        else
            echo "[FAIL] ${LABELS[$i]}  →  ${LOGFILES[$i]}"
            failed=$((failed + 1))
        fi
    done
    echo ""
    if [[ ${failed} -eq 0 ]]; then
        echo "  ALL ${#PIDS[@]} runs completed successfully."
    else
        echo "  ${failed}/${#PIDS[@]} runs FAILED. Check logs above."
    fi
    echo "  Logs dir: ${LOG_DIR}"
    return ${failed}
}

# ── ablation-name → config-path map (shared by the runners) ───────────────────
ABL_DIR="${ROOT}/backtest/stockbench/ablation"
BASE_CFG="${ROOT}/backtest/stockbench/config_darwintrade.yaml"
declare -A ABL_CFG=(
    [baseline]="${BASE_CFG}"
    [no-strategic]="${ABL_DIR}/config_darwintrade_no_strategic.yaml"
    [no-tactical]="${ABL_DIR}/config_darwintrade_no_tactical.yaml"
    [no-analyst-capsule]="${ABL_DIR}/config_darwintrade_no_analyst_capsule.yaml"
    [only-tactical]="${ABL_DIR}/config_darwintrade_only_tactical.yaml"
    [only-strategic]="${ABL_DIR}/config_darwintrade_only_strategic.yaml"
    [only-capsule]="${ABL_DIR}/config_darwintrade_only_capsule.yaml"
    [no-memory]="${ABL_DIR}/config_darwintrade_no_memory.yaml"
)
# Full 2^3 factorial over {tactical, strategic, capsule}.
FACTORIAL=(baseline no-strategic no-tactical no-analyst-capsule only-tactical only-strategic only-capsule no-memory)

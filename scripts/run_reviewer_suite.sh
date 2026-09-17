#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [core|sensitivity|diagnostics]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
SUITE="${2:-core}"
if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
else
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_INPUT}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

MOCA_BIN="${MOCA_BIN:-${REPO_ROOT}/.venv/bin/moca}"
if [[ ! -x "${MOCA_BIN}" ]]; then
  MOCA_BIN="$(command -v moca || true)"
fi
if [[ -z "${MOCA_BIN}" ]]; then
  echo "moca is not installed; run ${SCRIPT_DIR}/setup.sh first." >&2
  exit 1
fi

BASE_NAME="${EXPERIMENT_PREFIX:-$(basename "${CONFIG_PATH}" .yaml)}"
RANDOM_K="${RANDOM_K:-5}"
IFS=',' read -r -a SEEDS <<< "${SEEDS:-2026,2027,2028}"

run_experiment() {
  local name="$1"
  shift
  "${MOCA_BIN}" run --config "${CONFIG_PATH}" --set "experiment_name=${name}" "$@"
}

cd -- "${REPO_ROOT}"
case "${SUITE}" in
  core)
    for seed in "${SEEDS[@]}"; do
      run_experiment "${BASE_NAME}_moca_s${seed}" --set "runtime.seed=${seed}"
      run_experiment "${BASE_NAME}_kft_s${seed}" --set "runtime.seed=${seed}" --ablation k_ft
      run_experiment "${BASE_NAME}_vanilla_s${seed}" --set "runtime.seed=${seed}" --ablation vanilla_ft
      run_experiment "${BASE_NAME}_rank80_s${seed}" --set "runtime.seed=${seed}" --ablation vanilla_ft --ablation matched_rank_80
      run_experiment "${BASE_NAME}_random_k${RANDOM_K}_s${seed}" \
        --set "runtime.seed=${seed}" \
        --set "clustering.num_clusters=${RANDOM_K}" \
        --ablation random_partition
    done
    ;;
  sensitivity)
    seed="${SEEDS[0]}"
    for k in 2 3 5 8; do
      run_experiment "${BASE_NAME}_k${k}_s${seed}" \
        --set "runtime.seed=${seed}" --set "clustering.num_clusters=${k}"
    done
    for lambda in 0.03 0.1 0.3; do
      safe_lambda="${lambda//./p}"
      run_experiment "${BASE_NAME}_lambda${safe_lambda}_s${seed}" \
        --set "runtime.seed=${seed}" --set "objective.lambda_kl=${lambda}"
    done
    run_experiment "${BASE_NAME}_m5_s${seed}" --set "runtime.seed=${seed}" --ablation first_5_tokens
    run_experiment "${BASE_NAME}_nonspecial_s${seed}" \
      --set "runtime.seed=${seed}" --ablation non_special_uniform
    run_experiment "${BASE_NAME}_meanpool_s${seed}" \
      --set "runtime.seed=${seed}" --ablation mean_token_pooling
    ;;
  diagnostics)
    run_name="${RUN_NAME:-${BASE_NAME}_moca_s${SEEDS[0]}}"
    "${MOCA_BIN}" evaluate \
      --config "${CONFIG_PATH}" \
      --set "experiment_name=${run_name}" \
      --ablation semantic_entropy_evaluation \
      --split test \
      --output "runs/${run_name}/evaluation/test_semantic_entropy_metrics.json"
    "${MOCA_BIN}" generate \
      --config "${CONFIG_PATH}" \
      --set "experiment_name=${run_name}" \
      --split test \
      --force-wrong-route
    "${MOCA_BIN}" evaluate \
      --config "${CONFIG_PATH}" \
      --set "experiment_name=${run_name}" \
      --split test \
      --predictions "runs/${run_name}/predictions/test_forced_wrong.jsonl" \
      --output "runs/${run_name}/evaluation/test_forced_wrong_metrics.json"
    ;;
  *)
    echo "Unknown suite: ${SUITE}; expected core, sensitivity, or diagnostics" >&2
    exit 2
    ;;
esac

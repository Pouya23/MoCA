#!/usr/bin/env bash
#SBATCH --job-name=moca-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_INPUT="${CONFIG_PATH:-${1:-}}"
SPLIT="${SPLIT:-test}"
FIT_CALIBRATOR="${FIT_CALIBRATOR:-1}"
ABLATION="${ABLATION:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

if [[ -z "${CONFIG_INPUT}" ]]; then
  echo "Set CONFIG_PATH or pass CONFIG as the first argument." >&2
  exit 2
fi
if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_FILE="${CONFIG_INPUT}"
else
  CONFIG_FILE="${REPO_ROOT}/${CONFIG_INPUT}"
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

MOCA_BIN="${MOCA_BIN:-${REPO_ROOT}/.venv/bin/moca}"
if [[ ! -x "${MOCA_BIN}" ]]; then
  MOCA_BIN="$(command -v moca || true)"
fi
if [[ -z "${MOCA_BIN}" ]]; then
  echo "moca is not installed; run ${SCRIPT_DIR}/setup.sh on the cluster first." >&2
  exit 1
fi

COMMON_ARGS=()
if [[ -n "${ABLATION}" ]]; then
  COMMON_ARGS+=(--ablation "${ABLATION}")
fi
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  COMMON_ARGS+=(--set "experiment_name=${EXPERIMENT_NAME}")
fi

cd -- "${REPO_ROOT}"
if [[ "${SPLIT}" = "test" && "${FIT_CALIBRATOR}" = "1" ]]; then
  "${MOCA_BIN}" generate \
    --config "${CONFIG_FILE}" \
    --split validation \
    "${COMMON_ARGS[@]}"
  "${MOCA_BIN}" calibrate \
    --config "${CONFIG_FILE}" \
    --split validation \
    "${COMMON_ARGS[@]}"
fi
"${MOCA_BIN}" generate \
  --config "${CONFIG_FILE}" \
  --split "${SPLIT}" \
  "${COMMON_ARGS[@]}"
"${MOCA_BIN}" evaluate \
  --config "${CONFIG_FILE}" \
  --split "${SPLIT}" \
  "${COMMON_ARGS[@]}"

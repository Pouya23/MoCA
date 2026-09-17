#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ABLATION="${ABLATION:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
SPLIT="${SPLIT:-test}"

if [[ $# -ne 1 ]]; then
  echo "Usage: GPU_IDS=0,1,... $0 CONFIG" >&2
  echo "Optional environment: ABLATION, EXPERIMENT_NAME, SPLIT, MOCA_BIN." >&2
  exit 2
fi

CONFIG_INPUT="$1"
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
MOCA_PYTHON="$(dirname -- "${MOCA_BIN}")/python"
if [[ ! -x "${MOCA_PYTHON}" ]]; then
  MOCA_PYTHON="${PYTHON_BIN:-python3}"
fi

GPU_LIST="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${GPU_LIST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
if [[ -z "${GPU_LIST}" ]]; then
  echo "No GPUs found. Set GPU_IDS to a comma-separated device list." >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_IDS did not contain any devices." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU identifier: ${gpu}" >&2
    exit 2
  fi
done

COMMON_ARGS=()
if [[ -n "${ABLATION}" ]]; then
  COMMON_ARGS+=(--ablation "${ABLATION}")
fi
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  COMMON_ARGS+=(--set "experiment_name=${EXPERIMENT_NAME}")
fi

cd -- "${REPO_ROOT}"
"${MOCA_BIN}" prepare --config "${CONFIG_PATH}" "${COMMON_ARGS[@]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
  "${MOCA_BIN}" cluster --config "${CONFIG_PATH}" "${COMMON_ARGS[@]}"

RUN_DIR="$(
  "${MOCA_PYTHON}" -c \
    'import pathlib, sys; from moca.config import load_experiment_config; c=load_experiment_config(sys.argv[1], overrides=([f"experiment_name={sys.argv[2]}"] if sys.argv[2] else None), ablations=([sys.argv[3]] if sys.argv[3] else None)); print(pathlib.Path(c.run_dir).resolve())' \
    "${CONFIG_PATH}" "${EXPERIMENT_NAME}" "${ABLATION}"
)"
MANIFEST="${RUN_DIR}/clusters/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Cluster manifest not found after clustering: ${MANIFEST}" >&2
  exit 1
fi
NUM_EXPERTS="$(
  "${MOCA_PYTHON}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["num_clusters"]))' \
    "${MANIFEST}"
)"
if [[ "${NUM_EXPERTS}" -lt 1 ]]; then
  echo "Invalid num_clusters in ${MANIFEST}: ${NUM_EXPERTS}" >&2
  exit 1
fi

LOG_DIR="${RUN_DIR}/logs"
mkdir -p -- "${LOG_DIR}"
echo "Training ${NUM_EXPERTS} experts across ${#GPUS[@]} GPU(s)."

for ((wave_start=0; wave_start<NUM_EXPERTS; wave_start+=${#GPUS[@]})); do
  pids=()
  expert_ids=()
  for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    expert_id=$((wave_start + slot))
    if ((expert_id >= NUM_EXPERTS)); then
      break
    fi
    gpu="${GPUS[slot]}"
    echo "Launching expert ${expert_id} on GPU ${gpu}."
    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${MOCA_BIN}" train-expert \
          --config "${CONFIG_PATH}" \
          --expert-id "${expert_id}" \
          "${COMMON_ARGS[@]}"
    ) >"${LOG_DIR}/expert_${expert_id}.log" 2>&1 &
    pids+=("$!")
    expert_ids+=("${expert_id}")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
      echo "Expert ${expert_ids[index]} failed; see ${LOG_DIR}/expert_${expert_ids[index]}.log" >&2
      failed=1
    fi
  done
  if ((failed)); then
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
  "${MOCA_BIN}" generate \
    --config "${CONFIG_PATH}" \
    --split "${SPLIT}" \
    "${COMMON_ARGS[@]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
  "${MOCA_BIN}" evaluate \
    --config "${CONFIG_PATH}" \
    --split "${SPLIT}" \
    "${COMMON_ARGS[@]}"


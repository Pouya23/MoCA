#!/usr/bin/env bash
#SBATCH --job-name=moca-dispatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_INPUT="${CONFIG_PATH:-${1:-}}"
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
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable in the dispatcher job." >&2
  exit 1
fi

MOCA_BIN="${MOCA_BIN:-${REPO_ROOT}/.venv/bin/moca}"
MOCA_PYTHON="$(dirname -- "${MOCA_BIN}")/python"
if [[ ! -x "${MOCA_PYTHON}" ]]; then
  MOCA_PYTHON="${PYTHON_BIN:-python3}"
fi

RUN_DIR="$(
  cd -- "${REPO_ROOT}"
  "${MOCA_PYTHON}" -c \
    'import pathlib, sys; from moca.config import load_experiment_config; c=load_experiment_config(sys.argv[1], overrides=([f"experiment_name={sys.argv[2]}"] if sys.argv[2] else None), ablations=([sys.argv[3]] if sys.argv[3] else None)); print(pathlib.Path(c.run_dir).resolve())' \
    "${CONFIG_FILE}" "${EXPERIMENT_NAME}" "${ABLATION}"
)"
MANIFEST="${RUN_DIR}/clusters/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Cluster manifest not found: ${MANIFEST}" >&2
  exit 1
fi
NUM_EXPERTS="$(
  "${MOCA_PYTHON}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["num_clusters"]))' \
    "${MANIFEST}"
)"
if [[ ! "${NUM_EXPERTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid num_clusters in ${MANIFEST}: ${NUM_EXPERTS}" >&2
  exit 1
fi

SBATCH_COMMON=(
  --chdir="${REPO_ROOT}"
  --export="ALL,CONFIG_PATH=${CONFIG_FILE}"
)
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_COMMON+=(--account="${SBATCH_ACCOUNT}")
fi
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_COMMON+=(--partition="${SBATCH_PARTITION}")
fi
if [[ -n "${SBATCH_QOS:-}" ]]; then
  SBATCH_COMMON+=(--qos="${SBATCH_QOS}")
fi
if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then
  SBATCH_COMMON+=(--constraint="${SBATCH_CONSTRAINT}")
fi

train_job_raw="$(
  sbatch --parsable \
    "${SBATCH_COMMON[@]}" \
    --array="0-$((NUM_EXPERTS - 1))" \
    "${SCRIPT_DIR}/slurm_train_array.sh"
)"
train_job="${train_job_raw%%;*}"
evaluate_job_raw="$(
  sbatch --parsable \
    "${SBATCH_COMMON[@]}" \
    --dependency="afterok:${train_job}" \
    "${SCRIPT_DIR}/slurm_generate_evaluate.sh"
)"
evaluate_job="${evaluate_job_raw%%;*}"

echo "Discovered ${NUM_EXPERTS} experts from ${MANIFEST}"
echo "Submitted expert array job:  ${train_job} (0-$((NUM_EXPERTS - 1)))"
echo "Submitted generation/eval:   ${evaluate_job}"

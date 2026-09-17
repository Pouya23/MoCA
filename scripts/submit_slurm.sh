#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG" >&2
  echo "Optional: ABLATION, EXPERIMENT_NAME, SBATCH_ACCOUNT, SBATCH_PARTITION," >&2
  echo "SBATCH_QOS, SBATCH_CONSTRAINT, and SPLIT." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available on this host." >&2
  exit 1
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

SBATCH_COMMON=(
  --chdir="${REPO_ROOT}"
  --export="ALL,CONFIG_PATH=${CONFIG_PATH}"
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

prepare_job_raw="$(
  sbatch --parsable \
    "${SBATCH_COMMON[@]}" \
    "${SCRIPT_DIR}/slurm_prepare_cluster.sh"
)"
prepare_job="${prepare_job_raw%%;*}"
dispatch_job_raw="$(
  sbatch --parsable \
    "${SBATCH_COMMON[@]}" \
    --dependency="afterok:${prepare_job}" \
    "${SCRIPT_DIR}/slurm_dispatch.sh"
)"
dispatch_job="${dispatch_job_raw%%;*}"

echo "Submitted prepare/cluster job: ${prepare_job}"
echo "Submitted array dispatcher:    ${dispatch_job}"
echo "The dispatcher will read the selected k and submit training plus evaluation."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG EXPERT_ID [MOCA_OPTIONS...]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
EXPERT_ID="$2"
shift 2
if [[ ! "${EXPERT_ID}" =~ ^[0-9]+$ ]]; then
  echo "EXPERT_ID must be a non-negative integer, got: ${EXPERT_ID}" >&2
  exit 2
fi
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

cd -- "${REPO_ROOT}"
exec "${MOCA_BIN}" train-expert \
  --config "${CONFIG_PATH}" \
  --expert-id "${EXPERT_ID}" \
  "$@"


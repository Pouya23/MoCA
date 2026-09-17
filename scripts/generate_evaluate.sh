#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SPLIT="${SPLIT:-test}"
FIT_CALIBRATOR="${FIT_CALIBRATOR:-1}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [MOCA_OPTIONS...]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift
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
if [[ "${SPLIT}" = "test" && "${FIT_CALIBRATOR}" = "1" ]]; then
  "${MOCA_BIN}" generate --config "${CONFIG_PATH}" --split validation "$@"
  "${MOCA_BIN}" calibrate --config "${CONFIG_PATH}" --split validation "$@"
fi
"${MOCA_BIN}" generate --config "${CONFIG_PATH}" --split "${SPLIT}" "$@"
"${MOCA_BIN}" evaluate --config "${CONFIG_PATH}" --split "${SPLIT}" "$@"

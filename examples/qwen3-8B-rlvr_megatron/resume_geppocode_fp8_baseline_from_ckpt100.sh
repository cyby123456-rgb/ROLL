#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEPPO_EXP_NAME="${GEPPO_EXP_NAME:-qwen3-1p7b-geppo-code-fp8-baseline-s42}"
GEPPO_OUTPUT_ROOT="${GEPPO_OUTPUT_ROOT:-/mnt/shared-storage-user/liujinyi/lj/ROLL/output/${GEPPO_EXP_NAME}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${GEPPO_OUTPUT_ROOT}/checkpoints/20260727-011030/checkpoint-100}"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "[geppocode-baseline-resume] ERROR: checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${RESUME_CHECKPOINT}/pipeline/worker_state_pipeline.json" ]]; then
  echo "[geppocode-baseline-resume] ERROR: checkpoint is missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

echo "[geppocode-baseline-resume] exp_name=${GEPPO_EXP_NAME}"
echo "[geppocode-baseline-resume] output_root=${GEPPO_OUTPUT_ROOT}"
echo "[geppocode-baseline-resume] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[geppocode-baseline-resume] quantization=fp8, TIS=false, UV=false"

export GEPPO_EXP_NAME GEPPO_OUTPUT_ROOT

exec bash "${SCRIPT_DIR}/run_geppo_code_pipeline.sh" \
  "resume_from_checkpoint=${RESUME_CHECKPOINT}" \
  "$@"

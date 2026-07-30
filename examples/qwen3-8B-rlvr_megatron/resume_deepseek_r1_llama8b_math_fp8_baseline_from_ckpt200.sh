#!/bin/bash
set -euo pipefail

# Resume the DeepSeek-R1-Distill-Llama-8B FP8 baseline from checkpoint-200.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-deepseek-r1-distill-llama8b-math-fp8-baseline-no-tis-no-uv-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/deepseek-r1-distill-llama8b-math-fp8-baseline-no-tis-no-uv-s42/checkpoints/20260728-214919/checkpoint-200}"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "[deepseek-r1-llama8b-resume] ERROR: RESUME_CHECKPOINT not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[deepseek-r1-llama8b-resume] ERROR: RESUME_CHECKPOINT missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"

export EXP_NAME
export EXP_ROOT
export RESUME_CHECKPOINT

exec bash "${SCRIPT_DIR}/run_deepseek_r1_llama8b_math_fp8_baseline.sh" "$@"

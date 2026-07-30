#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-int8-tis-uvtestthre-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/qwen3-1p7b-math-int8-tis-uvtestthre-s42}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/Qwen3-1.7B}"

if [[ ! -d "${EXP_ROOT}" ]]; then
  echo "[qwen3-1p7b-resume] ERROR: EXP_ROOT not found: ${EXP_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-1p7b-resume] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  shopt -s nullglob
  checkpoints=("${EXP_ROOT}"/checkpoints/*/checkpoint-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    echo "[qwen3-1p7b-resume] ERROR: no checkpoint found under ${EXP_ROOT}/checkpoints" >&2
    exit 1
  fi
  mapfile -t checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort -V)
  RESUME_CHECKPOINT="${checkpoints[${#checkpoints[@]} - 1]}"
fi

if [[ ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[qwen3-1p7b-resume] ERROR: checkpoint is missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[qwen3-1p7b-resume] exp_name=${EXP_NAME}"
echo "[qwen3-1p7b-resume] exp_root=${EXP_ROOT}"
echo "[qwen3-1p7b-resume] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[qwen3-1p7b-resume] quantization=int8, template=config defaults, TIS=true, UV=false"

# Leave all data_args.template values untouched so rlvr_config.yaml supplies the defaults.
exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  "resume_from_checkpoint=${RESUME_CHECKPOINT}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=false \
  "+actor_infer.strategy_args.strategy_config.quantization=int8" \
  "$@"

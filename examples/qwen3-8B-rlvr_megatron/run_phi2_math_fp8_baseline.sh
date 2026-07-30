#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-phi2-math-fp8-baseline-no-tis-no-uv-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/phi-2}"

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[phi2] exp_name=${EXP_NAME}"
echo "[phi2] model_path=${MODEL_PATH}"
echo "[phi2] exp_root=${EXP_ROOT}"
echo "[phi2] quantization=fp8, train_infer_ratio_trunc_is=false, update_validation.enable=false"
echo "[phi2] launching RLVR pipeline..."

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  prompt_length=512 \
  response_length=1536 \
  train_infer_ratio_trunc_is=false \
  update_validation.enable=false \
  actor_train.data_args.template=base \
  actor_infer.data_args.template=base \
  reference.data_args.template=base \
  rewards.math_rule.data_args.template=base \
  "+actor_infer.strategy_args.strategy_config.quantization=fp8" \
  "$@"

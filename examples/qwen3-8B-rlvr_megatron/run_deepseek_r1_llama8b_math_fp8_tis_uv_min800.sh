#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-deepseek-r1-distill-llama8b-math-fp8-tis-uv-min800-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/DeepSeek-R1-Distill-Llama-8B}"
TEMPLATE="${TEMPLATE:-native}"

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[deepseek-r1-llama8b] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! grep -q '"chat_template"' "${MODEL_PATH}/tokenizer_config.json" 2>/dev/null; then
  echo "[deepseek-r1-llama8b] ERROR: ${MODEL_PATH}/tokenizer_config.json has no chat_template; set TEMPLATE=base or add a template override." >&2
  exit 1
fi

echo "[deepseek-r1-llama8b] exp_name=${EXP_NAME}"
echo "[deepseek-r1-llama8b] model_path=${MODEL_PATH}"
echo "[deepseek-r1-llama8b] exp_root=${EXP_ROOT}"
echo "[deepseek-r1-llama8b] template=${TEMPLATE} (uses tokenizer.apply_chat_template; DeepSeek tokenizer provides chat_template)"
echo "[deepseek-r1-llama8b] quantization=fp8, train_infer_ratio_trunc_is=true, update_validation.enable=true, update_validation.min_steps_for_threshold=800"
echo "[deepseek-r1-llama8b] launching RLVR pipeline..."

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=true \
  update_validation.min_steps_for_threshold=800 \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  "+actor_infer.strategy_args.strategy_config.quantization=fp8" \
  "$@"

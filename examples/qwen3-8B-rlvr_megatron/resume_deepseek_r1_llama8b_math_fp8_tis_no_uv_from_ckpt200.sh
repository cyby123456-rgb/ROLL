#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the new no-UV trajectory separate from its TIS+UV checkpoint source.
EXP_NAME="${EXP_NAME:-deepseek-r1-distill-llama8b-math-fp8-tis-no-uv-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/DeepSeek-R1-Distill-Llama-8B}"
TEMPLATE="${TEMPLATE:-native}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/deepseek-r1-distill-llama8b-math-fp8-tis-uv-min800-s42/checkpoints/20260729-142645/checkpoint-200}"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "[deepseek-r1-llama8b-tis-no-uv-resume] ERROR: checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${RESUME_CHECKPOINT}/pipeline/worker_state_pipeline.json" ]]; then
  echo "[deepseek-r1-llama8b-tis-no-uv-resume] ERROR: checkpoint is missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[deepseek-r1-llama8b-tis-no-uv-resume] ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! grep -q '"chat_template"' "${MODEL_PATH}/tokenizer_config.json" 2>/dev/null; then
  echo "[deepseek-r1-llama8b-tis-no-uv-resume] ERROR: ${MODEL_PATH}/tokenizer_config.json has no chat_template" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[deepseek-r1-llama8b-tis-no-uv-resume] exp_name=${EXP_NAME}"
echo "[deepseek-r1-llama8b-tis-no-uv-resume] exp_root=${EXP_ROOT}"
echo "[deepseek-r1-llama8b-tis-no-uv-resume] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[deepseek-r1-llama8b-tis-no-uv-resume] quantization=fp8, TIS=true, UV=false, max_steps=800"

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
  max_steps=800 \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=false \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  "+actor_infer.strategy_args.strategy_config.quantization=fp8" \
  "$@"

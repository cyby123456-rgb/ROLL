#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-deepseek-r1-distill-llama8b-math-fp8-baseline-no-tis-no-uv-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/DeepSeek-R1-Distill-Llama-8B}"
TEMPLATE="${TEMPLATE:-native}"
RESUME_ROOT="${RESUME_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/deepseek-r1-distill-llama8b-math-fp8-baseline-no-tis-no-uv-s42}"

if [[ ! -d "${RESUME_ROOT}" ]]; then
  echo "[deepseek-r1-llama8b] ERROR: RESUME_ROOT not found: ${RESUME_ROOT}" >&2
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  shopt -s nullglob
  checkpoints=("${RESUME_ROOT}"/checkpoints/*/checkpoint-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    echo "[deepseek-r1-llama8b] ERROR: no checkpoint found under ${RESUME_ROOT}/checkpoints" >&2
    exit 1
  fi
  IFS=$'\n' read -r -d '' -a checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort -V && printf '\0')
  RESUME_CHECKPOINT="${checkpoints[-1]}"
fi

if [[ ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[deepseek-r1-llama8b] ERROR: RESUME_CHECKPOINT missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

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
echo "[deepseek-r1-llama8b] resume_root=${RESUME_ROOT}"
echo "[deepseek-r1-llama8b] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[deepseek-r1-llama8b] template=${TEMPLATE} (uses tokenizer.apply_chat_template; DeepSeek tokenizer provides chat_template)"
echo "[deepseek-r1-llama8b] quantization=fp8, train_infer_ratio_trunc_is=false, update_validation.enable=false"
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
  "resume_from_checkpoint=${RESUME_CHECKPOINT}" \
  train_infer_ratio_trunc_is=false \
  update_validation.enable=false \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  "+actor_infer.strategy_args.strategy_config.quantization=fp8" \
  "$@"

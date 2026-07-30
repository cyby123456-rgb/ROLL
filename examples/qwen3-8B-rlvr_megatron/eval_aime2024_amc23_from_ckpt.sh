#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/qwen3-1p7b-math-fp16-ours-s42-0p00001/checkpoints/20260727-193725/checkpoint-200}"
EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-fp16-ours-s42-0p00001-eval-aime2024-amc23}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
EVAL_FILE="${EVAL_FILE:-data/math_benchmarks_aime2024_amc23.jsonl}"
EVAL_N="${EVAL_N:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EXP_ROOT}/aime2024_amc23_acc_qwen.jsonl}"

if [[ ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[eval-aime2024-amc23] ERROR: RESUME_CHECKPOINT missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/../../${EVAL_FILE}" && ! -f "${EVAL_FILE}" ]]; then
  echo "[eval-aime2024-amc23] ERROR: EVAL_FILE not found: ${EVAL_FILE}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[eval-aime2024-amc23] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[eval-aime2024-amc23] eval_file=${EVAL_FILE}"
echo "[eval-aime2024-amc23] eval_n=${EVAL_N}"
echo "[eval-aime2024-amc23] num_gpus=${NUM_GPUS}"
echo "[eval-aime2024-amc23] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[eval-aime2024-amc23] output_jsonl=${OUTPUT_JSONL}"

export CUDA_VISIBLE_DEVICES

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "resume_from_checkpoint=${RESUME_CHECKPOINT}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  ++only_rollout=true \
  ++only_rollout_source=validation \
  "++only_rollout_num_return_sequences=${EVAL_N}" \
  "++only_rollout_output_path=${OUTPUT_JSONL}" \
  "validation.data_args.file_name=${EVAL_FILE}" \
  validation.generating_args.num_return_sequences=1 \
  "num_gpus_per_node=${NUM_GPUS}" \
  "actor_train.device_mapping='list(range(0,${NUM_GPUS}))'" \
  "actor_infer.device_mapping='list(range(0,${NUM_GPUS}))'" \
  "reference.device_mapping='list(range(0,${NUM_GPUS}))'" \
  "rewards.math_rule.world_size=${NUM_GPUS}" \
  update_validation.enable=false \
  train_infer_ratio_trunc_is=false \
  "$@"

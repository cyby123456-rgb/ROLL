#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAY_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  ROLL_NVIDIA_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
  if [[ -d "${ROLL_NVIDIA_ROOT}" ]]; then
    ROLL_NVIDIA_LIBS="$(find "${ROLL_NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib ! -path '*/~*/*' -print | sort | paste -sd: -)"
    if [[ -n "${ROLL_NVIDIA_LIBS}" ]]; then
      RAY_LD_LIBRARY_PATH="${ROLL_NVIDIA_LIBS}${RAY_LD_LIBRARY_PATH:+:${RAY_LD_LIBRARY_PATH}}"
    fi
  fi
fi

EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-vllm-tis-uv-scheduler-test64-2g-debug-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/Qwen3-1.7B}"
TEMPLATE="${TEMPLATE:-qwen2_5}"
MAX_STEPS="${MAX_STEPS:-3}"
EVAL_STEPS="${EVAL_STEPS:-9999}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
UV_BATCH_SIZE="${UV_BATCH_SIZE:-64}"
UV_GENERATE_MODE="${UV_GENERATE_MODE:-scheduler}"
UV_GENERATE_CONCURRENCY="${UV_GENERATE_CONCURRENCY:-64}"
UV_MIN_STEPS="${UV_MIN_STEPS:-1}"
UV_THRESHOLD="${UV_THRESHOLD:--0.00001}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-transformer_engine}"
GENERATE_OPT_LEVEL="${GENERATE_OPT_LEVEL:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ "${UV_GENERATE_MODE}" != "legacy" && "${UV_GENERATE_MODE}" != "scheduler" ]]; then
  echo "[qwen3-1p7b-vllm-uv-test64-2g] ERROR: UV_GENERATE_MODE must be legacy or scheduler" >&2
  exit 1
fi

if [[ -z "${VALIDATION_FILE:-}" ]]; then
  if [[ -f "${ROOT_DIR}/data/test.json" ]]; then
    VALIDATION_FILE="data/test.json"
  else
    VALIDATION_FILE="data/test.jsonl"
  fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-1p7b-vllm-uv-test64-2g] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ROOT_DIR}/${VALIDATION_FILE}" && ! -f "${VALIDATION_FILE}" ]]; then
  echo "[qwen3-1p7b-vllm-uv-test64-2g] ERROR: VALIDATION_FILE not found: ${VALIDATION_FILE}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[qwen3-1p7b-vllm-uv-test64-2g] exp_name=${EXP_NAME}"
echo "[qwen3-1p7b-vllm-uv-test64-2g] validation_file=${VALIDATION_FILE}"
echo "[qwen3-1p7b-vllm-uv-test64-2g] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[qwen3-1p7b-vllm-uv-test64-2g] rollout_batch_size=${ROLLOUT_BATCH_SIZE}, n=${NUM_RETURN_SEQUENCES}, response_length=${RESPONSE_LENGTH}"
echo "[qwen3-1p7b-vllm-uv-test64-2g] uv_batch_size=${UV_BATCH_SIZE}, uv_generate_mode=${UV_GENERATE_MODE}, uv_generate_concurrency=${UV_GENERATE_CONCURRENCY}"

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline_2gpus.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "max_steps=${MAX_STEPS}" \
  "eval_steps=${EVAL_STEPS}" \
  "save_steps=${SAVE_STEPS}" \
  "rollout_batch_size=${ROLLOUT_BATCH_SIZE}" \
  "num_return_sequences_in_group=${NUM_RETURN_SEQUENCES}" \
  "response_length=${RESPONSE_LENGTH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  "+system_envs.LD_LIBRARY_PATH=${RAY_LD_LIBRARY_PATH}" \
  "+generate_opt_level=${GENERATE_OPT_LEVEL}" \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  rewards.math_rule.data_args.template=qwen3 \
  "validation.data_args.template=${TEMPLATE}" \
  "validation.data_args.file_name=${VALIDATION_FILE}" \
  "++actor_train.strategy_args.strategy_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL}" \
  "++reference.strategy_args.strategy_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL}" \
  actor_infer.strategy_args.strategy_name=vllm \
  "actor_infer.strategy_args.strategy_config.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=true \
  +update_validation.log_update_norm=false \
  "update_validation.batch_size=${UV_BATCH_SIZE}" \
  "update_validation.num_return_sequences=${NUM_RETURN_SEQUENCES}" \
  "update_validation.generate_mode=${UV_GENERATE_MODE}" \
  "update_validation.generate_concurrency=${UV_GENERATE_CONCURRENCY}" \
  "update_validation.min_steps_for_threshold=${UV_MIN_STEPS}" \
  update_validation.acceptance_advantage_threshold_mode=fixed \
  "update_validation.acceptance_advantage_threshold=${UV_THRESHOLD}" \
  "$@"

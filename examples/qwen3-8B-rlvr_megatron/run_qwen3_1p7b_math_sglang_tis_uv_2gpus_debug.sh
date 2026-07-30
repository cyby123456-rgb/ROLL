#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVIDIA_SITE_PACKAGES="${CONDA_PREFIX:-/root/.conda/envs/sglang}/lib/python3.10/site-packages/nvidia"
NVIDIA_LD_LIBRARY_PATH="${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${NVIDIA_SITE_PACKAGES}/nccl/lib:${NVIDIA_SITE_PACKAGES}/nvtx/lib:${NVIDIA_SITE_PACKAGES}/cufft/lib:${NVIDIA_SITE_PACKAGES}/curand/lib:${NVIDIA_SITE_PACKAGES}/cusolver/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/cusparselt/lib:${NVIDIA_SITE_PACKAGES}/nvjitlink/lib"
CUDA_LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64:${CUDA_HOME:-/usr/local/cuda}/lib"
export LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}:${NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-sglang-tis-uv-2g-debug-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/Qwen3-1.7B}"
TEMPLATE="${TEMPLATE:-qwen3}"
UV_THRESHOLD="${UV_THRESHOLD:-1000000000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-transformer_engine}"
VALIDATION_FILE="${VALIDATION_FILE:-/mnt/shared-storage-user/liujinyi/lj/ROLL/data/test.jsonl}"
EVAL_STEPS="${EVAL_STEPS:-9999}"
SAVE_STEPS="${SAVE_STEPS:-9999}"
GENERATE_OPT_LEVEL="${GENERATE_OPT_LEVEL:-0}"
MAX_STEPS="${MAX_STEPS:-3}"
UPDATE_VALIDATION_ENABLE="${UPDATE_VALIDATION_ENABLE:-true}"
UPDATE_VALIDATION_LOG_UPDATE_NORM="${UPDATE_VALIDATION_LOG_UPDATE_NORM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VALIDATION_FILE}" ]]; then
  echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] ERROR: VALIDATION_FILE not found: ${VALIDATION_FILE}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] exp_name=${EXP_NAME}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] exp_root=${EXP_ROOT}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] model_path=${MODEL_PATH}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] validation_file=${VALIDATION_FILE}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] actor_infer=sglang, mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}, tp_size=${SGLANG_TP_SIZE}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] megatron_transformer_impl=${MEGATRON_TRANSFORMER_IMPL}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] eval_steps=${EVAL_STEPS}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] save_steps=${SAVE_STEPS}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] generate_opt_level=${GENERATE_OPT_LEVEL}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] update_validation_enable=${UPDATE_VALIDATION_ENABLE}, log_update_norm=${UPDATE_VALIDATION_LOG_UPDATE_NORM}"
echo "[qwen3-1p7b-math-sglang-tis-uv-2g-debug] debug: max_steps=${MAX_STEPS} rollout_batch_size=2 n=2 response_length=2048 ga=1 force_rollback_threshold=${UV_THRESHOLD}"

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline_2gpus.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  "+system_envs.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  max_steps=${MAX_STEPS} \
  save_steps=${SAVE_STEPS} \
  eval_steps=${EVAL_STEPS} \
  "+generate_opt_level=${GENERATE_OPT_LEVEL}" \
  rollout_batch_size=2 \
  num_return_sequences_in_group=2 \
  response_length=2048 \
  update_validation.batch_size=2 \
  update_validation.num_return_sequences=2 \
  actor_train.training_args.gradient_accumulation_steps=1 \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  "validation.data_args.file_name=${VALIDATION_FILE}" \
  actor_infer.strategy_args.strategy_name=sglang \
  '~actor_infer.strategy_args.strategy_config.gpu_memory_utilization' \
  '~actor_infer.strategy_args.strategy_config.block_size' \
  "++actor_infer.strategy_args.strategy_config.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" \
  "++actor_infer.strategy_args.strategy_config.load_format=dummy" \
  "++actor_infer.strategy_args.strategy_config.tensor_parallel_size=${SGLANG_TP_SIZE}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=${UPDATE_VALIDATION_ENABLE} \
  "+update_validation.log_update_norm=${UPDATE_VALIDATION_LOG_UPDATE_NORM}" \
  update_validation.acceptance_advantage_threshold_mode=fixed \
  update_validation.min_steps_for_threshold=0 \
  "update_validation.acceptance_advantage_threshold=${UV_THRESHOLD}" \
  "$@"

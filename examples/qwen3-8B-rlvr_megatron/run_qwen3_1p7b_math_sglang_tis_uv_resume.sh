#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVIDIA_SITE_PACKAGES="/root/.conda/envs/sglang-py310/lib/python3.10/site-packages/nvidia"
NVIDIA_LD_LIBRARY_PATH="${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${NVIDIA_SITE_PACKAGES}/nccl/lib:${NVIDIA_SITE_PACKAGES}/nvtx/lib:${NVIDIA_SITE_PACKAGES}/cufft/lib:${NVIDIA_SITE_PACKAGES}/curand/lib:${NVIDIA_SITE_PACKAGES}/cusolver/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/cusparselt/lib:${NVIDIA_SITE_PACKAGES}/nvjitlink/lib"
export LD_LIBRARY_PATH="${NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-sglang-tis-uv-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/Qwen3-1.7B}"
TEMPLATE="${TEMPLATE:-qwen3}"
UV_THRESHOLD="${UV_THRESHOLD:--0.0005}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-1p7b-sglang-uv] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -d "${EXP_ROOT}" ]]; then
    echo "[qwen3-1p7b-sglang-uv] ERROR: EXP_ROOT not found and RESUME_CHECKPOINT is not set: ${EXP_ROOT}" >&2
    exit 1
  fi
  shopt -s nullglob
  checkpoints=("${EXP_ROOT}"/checkpoints/*/checkpoint-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    echo "[qwen3-1p7b-sglang-uv] ERROR: no checkpoint found under ${EXP_ROOT}/checkpoints" >&2
    exit 1
  fi
  IFS=$'\n' read -r -d '' -a checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort -V && printf '\0')
  RESUME_CHECKPOINT="${checkpoints[-1]}"
fi

if [[ ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[qwen3-1p7b-sglang-uv] ERROR: RESUME_CHECKPOINT missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[qwen3-1p7b-sglang-uv] exp_name=${EXP_NAME}"
echo "[qwen3-1p7b-sglang-uv] exp_root=${EXP_ROOT}"
echo "[qwen3-1p7b-sglang-uv] model_path=${MODEL_PATH}"
echo "[qwen3-1p7b-sglang-uv] resume_from_checkpoint=${RESUME_CHECKPOINT}"
echo "[qwen3-1p7b-sglang-uv] actor_infer=sglang, mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}, tp_size=${SGLANG_TP_SIZE}, TIS=true, UV=true, threshold=${UV_THRESHOLD}"
echo "[qwen3-1p7b-sglang-uv] launching RLVR pipeline..."

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  "+system_envs.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  "resume_from_checkpoint=${RESUME_CHECKPOINT}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=true \
  update_validation.acceptance_advantage_threshold_mode=fixed \
  "update_validation.acceptance_advantage_threshold=${UV_THRESHOLD}" \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  actor_infer.strategy_args.strategy_name=sglang \
  '~actor_infer.strategy_args.strategy_config.gpu_memory_utilization' \
  '~actor_infer.strategy_args.strategy_config.block_size' \
  "++actor_infer.strategy_args.strategy_config.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" \
  "++actor_infer.strategy_args.strategy_config.load_format=dummy" \
  "++actor_infer.strategy_args.strategy_config.tensor_parallel_size=${SGLANG_TP_SIZE}" \
  "$@"

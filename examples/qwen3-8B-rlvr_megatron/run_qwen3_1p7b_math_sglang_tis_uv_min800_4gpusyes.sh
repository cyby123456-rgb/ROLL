#!/bin/bash
set -euo pipefail
# set -x
# eval "$(/root/miniconda3/bin/conda shell.bash hook)"

# export PATH="/root/miniconda3/bin:$PATH"
# eval "$(conda shell.bash hook)"

cd /inspire/hdd/project/traffic-congestion-management/zhangqiaosheng-24045/dist/ROLL
# conda activate sglang

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVIDIA_SITE_PACKAGES="${CONDA_PREFIX:-/root/.conda/envs/sglang}/lib/python3.10/site-packages/nvidia"
NVIDIA_LD_LIBRARY_PATH="${NVIDIA_SITE_PACKAGES}/cublas/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${NVIDIA_SITE_PACKAGES}/nccl/lib:${NVIDIA_SITE_PACKAGES}/nvtx/lib:${NVIDIA_SITE_PACKAGES}/cufft/lib:${NVIDIA_SITE_PACKAGES}/curand/lib:${NVIDIA_SITE_PACKAGES}/cusolver/lib:${NVIDIA_SITE_PACKAGES}/cusparse/lib:${NVIDIA_SITE_PACKAGES}/cusparselt/lib:${NVIDIA_SITE_PACKAGES}/nvjitlink/lib"
CUDA_LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64:${CUDA_HOME:-/usr/local/cuda}/lib"
export LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}:${NVIDIA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

EXP_NAME="${EXP_NAME:-qwen3-1p7b-math-sglang-tis-uv-min800-4g-s42}"
EXP_ROOT="${EXP_ROOT:-/inspire/hdd/project/traffic-congestion-management/zhangqiaosheng-24045/dist/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/traffic-congestion-management/zhangqiaosheng-24045/dist/models/Qwen3-1.7B}"
TEMPLATE="${TEMPLATE:-qwen2_5}"
MAX_STEPS="${MAX_STEPS:-800}"
UV_MIN_STEPS="${UV_MIN_STEPS:-800}"
UV_THRESHOLD="${UV_THRESHOLD:--0.00001}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-fp8}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-transformer_engine}"
GENERATE_OPT_LEVEL="${GENERATE_OPT_LEVEL:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb}

echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] exp_name=${EXP_NAME}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] exp_root=${EXP_ROOT}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] model_path=${MODEL_PATH}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] actor_infer=sglang, mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}, tp_size=${SGLANG_TP_SIZE}, quantization=${SGLANG_QUANTIZATION}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] transformer_impl=${MEGATRON_TRANSFORMER_IMPL}, generate_opt_level=${GENERATE_OPT_LEVEL}, max_steps=${MAX_STEPS}"
echo "[qwen3-1p7b-sglang-tis-uv-min800-4g] TIS=true, UV=true, min_steps_for_threshold=${UV_MIN_STEPS}, threshold=${UV_THRESHOLD}"

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${EXP_NAME}" \
  "pretrain=${MODEL_PATH}" \
  "reward_pretrain=${MODEL_PATH}" \
  "max_steps=${MAX_STEPS}" \
  "output_dir=${EXP_ROOT}/output" \
  "logging_dir=${EXP_ROOT}/logs" \
  "checkpoint_config.output_dir=${EXP_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${EXP_ROOT}/wandb" \
  "+tracker_kwargs.name=${EXP_NAME}" \
  "+system_envs.LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"" \
  "+generate_opt_level=${GENERATE_OPT_LEVEL}" \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  rewards.math_rule.data_args.template=qwen3 \
  "validation.data_args.template=${TEMPLATE}" \
  "++actor_train.strategy_args.strategy_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL}" \
  "++reference.strategy_args.strategy_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL}" \
  actor_infer.strategy_args.strategy_name=sglang \
  '~actor_infer.strategy_args.strategy_config.gpu_memory_utilization' \
  '~actor_infer.strategy_args.strategy_config.block_size' \
  "++actor_infer.strategy_args.strategy_config.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" \
  "++actor_infer.strategy_args.strategy_config.load_format=dummy" \
  "++actor_infer.strategy_args.strategy_config.tensor_parallel_size=${SGLANG_TP_SIZE}" \
  "++actor_infer.strategy_args.strategy_config.quantization=${SGLANG_QUANTIZATION}" \
  train_infer_ratio_trunc_is=true \
  update_validation.enable=false \
  "update_validation.min_steps_for_threshold=${UV_MIN_STEPS}" \
  update_validation.acceptance_advantage_threshold_mode=fixed \
  "update_validation.acceptance_advantage_threshold=${UV_THRESHOLD}" \
  "$@"

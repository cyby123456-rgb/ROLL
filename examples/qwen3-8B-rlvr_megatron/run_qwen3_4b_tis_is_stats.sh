#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-qwen3-4b-tis-is-stats-ub2-s42}"
EXP_ROOT="${EXP_ROOT:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs/${EXP_NAME}}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/liujinyi/models_hf/Qwen3-4B}"
TEMPLATE="${TEMPLATE:-qwen3}"
MAX_STEPS="${MAX_STEPS:-800}"
TIS_UPPER_BOUND="${TIS_UPPER_BOUND:-2.0}"
RATIO_DUMP_DIR="${RATIO_DUMP_DIR:-${EXP_ROOT}/tis_is_dumps}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[qwen3-4b-tis-is] ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}"/{output,logs,checkpoints,wandb,tis_is_dumps}

echo "[qwen3-4b-tis-is] exp_name=${EXP_NAME}"
echo "[qwen3-4b-tis-is] exp_root=${EXP_ROOT}"
echo "[qwen3-4b-tis-is] model_path=${MODEL_PATH}"
echo "[qwen3-4b-tis-is] ratio_dump_dir=${RATIO_DUMP_DIR}"
echo "[qwen3-4b-tis-is] TIS upper_bound=${TIS_UPPER_BOUND}, max_steps=${MAX_STEPS}"

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
  "ratio_rollout_dump_dir=${RATIO_DUMP_DIR}" \
  "ratio_dump_include_token_ratios=true" \
  "train_infer_ratio_trunc_is=true" \
  "train_infer_ratio_trunc_is_low=0" \
  "train_infer_ratio_trunc_is_high=${TIS_UPPER_BOUND}" \
  "train_infer_ratio_trunc_is_normalize=false" \
  "update_validation.enable=false" \
  "actor_train.data_args.template=${TEMPLATE}" \
  "actor_infer.data_args.template=${TEMPLATE}" \
  "reference.data_args.template=${TEMPLATE}" \
  "rewards.math_rule.data_args.template=${TEMPLATE}" \
  "validation.data_args.template=${TEMPLATE}" \
  "+actor_infer.strategy_args.strategy_config.quantization=int8" \
  "$@"

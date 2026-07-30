#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEPPO_EXP_NAME="${GEPPO_EXP_NAME:-qwen3-1p7b-geppo-code-sglang-baseline-s42}"
GEPPO_OUTPUT_ROOT="${GEPPO_OUTPUT_ROOT:-${SCRIPT_DIR}/../../output/${GEPPO_EXP_NAME}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"

if [[ -n "${RESUME_CHECKPOINT:-}" && ! -d "${RESUME_CHECKPOINT}/pipeline" ]]; then
  echo "[geppocode-sglang-baseline] ERROR: RESUME_CHECKPOINT missing pipeline state: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args+=("resume_from_checkpoint=${RESUME_CHECKPOINT}")
fi

exec bash "${SCRIPT_DIR}/run_rlvr_pipeline.sh" \
  "exp_name=${GEPPO_EXP_NAME}" \
  "output_dir=${GEPPO_OUTPUT_ROOT}" \
  "logging_dir=${GEPPO_OUTPUT_ROOT}/logs" \
  "checkpoint_config.output_dir=${GEPPO_OUTPUT_ROOT}/checkpoints" \
  "tracker_kwargs.log_dir=${GEPPO_OUTPUT_ROOT}" \
  "+tracker_kwargs.name=${GEPPO_EXP_NAME}" \
  "${resume_args[@]}" \
  prompt_length=1600 \
  actor_train.data_args.file_name=/mnt/shared-storage-user/liujinyi/lj/geppocode/data/skywork-code/train/train_1p5b_code.pkl \
  actor_train.data_args.dataset_dir=. \
  train_infer_ratio_trunc_is=false \
  update_validation.enable=false \
  +actor_train.data_args.dataset_type=geppo_code \
  actor_train.data_args.messages=messages \
  '~actor_train.data_args.domain_interleave_probs.math_rule' \
  +actor_train.data_args.domain_interleave_probs.code_sandbox=1.0 \
  validation.data_args.file_name=/mnt/shared-storage-user/liujinyi/lj/geppocode/livecodebench/livecodebench_2408_2502.parquet \
  +validation.data_args.dataset_type=geppo_code \
  +validation.data_args.livecodebench_data_dir=/mnt/shared-storage-user/liujinyi/lj/geppocode/livecodebench/livecodebench_2408_2502 \
  +validation.data_args.messages=messages \
  actor_infer.strategy_args.strategy_name=sglang \
  '~actor_infer.strategy_args.strategy_config.gpu_memory_utilization' \
  '~actor_infer.strategy_args.strategy_config.block_size' \
  "++actor_infer.strategy_args.strategy_config.mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" \
  "++actor_infer.strategy_args.strategy_config.load_format=dummy" \
  "++actor_infer.strategy_args.strategy_config.tensor_parallel_size=${SGLANG_TP_SIZE}" \
  '~rewards.math_rule' \
  '+rewards.code_sandbox={use_local: true, worker_cls: roll.pipeline.rlvr.rewards.geppo_code_reward_worker.GeppoCodeRewardWorker, livecodebench_data_dir: /mnt/shared-storage-user/liujinyi/lj/geppocode/livecodebench/livecodebench_2408_2502, code_reward_timeout: 6, code_reward_binary: true, code_reward_power4: false, tag_included: [geppo_code], model_args: {model_name_or_path: ${reward_pretrain}}, data_args: {template: qwen2_5}, world_size: 4, infer_batch_size: 1}' \
  "$@"

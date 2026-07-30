#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  ROLL_NVIDIA_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
  if [[ -d "${ROLL_NVIDIA_ROOT}" ]]; then
    ROLL_NVIDIA_LIBS="$(find "${ROLL_NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
    if [[ -n "${ROLL_NVIDIA_LIBS}" ]]; then
      export LD_LIBRARY_PATH="${ROLL_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  fi
fi

# Restrict the local run to two GPUs unless the caller already selected a device set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

"${PYTHON_BIN}" examples/start_rlvr_pipeline.py \
  --config_path qwen3-8B-rlvr_megatron \
  --config_name rlvr_config \
  num_gpus_per_node=2 \
  save_steps=700 \
  'actor_train.device_mapping="list(range(0,2))"' \
  'actor_infer.device_mapping="list(range(0,2))"' \
  'reference.device_mapping="list(range(0,2))"' \
  rewards.math_rule.world_size=2 \
  "$@"

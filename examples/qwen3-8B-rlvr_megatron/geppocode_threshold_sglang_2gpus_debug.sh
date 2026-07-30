#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[geppocode_threshold_sglang_2gpus_debug.sh] renamed behavior: math debug uses ${SCRIPT_DIR}/run_qwen3_1p7b_math_sglang_tis_uv_2gpus_debug.sh" >&2
exec bash "${SCRIPT_DIR}/run_qwen3_1p7b_math_sglang_tis_uv_2gpus_debug.sh" "$@"

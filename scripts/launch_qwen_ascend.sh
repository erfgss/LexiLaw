#!/usr/bin/env bash
set -euo pipefail

NPU_COUNT="${NPU_COUNT:-8}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/sft/current_law_sft.json}"
OUTPUT="${OUTPUT:-outputs/qwen3.6-legal-ascend-lora}"

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
fi

python scripts/check_ascend.py
torchrun --nproc_per_node="${NPU_COUNT}" scripts/train_qwen_ascend.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --deepspeed ds_config_ascend_zero3.json

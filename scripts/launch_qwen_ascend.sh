#!/usr/bin/env bash
set -euo pipefail

NPU_COUNT="${NPU_COUNT:-2}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
DATA="${DATA:-data/sft/current_law_sft.generated.json}"
OUTPUT="${OUTPUT:-outputs/qwen3.6-legal-ascend-lora}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EPOCHS="${EPOCHS:-3.0}"
QLORA="${QLORA:-1}"
QLORA_ARGS=()
if [[ "${QLORA}" == "1" ]]; then
  QLORA_ARGS+=(--qlora)
fi

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
fi

python scripts/check_ascend.py
torchrun --nproc_per_node="${NPU_COUNT}" scripts/train_qwen_ascend.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --epochs "${EPOCHS}" \
  "${QLORA_ARGS[@]}" \
  --deepspeed ds_config_ascend_zero3.json

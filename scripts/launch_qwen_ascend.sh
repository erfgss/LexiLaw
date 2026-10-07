#!/usr/bin/env bash
set -euo pipefail

NPU_COUNT="${NPU_COUNT:-2}"
MODEL="${MODEL:-/workspace/Qwen3-8B}"
DATA="${DATA:-data/sft/current_law_sft.train.json}"
EVAL_DATA="${EVAL_DATA:-data/sft/current_law_sft.validation.json}"
OUTPUT="${OUTPUT:-outputs/qwen3-8b-legal-ascend-lora-fast}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EPOCHS="${EPOCHS:-3.0}"
QLORA="${QLORA:-0}"
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
  --eval-data "${EVAL_DATA}" \
  --output "${OUTPUT}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --epochs "${EPOCHS}" \
  "${QLORA_ARGS[@]}" \
  --deepspeed ds_config_ascend_zero3.json

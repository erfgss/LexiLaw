"""LoRA/QLoRA SFT entry point for Ascend/torch_npu.

Launch with scripts/launch_qwen_ascend.sh so each rank is assigned one NPU.
The model is intentionally not loaded with device_map='auto': DeepSpeed ZeRO-3
must own parameter placement and sharding in a multi-NPU job.
"""
import argparse
import os

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig
from trl import SFTConfig, SFTTrainer


def format_row(row, tokenizer):
    messages = []
    for turn in row["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/Qwen3-8B")
    parser.add_argument("--data", default="data/sft/current_law_sft.json")
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output", default="outputs/qwen3-8b-legal-ascend-lora-fast")
    parser.add_argument("--deepspeed", default="ds_config_ascend_zero3.json")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--qlora", action="store_true", help="Load an INT4 pre-quantized base and train only LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable; check CANN, torch and torch_npu.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda row: {"text": format_row(row, tokenizer)})
    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")
        eval_dataset = eval_dataset.map(lambda row: {"text": format_row(row, tokenizer)})

    # Register ZeRO-3 before loading so Transformers can shard weights during
    # from_pretrained instead of moving the full model to every NPU.
    ds_hf_config = HfDeepSpeedConfig(args.deepspeed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    if args.qlora:
        # The model directory must contain an Ascend-compatible INT4 loader.
        # Do not silently enable CUDA bitsandbytes on torch_npu.
        is_quantized = bool(
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "quantization_config", None) is not None
        )
        if not is_quantized:
            raise RuntimeError(
                "--qlora requires an Ascend-compatible pre-quantized model "
                "directory; do not use CUDA bitsandbytes on torch_npu."
            )
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=args.output,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        packing=False,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=True,
        tf32=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()

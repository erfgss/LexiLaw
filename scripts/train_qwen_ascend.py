"""BF16 LoRA SFT entry point for Ascend 910B/torch_npu.

Launch with scripts/launch_qwen_ascend.sh so each rank is assigned one NPU.
The model is intentionally not loaded with device_map='auto': DeepSpeed ZeRO-3
must own parameter placement and sharding in a multi-NPU job.
"""
import argparse
import os

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def format_row(row, tokenizer):
    messages = []
    for turn in row["conversations"]:
        role = "user" if turn["from"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="data/sft/current_law_sft.json")
    parser.add_argument("--output", default="outputs/qwen3.6-legal-ascend-lora")
    parser.add_argument("--deepspeed", default="ds_config_ascend_zero3.json")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable; check CANN, torch and torch_npu.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda row: {"text": format_row(row, tokenizer)})

    # Do not set device_map here. DeepSpeed places/shards the model per rank.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        tf32=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()

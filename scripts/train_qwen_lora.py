"""Minimal Qwen LoRA SFT entry point for the versioned legal pilot dataset."""
import argparse
import json

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
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", default="data/sft/current_law_sft.json")
    parser.add_argument("--output", default="outputs/qwen3-8b-legal-lora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda row: {"text": format_row(row, tokenizer)})
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
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
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=TrainingArguments(
            output_dir=args.output,
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="epoch",
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()

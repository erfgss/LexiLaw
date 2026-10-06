"""Inference for the version-aware Qwen legal adapter."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--adapter", default="outputs/qwen3.6-legal-lora")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    system = (
        "你是法律条文版本核对助手。优先依据提供的法律名称、版本、生效日期和检索证据回答。"
        "不能仅凭条号推断历史条文与现行条文对应；证据不足时明确说明无法确认。"
    )
    print("输入问题，输入 exit 退出。")
    while True:
        question = input("用户> ").strip()
        if question.lower() == "exit":
            break
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        answer = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"Qwen> {answer.strip()}")


if __name__ == "__main__":
    main()

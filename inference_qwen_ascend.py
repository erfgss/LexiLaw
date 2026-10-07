"""Single-NPU smoke inference for a Qwen LoRA adapter on Ascend.

The 35B checkpoint generally needs multi-NPU sharding for production inference;
this script is intended for a sharded/quantized local checkpoint or a smoke test.
"""
import argparse

import torch
import torch_npu  # noqa: F401
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/Qwen3-8B")
    parser.add_argument("--adapter", default="outputs/qwen3-8b-legal-ascend-qlora")
    parser.add_argument("--question", default="《劳动合同法》第三十七条现行规定是什么？")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    messages = [
        {"role": "system", "content": "你是法律条款版本核对助手，证据不足时不要猜测。"},
        {"role": "user", "content": args.question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()

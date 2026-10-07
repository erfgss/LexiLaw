"""Compare a Qwen base model or LoRA adapter on held-out SFT examples."""

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def final_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/Qwen3-8B")
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data/sft/current_law_sft.validation.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))[: args.limit]
    results = []
    for index, row in enumerate(rows, 1):
        user = next(turn["value"] for turn in row["conversations"] if turn["from"] == "user")
        reference = next(
            turn["value"] for turn in row["conversations"] if turn["from"] == "assistant"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        prediction = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        prediction = final_answer(prediction)
        normalized_prediction = normalize(prediction)
        normalized_reference = normalize(reference)
        similarity = SequenceMatcher(
            None, normalized_prediction, normalized_reference, autojunk=False
        ).ratio()
        exact_match = normalized_prediction == normalized_reference
        reference_contained = normalized_reference in normalized_prediction
        results.append(
            {
                "id": row.get("id", str(index)),
                "question": user,
                "reference": reference,
                "prediction": prediction,
                "similarity": similarity,
                "exact_match": exact_match,
                "reference_contained": reference_contained,
            }
        )
        print(
            f"[{index}/{len(rows)}] similarity={similarity:.4f} "
            f"exact={int(exact_match)} contained={int(reference_contained)}",
            flush=True,
        )

    count = len(results)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "examples": count,
        "mean_similarity": sum(x["similarity"] for x in results) / count,
        "exact_match_rate": sum(x["exact_match"] for x in results) / count,
        "reference_containment_rate": sum(x["reference_contained"] for x in results)
        / count,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Create a deterministic train/validation split for a JSON SFT dataset."""

import argparse
import json
import math
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sft/current_law_sft.generated.json")
    parser.add_argument("--train-output", default="data/sft/current_law_sft.train.json")
    parser.add_argument("--validation-output", default="data/sft/current_law_sft.validation.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")

    input_path = Path(args.input)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("input must be a JSON list containing at least two rows")

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    validation_size = math.ceil(len(shuffled) * args.validation_ratio)
    validation_rows = shuffled[:validation_size]
    train_rows = shuffled[validation_size:]

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path.write_text(
        json.dumps(validation_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"total={len(rows)} train={len(train_rows)} validation={len(validation_rows)} "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()

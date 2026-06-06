#!/usr/bin/env python3
"""Download benchmark datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "humaneval", "refactor"], default="gsm8k")
    parser.add_argument("--output", default="./data")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test[:50]")
        records = [{"question": r["question"], "answer": r["answer"]} for r in ds]
        (out / "gsm8k_sample.json").write_text(json.dumps(records, indent=2))
        print(f"Wrote {len(records)} GSM8K samples")
    elif args.dataset == "refactor":
        from jax_ssd.benchmarks.code_refactor import REFACTOR_TASKS

        (out / "refactor_tasks.json").write_text(json.dumps(REFACTOR_TASKS, indent=2))
        print(f"Wrote {len(REFACTOR_TASKS)} refactor tasks")
    else:
        print("HumanEval: use huggingface datasets openai/openai_humaneval when on TPU VM")


if __name__ == "__main__":
    main()

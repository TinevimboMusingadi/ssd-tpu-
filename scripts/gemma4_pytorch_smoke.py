#!/usr/bin/env python3
"""Gemma 4 E2B text smoke (CPU PyTorch) — works on 10GB TPU VM boot disk."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="What is gravity in one sentence?")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--model",
        default="google/gemma-4-E2B-it",
        help="Full E2B (~10GB); needs 50GB+ boot disk on TPU VM",
    )
    args = parser.parse_args()

    token = (os.getenv("HF_TOKEN") or "").strip() or None
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}\n")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print("Install: pip install torch transformers accelerate", file=sys.stderr)
        raise SystemExit(1) from exc

    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
    else:
        inputs = tok(args.prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

    new_ids = out[0, inputs["input_ids"].shape[-1] :]
    print(tok.decode(new_ids, skip_special_tokens=True))
    print("\nOK — Gemma 4 loads and generates (CPU). For TPU JAX use EasyDeL + Python 3.11+.")


if __name__ == "__main__":
    main()

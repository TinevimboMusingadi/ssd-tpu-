"""Compare AR, SD, SSD, and Instance modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jax_ssd.config import DecodeMode
from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams

DEFAULT_PROMPTS = [
    "What is speculative speculative decoding?",
    "Write a Python function to compute fibonacci.",
    "Explain TPU mesh allocation.",
]


def load_prompts(num: int) -> list[str]:
    prompts = DEFAULT_PROMPTS * ((num // len(DEFAULT_PROMPTS)) + 1)
    return prompts[:num]


def run_mode(mode: str, prompts: list[str], max_tokens: int) -> dict:
    llm = LLM.from_mode(mode)
    sampling = SamplingParams(max_new_tokens=max_tokens)
    outputs, metrics = llm.generate(prompts, sampling)
    llm.shutdown()
    return {
        "mode": mode,
        "metrics": metrics.to_dict(),
        "sample_output_len": len(outputs[0]) if outputs else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SSD-TPU benchmark harness")
    parser.add_argument("--mode", default="all", choices=["ar", "sd", "ssd", "instance", "all"])
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text (overrides defaults)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
        if args.num_prompts > 1:
            prompts = prompts * args.num_prompts
    else:
        prompts = load_prompts(args.num_prompts)
    modes = ["ar", "sd", "ssd", "instance"] if args.mode == "all" else [args.mode]

    results = []
    baseline_tps = None
    for mode in modes:
        r = run_mode(mode, prompts, args.max_tokens)
        tps = r["metrics"]["tokens_per_second"]
        if mode == "ar":
            baseline_tps = tps
        if baseline_tps and baseline_tps > 0:
            r["speedup_vs_ar"] = tps / baseline_tps
        results.append(r)

    report = {"prompts": args.num_prompts, "results": results}
    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        md = _to_markdown(report)
        Path(args.output).with_suffix(".md").write_text(md, encoding="utf-8")


def _to_markdown(report: dict) -> str:
    lines = ["# SSD-TPU Benchmark Report\n", f"Prompts: {report['prompts']}\n", "| Mode | tok/s | TTFT | cache hit | speedup vs AR |", "|------|-------|------|-----------|---------------|"]
    for r in report["results"]:
        m = r["metrics"]
        sup = r.get("speedup_vs_ar", 1.0)
        lines.append(
            f"| {r['mode']} | {m['tokens_per_second']:.1f} | {m['ttft_s']:.3f}s | {m['cache_hit_rate']:.1%} | {sup:.2f}x |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

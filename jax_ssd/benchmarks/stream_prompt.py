"""Stream decoded tokens for one prompt (no full TUI). Good for SSH."""

from __future__ import annotations

import argparse
import sys

from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams

PHYSICS_EXAMPLES = [
    "Explain Newton's second law F equals m a in plain English.",
    "What is the difference between speed and velocity?",
    "Why does light bend when it passes near a massive object?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream tokens for one prompt")
    parser.add_argument("--prompt", default=PHYSICS_EXAMPLES[0])
    parser.add_argument("--mode", default="ar", choices=["ar", "sd", "ssd", "instance"])
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    print(f"=== mode: {args.mode} ===", flush=True)
    print(f"=== prompt: {args.prompt} ===\n", flush=True)

    llm = LLM.from_mode(args.mode)

    def on_token(_tok: int, decoded: str) -> None:
        sys.stdout.write(decoded if decoded.endswith(" ") else decoded + " ")
        sys.stdout.flush()

    _, metrics = llm.generate(
        [args.prompt],
        SamplingParams(max_new_tokens=args.max_tokens),
        on_token=on_token,
    )
    llm.shutdown()

    print(f"\n\n--- {metrics.total_tokens} tokens, {metrics.tokens_per_second:.1f} tok/s ---", flush=True)


if __name__ == "__main__":
    main()

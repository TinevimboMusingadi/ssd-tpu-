"""QA smoke benchmark."""

from __future__ import annotations

from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams

QA_PROMPTS = [
    "What is the capital of France?",
    "Explain speculative decoding in one sentence.",
    "Summarize the benefits of JAX on TPU.",
]


def run_qa_smoke(mode: str = "ar", max_tokens: int = 32) -> dict:
    llm = LLM.from_mode(mode)
    prompts = [[ord(c) % 64 for c in p] for p in QA_PROMPTS]
    sampling = SamplingParams(max_new_tokens=max_tokens)
    outputs, metrics = llm.generate(prompts, sampling)
    llm.shutdown()
    return {"mode": mode, "metrics": metrics.to_dict(), "num_prompts": len(prompts)}


if __name__ == "__main__":
    import json

    print(json.dumps(run_qa_smoke("ssd"), indent=2))

"""Code refactoring benchmark for Instance-SSD."""

from __future__ import annotations

from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams

REFACTOR_TASKS = [
    {
        "name": "rename_variable",
        "prompt": "Refactor: rename getUserById to fetchUserById. def getUserById(user_id): return db.get(user_id)",
    },
    {
        "name": "extract_helper",
        "prompt": "Extract validation into a helper. def process(x): if x < 0: raise ValueError; return x * 2",
    },
    {
        "name": "add_guard",
        "prompt": "Add guard clause. def divide(a, b): return a / b",
    },
]


def run_code_refactor(mode: str = "instance", max_tokens: int = 48) -> dict:
    llm = LLM.from_mode(mode)
    sampling = SamplingParams(max_new_tokens=max_tokens)
    results = []
    for task in REFACTOR_TASKS:
        prompt = [ord(c) % 64 for c in task["prompt"]]
        outputs, metrics = llm.generate([prompt], sampling)
        results.append(
            {
                "task": task["name"],
                "tokens": len(outputs[0]),
                "tokens_per_second": metrics.tokens_per_second,
                "cache_hit_rate": metrics.cache_hit_rate,
            }
        )
    llm.shutdown()
    return {"mode": mode, "tasks": results}


if __name__ == "__main__":
    import json

    print(json.dumps(run_code_refactor(), indent=2))

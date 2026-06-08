#!/usr/bin/env python3
"""Load Gemma 4 E2B via EasyDeL 0.1.4.x + vInference on TPU."""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
from dotenv import load_dotenv
from transformers import AutoTokenizer

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
import easydel as ed


def main() -> None:
    load_dotenv()
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is gravity in one sentence?"
    max_new = int(os.getenv("GEMMA4_MAX_NEW", "16"))

    print("devices", jax.devices())
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=token)
    ids = tok.encode(prompt, add_special_tokens=True)
    print("prompt tokens", len(ids))

    fsdp = max(1, len(jax.devices()))
    print("loading eSurge EasyDeL/gemma-4-E2B-it (fsdp=%d)..." % fsdp, flush=True)
    if hasattr(ed, "eSurge"):
        dtype = jnp.bfloat16
        model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
            "EasyDeL/gemma-4-E2B-it",
            dtype=dtype,
            param_dtype=dtype,
            sharding_axis_dims=(1, 1, 1, fsdp, 1),
            sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
            auto_shard_model=True,
        )
        engine = ed.eSurge(
            model=model,
            processor=tok,
            max_model_len=2048,
            max_num_seqs=4,
            hbm_utilization=0.85,
            sharding_axis_dims=(1, 1, 1, fsdp, 1),
        )
        prompt = tok.decode(ids, skip_special_tokens=False)
        sampling = ed.SamplingParams(max_tokens=max_new, temperature=0.0, top_p=1.0)
        print("generating...", flush=True)
        outputs = engine.generate(prompt, sampling_params=sampling)
        out = outputs[0].outputs[0]
        text = getattr(out, "text", str(out))
        new_ids = tok.encode(text, add_special_tokens=False)
    else:
        dtype = jnp.bfloat16
        model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
            "EasyDeL/gemma-4-E2B-it",
            dtype=dtype,
            param_dtype=dtype,
            sharding_axis_dims=(1, fsdp, 1, 1),
            sharding_axis_names=("dp", "fsdp", "tp", "sp"),
            auto_shard_model=True,
        )
        vinf = ed.vInference(model=model, max_new_tokens=max_new)
        input_ids = jnp.array([ids], dtype=jnp.int32)
        sampling = ed.SamplingParams(max_tokens=max_new, temperature=0.0, top_p=1.0)
        print("generating...", flush=True)
        final = None
        for state in vinf.generate(input_ids, sampling_params=sampling):
            final = state
        if final is None:
            raise SystemExit("no generation state returned")
        sequences = getattr(final, "sequences", None) or getattr(final, "output_ids", None)
        if sequences is None:
            raise SystemExit("could not find sequences on final state")
        seq = jnp.asarray(sequences)[0]
        new_ids = [int(t) for t in seq[len(ids) :]]
        text = tok.decode(new_ids, skip_special_tokens=True)
    print("new token ids:", new_ids[:20])
    print("text:", text)
    print("OK")


if __name__ == "__main__":
    main()

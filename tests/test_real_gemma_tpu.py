"""Real Gemma integration tests on TPU (skipped unless SSD_USE_REAL_MODEL=1)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.tpu

requires_real = pytest.mark.skipif(
    os.getenv("SSD_USE_REAL_MODEL", "0").lower() not in ("1", "true", "yes"),
    reason="Set SSD_USE_REAL_MODEL=1 to run real Gemma tests",
)

requires_tpu = pytest.mark.skipif(
    os.getenv("JAX_PLATFORMS", "") != "tpu",
    reason="Real Gemma tests require JAX_PLATFORMS=tpu",
)


@requires_real
@requires_tpu
def test_ar_generates_real_tokens():
    from jax_ssd.config import DecodeMode
    from jax_ssd.llm import LLM
    from jax_ssd.sampling_params import SamplingParams

    llm = LLM.from_mode(DecodeMode.AR, use_toy_model=False)
    prompt = "What is two plus two?"
    outputs, metrics = llm.generate([prompt], SamplingParams(max_new_tokens=8))
    assert len(outputs[0]) > 0
    assert metrics.total_tokens > 0
    text = llm._engine.target.decode_tokens_to_str(outputs[0][:3])
    llm.shutdown()
    assert text.strip()


@requires_real
@requires_tpu
def test_sd_matches_ar_length_greedy():
    from jax_ssd.config import DecodeMode
    from jax_ssd.llm import LLM
    from jax_ssd.sampling_params import SamplingParams

    prompt = "Count to five."
    sampling = SamplingParams(max_new_tokens=12)

    ar = LLM.from_mode(DecodeMode.AR, use_toy_model=False)
    ar_out, _ = ar.generate([prompt], sampling)
    ar.shutdown()

    sd = LLM.from_mode(DecodeMode.SD, use_toy_model=False)
    sd_out, _ = sd.generate([prompt], sampling)
    sd.shutdown()

    assert len(ar_out[0]) == len(sd_out[0])

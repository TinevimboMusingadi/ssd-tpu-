"""Engine mode integration tests."""

from jax_ssd.config import DecodeMode
from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams


def test_ar_generates_tokens():
    llm = LLM.from_mode(DecodeMode.AR)
    prompt = [1, 2, 3, 4, 5]
    outputs, metrics = llm.generate([prompt], SamplingParams(max_new_tokens=8))
    llm.shutdown()
    assert len(outputs[0]) > 0
    assert metrics.total_tokens > 0


def test_sd_ssd_greedy_same_length_as_ar():
    prompt = [10, 20, 30]
    sampling = SamplingParams(max_new_tokens=12)
    ar = LLM.from_mode(DecodeMode.AR)
    ar_out, _ = ar.generate([prompt], sampling)
    ar.shutdown()

    sd = LLM.from_mode(DecodeMode.SD)
    sd_out, _ = sd.generate([prompt], sampling)
    sd.shutdown()

    ssd = LLM.from_mode(DecodeMode.SSD)
    ssd_out, m = ssd.generate([prompt], sampling)
    ssd.shutdown()

    assert len(ar_out[0]) == len(sd_out[0]) == len(ssd_out[0])

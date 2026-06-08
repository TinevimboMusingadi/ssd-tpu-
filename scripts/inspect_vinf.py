#!/usr/bin/env python3
import inspect

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
import easydel as ed

print("vInference init", inspect.signature(ed.vInference))
for name in ("generate", "stream", "__call__"):
    if hasattr(ed.vInference, name):
        print(name, inspect.signature(getattr(ed.vInference, name)))

if hasattr(ed, "SamplingParams"):
    print("SamplingParams", inspect.signature(ed.SamplingParams))
else:
    from easydel.inference.sampling_params import SamplingParams

    print("SamplingParams module", inspect.signature(SamplingParams))

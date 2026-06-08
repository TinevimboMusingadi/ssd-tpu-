#!/usr/bin/env python3
import inspect

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
import easydel as ed

names = [
    "AutoEasyDeLModelForCausalLM",
    "AutoEasyDeLModelForImageTextToText",
    "AutoEasyDeLModel",
]
for name in names:
    cls = getattr(ed, name, None)
    if cls is None:
        print(name, "MISSING")
        continue
    print(name, "from_pretrained", inspect.signature(cls.from_pretrained))

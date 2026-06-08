#!/usr/bin/env python3
import inspect

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
import easydel as ed

for name in ("vDriver", "oDriver", "vSurge", "vInference", "AutoEasyDeLModelForCausalLM"):
    obj = getattr(ed, name, None)
    if obj is None:
        print(name, "MISSING")
        continue
    print(name, "sig", inspect.signature(obj) if callable(obj) else type(obj))

try:
    from easydel.inference.vsurge import vSurge as VS
    from easydel.inference.vdriver import vDriver as VD

    print("module vSurge", inspect.signature(VS))
    print("module vDriver", inspect.signature(VD))
except Exception as exc:
    print("module import err", exc)

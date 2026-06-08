#!/usr/bin/env python3
import inspect

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
import easydel as ed

print("has eSurge", hasattr(ed, "eSurge"))
print("has vSurge", hasattr(ed, "vSurge"))
if hasattr(ed, "vSurge"):
    print("vSurge sig", inspect.signature(ed.vSurge))
if hasattr(ed, "eSurge"):
    print("eSurge sig", inspect.signature(ed.eSurge))

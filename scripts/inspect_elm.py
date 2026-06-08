#!/usr/bin/env python3
import inspect

from jax_ssd.compat.transformers_shim import apply_transformers_compat

apply_transformers_compat()
from easydel import eLargeModel

print("from_pretrained", inspect.signature(eLargeModel.from_pretrained))
elm = eLargeModel.from_pretrained("EasyDeL/gemma-4-E2B-it")
print("methods", [m for m in dir(elm) if "build" in m.lower() or "surge" in m.lower()])

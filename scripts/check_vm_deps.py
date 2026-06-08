#!/usr/bin/env python3
import easydel
import jax
import transformers

print("easydel", easydel.__version__)
print("transformers", transformers.__version__)
print("jax devices", jax.devices())

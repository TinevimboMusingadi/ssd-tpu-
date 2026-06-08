#!/usr/bin/env python3
import inspect

import easydel as ed

for name in ("eSurge", "vSurge", "vInference", "eLargeModel", "SamplingParams"):
    obj = getattr(ed, name, None)
    print(name, "FOUND" if obj else "MISSING", inspect.signature(obj) if callable(obj) else "")

#!/usr/bin/env python3
import transformers
import transformers.utils.generic as g
import transformers.utils.hub as h

print("transformers", transformers.__version__)
print("generic has working_or_temp_dir", hasattr(g, "working_or_temp_dir"))
print("hub attrs", [x for x in dir(h) if "temp" in x.lower() or "working" in x.lower()])

import transformers

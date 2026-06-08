#!/usr/bin/env python3
import os

from dotenv import load_dotenv

load_dotenv()
from jax_ssd.models.model_loader import load_model_adapter

use_toy = os.getenv("SSD_USE_TOY_MODEL", "").strip() in ("1", "true", "yes")
path = os.getenv("TARGET_MODEL_PATH", "")
print("SSD_USE_TOY_MODEL", os.getenv("SSD_USE_TOY_MODEL"))
print("TARGET_MODEL_PATH", path)
m = load_model_adapter(path, use_toy=use_toy, role="target")
print("adapter", type(m).__name__)
print("model_path", getattr(m, "model_path", None))

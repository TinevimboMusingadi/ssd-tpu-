"""Load target/draft model adapters from config paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from jax_ssd.models.base import DecodeModelAdapter
from jax_ssd.models.gemma_adapter import GemmaAdapter
from jax_ssd.models.toy_model import ToyModelAdapter

logger = logging.getLogger(__name__)


def default_target_path() -> str:
    return os.getenv("TARGET_MODEL_PATH", "./models/google_gemma-2b-it")


def default_draft_path() -> str:
    return os.getenv("DRAFT_MODEL_PATH", default_target_path())


def load_model_adapter(
    model_path: str | None,
    *,
    use_toy: bool = False,
    seed: int = 0,
    share_key: str | None = None,
) -> DecodeModelAdapter:
    if use_toy:
        logger.info("Using toy model adapter (SSD_USE_TOY_MODEL=1).")
        return ToyModelAdapter(seed=seed)

    path = model_path or default_target_path()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run: python scripts/download_models.py --preset gemma-2b "
            "(set HF_TOKEN in .env for gated Gemma weights)."
        )

    return GemmaAdapter(model_path=path, share_key=share_key)

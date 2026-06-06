"""Load target/draft model adapters from config paths."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from jax_ssd.models.base import DecodeModelAdapter
from jax_ssd.models.gemma_adapter import GemmaAdapter
from jax_ssd.models.toy_model import ToyModelAdapter

if TYPE_CHECKING:
    import jax
    from jax.sharding import Mesh

logger = logging.getLogger(__name__)


def default_target_path() -> str:
    return os.getenv(
        "TARGET_MODEL_PATH",
        "gs://your-project-ssd-tpu/models/google_gemma-7b-it",
    )


def default_draft_path() -> str:
    return os.getenv(
        "DRAFT_MODEL_PATH",
        "gs://your-project-ssd-tpu/models/google_gemma-2b-it",
    )


def load_model_adapter(
    model_path: str | None,
    *,
    use_toy: bool = False,
    seed: int = 0,
    share_key: str | None = None,
    mesh: Mesh | None = None,
    devices: tuple[jax.Device, ...] | None = None,
    role: str = "target",
) -> DecodeModelAdapter:
    if use_toy:
        logger.info("Using toy model adapter (%s, SSD_USE_TOY_MODEL=1).", role)
        return ToyModelAdapter(seed=seed)

    path = model_path or (default_target_path() if role == "target" else default_draft_path())

    try:
        return GemmaAdapter(
            model_path=path,
            share_key=share_key,
            mesh=mesh,
            devices=devices,
            role=role,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Run: python scripts/download_models.py --preset sd-pair-7b --gcs-uri $GCS_BUCKET"
        ) from exc

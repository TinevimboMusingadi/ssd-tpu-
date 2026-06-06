"""Qwen model adapter stub — Phase 2 via MaxText."""

from __future__ import annotations

import logging

from jax_ssd.models.gemma_adapter import GemmaAdapter

logger = logging.getLogger(__name__)


class QwenAdapter(GemmaAdapter):
    """Qwen3 adapter placeholder; delegates to toy/Gemma path until MaxText wired."""

    def __init__(self, model_path: str | None = None, **kwargs) -> None:
        super().__init__(model_path=model_path, vocab_size=151_936, **kwargs)
        logger.info(
            "QwenAdapter: MaxText integration is a Stage 9 item; using fallback backend."
        )

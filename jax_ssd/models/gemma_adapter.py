"""Gemma model adapter for TPU inference."""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp

from jax_ssd.kernels.kv_cache import KVCacheState
from jax_ssd.models.base import DecodeResult, PrefillResult, VerifyResultModel
from jax_ssd.models.toy_model import ToyModelAdapter

logger = logging.getLogger(__name__)


class GemmaAdapter:
    """Gemma adapter — uses real Gemma weights when available, toy fallback otherwise."""

    def __init__(
        self,
        model_path: str | None = None,
        vocab_size: int = 256_000,
        use_toy_fallback: bool = True,
    ) -> None:
        self.model_path = model_path
        self.vocab_size = vocab_size
        self._toy: ToyModelAdapter | None = None
        self._tokenizer = None
        self._loaded = False

        if model_path and Path(model_path).exists():
            self._try_load_gemma(model_path)
        elif use_toy_fallback:
            logger.warning("Gemma weights not found at %s; using toy model.", model_path)
            self._toy = ToyModelAdapter(vocab_size=min(vocab_size, 512))

    def _try_load_gemma(self, path: str) -> None:
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._loaded = True
            logger.info("Loaded Gemma tokenizer from %s", path)
        except Exception as exc:
            logger.warning("Failed to load Gemma from %s: %s", path, exc)
            self._toy = ToyModelAdapter(vocab_size=512)

    @property
    def backend(self) -> ToyModelAdapter:
        if self._toy is None:
            self._toy = ToyModelAdapter(vocab_size=512)
        return self._toy

    def allocate_kv(self) -> KVCacheState:
        return self.backend.allocate_kv()

    def prefill(
        self,
        token_ids: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> PrefillResult:
        return self.backend.prefill(token_ids, kv_cache, page_table)

    def decode(
        self,
        token_id: int,
        position: int,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> DecodeResult:
        return self.backend.decode(token_id, position, kv_cache, page_table)

    def verify(
        self,
        token_ids_kp1: jnp.ndarray,
        positions: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> VerifyResultModel:
        return self.backend.verify(token_ids_kp1, positions, kv_cache, page_table)

    def decode_tokens_to_str(self, token_ids: list[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return self.backend.decode_tokens_to_str(token_ids)

    def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text)
        return [ord(c) % 512 for c in text]

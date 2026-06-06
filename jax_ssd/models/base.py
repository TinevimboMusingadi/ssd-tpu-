"""Model adapter interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from jax_ssd.kernels.kv_cache import KVCacheState


@dataclass
class PrefillResult:
    logits: jax.Array
    kv_cache: KVCacheState
    hidden_states: jax.Array | None = None


@dataclass
class DecodeResult:
    logits: jax.Array
    kv_cache: KVCacheState
    next_token: int


@dataclass
class VerifyResultModel:
    logits: jax.Array  # [B, K+1, V]
    kv_cache: KVCacheState


class DecodeModelAdapter(Protocol):
    vocab_size: int

    def prefill(
        self,
        token_ids: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> PrefillResult: ...

    def decode(
        self,
        token_id: int,
        position: int,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> DecodeResult: ...

    def verify(
        self,
        token_ids_kp1: jnp.ndarray,
        positions: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> VerifyResultModel: ...

    def decode_tokens_to_str(self, token_ids: list[int]) -> str: ...

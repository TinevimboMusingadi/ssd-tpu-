"""Tiny decoder for algorithm tests without real weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_ssd.kernels.kv_cache import KVCacheState
from jax_ssd.models.base import DecodeResult, PrefillResult, VerifyResultModel


class ToyModelAdapter:
    """Deterministic toy LM: logits seeded by token ids for reproducible verify tests."""

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_blocks: int = 32,
        block_size: int = 16,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.standard_normal((hidden_dim, vocab_size)).astype(np.float32) * 0.1

    def _logits_from_tokens(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Produce logits [len, V] deterministically from token ids."""
        ids = np.array(token_ids, dtype=np.int32)
        h = np.zeros((len(ids), self.hidden_dim), dtype=np.float32)
        for i, t in enumerate(ids):
            h[i, t % self.hidden_dim] = 1.0
            h[i, (t * 7) % self.hidden_dim] = 0.5
        logits = h @ self.W
        return jnp.array(logits)

    def allocate_kv(self) -> KVCacheState:
        return KVCacheState.allocate(
            num_layers=self.num_layers,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=1,
            head_dim=self.hidden_dim,
        )

    def prefill(
        self,
        token_ids: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> PrefillResult:
        logits = self._logits_from_tokens(token_ids)
        return PrefillResult(logits=logits, kv_cache=kv_cache)

    def decode(
        self,
        token_id: int,
        position: int,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> DecodeResult:
        logits = self._logits_from_tokens(jnp.array([token_id]))
        next_token = int(jnp.argmax(logits[0]))
        return DecodeResult(logits=logits, kv_cache=kv_cache, next_token=next_token)

    def verify(
        self,
        token_ids_kp1: jnp.ndarray,
        positions: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> VerifyResultModel:
        logits = self._logits_from_tokens(token_ids_kp1)
        return VerifyResultModel(logits=logits[None, :, :], kv_cache=kv_cache)

    def draft_speculate(
        self,
        prefix_tokens: list[int],
        k: int,
    ) -> tuple[list[int], jnp.ndarray]:
        """Autoregressive draft speculation for K tokens."""
        tokens = list(prefix_tokens)
        all_logits = []
        for _ in range(k):
            logits = self._logits_from_tokens(jnp.array(tokens[-1:]))
            all_logits.append(logits[0])
            tok = int(jnp.argmax(logits[0]))
            tokens.append(tok)
        spec = tokens[-k:]
        return spec, jnp.stack(all_logits)

    def decode_tokens_to_str(self, token_ids: list[int]) -> str:
        return " ".join(f"t{tid}" for tid in token_ids)

    def commit_tokens(self, token_ids: list[int]) -> None:
        return

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

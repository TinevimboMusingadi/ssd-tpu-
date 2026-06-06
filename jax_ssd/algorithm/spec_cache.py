"""Tensor-backed draft speculation cache."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class SpecCache:
    """Fixed-size speculation cache keyed by [seq_id, accepted_index, recovery_token]."""

    keys: jax.Array  # [N, 3] int32
    tokens: jax.Array  # [N, K] int32
    logits: jax.Array  # [N, K, V] float32
    valid: jax.Array  # [N] bool
    k: int
    vocab: int

    @classmethod
    def create(cls, num_branches: int, k: int, vocab: int) -> SpecCache:
        return cls(
            keys=jnp.zeros((num_branches, 3), dtype=jnp.int32),
            tokens=jnp.zeros((num_branches, k), dtype=jnp.int32),
            logits=jnp.zeros((num_branches, k, vocab), dtype=jnp.float32),
            valid=jnp.zeros((num_branches,), dtype=bool),
            k=k,
            vocab=vocab,
        )

    def reset(self) -> SpecCache:
        return SpecCache(
            keys=self.keys,
            tokens=self.tokens,
            logits=self.logits,
            valid=jnp.zeros_like(self.valid),
            k=self.k,
            vocab=self.vocab,
        )

    def lookup(self, request_keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Lookup request keys [B, 3] against cache.

        Returns:
            hit: [B] bool
            idx: [B] int32 index into cache (-1 if miss)
            tokens: [B, K]
            logits: [B, K, V]
        """
        # [B, N, 3] == [B, 1, 3] vs [1, N, 3]
        matches = jnp.all(
            request_keys[:, None, :] == self.keys[None, :, :],
            axis=-1,
        )
        matches = matches & self.valid[None, :]
        hit = jnp.any(matches, axis=-1)
        idx = jnp.argmax(matches.astype(jnp.int32), axis=-1)
        idx = jnp.where(hit, idx, -1)

        tokens = self.tokens[idx]
        logits = self.logits[idx]
        tokens = jnp.where(hit[:, None], tokens, jnp.zeros_like(tokens))
        logits = jnp.where(hit[:, None, None], logits, jnp.zeros_like(logits))
        return hit, idx, tokens, logits

    def insert(
        self,
        slot: int,
        key: jax.Array,
        tokens: jax.Array,
        logits: jax.Array,
    ) -> SpecCache:
        return SpecCache(
            keys=self.keys.at[slot].set(key),
            tokens=self.tokens.at[slot].set(tokens),
            logits=self.logits.at[slot].set(logits),
            valid=self.valid.at[slot].set(True),
            k=self.k,
            vocab=self.vocab,
        )

    def insert_batch(
        self,
        start_slot: int,
        keys: jax.Array,
        tokens: jax.Array,
        logits: jax.Array,
    ) -> SpecCache:
        n = keys.shape[0]
        slots = jnp.arange(start_slot, start_slot + n)
        new = self
        for i in range(n):
            new = new.insert(int(slots[i]), keys[i], tokens[i], logits[i])
        return new

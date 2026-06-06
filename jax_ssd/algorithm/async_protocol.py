"""Async SSD request/response message types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import jax.numpy as jnp


class DraftCommand(IntEnum):
    PREFILL = 0
    SERVE = 1
    SHUTDOWN = 2


@dataclass
class DraftRequest:
    command: DraftCommand
    cache_keys: jnp.ndarray  # [B, 3] int32
    seq_lens: jnp.ndarray  # [B] int32
    page_tables: jnp.ndarray  # [B, max_pages] int32
    temperatures: jnp.ndarray  # [B] float32


@dataclass
class DraftResponse:
    cache_hit: jnp.ndarray  # [B] bool
    spec_tokens: jnp.ndarray  # [B, K] int32
    draft_logits: jnp.ndarray  # [B, K, V] float32


@dataclass
class VerifyOutcome:
    seq_id: int
    accepted_length: int
    recovery_token: int

    def as_key(self) -> tuple[int, int, int]:
        return (self.seq_id, self.accepted_length, self.recovery_token)

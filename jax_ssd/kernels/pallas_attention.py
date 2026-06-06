"""Pallas TPU attention kernels — Stage 9 optimization (profile-gated)."""

from __future__ import annotations

import logging

import jax.numpy as jnp

from jax_ssd.kernels.attention_ref import masked_dot_product_attention

logger = logging.getLogger(__name__)

PALLAS_ENABLED = False


def branch_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    force_pallas: bool = False,
) -> jnp.ndarray:
    """Branch/tree attention — uses Pallas when enabled and profiled."""
    if force_pallas and PALLAS_ENABLED:
        try:
            return _pallas_branch_attention(q, k, v, mask)
        except Exception as exc:
            logger.warning("Pallas kernel failed, falling back to reference: %s", exc)
    return masked_dot_product_attention(q, k, v, mask)


def _pallas_branch_attention(q, k, v, mask):
    """Placeholder for Pallas kernel — enable after profiling shows mask attention bottleneck."""
    raise NotImplementedError("Pallas branch attention not yet implemented; profile first.")

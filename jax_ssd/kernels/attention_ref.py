"""Dense masked attention reference implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def masked_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Scaled dot-product attention with optional additive mask."""
    d = q.shape[-1]
    scores = jnp.einsum("...qd,...kd->...qk", q, k) / jnp.sqrt(d)
    if mask is not None:
        scores = scores + mask
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("...qk,...kd->...qd", weights, v)

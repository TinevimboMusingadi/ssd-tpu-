"""Compress draft logits for host queue transfer — Stage 9 optimization."""

from __future__ import annotations

import jax.numpy as jnp


def compress_greedy_logits(
    draft_logits: jnp.ndarray,
    spec_tokens: jnp.ndarray,
) -> jnp.ndarray:
    """Greedy mode: only keep draft prob at sampled token positions [B, K]."""
    b, k, v = draft_logits.shape
    probs = jax_nn_softmax(draft_logits)
    tok_idx = spec_tokens[:, :k]
    return jnp.take_along_axis(probs, tok_idx[..., None], axis=-1).squeeze(-1)


def decompress_greedy_logits(
    token_probs: jnp.ndarray,
    spec_tokens: jnp.ndarray,
    vocab_size: int,
) -> jnp.ndarray:
    """Reconstruct sparse logits from token probs for verify (greedy only)."""
    b, k = spec_tokens.shape
    logits = jnp.full((b, k, vocab_size), -1e9)
    for i in range(k):
        logits = logits.at[:, i, spec_tokens[:, i]].set(
            jnp.log(token_probs[:, i] + 1e-12)
        )
    return logits


def jax_nn_softmax(x: jnp.ndarray) -> jnp.ndarray:
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)

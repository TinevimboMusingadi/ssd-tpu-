"""Saguaro branch prior: top-F recovery token selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def top_f_recovery_tokens(
    glue_logits: jax.Array,
    fan_out: int,
) -> jax.Array:
    """Select top-F recovery token candidates from draft glue logits.

    Args:
        glue_logits: [B, V] or [B, K+1, V] — use last position if 3D
        fan_out: F candidates per branch

    Returns:
        tokens: [B, F] int32
    """
    if glue_logits.ndim == 3:
        glue_logits = glue_logits[:, -1, :]
    _, top_idx = jax.lax.top_k(glue_logits, fan_out)
    return top_idx.astype(jnp.int32)


def build_branch_keys(
    seq_ids: jax.Array,
    accepted_indices: jax.Array,
    recovery_tokens: jax.Array,
) -> jax.Array:
    """Build cache keys [N, 3] from branch components."""
    return jnp.stack([seq_ids, accepted_indices, recovery_tokens], axis=-1).astype(jnp.int32)


def expand_fanout_branches(
    base_keys: jax.Array,
    fan_out_list: tuple[int, ...],
    recovery_candidates: jax.Array,
) -> tuple[jax.Array, int]:
    """Expand base batch into branch batch using fan_out_list.

    Args:
        base_keys: [B, 3]
        fan_out_list: length K+1 fanout per accepted length
        recovery_candidates: [B, max_F] top recovery tokens

    Returns:
        branch_keys: [B * MQ_LEN, 3]
        mq_len: total branches per sequence
    """
    mq_len = sum(fan_out_list)
    b = base_keys.shape[0]
    branches = []
    offset = 0
    for acc_idx, fan in enumerate(fan_out_list):
        for f in range(fan):
            key = base_keys.at[:, 1].set(acc_idx)
            if recovery_candidates.shape[1] > f:
                key = key.at[:, 2].set(recovery_candidates[:, f])
            branches.append(key)
        offset += fan
    if not branches:
        return base_keys, b
    return jnp.concatenate(branches, axis=0), mq_len

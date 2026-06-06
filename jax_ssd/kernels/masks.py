"""Branch/tree attention masks for async SSD draft decode."""

from __future__ import annotations

import jax.numpy as jnp


def build_branch_mask(
    batch: int,
    num_branches: int,
    trunk_len: int,
    branch_len: int,
) -> jnp.ndarray:
    """Dense branch attention mask [B*branches, trunk+branch, trunk+branch].

    Each branch attends to:
    - full verified trunk prefix
    - its glue/recovery path
    - only its own prior branch tokens (causal within branch)
    """
    total_q = trunk_len + branch_len
    mask = jnp.full((batch * num_branches, total_q, total_q), -jnp.inf)
    for b in range(batch * num_branches):
        # Trunk visible to all queries
        mask = mask.at[b, :, :trunk_len].set(0.0)
        # Causal within branch portion
        for i in range(branch_len):
            q_idx = trunk_len + i
            mask = mask.at[b, q_idx, trunk_len : trunk_len + i + 1].set(0.0)
    return mask


def build_verify_mask(seq_len: int, query_len: int) -> jnp.ndarray:
    """Mask for target verify pass: queries attend to full prefix + causal within query."""
    total = seq_len + query_len
    mask = jnp.full((query_len, total), -jnp.inf)
    mask = mask.at[:, :seq_len].set(0.0)
    for i in range(query_len):
        mask = mask.at[i, seq_len : seq_len + i + 1].set(0.0)
    return mask

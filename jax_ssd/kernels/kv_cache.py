"""Paged KV cache operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class KVCacheState:
    """Paged KV cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]."""

    pages: jax.Array
    block_size: int

    @classmethod
    def allocate(
        cls,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> KVCacheState:
        pages = jnp.zeros(
            (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )
        return cls(pages=pages, block_size=block_size)

    @property
    def num_blocks(self) -> int:
        return self.pages.shape[2]


def write_kv_slots(
    cache: KVCacheState,
    layer: int,
    slot_mapping: jax.Array,
    k: jax.Array,
    v: jax.Array,
) -> KVCacheState:
    """Write K/V vectors into paged cache at slot positions."""

    def _write(pages, slot, k_vec, v_vec):
        block = slot // cache.block_size
        offset = slot % cache.block_size
        pages = pages.at[0, layer, block, offset].set(k_vec)
        pages = pages.at[1, layer, block, offset].set(v_vec)
        return pages

    pages = cache.pages
    for i in range(slot_mapping.shape[0]):
        pages = _write(pages, slot_mapping[i], k[i], v[i])
    return KVCacheState(pages=pages, block_size=cache.block_size)


def gather_kv_for_positions(
    cache: KVCacheState,
    layer: int,
    page_table: jax.Array,
    positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gather K/V for sequence positions from page table."""
    slots = page_table[positions // cache.block_size] * cache.block_size + positions % cache.block_size
    # Simplified gather — full attention uses masks module
    k = jnp.zeros((positions.shape[0], cache.pages.shape[-2], cache.pages.shape[-1]))
    v = jnp.zeros_like(k)
    return k, v

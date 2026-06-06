"""Shape bucket padding for stable JAX compilation."""

from __future__ import annotations

import jax.numpy as jnp


def bucket_batch_size(batch: int, buckets: tuple[int, ...] = (1, 2, 4, 8, 16)) -> int:
    for b in buckets:
        if batch <= b:
            return b
    return buckets[-1]


def pad_to_bucket(
    arr: jnp.ndarray,
    target_batch: int,
    *,
    axis: int = 0,
) -> jnp.ndarray:
    """Pad array along batch axis to target_batch size."""
    current = arr.shape[axis]
    if current >= target_batch:
        return arr
    pad_len = target_batch - current
    pad_shape = list(arr.shape)
    pad_shape[axis] = pad_len
    pad = jnp.zeros(pad_shape, dtype=arr.dtype)
    return jnp.concatenate([arr, pad], axis=axis)

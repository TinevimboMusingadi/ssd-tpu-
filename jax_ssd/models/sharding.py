"""FSDP-style parameter sharding for Flax Gemma on TPU meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def fsdp_shard_params(params: object, mesh: Mesh | None) -> object:
    """Shard large arrays along the first mesh axis; replicate small arrays."""
    if mesh is None:
        return params

    axis = mesh.axis_names[0]
    shard_pspec = P(axis)
    replicate_pspec = P()
    shard_sharding = NamedSharding(mesh, shard_pspec)
    replicate_sharding = NamedSharding(mesh, replicate_pspec)

    def _shard_leaf(x):
        if not isinstance(x, (jnp.ndarray, jax.Array)):
            return x
        if x.ndim >= 2 and x.size > 8192:
            try:
                return jax.device_put(x, shard_sharding)
            except Exception:
                pass
        return jax.device_put(x, replicate_sharding)

    return jax.tree.map(_shard_leaf, params)


def primary_device(devices: tuple[jax.Device, ...]) -> jax.Device:
    return devices[0] if devices else jax.devices()[0]

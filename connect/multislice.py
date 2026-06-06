"""Multislice mesh utilities — Stage 9 scaling."""

from __future__ import annotations

import logging

import jax

logger = logging.getLogger(__name__)


def create_hybrid_mesh(
    num_slices: int,
    chips_per_slice: int,
    *,
    dcn_axis: str = "data",
) -> jax.sharding.Mesh | None:
    """Create hybrid device mesh for multislice TPU pods.

    Requires multiple TPU slices connected via DCN. Use only after single-slice SSD is stable.
    """
    try:
        from jax.experimental.mesh_utils import create_hybrid_device_mesh

        shape = (num_slices, chips_per_slice)
        mesh = create_hybrid_device_mesh(
            mesh_shape=shape,
            process_is_granule=False,
        )
        logger.info("Created hybrid mesh %s with axis %s", shape, dcn_axis)
        return mesh
    except Exception as exc:
        logger.warning("Multislice mesh unavailable: %s", exc)
        return None

"""Auto-split TPU devices into target and draft meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from jax.sharding import Mesh

from connect.slice_probe import DeviceTopology, probe_devices


@dataclass(frozen=True)
class MeshAllocation:
    target_devices: tuple[jax.Device, ...]
    draft_devices: tuple[jax.Device, ...]
    target_mesh: Mesh | None
    draft_mesh: Mesh | None
    policy: str
    warnings: tuple[str, ...]

    @property
    def supports_async_ssd(self) -> bool:
        return len(self.target_devices) >= 1 and len(self.draft_devices) >= 1

    def summary(self) -> str:
        return (
            f"target={len(self.target_devices)} draft={len(self.draft_devices)} "
            f"policy={self.policy} async_ssd={self.supports_async_ssd}"
        )


def _split_counts(n: int) -> tuple[int, int]:
    """Return (target_count, draft_count) for N devices."""
    if n >= 8:
        draft = max(1, n // 8)
        target = n - draft
    elif n == 4:
        target, draft = 3, 1
    elif n == 2:
        target, draft = 1, 1
    elif n == 1:
        target, draft = 1, 0
    else:
        draft = max(1, n // 4)
        target = n - draft
    return target, draft


def allocate_meshes(
    policy: Literal["auto", "all_target"] = "auto",
    topology: DeviceTopology | None = None,
) -> MeshAllocation:
    topology = topology or probe_devices()
    devices = tuple(jax.devices())
    n = len(devices)
    warnings: list[str] = []

    if policy == "all_target":
        target_devs = devices
        draft_devs: tuple[jax.Device, ...] = ()
        warnings.append("all_target policy: async SSD unavailable without draft devices.")
    else:
        t_count, d_count = _split_counts(n)
        target_devs = devices[:t_count]
        draft_devs = devices[t_count : t_count + d_count]
        if n == 1:
            warnings.append("Single device: only AR and sync SD; SSD runs sequentially.")

    target_mesh = _make_mesh(target_devs, "target") if target_devs else None
    draft_mesh = _make_mesh(draft_devs, "draft") if draft_devs else None

    return MeshAllocation(
        target_devices=target_devs,
        draft_devices=draft_devs,
        target_mesh=target_mesh,
        draft_mesh=draft_mesh,
        policy=policy,
        warnings=tuple(warnings),
    )


def _make_mesh(devices: tuple[jax.Device, ...], name: str) -> Mesh:
    """Build a Mesh whose device array ndim matches axis_names (JAX requirement)."""
    n = len(devices)
    if n == 0:
        raise ValueError(f"cannot build {name} mesh with zero devices")
    if n == 1:
        return Mesh(devices, axis_names=("model",))
    if n == 2:
        return Mesh(np.array(devices).reshape(1, 2), axis_names=("data", "model"))
    if n == 4:
        return Mesh(np.array(devices).reshape(2, 2), axis_names=("data", "model"))
    # 3, 5, 6, 7, ... — use one axis (e.g. v6e-4 split 3+1)
    return Mesh(devices, axis_names=("devices",))


class TPUConnector:
    """High-level connector: probe topology and allocate meshes."""

    def __init__(self, policy: Literal["auto", "all_target"] = "auto") -> None:
        self.policy = policy
        self._topology: DeviceTopology | None = None
        self._allocation: MeshAllocation | None = None

    def probe(self) -> DeviceTopology:
        self._topology = probe_devices()
        return self._topology

    def allocate_meshes(self, policy: str | None = None) -> MeshAllocation:
        policy = policy or self.policy  # type: ignore[assignment]
        self._allocation = allocate_meshes(policy=policy, topology=self.probe())
        return self._allocation

    def health_check(self) -> bool:
        topo = self.probe()
        return topo.device_count > 0

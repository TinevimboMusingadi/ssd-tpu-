"""Probe JAX devices and build a topology summary."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class DeviceTopology:
    device_count: int
    local_device_count: int
    platform: str
    device_kinds: tuple[str, ...]
    device_ids: tuple[int, ...]

    @property
    def is_tpu(self) -> bool:
        return self.platform == "tpu"

    def summary(self) -> str:
        kinds = ", ".join(sorted(set(self.device_kinds)))
        return (
            f"platform={self.platform} devices={self.device_count} "
            f"local={self.local_device_count} kinds=[{kinds}]"
        )


def probe_devices() -> DeviceTopology:
    devices = jax.devices()
    kinds = tuple(str(d.device_kind) for d in devices)
    ids = tuple(int(str(d).split(":")[-1]) if ":" in str(d) else i for i, d in enumerate(devices))
    platform = devices[0].platform if devices else "unknown"
    return DeviceTopology(
        device_count=jax.device_count(),
        local_device_count=jax.local_device_count(),
        platform=platform,
        device_kinds=kinds,
        device_ids=ids,
    )

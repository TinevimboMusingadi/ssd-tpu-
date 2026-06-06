"""TPU smoke test and diagnostics CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from connect.config import ConnectConfig
from connect.mesh_allocator import TPUConnector


def run_diagnostics() -> int:
    config = ConnectConfig.from_env()
    config.apply_jax_platforms()

    print("=== SSD-TPU Doctor ===")
    if config.gcp_project:
        print(f"GCP project: {config.gcp_project}")
    if config.tpu_zone:
        print(f"TPU zone: {config.tpu_zone}")
    if config.ssh_host:
        print(f"SSH host: {config.ssh_host}")

    connector = TPUConnector()
    if not connector.health_check():
        print("FAIL: no JAX devices detected.")
        return 1

    topo = connector.probe()
    print(f"Topology: {topo.summary()}")

    alloc = connector.allocate_meshes()
    print(f"Mesh allocation: {alloc.summary()}")
    for w in alloc.warnings:
        print(f"  warning: {w}")

    target_path = os.getenv("TARGET_MODEL_PATH", "./models/google_gemma-2b-it")
    if Path(target_path).exists():
        print(f"Model weights: {target_path} (OK)")
    elif os.getenv("SSD_USE_TOY_MODEL", "0").lower() in ("1", "true", "yes"):
        print("Model weights: toy mode (SSD_USE_TOY_MODEL=1)")
    else:
        print(f"Model weights: MISSING at {target_path}")
        print("  Run: python scripts/download_models.py --preset gemma-2b")

    try:
        import jax
        import jax.numpy as jnp

        x = jnp.ones((4, 4))
        y = jax.device_put(x)
        _ = (y + 1).block_until_ready()
        print("JAX compute smoke test: OK")
    except Exception as exc:
        print(f"JAX compute smoke test: FAIL ({exc})")
        return 1

    print("All checks passed.")
    return 0


def main() -> None:
    sys.exit(run_diagnostics())


if __name__ == "__main__":
    main()

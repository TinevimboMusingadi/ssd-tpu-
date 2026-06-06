"""TPU smoke test and diagnostics CLI."""

from __future__ import annotations

import argparse
import sys

import jax

from connect.config import ConnectConfig
from connect.gcs_storage import check_gcs_access, is_gcs_path, resolve_model_path
from connect.mesh_allocator import TPUConnector
from connect.profiles import get_profile


def run_diagnostics(*, smoke_model: bool = False) -> int:
    config = ConnectConfig.from_env()
    config.apply_jax_platforms()

    print("=== SSD-TPU Doctor ===")
    if config.gcp_project:
        print(f"GCP project: {config.gcp_project}")
    if config.tpu_zone:
        print(f"TPU zone: {config.tpu_zone}")
    if config.tpu_vm_name:
        print(f"TPU VM: {config.tpu_vm_name}")
    if config.tpu_slice_chips:
        print(f"Expected chips: {config.tpu_slice_chips}")

    try:
        profile = get_profile(config.model_profile)
        print(f"Model profile: {profile.name} ({profile.description})")
    except KeyError as exc:
        print(f"WARN: {exc}")
        profile = None

    if config.gcs_bucket:
        ok, msg = check_gcs_access(config.gcs_bucket)
        print(f"GCS bucket: {config.gcs_bucket} ({msg})")

    connector = TPUConnector()
    if not connector.health_check():
        print("FAIL: no JAX devices detected.")
        return 1

    topo = connector.probe()
    print(f"Topology: {topo.summary()}")

    if profile and topo.device_count < profile.min_chips:
        print(
            f"WARN: profile {profile.name} needs >={profile.min_chips} chips, "
            f"got {topo.device_count}"
        )

    alloc = connector.allocate_meshes()
    print(f"Mesh allocation: {alloc.summary()}")
    for w in alloc.warnings:
        print(f"  warning: {w}")

    if config.use_toy_model:
        print("Model weights: toy mode (SSD_USE_TOY_MODEL=1)")
    else:
        for label, path in (
            ("target", config.target_model_path),
            ("draft", config.draft_model_path),
        ):
            if not path:
                print(f"Model {label}: not configured")
                continue
            if is_gcs_path(path):
                print(f"Model {label}: {path} (GCS — sync via bootstrap_vm.sh)")
            else:
                from pathlib import Path

                if Path(path).exists():
                    print(f"Model {label}: {path} (OK)")
                else:
                    print(f"Model {label}: MISSING at {path}")
                    print(
                        "  Run: python scripts/download_models.py "
                        "--preset sd-pair-7b --gcs-uri $GCS_BUCKET"
                    )

    try:
        import jax.numpy as jnp

        x = jnp.ones((4, 4))
        y = jax.device_put(x)
        _ = (y + 1).block_until_ready()
        print("JAX compute smoke test: OK")
    except Exception as exc:
        print(f"JAX compute smoke test: FAIL ({exc})")
        return 1

    if smoke_model and not config.use_toy_model and config.target_model_path:
        try:
            from jax_ssd.models.model_loader import load_model_adapter

            print("Model forward smoke: loading target...")
            m = load_model_adapter(
                config.target_model_path,
                mesh=alloc.target_mesh,
                devices=alloc.target_devices,
                role="target",
            )
            ids = m.tokenize("Hello")
            pre = m.prefill(
                jnp.array(ids[:4], dtype=jnp.int32),
                m.allocate_kv(),
                jnp.zeros((1,), dtype=jnp.int32),
            )
            _ = pre.logits.block_until_ready()
            print("Model forward smoke: OK")
        except Exception as exc:
            print(f"Model forward smoke: FAIL ({exc})")
            return 1

    print("All checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-model",
        action="store_true",
        help="Load target model and run one prefill forward pass",
    )
    args = parser.parse_args()
    sys.exit(run_diagnostics(smoke_model=args.smoke_model))


if __name__ == "__main__":
    main()

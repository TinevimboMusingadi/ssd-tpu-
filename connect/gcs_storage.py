"""GCS path helpers and model sync for TPU VMs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def gcs_to_local_cache(gcs_path: str, cache_root: Path | None = None) -> Path:
    """Map gs://bucket/prefix/repo-name -> ~/.cache/ssd-tpu-models/repo-name."""
    if not is_gcs_path(gcs_path):
        return Path(gcs_path)
    root = cache_root or Path.home() / ".cache" / "ssd-tpu-models"
    name = gcs_path.rstrip("/").split("/")[-1]
    return root / name


def sync_gcs_to_local(gcs_path: str, local_path: Path | None = None) -> Path:
    """Rsync a GCS model prefix to local cache. Returns local path."""
    if not is_gcs_path(gcs_path):
        return Path(gcs_path)

    local = local_path or gcs_to_local_cache(gcs_path)
    local.mkdir(parents=True, exist_ok=True)

    # Skip sync if config.json already present
    if (local / "config.json").exists():
        return local

    print(f"Syncing {gcs_path} -> {local} ...")
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_path, str(local)],
        check=True,
    )
    return local


def upload_local_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """Upload a local model directory to GCS."""
    gcs_dest = gcs_uri.rstrip("/") + "/" + local_path.name
    print(f"Uploading {local_path} -> {gcs_dest} ...")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", str(local_path), gcs_dest],
        check=True,
    )


def check_gcs_access(gcs_bucket: str | None) -> tuple[bool, str]:
    if not gcs_bucket:
        return False, "GCS_BUCKET not set"
    if not is_gcs_path(gcs_bucket):
        return False, f"Invalid GCS_BUCKET: {gcs_bucket}"
    try:
        subprocess.run(
            ["gcloud", "storage", "ls", gcs_bucket],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, "OK"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return False, str(exc)


def resolve_model_path(path: str | None, default_gcs_prefix: str | None = None) -> str:
    """Return local path, syncing from GCS if needed."""
    if not path:
        raise FileNotFoundError("Model path not configured")
    if is_gcs_path(path):
        return str(sync_gcs_to_local(path))
    if Path(path).exists():
        return path
    if default_gcs_prefix and os.getenv("GCS_BUCKET"):
        bucket = os.getenv("GCS_BUCKET", "").rstrip("/")
        prefix = os.getenv("GCS_MODEL_PREFIX", "models")
        guess = f"{bucket}/{prefix}/{Path(path).name}"
        if is_gcs_path(guess):
            return str(sync_gcs_to_local(guess))
    raise FileNotFoundError(f"Model not found: {path}")

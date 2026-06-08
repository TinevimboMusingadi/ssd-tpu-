#!/usr/bin/env python3
"""Check EasyDeL Gemma4 JAX snapshot has text head weights (lm_head)."""

from __future__ import annotations

import sys
from pathlib import Path


def _snapshot_dir() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo = hub / "models--EasyDeL--gemma-4-E2B-it"
    if not repo.exists():
        return None
    snaps = repo / "snapshots"
    if not snaps.exists():
        return None
    candidates = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    snap = _snapshot_dir()
    if snap is None:
        print("MISSING: no EasyDeL/gemma-4-E2B-it HF cache")
        print("Run: hf download EasyDeL/gemma-4-E2B-it")
        return 1

    print(f"snapshot: {snap}")
    lm_paths = list(snap.rglob("*lm_head*"))
    embed_paths = list(snap.rglob("*embed*kernel*"))[:5]
    zarray_count = sum(1 for _ in snap.rglob(".zarray"))

    print(f"zarr arrays: {zarray_count}")
    print(f"lm_head paths: {len(lm_paths)}")
    for p in lm_paths[:8]:
        print(f"  {p.relative_to(snap)}")

    if embed_paths:
        print("embed samples:")
        for p in embed_paths[:3]:
            print(f"  {p.relative_to(snap)}")

    has_lm_kernel = any("lm_head" in str(p) and "kernel" in str(p) for p in lm_paths)
    if has_lm_kernel:
        print("OK: lm_head.kernel present — generation should work")
        return 0

    print("WARN: lm_head.kernel not found in snapshot")
    print("Gemma may use tied embeddings; EasyDeL still warns and output can be empty.")
    print("Try re-download: hf download EasyDeL/gemma-4-E2B-it")
    print("Or restore from GCS: bash scripts/restore_gemma4_from_gcs.sh")
    return 1


if __name__ == "__main__":
    sys.exit(main())

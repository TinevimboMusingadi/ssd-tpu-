#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PRESETS: dict[str, str] = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-7b": "google/gemma-7b-it",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="HF repo id, e.g. google/gemma-2b-it")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Shortcut for common Gemma checkpoints",
    )
    parser.add_argument("--output", default="./models")
    args = parser.parse_args()

    repo = args.repo or (PRESETS[args.preset] if args.preset else None)
    if not repo:
        parser.error("Provide --repo or --preset")

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    out = Path(args.output) / repo.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo} -> {out}")
    if not token:
        print("Tip: set HF_TOKEN in .env (accept Gemma license on huggingface.co first).")

    path = snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        token=token,
    )
    print(f"Downloaded to {path}")
    print(f"Set in .env: TARGET_MODEL_PATH={out}")


if __name__ == "__main__":
    main()

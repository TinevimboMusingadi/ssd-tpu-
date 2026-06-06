#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PRESETS: dict[str, str] = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-7b": "google/gemma-7b-it",
}


def main() -> None:
    load_dotenv()

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
        print("ERROR: HF_TOKEN not set.")
        print("1. Accept license: https://huggingface.co/google/gemma-2b-it")
        print("2. Create token: https://huggingface.co/settings/tokens")
        print("3. Add to ~/ssd-tpu-/.env:  HF_TOKEN=hf_...")
        print("   Or run from Windows: .\\scripts\\push_hf_token.ps1")
        sys.exit(1)

    print("HF_TOKEN found — starting download...")

    path = snapshot_download(
        repo_id=repo,
        local_dir=str(out),
        token=token,
    )
    print(f"Downloaded to {path}")
    print(f"Set in .env: TARGET_MODEL_PATH={out}")


if __name__ == "__main__":
    main()

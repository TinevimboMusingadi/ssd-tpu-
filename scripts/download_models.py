#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. google/gemma-2b")
    parser.add_argument("--output", default="./models")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.output) / args.repo.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.repo, local_dir=str(out))
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()

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

# Default speculative-decoding pair for v6e-4: larger target + smaller draft
SD_PAIR = ("google/gemma-2-2b-it", "google/gemma-2b-it")


def _download_one(repo: str, output: Path, token: str) -> Path:
    from huggingface_hub import snapshot_download

    out = output / repo.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo} -> {out}")

    try:
        return Path(
            snapshot_download(
                repo_id=repo,
                local_dir=str(out),
                token=token,
            )
        )
    except Exception as exc:
        msg = str(exc)
        print(f"Download failed: {exc}")
        if "403" in msg or "gated" in msg.lower() or "401" in msg:
            print()
            print("Fix Hugging Face access:")
            print(f"  1. Open https://huggingface.co/{repo} and click 'Agree and access'")
            print("  2. Use a Classic token (Read) OR a fine-grained token with")
            print("     'Access public gated repositories' enabled:")
            print("     https://huggingface.co/settings/tokens")
            print("  3. Update HF_TOKEN in .env, then re-run:")
            print("     python scripts/push_hf_token.py   # from Windows")
            print("     python scripts/download_models.py --preset sd-pair")
        sys.exit(1)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="HF repo id, e.g. google/gemma-2b-it")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS) + ["sd-pair"],
        help="Shortcut: sd-pair = Gemma-2.2B target + Gemma-2B draft (v6e-4)",
    )
    parser.add_argument("--output", default="./models")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.")
        print("1. Accept Gemma licenses on huggingface.co (both target and draft repos)")
        print("2. Create token: https://huggingface.co/settings/tokens")
        print("3. Add HF_TOKEN to .env, then: python scripts/push_hf_token.py")
        sys.exit(1)

    print("HF_TOKEN found — starting download...")
    output = Path(args.output)

    if args.preset == "sd-pair":
        target_repo, draft_repo = SD_PAIR
        target_path = _download_one(target_repo, output, token)
        draft_path = _download_one(draft_repo, output, token)
        print()
        print("SD pair ready. Set in .env:")
        print(f"  TARGET_MODEL_PATH={target_path}")
        print(f"  DRAFT_MODEL_PATH={draft_path}")
        return

    repo = args.repo or (PRESETS[args.preset] if args.preset else None)
    if not repo:
        parser.error("Provide --repo or --preset")

    path = _download_one(repo, output, token)
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face (optionally upload to GCS)."""

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

SD_PAIR_2B = ("google/gemma-2-2b-it", "google/gemma-2b-it")
SD_PAIR_7B = ("google/gemma-7b-it", "google/gemma-2b-it")
SD_PAIR_GEMMA4_E2B = ("google/gemma-4-E2B-it", "google/gemma-4-E2B-it-assistant")

PAIR_PRESETS: dict[str, tuple[str, str]] = {
    "sd-pair": SD_PAIR_2B,
    "sd-pair-2b": SD_PAIR_2B,
    "sd-pair-7b": SD_PAIR_7B,
    "sd-pair-gemma4-e2b": SD_PAIR_GEMMA4_E2B,
}


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
        _print_hf_access_help(repo, token)
        sys.exit(1)


def _print_hf_access_help(repo: str, token: str) -> None:
    print()
    print("Fix Hugging Face access:")
    print(f"  1. Open https://huggingface.co/{repo} and click 'Agree and access'")
    print("  2. Token must be Classic (Read) OR fine-grained with")
    print("     'Read access to contents of all public gated repos' enabled")
    try:
        import requests

        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.ok:
            auth = r.json().get("auth", {}).get("accessToken", {})
            if auth.get("role") == "fineGrained":
                fg = auth.get("fineGrained", {})
                if not fg.get("canReadGatedRepos"):
                    print("  3. Your fine-grained token has canReadGatedRepos=False — create a new token")
    except Exception:
        pass
    print("  4. Update HF_TOKEN in .env, then: py scripts/push_hf_token.py")


def _maybe_upload(local_path: Path, gcs_uri: str | None) -> str:
    if not gcs_uri:
        return str(local_path)
    from connect.gcs_storage import upload_local_to_gcs

    dest_base = gcs_uri.rstrip("/")
    upload_local_to_gcs(local_path, dest_base)
    gcs_path = f"{dest_base}/{local_path.name}"
    print(f"Uploaded to {gcs_path}")
    return gcs_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="HF repo id, e.g. google/gemma-7b-it")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS) + sorted(PAIR_PRESETS),
        help="sd-pair-7b = Gemma-7B target + Gemma-2B draft (v6e-16)",
    )
    parser.add_argument("--output", default="./models")
    parser.add_argument(
        "--gcs-uri",
        default=os.getenv("GCS_BUCKET"),
        help="Upload to GCS after download (e.g. gs://project-ssd-tpu/models)",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.")
        sys.exit(1)

    gcs_base = args.gcs_uri
    if gcs_base and not gcs_base.endswith("/models"):
        prefix = os.getenv("GCS_MODEL_PREFIX", "models")
        gcs_base = f"{gcs_base.rstrip('/')}/{prefix}"

    print("HF_TOKEN found — starting download...")
    output = Path(args.output)

    if args.preset in PAIR_PRESETS:
        target_repo, draft_repo = PAIR_PRESETS[args.preset]
        target_local = _download_one(target_repo, output, token)
        draft_local = _download_one(draft_repo, output, token)
        target_final = _maybe_upload(target_local, gcs_base)
        draft_final = _maybe_upload(draft_local, gcs_base)
        print()
        print("SD pair ready. Set in .env:")
        print(f"  TARGET_MODEL_PATH={target_final}")
        print(f"  DRAFT_MODEL_PATH={draft_final}")
        print(f"  MODEL_PROFILE={args.preset}")
        return

    repo = args.repo or (PRESETS[args.preset] if args.preset else None)
    if not repo:
        parser.error("Provide --repo or --preset")

    path = _download_one(repo, output, token)
    final = _maybe_upload(path, gcs_base)
    print(f"Downloaded to {final}")


if __name__ == "__main__":
    main()

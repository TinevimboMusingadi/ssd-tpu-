#!/usr/bin/env python3
"""Verify HF token can read gated Gemma repos before downloading."""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    token = token.strip("\ufeff").strip("\r\n")
    if not token or not token.startswith("hf_"):
        print("ERROR: set HF_TOKEN=hf_... in .env")
        sys.exit(1)

    who = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if not who.ok:
        print(f"ERROR: invalid token (HTTP {who.status_code})")
        sys.exit(1)

    info = who.json()
    print(f"HF user: {info.get('name')}")
    auth = info.get("auth", {}).get("accessToken", {})
    if auth.get("role") == "fineGrained" and not auth.get("fineGrained", {}).get(
        "canReadGatedRepos"
    ):
        print("ERROR: fine-grained token cannot read gated repos.")
        print("Create a Classic Read token or enable gated-repo read on fine-grained.")
        sys.exit(1)

    repo = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2b-it"
    r = requests.head(
        f"https://huggingface.co/{repo}/resolve/main/config.json",
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=True,
        timeout=30,
    )
    if r.status_code == 200:
        print(f"OK: can download {repo}")
        return
    print(f"ERROR: cannot download {repo} (HTTP {r.status_code})")
    print(f"Accept license at https://huggingface.co/{repo}")
    sys.exit(1)


if __name__ == "__main__":
    main()

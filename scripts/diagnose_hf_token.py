#!/usr/bin/env python3
"""Diagnose HF_TOKEN issues on VM vs local .env (no secrets printed)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _mask(token: str) -> str:
    if len(token) < 12:
        return "(too short)"
    return f"{token[:7]}...{token[-4:]}"


def _clean(token: str) -> str:
    return token.strip().strip("\ufeff").strip("\r\n").strip('"' "'")


def _read_env_file(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return _clean(val)
    return None


def _http_status(url: str, token: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _whoami(token: str) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return exc.code, body[:200]


def _report_token(label: str, token: str | None) -> None:
    if not token:
        print(f"  {label}: (missing)")
        return
    print(f"  {label}: len={len(token)} mask={_mask(token)}")
    if not token.startswith("hf_"):
        print(f"    WARN: does not start with hf_")
    if "\r" in token or "\n" in token:
        print("    WARN: contains newline characters")
    if token != _clean(token):
        print("    WARN: has leading/trailing whitespace or quotes")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"

    print("=== HF token diagnostic ===")
    print(f"cwd: {root}")
    print(f".env exists: {env_path.exists()}")

    file_token = _read_env_file(env_path)
    env_token = _clean(os.getenv("HF_TOKEN", "") or "") or None
    hub_token = _clean(os.getenv("HUGGING_FACE_HUB_TOKEN", "") or "") or None

    print("\n--- token sources ---")
    _report_token("from .env file", file_token)
    _report_token("from $HF_TOKEN", env_token)
    _report_token("from $HUGGING_FACE_HUB_TOKEN", hub_token)

    if file_token and env_token and file_token != env_token:
        print("\n  MISMATCH: .env file token != $HF_TOKEN in shell")
        print("  Fix: run 'unset HF_TOKEN' then 'source .env', or sync .env to VM")

    token = file_token or env_token or hub_token
    if not token:
        print("\nERROR: no HF_TOKEN found. Set HF_TOKEN=hf_... in .env")
        return 1

    print("\n--- API check (authoritative) ---")
    status, info = _whoami(token)
    if status != 200:
        print(f"  whoami-v2: HTTP {status}")
        if isinstance(info, str):
            print(f"  body: {info}")
        print("\n  Token rejected by Hugging Face API.")
        print("  Colab login is separate — updating local .env does NOT update the VM.")
        print("  From Windows: python scripts/push_hf_token.py")
        print("  Or: gcloud compute scp .env VM:~/ssd-tpu-/.env --zone=us-east5-b")
        return 1

    name = info.get("name", "?")
    print(f"  whoami-v2: OK (user={name})")
    auth = info.get("auth", {}).get("accessToken", {})
    if auth.get("role") == "fineGrained":
        can_gated = auth.get("fineGrained", {}).get("canReadGatedRepos")
        print(f"  fine-grained token, canReadGatedRepos={can_gated}")
        if not can_gated:
            print("  ERROR: enable gated-repo read or use Classic Read token")
            return 1

    repos = [
        "google/gemma-4-E2B-it",
        "EasyDeL/gemma-4-E2B-it",
    ]
    print("\n--- repo access ---")
    ok = True
    for repo in repos:
        url = f"https://huggingface.co/{repo}/resolve/main/config.json"
        code = _http_status(url, token)
        status_txt = "OK" if code in (200, 307) else "FAIL"
        print(f"  {repo}: HTTP {code} ({status_txt})")
        if code not in (200, 307):
            ok = False
            print(f"    Accept license: https://huggingface.co/{repo}")

    if ok:
        print("\nALL CHECKS PASSED — token works. Run:")
        print("  hf download EasyDeL/gemma-4-E2B-it")
        return 0

    print("\nToken valid but repo access failed — accept model license on huggingface.co")
    return 1


if __name__ == "__main__":
    sys.exit(main())

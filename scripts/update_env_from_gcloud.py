#!/usr/bin/env python3
"""Sync TPU_SSH_HOST / TPU_SSH_USER in .env from an existing gcloud VM."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def write_env_key(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out: list[str] = []
    found = False
    for line in lines:
        if re.match(rf"^\s*{re.escape(key)}\s*=", line):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def gcloud(*args: str) -> str:
    cmd = ["gcloud", *args]
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm", default=None)
    parser.add_argument("--env", default=str(ENV_PATH))
    args = parser.parse_args()

    env = load_env(Path(args.env))
    project = env.get("GCP_PROJECT")
    zone = env.get("TPU_ZONE")
    vm = args.vm or env.get("TPU_VM_NAME", "ssd-tpu-v6e-vm")
    if not project or not zone:
        print("GCP_PROJECT and TPU_ZONE required in .env", file=sys.stderr)
        return 1

    ip = gcloud(
        "compute", "instances", "describe", vm,
        f"--project={project}",
        f"--zone={zone}",
        "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
    )
    account = gcloud("config", "get-value", "account")
    user = account.split("@")[0] if "@" in account else account

    env_path = Path(args.env)
    write_env_key(env_path, "TPU_SSH_HOST", ip)
    write_env_key(env_path, "TPU_SSH_USER", user)
    write_env_key(env_path, "TPU_VM_NAME", vm)
    print(f"Updated {env_path}: TPU_SSH_HOST={ip} TPU_SSH_USER={user}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

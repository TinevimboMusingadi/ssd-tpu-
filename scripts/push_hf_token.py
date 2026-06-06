#!/usr/bin/env python3
"""Push HF_TOKEN from local .env to the TPU VM via gcloud ssh."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def _update_env_text(lines: list[str], key: str, value: str) -> list[str]:
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token or not token.startswith("hf_"):
        print("ERROR: set HF_TOKEN=hf_... in .env", file=sys.stderr)
        sys.exit(1)

    project = os.getenv("GCP_PROJECT", "tpu-builder1")
    zone = os.getenv("TPU_ZONE", "us-east5-a")
    vm = os.getenv("TPU_VM_NAME", "ssd-tpu-v6e-vm")

    b64 = base64.b64encode(token.encode()).decode()
    py = f"""
import base64
from pathlib import Path
token = base64.b64decode("{b64}").decode()
path = Path.home() / "ssd-tpu-" / ".env"
lines = path.read_text().splitlines() if path.exists() else []
for key, val in [
    ("HF_TOKEN", token),
    ("TARGET_MODEL_PATH", "./models/google_gemma-2b-it"),
    ("DRAFT_MODEL_PATH", "./models/google_gemma-2b-it"),
]:
    replaced = False
    new_lines = []
    for line in lines:
        if line.startswith(key + "="):
            new_lines.append(f"{{key}}={{val}}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{{key}}={{val}}")
    lines = new_lines
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\\n".join(lines) + "\\n")
print("HF_TOKEN synced to VM .env")
"""
    script_b64 = base64.b64encode(py.encode()).decode()
    remote = f"echo {script_b64} | base64 -d | python3"

    gcloud = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if not Path(gcloud).exists():
        gcloud = "gcloud"

    cmd = [
        gcloud,
        "compute",
        "ssh",
        vm,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={remote}",
    ]
    print(f"Pushing HF_TOKEN to {vm}...")
    subprocess.run(cmd, check=True, shell=False)
    print("Done. On VM: python scripts/download_models.py --preset gemma-2b")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Push full .env profile from Windows to the TPU VM via gcloud ssh."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from connect.profiles import get_profile, profile_env_defaults


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token or not token.startswith("hf_"):
        print("ERROR: set HF_TOKEN=hf_... in .env", file=sys.stderr)
        sys.exit(1)

    project = os.getenv("GCP_PROJECT", "tpu-builder1")
    zone = os.getenv("TPU_ZONE", "us-east5-a")
    vm = os.getenv("TPU_VM_NAME", "ssd-tpu-v6e-8-vm")
    gcs = os.getenv("GCS_BUCKET")
    profile_name = os.getenv("MODEL_PROFILE", "sd-pair-7b")

    profile = get_profile(profile_name)
    defaults = profile_env_defaults(profile, gcs)

    env_updates: dict[str, str] = {
        "HF_TOKEN": token,
        "MODEL_PROFILE": profile.name,
        "JAX_PLATFORMS": os.getenv("JAX_PLATFORMS", "tpu"),
        **defaults,
    }
    for key in (
        "GCP_PROJECT",
        "TPU_ZONE",
        "TPU_VM_NAME",
        "TPU_SLICE_CHIPS",
        "GCS_BUCKET",
        "GCS_MODEL_PREFIX",
        "SSD_SHARDING_BACKEND",
    ):
        val = os.getenv(key)
        if val:
            env_updates[key] = val

    payload_b64 = base64.b64encode(json.dumps(env_updates).encode()).decode()
    remote = f"echo {payload_b64} | base64 -d | python3 -c \""
    remote += (
        "import json,sys,base64; from pathlib import Path; "
        "data=json.loads(sys.stdin.read()); "
        "p=Path.home()/'ssd-tpu-'/'.env'; "
        "lines=p.read_text().splitlines() if p.exists() else []; "
        "keys=set(); "
        "out=[]; "
        "[out.append(f'{k}={data[k]}') or keys.add(k) for k in data if not any(l.startswith(k+'=') for l in lines)]; "
        "for line in lines: "
        "  k=line.split('=',1)[0] if '=' in line else ''; "
        "  out.append(f'{k}={data[k]}' if k in data else line); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(chr(10).join(out)+chr(10)); "
        "print('Profile synced:', list(data.keys()))\""
    )

    gcloud = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if not Path(gcloud).exists():
        gcloud = "gcloud"

    # Simpler remote script via heredoc payload
    py_script = f"""
import json, base64
from pathlib import Path
data = json.loads(base64.b64decode("{payload_b64}").decode())
path = Path.home() / "ssd-tpu-" / ".env"
lines = path.read_text().splitlines() if path.exists() else []
out = []
seen = set()
for line in lines:
    if "=" not in line or line.strip().startswith("#"):
        out.append(line)
        continue
    key = line.split("=", 1)[0]
    if key in data:
        out.append(f"{{key}}={{data[key]}}")
        seen.add(key)
    else:
        out.append(line)
for key, val in data.items():
    if key not in seen:
        out.append(f"{{key}}={{val}}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\\n".join(out) + "\\n")
print("Synced keys:", ", ".join(sorted(data.keys())))
"""
    script_b64 = base64.b64encode(py_script.encode()).decode()
    remote_cmd = f"echo {script_b64} | base64 -d | python3"

    cmd = [
        gcloud,
        "compute",
        "ssh",
        vm,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={remote_cmd}",
    ]
    print(f"Pushing profile '{profile.name}' to {vm}...")
    subprocess.run(cmd, check=True, shell=False)
    print("Done. On VM: ./scripts/bootstrap_vm.sh --profile", profile.name)


if __name__ == "__main__":
    main()

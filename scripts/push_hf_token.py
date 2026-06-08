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


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse .env file directly (ignores stale shell HF_TOKEN)."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    out: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip().strip("\r\n").strip('"' "'")
    return out


def _mask(token: str) -> str:
    return f"{token[:7]}...{token[-4:]}" if len(token) >= 12 else "(too short)"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    file_vars = _read_env_file(env_file)
    # File wins over shell env (Windows often has stale HF_TOKEN exported).
    load_dotenv(env_file, override=True)

    token = file_vars.get("HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token or not token.startswith("hf_"):
        print("ERROR: set HF_TOKEN=hf_... in .env", file=sys.stderr)
        sys.exit(1)

    shell_token = os.environ.get("HF_TOKEN", "")
    if shell_token and shell_token != token:
        print(f"NOTE: shell HF_TOKEN ({_mask(shell_token)}) ignored; using .env ({_mask(token)})")

    project = file_vars.get("GCP_PROJECT") or os.getenv("GCP_PROJECT", "tpu-builder1")
    zone = file_vars.get("TPU_ZONE") or os.getenv("TPU_ZONE", "us-east5-b")
    vm = file_vars.get("TPU_VM_NAME") or os.getenv("TPU_VM_NAME", "ssd-tpu-v6e-4-vm")
    gcs = file_vars.get("GCS_BUCKET") or os.getenv("GCS_BUCKET")
    profile_name = file_vars.get("MODEL_PROFILE") or os.getenv("MODEL_PROFILE", "sd-pair-7b")

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
    print(f"Pushing HF_TOKEN {_mask(token)} to {vm} ({zone})...")
    subprocess.run(cmd, check=True, shell=False)
    print("Done. On VM run:")
    print("  unset HF_TOKEN && source .env && python scripts/diagnose_hf_token.py")


if __name__ == "__main__":
    main()

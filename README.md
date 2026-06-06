# SSD on Google TPU

JAX/TPU implementation of **Speculative Speculative Decoding (SSD / Saguaro)** — porting the [CUDA reference](https://github.com/tanishqkumar/ssd) to Google TPUs with Gemma model support, flexible slice connection, and a live terminal UI.

## Algorithms

| Mode | Description |
|------|-------------|
| `ar` | Autoregressive target-only baseline |
| `sd` | Synchronous speculative decoding |
| `ssd` | Async SSD / Saguaro with speculation cache |
| `instance` | Instance-SSD: retrieval-based speculation for code refactoring |

## Requirements

Python 3.10–3.12 recommended (JAX TPU wheels). Use a TPU VM for production runs.

## Quick start

```bash
# Local / CPU smoke test (Windows: use project venv)
py -m venv .venv
.\.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python -m connect.diagnostics
pytest tests/ -q

# TPU VM setup (after git clone + cp .env.example .env)
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
./scripts/setup_tpu_vm.sh
source .venv/bin/activate
```

Copy `.env.example` to `.env` and set `GCP_PROJECT` and `TPU_ZONE`.

### Windows: provision VM + auto-fill SSH in `.env`

```powershell
# Fix: FLEX_START request-valid-for-duration max is 2h (not 4h)
.\scripts\provision_tpu.ps1

# SSH into VM (also writes ~/.ssh/config entry "ssd-tpu")
.\scripts\connect_ssh.ps1
```

This reads `GCP_PROJECT=tpu-builder1` from `.env`, creates `ssd-tpu-v6e-vm`, and writes `TPU_SSH_HOST` / `TPU_SSH_USER` back automatically.

## TPU connect

```bash
python -m connect.diagnostics
# or
ssd-tpu-doctor
```

Auto-partitions devices: most chips → target mesh, remainder → draft mesh.

## Benchmarks

```bash
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 10
```

## Live TUI

```bash
python -m tui.app --prompt "Refactor getUserById to fetchUserById"
```

Four panels stream **decoded tokens live** as each algorithm generates them.

## Layout

- `connect/` — TPU SSH, probe, mesh allocation
- `jax_ssd/` — inference engine, algorithms, models, kernels
- `tui/` — terminal UI with live token streaming
- `tests/` — parity and unit tests
- `scripts/` — TPU provisioning and model download
- `docs/` — architecture and algorithm notes

## Citation

```bibtex
@misc{kumar2026speculativespeculativedecoding,
  title={Speculative Speculative Decoding},
  author={Tanishq Kumar and Tri Dao and Avner May},
  year={2026},
  eprint={2603.03251},
  archivePrefix={arXiv},
}
```

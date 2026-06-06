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
./scripts/setup_tpu_vm.sh          # no PyTorch — saves ~2GB on 10GB boot disk
source .venv/bin/activate

# If a prior install failed with "No space left on device":
chmod +x scripts/recover_tpu_install.sh && ./scripts/recover_tpu_install.sh
```

> **Disk note:** TPU VM boot disks are small (~10GB). Do not `pip install -e ".[parity]"` on the VM — that pulls PyTorch+CUDA. Parity tests skip torch automatically.

Copy `.env.example` to `.env` and set `GCP_PROJECT`, `TPU_ZONE`, and `HF_TOKEN`.

### Download Gemma weights (required for real inference)

1. Accept the [Gemma license](https://huggingface.co/google/gemma-2b-it) on Hugging Face.
2. Create an HF token and set `HF_TOKEN` in `.env`.
3. On the TPU VM:

```bash
python scripts/download_models.py --preset gemma-2b
# or: chmod +x scripts/download_gemma.sh && ./scripts/download_gemma.sh
```

Default path: `./models/google_gemma-2b-it` (fits a v6e-4 slice). Tests use the toy model automatically (`SSD_USE_TOY_MODEL=1` in pytest).

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
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 2
python -m jax_ssd.benchmarks.stream_prompt --prompt "Explain Newton's second law"
```

## Live TUI

```bash
python -m tui.app --prompt "Explain quantum entanglement in simple terms"
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

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

## TPU Builders workflow (v6e-16 + Gemma 7B/2B + GCS)

### 1. Windows setup

```powershell
copy .env.example .env
# Edit: GCP_PROJECT, TPU_ZONE, HF_TOKEN

.\scripts\setup_gcs.ps1
python scripts/download_models.py --preset sd-pair-7b --gcs-uri gs://YOUR_PROJECT-ssd-tpu/models
.\scripts\teardown_vm.ps1 -VmName ssd-tpu-v6e-vm          # delete old 4-chip VM
.\scripts\provision_tpu.ps1 -ChipCount 16 -VmName ssd-tpu-v6e-16-vm
python scripts/push_hf_token.py
```

### 2. VM bootstrap (one command)

```bash
git clone https://github.com/TinevimboMusingadi/ssd-tpu-.git ~/ssd-tpu-
cd ~/ssd-tpu-
chmod +x scripts/bootstrap_vm.sh
./scripts/bootstrap_vm.sh --profile sd-pair-7b
```

Bootstrap installs JAX TPU + project deps (no PyTorch), syncs models from GCS, runs diagnostics and tests.

### 3. Model pair (v6e-16: 14 target + 2 draft chips)

| Role | Model | Default path |
|------|-------|--------------|
| Target (verifier) | Gemma-7B-IT | `gs://.../models/google_gemma-7b-it` |
| Draft (speculator) | Gemma-2B-IT | `gs://.../models/google_gemma-2b-it` |

Accept licenses for both models on Hugging Face. Use a Classic HF token with Read access.

### 4. Run inference

```bash
export JAX_PLATFORMS=tpu
export SSD_USE_TOY_MODEL=0
python -m connect.diagnostics --smoke-model
python -m jax_ssd.benchmarks.stream_prompt --prompt "Explain Newton's second law"
python -m tui.app --prompt "What is quantum entanglement?"
```

## Local dev (CPU, toy model)

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Tests use the toy model automatically unless `SSD_USE_REAL_MODEL=1`.

## Scripts

| Script | Purpose |
|--------|---------|
| `provision_tpu.ps1` | Create v6e/v5p VM (`-ChipCount 16`) |
| `teardown_vm.ps1` | Delete old VM |
| `setup_gcs.ps1` | Create GCS bucket + IAM |
| `bootstrap_vm.sh` | Master VM install (JAX, deps, GCS sync, tests) |
| `download_models.py` | HF download + optional GCS upload |
| `push_hf_token.py` | Sync full profile to VM `.env` |

## Layout

- `connect/` — TPU probe, mesh allocation, GCS, profiles
- `jax_ssd/` — inference engine, algorithms, sharded Gemma adapters
- `tui/` — terminal UI with live token streaming
- `tests/` — unit tests + `test_real_gemma_tpu.py` (TPU only)
- `scripts/` — provisioning, bootstrap, GCS

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

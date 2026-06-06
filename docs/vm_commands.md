# TPU VM — copy-paste commands

## Every new SSH session

```bash
cd ~/ssd-tpu-
source .venv/bin/activate
export JAX_PLATFORMS=tpu
export SSD_USE_TOY_MODEL=0
```

## First-time bootstrap (v6e-16)

```bash
git pull
chmod +x scripts/bootstrap_vm.sh
./scripts/bootstrap_vm.sh --profile sd-pair-7b
```

## Health check

```bash
python -m connect.diagnostics
python -m connect.diagnostics --smoke-model
```

## Real model tests

```bash
export SSD_USE_REAL_MODEL=1
pytest tests/test_real_gemma_tpu.py -m tpu -q
```

## Stream tokens (physics prompt)

```bash
python -m jax_ssd.benchmarks.stream_prompt --mode ar \
  --prompt "Explain Newton's second law" --max-tokens 32
```

## Benchmark all modes

```bash
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 2 \
  --prompt "What is the speed of light"
```

## Live TUI

```bash
python -m tui.app --prompt "Explain quantum entanglement in simple terms"
```

Quit with `q`.

## Recover from disk-full / broken venv

```bash
./scripts/recover_tpu_install.sh --profile sd-pair-7b
```

## Windows: provision + push profile

```powershell
.\scripts\provision_tpu.ps1 -ChipCount 16
python scripts/push_hf_token.py
```

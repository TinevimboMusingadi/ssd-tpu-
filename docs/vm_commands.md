# TPU VM — copy-paste commands

SSH in, then copy **one block at a time** (no extra spaces in flags).

## Every new SSH session

```bash
cd ~/ssd-tpu-
source .venv/bin/activate
export JAX_PLATFORMS=tpu
```

## Pull latest fixes

```bash
git pull
```

## Health check

```bash
python -m connect.diagnostics
```

## Tests

```bash
pytest tests/ -q
```

Note: `tests/` not `test/`. Flag is `-q` not `-q` on its own line.

## Benchmark

```bash
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 3
```

Note: `jax_ssd` not `jax_sdd`. Use `--mode` not `-- mode` (no space).

## Stream tokens (physics prompt, no TUI)

```bash
python -m jax_ssd.benchmarks.stream_prompt --mode ar --prompt "Explain Newton second law"
```

Modes: `ar`, `sd`, `ssd`, `instance`

## Benchmark with custom prompt

```bash
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode ar --prompt "What is quantum entanglement" --num-prompts 1
```

## Live TUI

```bash
python -m tui.app --prompt "Explain Newton second law in simple terms"
```

Quit with `q`.

## Full recover + install

```bash
chmod +x scripts/recover_tpu_install.sh
./scripts/recover_tpu_install.sh
```

## Run everything

```bash
bash scripts/vm_commands.sh
```

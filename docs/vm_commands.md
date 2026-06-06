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

## Live TUI

```bash
python -m tui.app --prompt "Refactor getUserById to fetchUserById"
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

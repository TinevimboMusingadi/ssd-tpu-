# Architecture

## Layers

1. **connect/** — TPU probe, SSH tunnel, mesh allocation
2. **jax_ssd/algorithm/** — verify, spec cache, branch/instance priors
3. **jax_ssd/runtime/** — engine, scheduler, workers, metrics
4. **jax_ssd/models/** — Gemma/Qwen/toy adapters
5. **jax_ssd/kernels/** — KV cache, masks, buckets, Pallas stubs
6. **tui/** — live four-panel token streaming demo

## Data flow (SSD)

Target worker verifies K tokens while draft worker rebuilds speculation cache for predicted outcomes. Host queues carry compact keys; logits optional in greedy mode.

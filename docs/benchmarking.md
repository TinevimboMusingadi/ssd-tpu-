# Benchmarking

```bash
python -m jax_ssd.benchmarks.compare_ar_sd_ssd --mode all --num-prompts 10 --output results.json
python -m jax_ssd.benchmarks.qa_smoke
python -m jax_ssd.benchmarks.code_refactor
```

Reports tokens/s, TTFT, cache hit rate, and speedup vs AR.

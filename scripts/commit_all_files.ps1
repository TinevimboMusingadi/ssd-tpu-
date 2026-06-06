# One commit per file (excludes .gitignore patterns). Do not run twice on clean tree.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$commits = @(
    @{ Path = ".gitignore"; Message = "Add gitignore rules for env, venv, and local context" },
    @{ Path = "README.md"; Message = "Add project README with setup and TPU quickstart" },
    @{ Path = "pyproject.toml"; Message = "Add pyproject with JAX, Textual, and CLI entrypoints" },
    @{ Path = ".env.example"; Message = "Add environment variable template for GCP and SSD config" },
    @{ Path = ".env copy.example"; Message = "Add secondary env template copy for local testing" },
    @{ Path = ".cursor/plans/ssd_tpu_port_plan_89484783.plan.md"; Message = "Add SSD TPU port implementation plan" },
    @{ Path = "connect/__init__.py"; Message = "Add connect package exports for TPU connector" },
    @{ Path = "connect/config.py"; Message = "Add ConnectConfig loader from dotenv" },
    @{ Path = "connect/slice_probe.py"; Message = "Add JAX device topology probe" },
    @{ Path = "connect/mesh_allocator.py"; Message = "Add auto target/draft TPU mesh allocator" },
    @{ Path = "connect/ssh_tunnel.py"; Message = "Add SSH tunnel helper for Remote-SSH dev" },
    @{ Path = "connect/diagnostics.py"; Message = "Add ssd-tpu-doctor diagnostics CLI" },
    @{ Path = "connect/multislice.py"; Message = "Add multislice hybrid mesh utilities for scaling" },
    @{ Path = "jax_ssd/__init__.py"; Message = "Add jax_ssd public package exports" },
    @{ Path = "jax_ssd/config.py"; Message = "Add SSDConfig and DecodeMode runtime settings" },
    @{ Path = "jax_ssd/sampling_params.py"; Message = "Add SamplingParams dataclass for generation" },
    @{ Path = "jax_ssd/llm.py"; Message = "Add public LLM API wrapper" },
    @{ Path = "jax_ssd/algorithm/__init__.py"; Message = "Add algorithm package exports" },
    @{ Path = "jax_ssd/algorithm/verify.py"; Message = "Add JAX speculative decoding verify kernel" },
    @{ Path = "jax_ssd/algorithm/spec_cache.py"; Message = "Add tensor-backed draft speculation cache" },
    @{ Path = "jax_ssd/algorithm/branch_prior.py"; Message = "Add Saguaro branch prior and top-F selection" },
    @{ Path = "jax_ssd/algorithm/instance_prior.py"; Message = "Add Instance-SSD context span retrieval prior" },
    @{ Path = "jax_ssd/algorithm/async_protocol.py"; Message = "Add async SSD draft request/response types" },
    @{ Path = "jax_ssd/runtime/__init__.py"; Message = "Add runtime package exports" },
    @{ Path = "jax_ssd/runtime/sequence.py"; Message = "Add per-request sequence state model" },
    @{ Path = "jax_ssd/runtime/page_manager.py"; Message = "Add paged KV block allocator" },
    @{ Path = "jax_ssd/runtime/scheduler.py"; Message = "Add inference request scheduler" },
    @{ Path = "jax_ssd/runtime/metrics.py"; Message = "Add tokens/s and cache hit metrics collector" },
    @{ Path = "jax_ssd/runtime/workers.py"; Message = "Add async draft worker with host queues" },
    @{ Path = "jax_ssd/runtime/engine.py"; Message = "Add LLM engine for AR, SD, SSD, and Instance modes" },
    @{ Path = "jax_ssd/kernels/__init__.py"; Message = "Add kernels package exports" },
    @{ Path = "jax_ssd/kernels/buckets.py"; Message = "Add JIT shape bucket padding helpers" },
    @{ Path = "jax_ssd/kernels/kv_cache.py"; Message = "Add paged KV cache read/write operations" },
    @{ Path = "jax_ssd/kernels/masks.py"; Message = "Add branch and verify attention masks" },
    @{ Path = "jax_ssd/kernels/attention_ref.py"; Message = "Add dense masked attention reference" },
    @{ Path = "jax_ssd/kernels/pallas_attention.py"; Message = "Add Pallas attention stub for Stage 9 optimization" },
    @{ Path = "jax_ssd/kernels/logits_compress.py"; Message = "Add draft logits compression for host queues" },
    @{ Path = "jax_ssd/kernels/profile.py"; Message = "Add profiling helpers to gate optimizations" },
    @{ Path = "jax_ssd/models/__init__.py"; Message = "Add models package exports" },
    @{ Path = "jax_ssd/models/base.py"; Message = "Add DecodeModelAdapter protocol interface" },
    @{ Path = "jax_ssd/models/toy_model.py"; Message = "Add toy LM adapter for algorithm tests" },
    @{ Path = "jax_ssd/models/gemma_adapter.py"; Message = "Add Gemma model adapter with toy fallback" },
    @{ Path = "jax_ssd/models/qwen_adapter.py"; Message = "Add Qwen adapter stub for MaxText phase" },
    @{ Path = "jax_ssd/benchmarks/__init__.py"; Message = "Add benchmarks package marker" },
    @{ Path = "jax_ssd/benchmarks/qa_smoke.py"; Message = "Add QA smoke benchmark entrypoint" },
    @{ Path = "jax_ssd/benchmarks/code_refactor.py"; Message = "Add code refactor benchmark for Instance-SSD" },
    @{ Path = "jax_ssd/benchmarks/compare_ar_sd_ssd.py"; Message = "Add AR vs SD vs SSD comparison harness" },
    @{ Path = "tui/__init__.py"; Message = "Add TUI package marker" },
    @{ Path = "tui/theme.py"; Message = "Add terminal UI color theme" },
    @{ Path = "tui/token_anim.py"; Message = "Add live token stream state for panels" },
    @{ Path = "tui/panels.py"; Message = "Add per-algorithm Textual output panels" },
    @{ Path = "tui/app.py"; Message = "Add four-panel SSD live demo TUI application" },
    @{ Path = "reference/cuda_ssd/__init__.py"; Message = "Add CUDA reference package for parity tests" },
    @{ Path = "reference/cuda_ssd/verify_torch.py"; Message = "Add PyTorch verify reference for JAX parity" },
    @{ Path = "tests/test_verify_parity.py"; Message = "Add JAX vs PyTorch verify parity tests" },
    @{ Path = "tests/test_spec_cache.py"; Message = "Add speculation cache unit tests" },
    @{ Path = "tests/test_mesh_allocator.py"; Message = "Add mesh allocator unit tests" },
    @{ Path = "tests/test_page_rollback.py"; Message = "Add page rollback unit tests" },
    @{ Path = "tests/test_instance_prior.py"; Message = "Add Instance-SSD prior unit tests" },
    @{ Path = "tests/test_engine_modes.py"; Message = "Add engine AR/SD/SSD integration tests" },
    @{ Path = "scripts/provision_tpu.sh"; Message = "Add bash TPU VM provision script for Linux" },
    @{ Path = "scripts/setup_tpu_vm.sh"; Message = "Add TPU VM JAX install setup script" },
    @{ Path = "scripts/download_models.py"; Message = "Add Hugging Face model download script" },
    @{ Path = "scripts/download_datasets.py"; Message = "Add benchmark dataset download script" },
    @{ Path = "scripts/load_env.ps1"; Message = "Add PowerShell dotenv loader for Windows scripts" },
    @{ Path = "scripts/update_env.ps1"; Message = "Add PowerShell helper to patch .env keys" },
    @{ Path = "scripts/provision_tpu.ps1"; Message = "Add Windows TPU provision and env auto-fill script" },
    @{ Path = "scripts/connect_ssh.ps1"; Message = "Add Windows SSH connect and config writer" },
    @{ Path = "scripts/update_env_from_gcloud.py"; Message = "Add Python gcloud to .env SSH sync utility" },
    @{ Path = "docs/architecture.md"; Message = "Add architecture overview documentation" },
    @{ Path = "docs/tpu_connect.md"; Message = "Add TPU connect and provision documentation" },
    @{ Path = "docs/algorithms.md"; Message = "Add AR, SD, SSD, and Instance algorithm docs" },
    @{ Path = "docs/benchmarking.md"; Message = "Add benchmarking usage documentation" }
)

$n = 0
foreach ($c in $commits) {
    $path = $c['Path']
    if (-not (Test-Path $path)) {
        Write-Warning "Skip missing: $path"
        continue
    }
    $msg = $c['Message']
    git add -- $path
    git commit -m $msg
    if ($LASTEXITCODE -ne 0) { throw "Commit failed for $path" }
    $n++
    Write-Host "[$n] $path"
}

Write-Host "Done: $n commits."

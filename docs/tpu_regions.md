# TPU regions and slice sizes (SSD-TPU)

Based on [TPU Builders Getting Started Guide](../context_files/TPU%20Builders%20Getting%20Started%20Guide%20(1).md) and live `gcloud` catalog for `tpu-builder1`.

## Key fact: there is no `ct6e-standard-16t`

A single FLEX_START **GCE VM** does not offer 16 v6e chips. The catalog max is:

| Machine type | Chips | Available zones (sample) |
|--------------|-------|--------------------------|
| `ct6e-standard-1t` | 1 | us-east5-a/b/c, us-central1-a/b/c, … |
| `ct6e-standard-4t` | 4 | same |
| `ct6e-standard-8t` | **8 (max)** | same |

The Builders table **"Max Slice Size 8x16"** means **pod multislice** (up to 128 chips in a pod topology), not a VM named `16t`.

## Recommended zones (TPU Builders FLEX_START)

| Family | Zones | GCE type | Notes |
|--------|-------|----------|-------|
| **v6e** | `us-east5-a`, `us-east5-b` | `ct6e-standard-{1,4,8}t` | Best for Builders; 8-chip max per VM |
| **v6e** | `us-central1-a` | `ct6e-standard-4t` | Builders doc: 8×8 pod max |
| **v5p** | `us-east5-a` | `ct5p-hightpu-{1,2,4}t` | Doc says 128 chips via multislice; higher $/chip |

## SSD-TPU mesh split by chip count

| Chips | Target | Draft | Good for |
|-------|--------|-------|----------|
| 8 | 7 | 1 | **Gemma 7B + 2B** (`sd-pair-7b`) |
| 4 | 3 | 1 | Gemma 2.2B + 2B (`sd-pair-2b`) |
| 1 | 1 | 0 | AR only |

## Discover capacity in your project

```powershell
.\scripts\list_tpu_capacity.ps1
.\scripts\list_tpu_capacity.ps1 -Family v5p
```

## Provision examples

```powershell
# Best option for 7B+2B today
.\scripts\provision_tpu.ps1 -ChipCount 8 -VmName ssd-tpu-v6e-8-vm -Zone us-east5-a

# Try sibling zone if stockout
.\scripts\provision_tpu.ps1 -ChipCount 8 -Zone us-east5-b

# v5p (different pricing, 4-chip VM max in catalog)
.\scripts\provision_tpu.ps1 -Family v5p -ChipCount 4 -Zone us-east5-a
```

## Dual 8+8 VMs (16 chips total)

When one 8-chip VM is tight for Gemma 7B, use **two** `ct6e-standard-8t` VMs:

| VM | Chips | Role | Env |
|----|-------|------|-----|
| `ssd-tpu-target-8-vm` | 8 | Gemma 7B sharded across all 8 | `SSD_TPU_ROLE=target` |
| `ssd-tpu-draft-8-vm` | 8 | Gemma 2B draft | `SSD_TPU_ROLE=draft` |

```powershell
.\scripts\provision_dual_8.ps1 -Zone us-east5-b
```

On each VM after bootstrap:

```bash
# Target VM only
export SSD_TPU_ROLE=target
python -m jax_ssd.benchmarks.stream_prompt --mode ar --prompt "Hello"

# Draft VM only (smoke)
export SSD_TPU_ROLE=draft
python -m connect.diagnostics --smoke-model
```

Cross-VM SD/SSD (target on one VM, draft on another) is a future step; **AR on target VM** works today.

## Need more than 8 chips per VM?

1. Email **tpu-builders-support@google.com** (chips, duration, use case).
2. Consider **v5p multislice** (advanced; not yet wired in SSD-TPU scripts).
3. GCS bucket is **region-wide** — you can move VMs across zones without re-uploading weights.

# TPU VM connect & overnight runs

## Connect (from your PC)

**Windows PowerShell:**

```powershell
.\scripts\connect_vm.ps1
```

**Git Bash / Linux / VM:**

```bash
bash scripts/connect_vm.sh
```

Reads `GCP_PROJECT`, `TPU_ZONE`, `TPU_VM_NAME` from `.env` (see `.env.example`).

## Push secrets to VM

`.env` is not in git. After editing `HF_TOKEN` locally:

```powershell
py -3 scripts/push_hf_token.py
```

## Overnight automation (while you sleep)

From your PC (pushes token + starts background job on VM):

```powershell
.\scripts\start_overnight_on_vm.ps1
```

On the VM directly:

```bash
cd ~/ssd-tpu-
nohup bash scripts/overnight_gemma4.sh > logs/overnight-nohup.log 2>&1 &
tail -f logs/overnight-*.log
```

## When done

```bash
bash scripts/stop_gemma4_vm.sh
```

Weights stay in `~/.cache/huggingface/` until you delete the VM. Backup once:

```bash
bash scripts/snapshot_gemma4_to_gcs.sh
```

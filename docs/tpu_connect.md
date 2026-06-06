# TPU Connect

## Doctor

```bash
python -m connect.diagnostics
```

## Provision

**Windows (reads `.env`, auto-fills SSH):**

```powershell
.\scripts\provision_tpu.ps1
.\scripts\connect_ssh.ps1
```

**Linux / macOS:**

```bash
./scripts/provision_tpu.sh v6e us-east5-a
./scripts/setup_tpu_vm.sh
```

Note: `request-valid-for-duration` max is **2h** for FLEX_START (not 4h).

## SSH

Set in `.env`:

```
TPU_SSH_HOST=your-vm-ip
TPU_SSH_USER=your-user
JAX_PLATFORMS=tpu
```

Use Cursor Remote-SSH to develop on the TPU VM directly.

## Mesh allocation

| Devices | Target | Draft |
|---------|--------|-------|
| 8+ | N-1 or N-N//8 | 1+ |
| 4 | 3 | 1 |
| 2 | 1 | 1 |
| 1 | 1 | 0 (SSD sequential) |

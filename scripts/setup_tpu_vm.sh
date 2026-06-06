#!/usr/bin/env bash
# Thin wrapper — delegates to bootstrap_vm.sh (no model sync).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$ROOT/scripts/bootstrap_vm.sh"
exec "$ROOT/scripts/bootstrap_vm.sh" --skip-models "$@"

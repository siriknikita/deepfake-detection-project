#!/usr/bin/env bash
#
# cluster_bootstrap.sh — set up a fresh GPU-cluster node for Hyperplane-Forge.
#
# Run this once after sshing into the cluster. It installs the Rust toolchain,
# uv, ffmpeg, builds the Rust extension into a project-local virtualenv, and
# verifies the install with the test suite.
#
# Usage:
#   ./scripts/cluster_bootstrap.sh [--no-tests]
#
# Assumes:
#   - You have already cloned the repo and `cd`ed to its root.
#   - The cluster node has a working CUDA driver (otherwise the PyTorch
#     install will resolve to the CPU-only wheel; the pipeline still runs,
#     just not on GPU).
#
# Idempotent — safe to re-run.

set -euo pipefail

run_tests=1
for arg in "$@"; do
    case "$arg" in
        --no-tests) run_tests=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

# ------------ Rust toolchain ------------
if ! command -v cargo >/dev/null 2>&1; then
    log "installing rustup + stable toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
else
    log "rust toolchain present: $(cargo --version)"
fi

# ------------ uv (Python package manager) ------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    log "uv present: $(uv --version)"
fi

# ------------ ffmpeg (frame extraction) ------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg not found — install it via your cluster's module system or apt"
    log "  e.g. \`module load ffmpeg\` or \`sudo apt install ffmpeg\`"
    log "  (continuing anyway — frame extraction will fail later if missing)"
else
    log "ffmpeg present: $(ffmpeg -version | head -1)"
fi

# ------------ Project venv + Rust extension ------------
log "creating .venv with uv"
uv venv .venv

log "installing project (this builds the Rust extension via maturin)"
uv pip install --python .venv/bin/python -e ".[dev]"

# ------------ CUDA detection ------------
log "checking CUDA availability"
uv run --python .venv/bin/python python - <<'PY'
import torch
print(f"  torch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB")
PY

# ------------ Test suites ------------
if [[ $run_tests -eq 1 ]]; then
    log "running Rust tests"
    cargo test --release
    log "running Python tests"
    uv run --python .venv/bin/python pytest tests/ -q
else
    log "skipping tests (--no-tests)"
fi

log "done. Activate the venv with: source .venv/bin/activate"

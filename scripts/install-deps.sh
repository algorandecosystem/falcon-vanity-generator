#!/usr/bin/env bash
#
# Install/verify build dependencies for pq-vanity.
#   - Rust toolchain (installed via rustup if missing) — REQUIRED.
#   - A C compiler (for the vendored Falcon reference) — REQUIRED.
#   - CUDA toolkit (nvcc) — OPTIONAL, only for the GPU backend.
#   - Go — OPTIONAL, only for the off-curve differential test.
#
# Safe to re-run; it only installs Rust, and only if absent.
set -euo pipefail

have() { command -v "$1" >/dev/null 2>&1; }
say() { printf '[deps] %s\n' "$*"; }

# --- Rust (required) -------------------------------------------------------
if have cargo || [ -x "$HOME/.cargo/bin/cargo" ]; then
	ver="$( (cargo --version 2>/dev/null) || "$HOME/.cargo/bin/cargo" --version)"
	say "Rust: present ($ver)"
else
	say "Installing Rust via rustup (minimal profile, stable toolchain)..."
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
		| sh -s -- -y --profile minimal --default-toolchain stable
	say "Rust installed. Add it to this shell with:  source \"\$HOME/.cargo/env\""
fi

# --- C compiler (required for vendored Falcon C) ---------------------------
if have cc || have gcc || have clang; then
	say "C compiler: present"
else
	say "WARNING: no C compiler found (cc/gcc/clang). Install one, e.g.:"
	say "  Debian/Ubuntu:  sudo apt-get install -y build-essential"
fi

# --- CUDA toolkit (optional, for the GPU backend) --------------------------
NVCC="$(command -v nvcc 2>/dev/null || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && [ -n "${CUDA_PATH:-}" ] && [ -x "$CUDA_PATH/bin/nvcc" ] && NVCC="$CUDA_PATH/bin/nvcc"
if [ -n "$NVCC" ]; then
	rel="$("$NVCC" --version | grep -oE 'release [0-9.]+' | head -1 || true)"
	say "CUDA: present ($rel) at $NVCC"
	say "  -> build the GPU backend with:  make build-cuda"
else
	say "CUDA toolkit (nvcc) NOT found — the CPU build (make build) works without it."
	say "  For the GPU backend, install the NVIDIA CUDA toolkit (>=12.8 for sm_120/sm_121),"
	say "  put nvcc on PATH or set CUDA_PATH, then run:  make build-cuda"
fi

# --- Go (optional, dev-only) ----------------------------------------------
if have go; then
	say "Go: present (optional — used by tools/offcurve-oracle)"
else
	say "Go: not found (optional — only for the off-curve differential test)"
fi

say "Done. Next:  make build   (CPU)   or   make build-cuda   (GPU)"

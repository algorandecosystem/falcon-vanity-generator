#!/usr/bin/env bash
#
# Work around a CUDA (<= 13.1) / glibc (>= 2.41) header clash:
#
#   CUDA's crt/math_functions.h declares rsqrt()/rsqrtf() WITHOUT an exception
#   specification, but modern glibc declares them `noexcept(true)`. nvcc's
#   front-end (cudafe++) then rejects the mismatch:
#     "error: exception specification is incompatible ... rsqrt"
#
# This adds `noexcept(true)` to CUDA's four rsqrt/rsqrtf declarations so they
# match glibc. Idempotent; keeps a one-time `.f1bak` backup. Needs write access
# to the CUDA install (run as its owner, typically root).
#
# Refs: NVIDIA dev forums; ggml-org/llama.cpp#19100 (GCC 15 + CUDA 13.1).
# Undo: restore the .f1bak backup.
set -euo pipefail

NVCC="$(command -v nvcc 2>/dev/null || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -n "${CUDA_PATH:-}" ] && [ -x "$CUDA_PATH/bin/nvcc" ] && NVCC="$CUDA_PATH/bin/nvcc"
[ -n "$NVCC" ] || { echo "nvcc not found — nothing to patch."; exit 0; }

CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
H="$CUDA_ROOT/targets/$(uname -m)-linux/include/crt/math_functions.h"
[ -f "$H" ] || { echo "CUDA header not found: $H"; exit 1; }

if grep -q 'rsqrt(double x) noexcept' "$H"; then
	echo "already patched: $H"
	exit 0
fi

cp -n "$H" "$H.f1bak" 2>/dev/null || true
sed -i -E \
	-e 's/rsqrt\(double x\);/rsqrt(double x) noexcept(true);/' \
	-e 's/rsqrtf\(float x\);/rsqrtf(float x) noexcept(true);/' \
	-e 's/rsqrt\(double a\)\)/rsqrt(double a) noexcept(true))/' \
	-e 's/rsqrtf\(float a\)\)/rsqrtf(float a) noexcept(true))/' \
	"$H"

if grep -q 'rsqrt(double x) noexcept' "$H"; then
	echo "patched: $H"
	echo "backup:  $H.f1bak"
else
	echo "WARNING: patch did not apply as expected; check $H" >&2
	exit 1
fi

# CUDA kernel notes (`pq_vanity.cu`)

The kernel reproduces the **public half** of Falcon-1024 keygen plus the PQ
address hash — integer + FP32 only, no FP64 (Blackwell is ~1:64 FP64). The CPU
(`pq-core`) is byte-exact and KAT-validated; the kernel is validated
bit-for-bit against it (`pq-vanity gpu-selftest`).

## Build

```bash
cargo build -p pq-vanity --release --features cuda
# override the arch list (default 120,121):
PQ_CUDA_ARCHS=120,121 cargo build -p pq-vanity --release --features cuda
```

`build.rs` auto-detects `nvcc` (`CUDA_PATH` / `/usr/local/cuda` / PATH) and
filters requested arches against what the local toolkit supports. Without
nvcc, the crate builds a host stub and `pq-vanity gpu` reports that GPU
support isn't compiled in. `-Xptxas -v` is on to surface register/spill
counts. Block size is tunable at runtime via `PQ_GPU_BLOCK` (default 256).

### Toolchain gotchas

- **CUDA ≤ 13.1 + glibc ≥ 2.41:** `crt/math_functions.h` declares
  `rsqrt`/`rsqrtf` without an exception spec while glibc declares them
  `noexcept(true)`; nvcc's front-end rejects the mismatch. One-time fix (adds
  `noexcept(true)` to CUDA's four declarations, with a backup):
  `make patch-cuda-rsqrt`. `build.rs` detects this error and points here.
- **nvcc 13 `-O3` miscompiles `for(;;)`+`continue` rejection-sampling loops**
  (stored one element past a buffer). Use `do { } while(...)` in device
  sampling loops. Caught by `compute-sanitizer` and the bit-exact selftest.

## Device ↔ reference mapping

Every device crypto function is a port of the pinned vendored reference:

| Device fn (`pq_vanity.cu`) | Ported from | Notes |
|---|---|---|
| `keccakf` / sampler shake | `vendor/falcon/shake.c` | Keccak-f[1600], rate 136; fully unrolled with `constexpr` index tables so the state registerizes. Specialized to the sampler's use: absorb exactly one 32-byte seed, squeeze LE u64s through a 17-lane pool. Byte stream identical to the generic SHAKE256. |
| `dev_mkgauss` / `dev_poly_small_mkgauss` | `vendor/falcon/keygen.c` (`mkgauss`, `poly_small_mkgauss`) | n=1024: two u64 draws per coefficient; constant-time walk over `GAUSS_1024_12289`; parity fix on the last coefficient. |
| `mq_NTT` / `mq_iNTT` / `dev_compute_public` | `vendor/falcon/vrfy.c`, `keygen.c` (`Zf(compute_public)`) | Integer Montgomery NTT mod 12289; reject if `f` is not invertible. |
| `dev_modq_encode` | `vendor/falcon/codec.c` (`Zf(modq_encode)`) | 14-bit big-endian packing, header byte `0x0A`, 1793 bytes. |
| `sha512_compress` / `dev_pq_addr32` | FIPS 180-4 | SHA-512/256 of `"PQA"‖"f1"‖salt‖pk` (1799 bytes); rolling 16-word schedule, statically indexed. |
| `dev_bnorm_ok` | `vendor/falcon/keygen.c` norm check, reformulated | FP32 forward-FFT filter via Parseval: `bnorm = (q²/n)·Σ 1/(|f̂|²+|ĝ|²)` over the negacyclic bins — no iFFT, both real polys packed into one complex FFT. Used by the (currently unused) retry path; see DESIGN §7. |
| `base32_prefix_match` | RFC 4648 | Host passes the prefix as 5-bit indices; valid to 51 chars. |

## Consistency contract

The address depends only on `h = g·f⁻¹ mod q`. The kernel computes `h` from
the **first** sampled `(f,g)`; on a hit the CPU re-derives the same `(f,g)`
from the entropy (same `mkgauss` over the same SHAKE256 stream), runs the full
acceptance (norm checks + `solve_NTRU`), assembles and round-trips the key,
and only then emits it. `(f,g)` the reference keygen would reject are simply
discarded at hit time — the device sampler must be bit-identical to the CPU
sampler, and nothing else on the device is trusted.

## Validation (`pq-vanity gpu-selftest`)

Stage-by-stage device-vs-oracle comparison, zero mismatches required:

1. sampler — device `(f,g)` == `pq_core::sample_fg` (200k seeds)
2. pubkey — device `pk` == `pq_core::pubkey_from_fg`
3. addr32 — device address == `pq_core::pq_address` at several salts
4. retry/bnorm — device accept-attempt index == the CPU oracle
   `pq_core::first_visible_accept` (FP32-vs-FPEMU borderline tolerance;
   measured zero borderline cases)

Run `compute-sanitizer ./target/release/pq-vanity gpu-selftest --items 10000`
after kernel changes.

## Optimization state & backlog

See DESIGN §7 for measured findings (Keccak/SHA registerization ~5×; why the
kernel scans first samples instead of retrying in-seed). Backlog: cooperative
warp-per-key kernel with the NTT in shared memory (the sampler+NTT is the
wall), in-kernel off-curve, multi-GPU, suffix/contains patterns, per-key SHA
midstate caching for the salt sweep.

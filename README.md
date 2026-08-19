# falcon-vanity-generator

GPU-accelerated **vanity address generator** for Algorand **native
post-quantum accounts** (Falcon signatures).

It searches over Falcon public keys for one whose derived PQ address starts
with a chosen base32 pattern, then emits the account's **25-word mnemonic**
(importable with `algokey pq import -m`) along with the matching salt.

Today it implements scheme **`f1` (Deterministic Falcon-1024)**; the `pq-*`
crate/binary naming is deliberate — future Falcon schemes (`f2`, `f3`, …) will
live in the same tool.

The derivation tracks the PQ-account spec as merged to go-algorand master
(PR #6639): keys derive from 32-byte mnemonic entropy via
`seed = SHA512_256("PQK" || "f1" || entropy)`, then deterministic Falcon-1024
keygen; `addr32 = SHA512_256("PQA" || "f1" || salt || pk)`; the **canonical**
salt is the lowest one in `0..=255` whose address is off-curve compliant.

## Correctness

Everything is pinned and cross-checked, end to end:

- The CPU path reproduces go-algorand's **known-answer tests** byte-for-byte
  (address KATs, the end-to-end entropy→address KAT, mnemonic codec KATs),
  against spec files vendored at an exact upstream commit.
- The **CUDA pipeline** (SHAKE256 + Gaussian sampler + NTT mod 12289 +
  pk-encode + SHA-512/256) is validated **bit-exact against the CPU oracle**
  by `pq-vanity gpu-selftest`, stage by stage, and is `compute-sanitizer`
  clean.
- The off-curve compliance predicate is differential-tested against
  `filippo.io/edwards25519` (the exact library go-algorand uses).
- Every GPU hit is **re-derived and verified on the CPU** (reference keygen +
  signature round-trip) before it is reported.

## Layout

```
crates/pq-core          byte-exact derivation + Falcon reference (CPU oracle), KAT-tested
crates/pq-cuda          CUDA kernel build + safe Rust wrapper (host stub without nvcc)
crates/pq-vanity        CLI: derive / search / bench / gpu / gpu-selftest
cuda/                   the kernel (pq_vanity.cu) + kernel notes
vendor/falcon           Falcon det1024 C reference (MIT), pinned, compiled & linked
vendor/go-algorand-ref  PQ-account spec + KATs (AGPL), pinned, reference only
tools/offcurve-oracle   Go ground-truth for the off-curve predicate (dev only)
docs/DESIGN.md          spec + architecture
```

## Quick start

```bash
make deps          # install the Rust toolchain (reports CUDA/Go status)
make build         # CPU release binary  -> target/release/pq-vanity
make build-cuda    # GPU release binary (needs nvcc; sm_120 + sm_121)
make test          # KATs, formats, fast-path shim
make selftest      # validate the CUDA pipeline vs the CPU oracle (after build-cuda)
make help          # list all targets
```

GPU requirements: an NVIDIA Blackwell-class GPU (`sm_120` / `sm_121`) and CUDA
12.8+ (13.x recommended). On CUDA ≤ 13.1 with glibc ≥ 2.41 the toolkit headers
need a one-time fix: `make patch-cuda-rsqrt`. Other architectures can be added
with `make build-cuda ARCHS=<list>` — the kernel is integer + FP32 only and
does not depend on Blackwell features.

## Usage

```bash
# Search on the GPU (the host completes + verifies each hit, emits the key):
pq-vanity gpu --prefix ALGO --count 1 --out ./hits

# CPU-only search (slow reference baseline, same output format):
pq-vanity search --prefix ALGO --out ./hits

# Re-derive a known address from mnemonic entropy (sanity check vs KATs):
pq-vanity derive --entropy-hex 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20
# => ZEJ4BLG3XWAUUZQGCEDJLYIC6D2NCWHRSX5DJMDPE54PXXR7G3PCQTARXU  (canonical)
# also accepts --entropy-byte N and --mnemonic "25 words ..."

# Validate the device pipeline against the CPU oracle:
pq-vanity gpu-selftest --items 200000

# CPU throughput sanity check:
pq-vanity bench --seconds 5
```

The pattern alphabet is Algorand's base32: `A–Z` and `2–7` (no `0`, `1`, `8`,
`9`). Each extra character multiplies the search time by 32.

### Using a found address

Each hit prints (and writes to `<ADDRESS>.pqhit`) the vanity address, its
salt, the canonical salt/address for the same key, and the secret **entropy +
25-word mnemonic**. Import into algokey and sign:

```bash
algokey pq import -m "<25-word mnemonic>" -k my.pqkey
algokey pq sign -k my.pqkey -t in.tx -o out.tx
```

### Canonical vs non-canonical salts

A PQ address commits to a **salt byte** as well as the public key, so one key
has up to 256 valid addresses. Algorand tooling defines the **canonical** salt
as the lowest compliant one, and `algokey` always signs as the canonical-salt
address — there is no `--salt` flag.

**By default, `search` and `gpu` match only the canonical salt**, so every hit
works with stock algokey exactly as printed.

Pass `--allow-non-canonical` to also accept matches at non-canonical compliant
salts. This gives ~128× more candidate addresses per key (so hits come much
faster), **but be aware what you are buying**:

> **Non-canonical addresses need custom signing tooling.** A non-canonical
> address is fully consensus-valid — the salt travels inside the `PQSig`
> envelope and the protocol accepts any compliant salt — but `algokey` will
> only ever sign as the *canonical* address of the imported key. To spend from
> a non-canonical vanity address you must build/patch a signer that sets your
> salt in the `PQSig` instead of the keyfile's canonical salt. The `.pqhit`
> record always shows both the matched and the canonical salt so the two
> addresses can't be confused.

## Benchmarks

Measured with the commands shown below on four GPUs (single GPU each) and four
CPUs. Keys/s counts candidate Falcon keys scanned; a "candidate" becomes usable
only if the reference keygen accepts it (~5.2% of first samples — the GPU
scans fresh seeds and lets the CPU host verify hits; the CPU baseline runs the
full keygen per key, so every CPU key is usable).

### Canonical mode (default)

One algokey-ready candidate address per usable key. Expected time to a hit for
an `L`-character prefix ≈ `32^L / candidates_per_second`.

`pq-vanity gpu --prefix ALG --batches 4 --count 9999 --max-hits 16384 --allow-sibling-hits`

| Device | keys/s | usable candidates/s | measured `ALG` (3-char) hits | est. 5-char hit |
|---|---|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell (Max-Q) | 2.40 M | ~125 k | 33 in 7.0 s | ~4.5 min |
| NVIDIA RTX PRO 4500 Blackwell | 1.31 M | ~68 k | 24 in 12.8 s | ~8 min |
| NVIDIA GeForce RTX 5070 | 0.87 M | ~45 k | 40 in 19.3 s | ~12 min |
| NVIDIA GB10 (DGX Spark) | 0.78 M | ~41 k | 17 in 21.4 s | ~14 min |
| CPU (Ryzen 9 9950X, 31 threads, `search`) | 806 | 806 | — | ~12 h |
| CPU (Ryzen 9 7950X, 31 threads, `search`) | 718 | 718 | — | ~13 h |
| CPU (Xeon Gold 5412U, 47 threads, `search`) | 694 | 694 | — | ~13 h |
| CPU (i7-10700, 15 threads, `search`) | 236 | 236 | — | ~39 h |

### Non-canonical mode (`--allow-non-canonical`)

Up to 256 addresses per usable key (salts swept on-device); hits at any
compliant salt. Expected addresses examined per usable hit ≈ `32^(L+1)`.

`pq-vanity gpu --prefix ALGO --allow-non-canonical --batches 3 --count 9999 --allow-sibling-hits`

| Device | keys/s | addresses/s | measured `ALGO` (4-char) hits |
|---|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell (Max-Q) | 1.03 M | 263 M | 91 in 12.2 s |
| NVIDIA RTX PRO 4500 Blackwell | 0.57 M | 146 M | 101 in 22.1 s |
| NVIDIA GeForce RTX 5070 | 0.37 M | 95.5 M | 93 in 33.7 s |
| NVIDIA GB10 (DGX Spark) | 0.30 M | 76.9 M | 96 in 41.9 s |
| CPU (Ryzen 9 9950X, 31 threads, `search`) | ~806 | ~206 k | — |
| CPU (Ryzen 9 7950X, 31 threads, `search`) | ~718 | ~184 k | — |
| CPU (Xeon Gold 5412U, 47 threads, `search`) | ~694 | ~178 k | — |
| CPU (i7-10700, 15 threads, `search`) | ~236 | ~60 k | — |

Canonical mode reports lower addresses/s by design: it only ever checks one
address per key (sweeping salts `0..=31` on-device to locate the canonical
one), trading raw address throughput for hits that stock algokey can sign.

## Security

A matched key is a full-entropy valid Falcon key; vanity search only *selects*
among valid keys and does not weaken them. Never commit mnemonics/keys — the
repo's `.gitignore` excludes `hits/`, `out/`, and key files, and hit records
are written only to the `--out` directory you choose. Treat every `.pqhit`
file as a secret: it contains the mnemonic.

Seed entropy is mixed from multiple independent sources with SHA-512/256: the
OS CSPRNG (`getrandom`, mandatory), the CPU DRNG (`RDSEED` on x86-64 / `RNDR`
on aarch64), timing jitter, a TPM 2.0 (`/dev/tpmrm0`) when present, and
optional user input (`--extra-entropy <file|string>`) — the mix is
unpredictable as long as *any one* source is. Missing sources are skipped; the
live set is printed at startup and recorded in each hit file. The GPU re-draws
the cached sources (TPM, jitter) every batch and by default emits **at most
one hit per GPU batch**, because same-batch keys share 24 of their 32
base-entropy bytes (`--allow-sibling-hits` accepts them anyway — they are
full-strength keys, just not independent of each other). The per-batch
counter starts at a random offset and wraps mod 2^64, so every emitted
entropy is uniformly distributed — no zero-suffix pattern fingerprints the
generator. The CPU search re-draws the cached sources after every hit.

## License

Project code: **MIT** (see `LICENSE`). Vendored third-party code keeps its own
license: `vendor/falcon` (MIT, compiled and linked), `vendor/go-algorand-ref`
(AGPL-3.0, **reference only — never compiled or linked**). See
`vendor/NOTICE.md` for exact upstream pins.

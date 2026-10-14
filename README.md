# falcon-vanity-generator

GPU-accelerated **vanity address generator** for Algorand **native
post-quantum accounts** (Falcon signatures).

It searches over Falcon public keys for one whose derived PQ address starts
and/or ends with chosen base32 patterns — one target or up to 1024 at once —
then emits the account's **25-word mnemonic** (importable with
`algokey pq import -m`) along with the matching salt.

Today it implements scheme **`f1` (Deterministic Falcon-1024)**; the `pq-*`
crate/binary naming is deliberate — future Falcon schemes (`f2`, `f3`, …) will
live in the same tool. It also generates **classic Algorand (Ed25519)**
vanity addresses with `--scheme ed25519` — see [Classic accounts](#classic-accounts-ed25519).

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
crates/pq-core          byte-exact derivation + Falcon reference (CPU oracle), KAT-tested;
                        classic Ed25519 oracle (ed25519-dalek, RFC 8032 + Go differential)
crates/pq-cuda          CUDA kernel build + safe Rust wrapper (host stub without nvcc)
crates/pq-vanity        CLI: derive / search / bench / gpu / gpu-selftest
cuda/                   the kernels (pq_vanity.cu, ed25519_vanity.cu) + kernel notes
vendor/falcon           Falcon det1024 C reference (MIT), pinned, compiled & linked
vendor/ed25519-cuda     orlp/ed25519 CUDA port from solana-perf-libs (zlib/Apache-2.0), pinned, unmodified
vendor/go-algorand-ref  PQ-account spec + KATs (AGPL), pinned, reference only
tools/offcurve-oracle   Go ground-truth for the off-curve predicate (dev only)
tools/classic-oracle    Go ground-truth for classic seed → pubkey → address (dev only)
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

### GPU memory

Memory use depends on the GPU's size, not on the search: seeds are derived
on-device, the hit buffer is ~135 KB and the target table a fixed 32 KB, so
`--items`, the pattern, and the number of targets make no difference. Almost
all of it is per-thread stack, which CUDA reserves for every thread that can
be resident at once (1,536 per SM):

```
memory ≈ 94 MiB (CUDA context) + stack_per_thread × SMs × 1536
  ed25519 search kernel: 2,960 B/thread → ~4.3 MiB per SM
  f1 search kernel:      8,304 B/thread → ~12.2 MiB per SM
```

| GPU | SMs | ed25519 | f1 |
|---|---|---|---|
| GeForce RTX 5070 (measured) | 48 | 302 MiB | 678 MiB |
| GB10 (DGX Spark) | 48 | ~0.3 GB | ~0.7 GB |
| RTX PRO 4500 Blackwell | 82 | ~0.45 GB | ~1.1 GB |
| RTX PRO 6000 Blackwell | 188 | ~0.9 GB | ~2.4 GB |

Rows other than the RTX 5070 are computed from the formula. On a GPU shared
with another process, keep that much **free** memory: a run that cannot get
it fails with `CUDA runtime error (code -2)` (`cudaErrorMemoryAllocation`).
For example, a vLLM server claims a fixed share of the GPU at startup
(`--gpu-memory-utilization`, often 0.9 or more) and holds it while idle;
`nvidia-smi` shows what is free.

## Usage

```bash
# Search on the GPU (the host completes + verifies each hit, emits the key):
pq-vanity gpu --prefix ALGO --count 1 --out ./hits

# Match the end too, or several targets at once (see below):
pq-vanity gpu --prefix ALGO --suffix XYZA                # ALGO…XYZA
pq-vanity gpu --pattern ALGO-XYZA,BEEF..CAFE,-QE         # explicit pairs
pq-vanity gpu --prefix ALGO,BEEF,CAFE                    # any of these, one hit each

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

### Prefixes, suffixes, and multi-target search

Addresses are usually shown as `ABCD…WXYZ`, so both ends can be targeted.
All three forms work on `gpu` and `search`, for both schemes:

```bash
pq-vanity gpu --prefix ALGO                      # one prefix
pq-vanity gpu --suffix XYZA                      # one suffix
pq-vanity gpu --prefix ALGO,BEEF,CAFE            # any of these prefixes
pq-vanity gpu --prefix AB,CD --suffix QE,4A      # every prefix × suffix (4 targets)
pq-vanity gpu --pattern ABCD-WXYA,EFGH..QRSE     # explicit prefix+suffix pairs
pq-vanity gpu --pattern ALGO-,-XYZA,AB-QE        # either side may be empty
```

| Option | Meaning |
|---|---|
| `--prefix P1,P2,…` | address starts with any listed prefix |
| `--suffix S1,S2,…` | address ends with any listed suffix |
| `--prefix … --suffix …` | both ends: every prefix × suffix combination is a target |
| `--pattern P-S,P..S,…` | explicit prefix+suffix pairs; `-` or `..` separates, either side may be empty (`ALGO-`, `-XYZA`); cannot be combined with `--prefix`/`--suffix` |
| `--count N` | stop after `N` hits (default: 1, or one per target in a multi-target search) |
| `--keep-found` | multi-target: keep searching targets that already have a hit |

Patterns are case-insensitive. Each hit prints which target it `matched:`.

- A comma anywhere makes it a **multi-target** search. All targets are matched
  in the same pass over the keys, so finding one hit for each of `M` targets
  takes about `T·(1 + 1/2 + … + 1/M)` instead of `M·T` searched one after another
  (≈2.9× instead of 10× for 10 targets).
  Each target is retired after its first hit and the search ends when all are
  found; `--count N` stops earlier, `--keep-found` keeps found targets active.
  Up to 1024 targets (after prefix × suffix expansion).
- **The last character of an address is always one of `AEIMQUY4`.** The 36
  bytes of `addr32 || checksum` fill 58 base32 characters with 2 padding bits,
  so the final character carries only 3 data bits. Suffixes ending in any other
  character are rejected; valid ones are 4× cheaper than a prefix of the same
  length (`32^(S-1)·8` keys).
- A suffix lies in the checksum (`SHA512_256(addr32)`), so matching it needs
  one extra hash. Kernels and CPU matchers compute it **only when a suffix
  target's prefix already matched**: prefix-only searches never hash, and a
  prefix+suffix pair hashes ~1 in `32^P` keys. Measured on an RTX 5070
  (ed25519): prefix-only 48.1 M/s, prefix+suffix 47.9 M/s, suffix-only
  46.1 M/s, 100 prefixes 46.8 M/s.
- GPU limits: prefixes up to 12 characters, suffixes up to 13 — far beyond
  anything feasible to find.
- The one-hit-per-batch entropy rule applies across all targets: two targets
  never get keys from the same batch.

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

## Classic accounts (Ed25519)

`--scheme ed25519` (alias `classic`) on `gpu`, `search`, `derive`, `bench` and
`gpu-selftest` searches ordinary Algorand accounts. A classic address *is* the
Ed25519 public key, so there is no salt and no off-curve rule; the 32-byte
seed is the mnemonic key.

```bash
pq-vanity gpu    --scheme ed25519 --prefix ALGO --out ./hits
pq-vanity gpu    --scheme ed25519 --pattern ALGO-XYZA,BEEF-     # pairs / multi-target
pq-vanity search --scheme ed25519 --prefix ALGO --suffix XYZA   # CPU baseline
pq-vanity gpu-selftest --scheme ed25519 --items 200000
pq-vanity derive --scheme ed25519 --mnemonic "25 words ..."

goal account import -m "<25-word mnemonic>"       # or: algokey import -m ... -f key
```

Hits are written to `<ADDRESS>.edhit` (seed + mnemonic — secret). The same
entropy hygiene applies as for `f1` (multi-source mix, random wrapping
counter, one hit per batch); the CPU search also scans in 2^15-key runs per
mix output, emitting at most one key per run, because per-key mixing would
otherwise dominate the ~25 µs/key/core keygen cost.

Device keygen is the orlp/ed25519 CUDA port from solana-perf-libs, vendored
unmodified; `cuda/ed25519_vanity.cu` adds a variable-time base-table lookup
and a Montgomery batch inversion (8 keys per thread share one field
inversion; +51% vs. one key per thread). Every device hit is re-derived with
`ed25519-dalek` and sign/verify round-tripped before it is reported;
`gpu-selftest` checks both device paths bit-for-bit plus the search kernel
end to end. Addresses from hits were confirmed with stock `algokey import`.

| Device | keys/s | est. 6-char hit | est. 7-char hit |
|---|---|---|---|
| NVIDIA GeForce RTX 5070 (`gpu`) | 48 M | ~22 s | ~12 min |
| CPU (i5-12600KF, 15 threads, `search`) | 0.6 M | ~30 min | ~16 h |

### Comparison with rsagg

[rsagg](https://github.com/dragmz/rsagg) is an OpenCL classic-account vanity
generator. Same GPU (RTX 5070), same 10-char prefix (no hits, so pure
keygen + match throughput), rsagg at commit `3ffe417`:

| Tool | Command | keys/s |
|---|---|---|
| pq-vanity | `gpu --scheme ed25519 --prefix ZZZZZZZZZZ --batches 12` | **45.9 M** (48 M on an idle host) |
| rsagg | `generate --benchmark --batch 4194304 ZZZZZZZZZZ` | 27.2 M |
| rsagg | `generate --benchmark --batch 1611936 ZZZZZZZZZZ` | 25.8 M |

pq-vanity is **~1.7× faster**. Both compute the same seed → SHA-512 →
fixed-base scalar multiplication; the difference is in what surrounds it:

- **Inversion:** one field inversion is shared by 8 keys (Montgomery batch
  trick); rsagg inverts once per key.
- **Base-point table:** a direct lookup per window; rsagg uses the
  constant-time select that scans all 8 entries.
- **Matching:** prefixes are compared as bit masks on the raw key and the
  checksum is never computed for prefix-only targets; rsagg base32-encodes the
  full 54-character address for every key.
- **Seeds:** derived on-device from one mixed base entropy per batch; rsagg
  generates every seed on the host and copies it to the GPU each batch.
- **Startup:** the kernel is compiled at build time; rsagg's OpenCL kernel is
  JIT-compiled at first run (~90 s on this machine).

## Benchmarks

The tables below are for `f1` (post-quantum) keys.

Measured with the commands shown below on four GPUs (single GPU each) and six
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
| NVIDIA RTX PRO 6000 Blackwell (Max-Q) | 2.42 M | ~126 k | 22 in 6.9 s | ~4.5 min |
| NVIDIA RTX PRO 4500 Blackwell | 1.31 M | ~68 k | 14 in 12.8 s | ~8 min |
| NVIDIA GeForce RTX 5070 | 0.86 M | ~45 k | 36 in 19.5 s | ~13 min |
| NVIDIA GB10 (DGX Spark) | 0.79 M | ~41 k | 22 in 21.3 s | ~14 min |
| CPU (Ryzen 9 9950X, 31 threads, `search`) | 806 | 806 | — | ~12 h |
| CPU (Xeon Gold 5412U, 47 threads, `search`) | 691 | 691 | — | ~13 h |
| CPU (Ryzen 9 7950X, 31 threads, `search`) | 624 | 624 | — | ~15 h |
| CPU (GB10 20-core Arm, 19 threads, `search`) | 563 | 563 | — | ~17 h |
| CPU (i5-12600KF, 15 threads, `search`) | 331 | 331 | — | ~28 h |
| CPU (i7-10700, 15 threads, `search`) | 247 | 247 | — | ~38 h |

### Non-canonical mode (`--allow-non-canonical`)

Up to 256 addresses per usable key (salts swept on-device); hits at any
compliant salt. Expected addresses examined per usable hit ≈ `32^(L+1)`.

`pq-vanity gpu --prefix ALGO --allow-non-canonical --batches 3 --count 9999 --allow-sibling-hits`

| Device | keys/s | addresses/s | measured `ALGO` (4-char) hits |
|---|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell (Max-Q) | 1.03 M | 263 M | 80 in 12.3 s |
| NVIDIA RTX PRO 4500 Blackwell | 0.57 M | 145 M | 85 in 22.2 s |
| NVIDIA GeForce RTX 5070 | 0.37 M | 94.2 M | 79 in 34.2 s |
| NVIDIA GB10 (DGX Spark) | 0.30 M | 76.8 M | 74 in 41.9 s |
| CPU (Ryzen 9 9950X, 31 threads, `search`) | ~806 | ~206 k | — |
| CPU (Xeon Gold 5412U, 47 threads, `search`) | ~691 | ~177 k | — |
| CPU (Ryzen 9 7950X, 31 threads, `search`) | ~624 | ~160 k | — |
| CPU (GB10 20-core Arm, 19 threads, `search`) | ~563 | ~144 k | — |
| CPU (i5-12600KF, 15 threads, `search`) | ~331 | ~85 k | — |
| CPU (i7-10700, 15 threads, `search`) | ~247 | ~63 k | — |

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
license: `vendor/falcon` (MIT, compiled and linked), `vendor/ed25519-cuda`
(zlib + Apache-2.0, compiled and linked), `vendor/go-algorand-ref`
(AGPL-3.0, **reference only — never compiled or linked**). See
`vendor/NOTICE.md` for exact upstream pins.

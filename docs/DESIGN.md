# pq-vanity — Design & Source-of-Truth

GPU-accelerated vanity address generator for **Algorand native post-quantum
accounts** (`pq` sig type, scheme `f1` = Deterministic Falcon-1024). Find a
Falcon-1024 keypair whose derived PQ address matches a chosen base32
prefix/suffix.

This document is the authoritative spec for the project. It records the
**byte-exact** derivation (which is consensus/hash critical), the pinned
upstream commits it was reproduced from, and the architecture. Read this
before changing anything in `crates/pq-core`.

---

## 1. Target hardware

Default build targets:

- **`sm_120`** (e.g. RTX 50-series, RTX PRO 6000 Blackwell)
- **`sm_121`** (e.g. GB10 / DGX Spark)

Blackwell is ~1:64 FP64, so the **GPU hot loop is integer + FP32 only — no
FP64 anywhere**. Falcon's FP64 dependencies (the keygen orthogonalized-norm
check and the NTRU solve) stay on the **CPU** (run once per hit). Other
architectures can be added via `PQ_CUDA_ARCHS` — nothing in the kernel is
Blackwell-specific. The CUDA backend is a feature-gated, separately-buildable
component; the CPU path is fully functional and tested standalone on machines
without a GPU.

---

## 2. Pinned upstream sources

| What | Repo | Commit | Used for |
|---|---|---|---|
| PQ account derivation (spec + KATs) | [`algorand/go-algorand`](https://github.com/algorand/go-algorand) master (PR [#6639](https://github.com/algorand/go-algorand/pull/6639) **merged** 2026-07-10) | `551fd57a5f91678d8f592305b16d00efb0743b42` | Reference only (AGPL). Vendored under `vendor/go-algorand-ref/`. |
| Falcon `det1024` C reference | [`algorand/falcon`](https://github.com/algorand/falcon) | `ce15e75bceb372867daf6b8e81918ab6978686eb` = tag **`v0.1.0`** (the release go-algorand's go.mod pins) | Compiled + linked (MIT). Vendored under `vendor/falcon/`. |

> The PR is merged to master for the upcoming consensus release; the gate is
> the `PQSchemeEnabled` consensus parameter. Before relying on generated keys
> in production, re-diff `vendor/go-algorand-ref/` against current go-algorand
> master and re-run the KAT tests — a derivation change upstream would change
> every address.

---

## 3. Byte-exact derivation (consensus-critical)

All of the following is reproduced in `crates/pq-core/src/address.rs` (and
`mnemonic.rs`) and locked down by tests against the upstream known-answer
vectors.

### 3.0 Key derivation (entropy → seed → keypair)

```
entropy = 32 random bytes            // crypto.Seed; ↔ 25-word Algorand mnemonic
seed32  = SHA512_256( "PQK" || "f1" || entropy )   // derivePQKeySeed
(pk,sk) = falcon_det1024_keygen( SHAKE256-PRNG(seed32) )   // FalconSeedSize = 32
```

- `"PQK"` — `protocol.PostQuantumKey` HashID (`protocol/hash.go`).
- `derivePQKeySeed` — `cmd/algokey/pq_scheme.go`; the entropy is what
  `algokey pq import -m` consumes as a 25-word mnemonic
  (`crypto/passphrase.KeyToMnemonic`, ported in `pq-core::mnemonic` and locked
  by the zero-key KAT + the "venue" wordlist checksum).
- Key files written by algokey hold the derived keys and the **canonical salt**;
  they never store the entropy.

### 3.1 Address hash

```
addr32 = SHA512_256( "PQA" || "f1" || salt || pk )
```

- `"PQA"` — `protocol.PostQuantumAddress` HashID domain separator
  (`protocol/hash.go`). Prepended by `crypto.HashObj`.
- `"f1"` — `protocol.PQSchemeFalcon1024` (`protocol/pq_scheme.go`,
  `PQScheme [2]byte`).
- `salt` — **1 byte** (`pqAddressSaltSize = 1`, `PQAddressSalt uint8`).
- `pk` — the **1793-byte** Falcon-1024 public key (`FalconPublicKeySize =
  cfalcon.PublicKeySize = FALCON_PUBKEY_SIZE(10) = 1793`).
- `SHA512_256` is `crypto.Hash` (Go `sha512.Sum512_256`).

The preimage layout is locked by `TestPQAddressPreimage`:
`pqAddressPreimage{f1, 0x7f, {ab,cd,ef}}.ToBeHashed()` →
HashID `"PQA"`, payload `f1 7f ab cd ef`.

### 3.2 Display string (`basics.Address.String`)

```
display = base32_nopad_std( addr32 || checksum4 )      // 58 ASCII chars
checksum4 = last 4 bytes of SHA512_256(addr32)
```

`base32_nopad_std` is RFC-4648 base32 (`ABCDEFGHIJKLMNOPQRSTUVWXYZ234567`),
no padding (`data_encoding::BASE32_NOPAD`).

### 3.3 Off-curve compliance (`addr.IsPQCompliant`)

```
compliant(addr32) = !IsEdwards25519Point(addr32)
```

`IsEdwards25519Point` is `filippo.io/edwards25519`'s `Point.SetBytes` succeeding
— the **broad** acceptance set (non-canonical encodings accepted, **no**
prime-order-subgroup check). We use `curve25519-dalek`'s `decompress`, which has
been **differential-tested to agree on 300k+ random vectors plus the full
non-canonical `y ≥ p` band and the `x = 0, sign = 1` case** (0 mismatches) — see
`tools/offcurve-oracle/` + `crates/pq-core/tests/offcurve_differential.rs`.
~50% of random 32-byte strings are off-curve (compliant), matching upstream.

### 3.4 Known-answer tests (ground truth)

Falcon seed = `crypto.FalconSeed` (**32** bytes), `seed[0] = firstSeedByte`,
rest 0 (raw seed, no "PQK" step — mirrors upstream `pq_address_test.go`):

| firstSeedByte | salt | address | compliant |
|---|---|---|---|
| 3 | 0 | `KJGJA2DTCQH6LT2I2OH2YO5GIIBFC6JHX5O6UPA5ZZ5ZURFT3LHKMTRCEM` | yes |
| 1 | 1 | `GYBWVYVQIQF6CO7BUMG4UQ66DQYHASFOCA2P7PBYOIPKGWUZIBX4KA3TP4` | yes |
| 0 | 255 | `YJFADDEP6Z3WAWY6ZMLN6MF4T4NK3BXKCVLPCYB6C4SQHE76LLQSZ5JG7Q` | yes |
| 2 | 2 | `II4DO6IIP3EAEQMWJEOLOUU3VBRVCH3WF4MX6UCRUD36DOQJ3YSHA2DV5A` | yes |
| 255 | 255 | `3JXWI6BYYEO6WO6M7TC4SOZAZUWAD4RQO5GJ2ED6MYIEVVLOVJOETMGG4A` | yes |
| 1 | 0 | `FLX4VRWXQ65HD5G5BI2EPHJWMERHA2EBBQ7XMTZLATXH4XEOWPQSIYVIF4` | no |

End-to-end (mnemonic entropy `{1,2,…,32}` → "PQK" seed → keygen → canonical
salt): `ZEJ4BLG3XWAUUZQGCEDJLYIC6D2NCWHRSX5DJMDPE54PXXR7G3PCQTARXU`
(`cmd/algokey/pq_test.go` TestPQGenerate).

All reproduced **exactly** by `pq-core` (`address::tests`).

---

## 4. Salt semantics — why we can grind keys and emit an explicit salt

`PQSig` (`data/transactions/pqsig.go`) carries an **explicit 1-byte `Salt`**
(`codec:"slt"`). `AuthorizerAddress()` derives the address from the explicit
`(scheme, salt, pk)`. Consensus `PQSig.Verify` accepts **any** salt whose
derived address equals the authorizer — there is **no canonical-salt enforcement
at consensus**, and off-curve compliance is **not** a consensus rule.

`basics.CanonicalPQAddressSalt` (lowest compliant salt) is what merged
`algokey pq` tooling uses **exclusively** — key files store the canonical salt
and `sign` has no salt override. Any compliant explicit salt remains valid at
consensus (it travels in the `PQSig`), but needs custom signing tooling.

**Consequences for the search:**
- We grind over **Falcon public keys** (salt's 1 byte / ~128 compliant values is
  far too little diversity for a vanity prefix).
- For each candidate key a match may exist at **any** salt `0..=255` whose
  address is **off-curve compliant**. We emit the entropy (mnemonic) plus that
  **explicit salt**.
- **Tooling caveat (as merged):** `algokey pq` has **no `--salt` flag** — it
  always signs as the *canonical* (lowest compliant) salt stored in the key
  file. `search` and `gpu` therefore match **only the canonical salt by
  default**, so every hit is directly usable with stock algokey; on the GPU
  this also shortens the device salt sweep to `0..=31` (canonical salt is >31
  with probability ~2^-32), skipping raw hits the host would filter anyway.
  A non-canonical-salt hit is consensus-valid (the salt travels in the `PQSig`
  envelope) but needs custom signing tooling; `--allow-non-canonical` accepts
  those too (~128× more candidate addresses per key — ~128 compliant salts vs
  one canonical). The hit report always shows both the matched and canonical
  salts.

---

## 5. Falcon `det1024` facts (from `vendor/falcon`)

- Public key: **1793 B**; private key: **2305 B**; compressed sig max: 1423 B.
- Keygen: `shake256_init_prng_from_seed(seed, len)` → `falcon_det1024_keygen(rng,
  sk, pk)` (= `falcon_keygen_make(rng, logn=10, …)`). This is exactly what
  `cfalcon.GenerateKey(seed)` does.
- `config.h` pins determinism: `FALCON_FPEMU=1`, `FALCON_FPNATIVE=0`,
  `FALCON_AVX2=0`, `FALCON_FMA=0`, **`FALCON_KG_CHACHA20` undefined** → keygen
  samples `(f,g)` **directly from a SHAKE256 stream** seeded by the seed. (If
  CHACHA20 were enabled the KATs would change — do not enable it.)
- Inside `Zf(keygen)` (`keygen.c`): sample `f,g` via `poly_small_mkgauss` /
  `mkgauss` using the `gauss_1024_12289[]` distribution table; reject if `f` is
  not invertible mod `q` or the norm/orthogonality bound fails; `solve_NTRU` to
  get `(F,G)`; encode. `h = g·f⁻¹ mod q` is the public key.

---

## 6. Architecture

The address depends **only** on `pk = encode(h)`, with `h = g·f⁻¹ mod q`
(`q = 12289`). Computing `h` does **not** need the expensive NTRU solve. So:

### GPU hot loop (per work-item) — `sm_120` / `sm_121a`, integer-only
1. `entropy = f(base_entropy, global_id, counter)`;
   `seed = SHA512_256("PQK" || "f1" || entropy)`
2. SHAKE256 → sample `(f, g)` (Falcon discrete Gaussian, n=1024) — **the same
   `mkgauss` + `gauss_1024_12289` sampler the reference uses**, reading the
   SHAKE256 stream.
3. NTT mod 12289; skip if `f` not invertible; `h = g · f⁻¹`.
4. encode `pk` (byte-exact, `codec.c modq_encode` layout).
5. `preimage = "PQA" || "f1" || salt || pk`; for `salt` in target range:
   `SHA512_256` → `addr32` → base32 leading chars → compare to target.
6. on match: atomically append `(entropy, salt)` to an output buffer.

### CPU host (Rust)
- batch launcher, seed-space management, hit collection (`crates/pq-vanity`).
- per hit: recompute `seed` from the entropy ("PQK" hash), re-derive `(f,g)`
  with the **same** sampler → run `solve_NTRU` + norm check (the
  `complete_from_fg` shim over `keygen.c`) → assemble the private key →
  **verify round-trip** (sign/verify, det mode) → recompute the address via the
  vendored derivation → off-curve check → emit the hit record (address, salt,
  canonical salt/address, entropy hex + 25-word mnemonic).

### Consistency guarantee
`h` depends only on `(f,g)`. On a hit the CPU re-derives the **same** `(f,g)`
from the entropy and runs `solve_NTRU` → identical `h` → identical address. So the
GPU sampler need not reproduce the reference *keygen*'s multi-round rejection
sequencing (which would require running `solve_NTRU` on the GPU — infeasible and
FP-heavy). It only must (a) sample a correct Falcon discrete Gaussian
deterministically from the seed, identically to the CPU, and (b) match the `pk`
encoding byte-for-byte. Rare `(f,g)` that fail norm/NTRU-solvability are simply
discarded at CPU-completion time.

### Off-curve handling
- **v1**: off-curve check on the CPU at hits (≈2× more raw matches needed since
  ~half of salts are compliant; cheap because there are few hits).
- **v2**: port Ed25519 decompress (sqrt mod 2²⁵⁵−19) into the kernel to drop the
  2× and avoid emitting non-compliant candidates.

---

## 7. Implementation & performance notes

**Components** (each validated as described in §9):

- Fast-path shim (`vendor-shim/pqv_shim.c`, exposed via `pq-core::fast`):
  `sample_fg` (seed → first-attempt (f,g)), `pubkey_from_fg` (the public half
  the GPU mirrors), `complete_from_fg` (single-attempt acceptance +
  `solve_NTRU`). Validated bit-exact against the reference **both directions**
  over 4000 seeds: whenever the reference keygen uses its first attempt, our
  completed key equals it (pk+sk); otherwise we reject it too.
- CUDA kernel: SHAKE256 + sampler + NTT mod 12289 + pk-encode + SHA-512/256 +
  base32 prefix match. `gpu-selftest` confirms device `(f,g)`, `pk`, `addr32`,
  and the retry/bnorm predicate == oracle (200k / 2000 seeds);
  `compute-sanitizer` clean.
- Host batching + hit completion: `pq-vanity gpu` loops device batches,
  re-derives each raw hit on the CPU (all cores in parallel), checks
  prefix + compliance (+ canonical salt in canonical mode), round-trips the
  key, and emits the hit record + mnemonic.

**Performance findings** (thread-per-key kernel; GPU models noted because the
numbers are device-relative):

- The per-key cost is **dominated by the sampler (+NTT)**; each extra salt is
  a near-free SHA-512/256. Non-canonical mode therefore sweeps all 256 salts
  to amortize the expensive part; canonical mode sweeps `0..=31` (the
  canonical salt is >31 with probability ~2⁻³²).
- **Registerized Keccak + SHA-512: ~5×.** A rolled `keccakf` keeps its 25-lane
  state in runtime-indexed local memory, which made the SHAKE256 sampler 92%
  of kernel time. Fully unrolling with `constexpr` index tables (state →
  registers), specializing the sampler shake (absorb exactly one 32-byte seed,
  squeeze LE u64s via a 17-lane pool), and giving `sha512_compress` the
  statically-indexed rolling 16-word schedule took an RTX 5070 from 0.21 →
  1.09 Mkeys/s (canonical mode) and 0.15 → 0.40 Mkeys/s (256 salts),
  bit-exact.
- **In-seed retry: tried, measured, rejected.** Mirroring the reference
  keygen's resampling on-device (integer checks exact; the orthogonalized
  bnorm as an FP32 forward-FFT filter via Parseval,
  `(q²/n)·Σ 1/(|f̂|²+|ĝ|²)` — validated against the FPEMU classifier, 0
  borderline in 2000 seeds) would make ~every seed yield the exact
  `algokey`-derivable key. But with p≈5.9% per-attempt visible acceptance, a
  thread-per-key warp waits for its slowest lane: E[max of 32 geometric(p)] ≈
  67–80 attempts vs E=17 mean — measured **3× worse per usable key** than
  first-sample scanning. Fresh seeds are free, so scanning wins. The FP32
  bnorm, retry loop, and CPU oracle (`pqv_first_visible_accept`) stay
  in-tree, selftest-validated, as groundwork for a cooperative kernel (where
  per-lane tails don't lockstep-stall). First-sample stage rates (4000
  seeds): sqnorm rejects 48.8%, bnorm 44.7%, invertibility 0.6%, solve 0.65%;
  P(accept | GPU-visible checks pass) = 89%; overall acceptance ≈ 5.2%.
- Things that did NOT help (measured): `__launch_bounds__`/register capping
  (the sampler/NTT want the registers for ILP); block-size tuning (within
  noise; `PQ_GPU_BLOCK`); a squared-norm pre-filter to skip keygen-doomed
  `(f,g)` — **warp lockstep** means skipping lanes doesn't shorten the warp.
- Remaining backlog (the sampler+NTT is the wall — the kernel is
  latency-bound at ~25% occupancy with a large per-thread local working set):
  a **cooperative warp-per-key kernel with the NTT in shared memory** (biggest
  lever; also unlocks the retry/compaction wins above, theoretical ~1.6× floor
  over first-sample scanning); in-kernel off-curve; multi-GPU;
  suffix/contains patterns.

> nvcc 13 `-O3` miscompiled a `for(;;)`+`continue`/`break` rejection-sampling
> loop (stored one element past the buffer); use `do { } while(...)` in device
> sampling loops. Caught by `compute-sanitizer` + the bit-exact selftest.

The CPU `search` (`crates/pq-vanity`) remains a **correct, slow baseline** and
the bit-exact oracle for the kernel. The GPU `gpu` path is the throughput path.

---

## 8. Feasibility — MEASURED

Current single-GPU throughput is benchmarked in the README (canonical and
non-canonical tables, three GPUs + a CPU baseline). Headline: an RTX PRO 6000
Blackwell (Max-Q) does **2.40 Mkeys/s ≈ 125k algokey-ready candidates/s** in
canonical mode and **263 M addr/s** with `--allow-non-canonical`.

Expected search cost, base32 alphabet = 32 symbols → 5 bits/char:

- **Canonical mode:** one candidate address per usable key → expected
  `32^L / candidates_per_second` for an `L`-char prefix.
- **Non-canonical mode:** empirically ~`32^(L+1)` addresses per usable hit
  (prefix match `32^-L` × keygen completion ≈5% × off-curve ≈50% ≈
  `1/32^(L+1)`).

Expected time to one hit on the RTX PRO 6000 (Max-Q):

| prefix chars | canonical (125k cand/s) | non-canonical (263 M addr/s) |
|---|---|---|
| 4 | ~8 s | < 1 s |
| 5 | ~4.5 min | ~4 s |
| 6 | ~2.4 h | ~2 min |
| 7 | ~3.2 days | ~1.2 h |
| 8 | ~100 days | ~1.5 days |

Scales ~linearly across GPUs (multi-GPU not yet wired). Throughput is
bottlenecked by the per-key sampler+NTT (see §7) — a cooperative warp-per-key
kernel is the path to higher rates.

---

## 9. Verification

- **Automated (run `cargo test`):** KAT addresses (merged vectors incl. the
  end-to-end entropy→canonical-address KAT), "PQK" preimage layout, canonical
  salt, sign/verify round-trip, pubkey-coefficient range, mnemonic codec
  (zero-key KAT, wordlist checksum, round-trips).
- **Off-curve differential (run manually, needs Go):**
  `cargo test -p pq-core --test offcurve_differential -- --ignored --nocapture`.
- **Not yet automated — `algokey pq` interop (manual):** build `algokey` from
  master (cgo) and confirm `algokey pq import -m "<our mnemonic>"` re-derives
  the same key and canonical address. The derivation is reproduced from
  `cmd/algokey/pq_scheme.go` + `crypto/passphrase` and locked by the vendored
  KATs, but is not yet exercised against the real binary in CI.

---

## 10. Security & licensing

- A matched key is a full-entropy valid Falcon key — vanity only **selects**
  among valid keys; it does not weaken them. Entropy must be CSPRNG-quality.
  Matched private keys must be stored securely and never committed (see
  `.gitignore`).
- `vendor/falcon/` is **MIT** (Falcon Project / Algorand) and is compiled+linked.
- `vendor/go-algorand-ref/` is **AGPL-3.0** (Algorand Foundation), included
  **unmodified, with headers, as pinned reference material** for spec-tracking
  and KAT validation. It is **not compiled or linked** into any binary. The Rust
  code reproduces the *algorithm/byte-format* (an interface/spec), not upstream
  expression. Get legal review before redistribution beyond research use.

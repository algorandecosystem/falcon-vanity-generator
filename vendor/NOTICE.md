# Vendored third-party sources

Both trees are pinned to exact upstream commits for byte-exact reproducibility.
Re-fetch with the commands below and diff to verify integrity.

## vendor/falcon — Falcon `det1024` C reference

- Upstream: https://github.com/algorand/falcon
- Pinned commit: `ce15e75bceb372867daf6b8e81918ab6978686eb` (see `falcon/PINNED_COMMIT`)
- License: **MIT** (Falcon Project / Algorand) — see `falcon/LICENSE`.
- Status: **compiled and linked** into `pq-core` via `crates/pq-core/build.rs`.
  Determinism knobs are pinned in `falcon/config.h` (`FALCON_FPEMU=1`,
  `FALCON_AVX2=0`, `FALCON_FMA=0`, `FALCON_KG_CHACHA20` undefined). Do not change
  them — they are what produce the KAT-matching keys.

```bash
git clone https://github.com/algorand/falcon && cd falcon
git checkout ce15e75bceb372867daf6b8e81918ab6978686eb
```

## vendor/go-algorand-ref — native PQ account derivation (spec + KATs)

- Upstream: https://github.com/algorand/go-algorand (PR #6639 **merged to
  master** 2026-07-10, merge commit `569ae3d4`)
- Pinned commit: `551fd57a5f91678d8f592305b16d00efb0743b42` (master at vendoring)
- License: **AGPL-3.0** (Algorand Foundation) — headers retained in each file.
- Status: **reference only**. These files are *not* compiled or linked into any
  binary. They are the source of truth for the key/address derivation and the
  known-answer tests reproduced in `crates/pq-core`. Only the subset of files
  needed to pin the derivation is vendored. `crypto/passphrase/*` pins the
  25-word mnemonic codec (`KeyToMnemonic`) ported to `pq-core::mnemonic`.

```bash
SHA=551fd57a5f91678d8f592305b16d00efb0743b42
for f in protocol/hash.go protocol/pq_scheme.go \
         crypto/pq_scheme.go crypto/falconWrapper.go crypto/curve25519.go \
         crypto/passphrase/passphrase.go crypto/passphrase/wordlist.go \
         crypto/passphrase/errors.go \
         data/basics/pq_address.go data/basics/pq_address_test.go \
         data/basics/address.go data/transactions/pqsig.go \
         cmd/algokey/pq.go cmd/algokey/pq_key.go cmd/algokey/pq_scheme.go \
         cmd/algokey/pq_test.go cmd/algokey/common.go ; do
  curl -sSL "https://raw.githubusercontent.com/algorand/go-algorand/$SHA/$f" \
    --create-dirs -o "vendor/go-algorand-ref/$f"
done
```

Derivation pipeline as merged (differs from the earlier PR draft):

```text
entropy[32]  — random; encodes to the standard 25-word Algorand mnemonic
seed[32]     = SHA512_256("PQK" || "f1" || entropy)          (protocol.PostQuantumKey)
(pk, sk)     = falcon_det1024_keygen(SHAKE256-PRNG(seed))    (FalconSeedSize = 32, was 48)
addr32       = SHA512_256("PQA" || "f1" || salt || pk)       (unchanged)
canonical    = lowest salt in 0..=255 with !IsEdwards25519Point(addr32)
```

`algokey pq` now imports from the mnemonic only (`import -m`); the old armored
key-file format and the `--salt` flags are gone — tooling always signs with the
canonical salt stored in the key file (consensus still accepts any salt carried
in a `PQSig`).

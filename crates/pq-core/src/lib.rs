//! `pq-core` — byte-exact derivation of Algorand native post-quantum (`f1`,
//! Deterministic Falcon-1024) account addresses, plus the vendored Falcon
//! reference keygen used as the CPU oracle.
//!
//! The consensus-critical pieces (entropy → "PQK" seed hash → keygen, address
//! derivation, off-curve compliance, key sizes) are reproduced from go-algorand
//! PR #6639 as merged to master and validated against its known-answer tests.
//! The Falcon keygen/sign/verify come from the vendored `algorand/falcon` C
//! reference (linked, MIT; pinned at the `v0.1.0` release go-algorand uses).
//! See `docs/DESIGN.md`.

pub mod address;
pub mod falcon;
pub mod fast;
pub mod mnemonic;

pub use address::{
    canonical_pq_salt, is_edwards25519_point, keygen_seed_from_entropy, pq_address,
    pq_address_preimage, sha512_256, Address, ENTROPY_SIZE, HASHID_PQ_ADDRESS, HASHID_PQ_KEY,
    SCHEME_F1,
};
pub use mnemonic::{key_to_mnemonic, mnemonic_to_key};
pub use falcon::{
    keygen_from_seed, pubkey_coeffs, verify_with_pubkey, FalconError, Keypair, N, PRIVKEY_SIZE,
    PUBKEY_SIZE, Q, SEED_SIZE, SIG_COMPRESSED_MAXSIZE,
};
pub use fast::{
    complete_from_fg, first_sample_stage, first_visible_accept, pubkey_from_fg, sample_fg,
    sample_fg_nth, Fg, SampleStage,
};

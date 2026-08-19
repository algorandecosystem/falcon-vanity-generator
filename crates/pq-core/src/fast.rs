//! Fast-path keygen primitives — the exact integer-only public
//! half the GPU reproduces, plus the CPU-side completion used at a hit.
//!
//! Bindings to `vendor-shim/pqv_shim.c`:
//! - [`sample_fg`]        — seed → (f,g), keygen's first Gaussian attempt.
//! - [`pubkey_from_fg`]    — (f,g) → pk (public half only; what the GPU mirrors).
//! - [`complete_from_fg`]  — (f,g) → full keypair (acceptance + NTRU solve).
//!
//! The address depends only on `h = g/f`, so [`pubkey_from_fg`] and the pk
//! inside [`complete_from_fg`] are byte-identical. A GPU hit gives a seed; the
//! host runs `complete_from_fg(sample_fg(seed))` to obtain (and verify) the key.

use crate::falcon::{Keypair, N, PRIVKEY_SIZE, PUBKEY_SIZE};
use core::ffi::{c_int, c_void};

extern "C" {
    fn pqv_sample_fg(seed: *const c_void, seed_len: usize, f: *mut i8, g: *mut i8) -> c_int;
    fn pqv_pubkey_from_fg(f: *const i8, g: *const i8, pk: *mut u8) -> c_int;
    fn pqv_complete_from_fg(f: *const i8, g: *const i8, pk: *mut u8, sk: *mut u8) -> c_int;
    fn pqv_first_sample_stage(f: *const i8, g: *const i8) -> c_int;
    fn pqv_sample_fg_nth(
        seed: *const c_void,
        seed_len: usize,
        n: u32,
        f: *mut i8,
        g: *mut i8,
    ) -> c_int;
    fn pqv_first_visible_accept(seed: *const c_void, seed_len: usize, max_attempts: u32) -> c_int;
}

/// The first attempt (0-based, SHAKE stream continuing) passing the
/// GPU-visible acceptance predicate (coef limit, sqnorm, bnorm, invertible;
/// NOT solve_NTRU). `None` if no attempt within `max_attempts` passes.
/// Oracle for the CUDA kernel's in-seed retry loop.
pub fn first_visible_accept(seed: &[u8], max_attempts: u32) -> Option<u32> {
    // SAFETY: seed ptr/len describe a valid slice.
    let r = unsafe {
        pqv_first_visible_accept(seed.as_ptr() as *const c_void, seed.len(), max_attempts)
    };
    u32::try_from(r).ok()
}

/// Which keygen acceptance stage rejects `(f, g)` (reference keygen order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStage {
    Accepted,
    CoefLimit,
    SqNorm,
    BNorm,
    NotInvertible,
    SolveFailed,
}

/// Classify one Gaussian attempt against the reference acceptance stages.
pub fn first_sample_stage(fg: &Fg) -> SampleStage {
    // SAFETY: f/g are N bytes each.
    let s = unsafe { pqv_first_sample_stage(fg.f.as_ptr(), fg.g.as_ptr()) };
    match s {
        0 => SampleStage::Accepted,
        1 => SampleStage::CoefLimit,
        2 => SampleStage::SqNorm,
        3 => SampleStage::BNorm,
        4 => SampleStage::NotInvertible,
        _ => SampleStage::SolveFailed,
    }
}

/// The `n`-th (0-based) Gaussian attempt for `seed`, with the SHAKE stream
/// continuing across attempts exactly as in the reference keygen loop.
/// `sample_fg_nth(seed, 0) == sample_fg(seed)`.
pub fn sample_fg_nth(seed: &[u8], n: u32) -> Fg {
    let mut f = [0i8; N];
    let mut g = [0i8; N];
    // SAFETY: f/g are N bytes each; seed ptr/len describe a valid slice.
    unsafe {
        pqv_sample_fg_nth(
            seed.as_ptr() as *const c_void,
            seed.len(),
            n,
            f.as_mut_ptr(),
            g.as_mut_ptr(),
        );
    }
    Fg { f, g }
}

/// The secret polynomials `(f, g)` sampled from a seed (Falcon discrete
/// Gaussian, n=1024), as small signed coefficients.
#[derive(Clone)]
pub struct Fg {
    pub f: [i8; N],
    pub g: [i8; N],
}

impl core::fmt::Debug for Fg {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.write_str("Fg(<redacted secret polynomials>)")
    }
}

/// Sample `(f, g)` from `seed` exactly as keygen's first attempt. Deterministic;
/// the GPU sampler must match this bit-for-bit.
pub fn sample_fg(seed: &[u8]) -> Fg {
    let mut f = [0i8; N];
    let mut g = [0i8; N];
    // SAFETY: f/g are N bytes each; seed ptr/len describe a valid slice.
    unsafe {
        pqv_sample_fg(
            seed.as_ptr() as *const c_void,
            seed.len(),
            f.as_mut_ptr(),
            g.as_mut_ptr(),
        );
    }
    Fg { f, g }
}

/// The 1793-byte public key for `(f, g)` (`h = g·f⁻¹ mod q`). `None` if `f` is
/// not invertible mod q. This is exactly what the GPU hot loop computes.
pub fn pubkey_from_fg(fg: &Fg) -> Option<Box<[u8; PUBKEY_SIZE]>> {
    let mut pk = Box::new([0u8; PUBKEY_SIZE]);
    // SAFETY: f/g are N bytes; pk is PUBKEY_SIZE.
    let ok = unsafe { pqv_pubkey_from_fg(fg.f.as_ptr(), fg.g.as_ptr(), pk.as_mut_ptr()) };
    (ok != 0).then_some(pk)
}

/// Complete `(f, g)` to a full keypair (runs the keygen acceptance checks +
/// `solve_NTRU`). `None` if `(f, g)` is rejected — the caller discards the hit.
pub fn complete_from_fg(fg: &Fg) -> Option<Keypair> {
    let mut public_key = Box::new([0u8; PUBKEY_SIZE]);
    let mut private_key = Box::new([0u8; PRIVKEY_SIZE]);
    // SAFETY: f/g are N bytes; pk/sk are the exact sizes the C side writes.
    let ok = unsafe {
        pqv_complete_from_fg(
            fg.f.as_ptr(),
            fg.g.as_ptr(),
            public_key.as_mut_ptr(),
            private_key.as_mut_ptr(),
        )
    };
    (ok != 0).then_some(Keypair {
        public_key,
        private_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::{keygen_from_seed, pubkey_coeffs, Q, SEED_SIZE};

    fn seed_for(i: u32) -> [u8; SEED_SIZE] {
        let mut s = [0u8; SEED_SIZE];
        s[..4].copy_from_slice(&i.to_le_bytes());
        s
    }

    /// Rigorous predicate validation, both directions, against the reference.
    ///
    /// The reference `keygen_from_seed` resamples until an attempt is accepted.
    /// Our `sample_fg` is exactly its FIRST attempt, and `pubkey_from_fg` is the
    /// pk of that first attempt. So:
    ///   - if the first-attempt pk equals the reference key's pk, the reference
    ///     ACCEPTED the first attempt ⇒ `complete_from_fg` MUST accept and yield
    ///     that exact (pk, sk)  [proves: correct, and not over-strict];
    ///   - otherwise the reference REJECTED the first attempt ⇒ `complete_from_fg`
    ///     MUST also reject it (return None)  [proves: not over-lenient].
    /// This pins our single-attempt accept/reject predicate to the reference's,
    /// in both directions, byte-for-byte on the produced key.
    #[test]
    fn single_attempt_predicate_matches_reference() {
        // Validated on 4000 seeds during development (both directions, 0
        // mismatches); kept at 500 here so `cargo test` stays fast in debug.
        const SEEDS: u32 = 500;
        let mut used_first = 0u32;
        for i in 0..SEEDS {
            let seed = seed_for(i);
            let fg = sample_fg(&seed);
            let ref_kp = keygen_from_seed(&seed).unwrap();
            let first_pk = pubkey_from_fg(&fg);
            let completed = complete_from_fg(&fg);

            let reference_used_first = first_pk.as_deref() == Some(ref_kp.public_key.as_ref());

            if reference_used_first {
                used_first += 1;
                let kp = completed.unwrap_or_else(|| {
                    panic!("seed {i}: reference used first attempt but complete rejected it")
                });
                assert_eq!(kp.public_key, ref_kp.public_key, "pk mismatch, seed {i}");
                assert_eq!(kp.private_key, ref_kp.private_key, "sk mismatch, seed {i}");
            } else {
                assert!(
                    completed.is_none(),
                    "seed {i}: reference rejected first attempt but complete accepted it"
                );
            }
        }
        assert!(
            used_first > 10,
            "only {used_first}/{SEEDS} seeds used the first attempt — sampler likely wrong"
        );
        eprintln!(
            "predicate matches reference on all {SEEDS} seeds ({used_first} used first attempt)"
        );
    }

    /// The public half (what the GPU computes) is byte-identical to the pk in
    /// the completed key, whenever completion succeeds.
    #[test]
    fn public_half_matches_completed_pk() {
        let mut checked = 0u32;
        for i in 0..500u32 {
            let fg = sample_fg(&seed_for(i));
            if let Some(kp) = complete_from_fg(&fg) {
                let pk_half = pubkey_from_fg(&fg).expect("f invertible since completion succeeded");
                assert_eq!(
                    pk_half.as_ref(),
                    kp.public_key.as_ref(),
                    "public-half pk != completed pk, seed {i}"
                );
                checked += 1;
            }
        }
        assert!(checked > 10, "too few completed keys to check ({checked})");
    }

    /// A completed fast key round-trips (sign/verify), and its decoded h is valid.
    #[test]
    fn completed_key_roundtrips() {
        for i in 0..400u32 {
            let fg = sample_fg(&seed_for(i));
            if let Some(kp) = complete_from_fg(&fg) {
                let h = pubkey_coeffs(&kp.public_key).unwrap();
                assert!(h.iter().all(|&c| c < Q));
                let sig = kp.sign(b"f1-fast-path").unwrap();
                kp.verify(b"f1-fast-path", &sig).unwrap();
                return; // one full round-trip is enough
            }
        }
        panic!("no accepted key in 400 seeds");
    }
}

//! Safe Rust bindings to the vendored Falcon `det1024` reference C.
//!
//! This is the **CPU oracle**: it produces byte-exact public/private keys from
//! a seed, exactly as `github.com/algorand/falcon` (`cfalcon.GenerateKey`) does,
//! which is what go-algorand's `crypto.GenerateFalconSigner` calls. Everything
//! the GPU fast-path computes is validated bit-for-bit against this.
//!
//! Sizes and the seed→key path are pinned from the vendored headers:
//! - `FALCON_DET1024_PUBKEY_SIZE`  = 1793
//! - `FALCON_DET1024_PRIVKEY_SIZE` = 2305
//! - `falcon_det1024_keygen(rng, sk, pk)` where `rng` is a SHAKE256 context
//!   seeded via `shake256_init_prng_from_seed(seed, seed_len)`.

use core::ffi::{c_int, c_void};

/// Falcon-1024 public key size in bytes (`FALCON_PUBKEY_SIZE(10)`).
pub const PUBKEY_SIZE: usize = 1793;
/// Falcon-1024 private key size in bytes (`FALCON_PRIVKEY_SIZE(10)`).
pub const PRIVKEY_SIZE: usize = 2305;
/// Max compressed-signature size for det1024 (`...MAXSIZE-40+1`).
pub const SIG_COMPRESSED_MAXSIZE: usize = 1423;
/// Ring degree.
pub const N: usize = 1024;
/// Falcon modulus.
pub const Q: u16 = 12289;
/// Seed length used by go-algorand's `crypto.FalconSeed` (`FalconSeedSize`).
/// 32 bytes since PR #6639 merged (SHA-512/256 digest of the PQK preimage);
/// the earlier draft used 48.
pub const SEED_SIZE: usize = 32;

/// Opaque SHAKE256 context: `typedef struct { uint64_t opaque_contents[26]; }`
/// (208 bytes). We never inspect its fields, only hand a pointer to the C side.
#[repr(C)]
struct Shake256Context {
    opaque_contents: [u64; 26],
}

impl Shake256Context {
    #[inline]
    fn zeroed() -> Self {
        Shake256Context {
            opaque_contents: [0u64; 26],
        }
    }
}

extern "C" {
    fn shake256_init_prng_from_seed(sc: *mut Shake256Context, seed: *const c_void, seed_len: usize);
    fn falcon_det1024_keygen(
        rng: *mut Shake256Context,
        privkey: *mut c_void,
        pubkey: *mut c_void,
    ) -> c_int;
    fn falcon_det1024_sign_compressed(
        sig: *mut c_void,
        sig_len: *mut usize,
        privkey: *const c_void,
        data: *const c_void,
        data_len: usize,
    ) -> c_int;
    fn falcon_det1024_verify_compressed(
        sig: *const c_void,
        sig_len: usize,
        pubkey: *const c_void,
        data: *const c_void,
        data_len: usize,
    ) -> c_int;
    fn falcon_det1024_pubkey_coeffs(h: *mut u16, pubkey: *const c_void) -> c_int;
}

/// A Falcon-1024 keypair in the canonical reference encoding.
#[derive(Clone)]
pub struct Keypair {
    pub public_key: Box<[u8; PUBKEY_SIZE]>,
    pub private_key: Box<[u8; PRIVKEY_SIZE]>,
}

impl core::fmt::Debug for Keypair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print private key bytes.
        f.debug_struct("Keypair")
            .field("public_key_len", &PUBKEY_SIZE)
            .field("private_key", &"<redacted>")
            .finish()
    }
}

/// Errors from the Falcon reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FalconError {
    Keygen(i32),
    Sign(i32),
    Verify(i32),
    PubkeyDecode(i32),
}

impl core::fmt::Display for FalconError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FalconError::Keygen(c) => write!(f, "falcon keygen failed (code {c})"),
            FalconError::Sign(c) => write!(f, "falcon sign failed (code {c})"),
            FalconError::Verify(c) => write!(f, "falcon verify failed (code {c})"),
            FalconError::PubkeyDecode(c) => write!(f, "falcon pubkey decode failed (code {c})"),
        }
    }
}

impl std::error::Error for FalconError {}

/// Generate a Falcon-1024 keypair deterministically from `seed`.
///
/// Mirrors `cfalcon.GenerateKey(seed)`: initialise a SHAKE256 PRNG from the seed
/// bytes, then run `falcon_det1024_keygen`. go-algorand uses a 32-byte seed
/// (`FalconSeedSize`, derived from entropy via the `"PQK"` hash — see
/// [`crate::address::keygen_seed_from_entropy`]); any non-empty length is
/// accepted by the reference.
pub fn keygen_from_seed(seed: &[u8]) -> Result<Keypair, FalconError> {
    let mut public_key = Box::new([0u8; PUBKEY_SIZE]);
    let mut private_key = Box::new([0u8; PRIVKEY_SIZE]);

    // SAFETY: the context is fully initialised by the C call before use; the
    // seed pointer/len are valid for the duration of the init call; key buffers
    // are exactly the sizes the C side writes.
    let rc = unsafe {
        let mut rng = Shake256Context::zeroed();
        let (seed_ptr, seed_len) = if seed.is_empty() {
            (core::ptr::null(), 0usize)
        } else {
            (seed.as_ptr() as *const c_void, seed.len())
        };
        shake256_init_prng_from_seed(&mut rng, seed_ptr, seed_len);
        falcon_det1024_keygen(
            &mut rng,
            private_key.as_mut_ptr() as *mut c_void,
            public_key.as_mut_ptr() as *mut c_void,
        )
    };
    if rc != 0 {
        return Err(FalconError::Keygen(rc as i32));
    }
    Ok(Keypair {
        public_key,
        private_key,
    })
}

/// Decode a public key into its 1024 coefficients `h[i] in [0, q)`.
/// Mirrors `cfalcon.PublicKey.Coefficients()`.
pub fn pubkey_coeffs(public_key: &[u8; PUBKEY_SIZE]) -> Result<[u16; N], FalconError> {
    let mut h = [0u16; N];
    // SAFETY: h is N u16; pubkey is the exact size the C decoder reads.
    let rc = unsafe {
        falcon_det1024_pubkey_coeffs(h.as_mut_ptr(), public_key.as_ptr() as *const c_void)
    };
    if rc != 0 {
        return Err(FalconError::PubkeyDecode(rc as i32));
    }
    Ok(h)
}

impl Keypair {
    /// Deterministically sign `msg` (compressed det1024 format). Used to verify
    /// that a ground key actually round-trips before we emit it.
    pub fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, FalconError> {
        let mut sig = vec![0u8; SIG_COMPRESSED_MAXSIZE];
        let mut sig_len: usize = 0;
        let (msg_ptr, msg_len) = if msg.is_empty() {
            (core::ptr::null(), 0usize)
        } else {
            (msg.as_ptr() as *const c_void, msg.len())
        };
        // SAFETY: sig buffer is MAXSIZE; private key is the exact size.
        let rc = unsafe {
            falcon_det1024_sign_compressed(
                sig.as_mut_ptr() as *mut c_void,
                &mut sig_len,
                self.private_key.as_ptr() as *const c_void,
                msg_ptr,
                msg_len,
            )
        };
        if rc != 0 {
            return Err(FalconError::Sign(rc as i32));
        }
        sig.truncate(sig_len);
        Ok(sig)
    }

    /// Verify a compressed det1024 signature against this keypair's public key.
    pub fn verify(&self, msg: &[u8], sig: &[u8]) -> Result<(), FalconError> {
        verify_with_pubkey(&self.public_key, msg, sig)
    }
}

/// Verify a compressed det1024 signature against a standalone public key.
pub fn verify_with_pubkey(
    public_key: &[u8; PUBKEY_SIZE],
    msg: &[u8],
    sig: &[u8],
) -> Result<(), FalconError> {
    let (msg_ptr, msg_len) = if msg.is_empty() {
        (core::ptr::null(), 0usize)
    } else {
        (msg.as_ptr() as *const c_void, msg.len())
    };
    // SAFETY: pointers/lengths describe valid slices for the call's duration.
    let rc = unsafe {
        falcon_det1024_verify_compressed(
            sig.as_ptr() as *const c_void,
            sig.len(),
            public_key.as_ptr() as *const c_void,
            msg_ptr,
            msg_len,
        )
    };
    if rc != 0 {
        return Err(FalconError::Verify(rc as i32));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_with_first_byte(b: u8) -> [u8; SEED_SIZE] {
        let mut s = [0u8; SEED_SIZE];
        s[0] = b;
        s
    }

    #[test]
    fn keygen_is_deterministic() {
        let kp1 = keygen_from_seed(&seed_with_first_byte(0)).unwrap();
        let kp2 = keygen_from_seed(&seed_with_first_byte(0)).unwrap();
        assert_eq!(kp1.public_key, kp2.public_key);
        assert_eq!(kp1.private_key, kp2.private_key);
    }

    #[test]
    fn keygen_different_seeds_differ() {
        let kp0 = keygen_from_seed(&seed_with_first_byte(0)).unwrap();
        let kp1 = keygen_from_seed(&seed_with_first_byte(1)).unwrap();
        assert_ne!(kp0.public_key, kp1.public_key);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let kp = keygen_from_seed(&seed_with_first_byte(7)).unwrap();
        let msg = b"pq-vanity round-trip";
        let sig = kp.sign(msg).unwrap();
        kp.verify(msg, &sig).unwrap();
        // Tampered message must fail.
        assert!(kp.verify(b"pq-vanity round-trip!", &sig).is_err());
    }

    #[test]
    fn pubkey_coeffs_in_range() {
        let kp = keygen_from_seed(&seed_with_first_byte(3)).unwrap();
        let h = pubkey_coeffs(&kp.public_key).unwrap();
        assert!(h.iter().all(|&c| c < Q));
    }
}

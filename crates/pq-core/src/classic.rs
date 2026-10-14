//! Classic Algorand (Ed25519) accounts — the CPU oracle for the classic
//! vanity search.
//!
//! go-algorand derives a classic account with libsodium's
//! `crypto_sign_ed25519_seed_keypair` (`crypto.GenerateSignatureSecrets`),
//! which is plain RFC 8032 key generation:
//!
//! ```text
//! seed[32]  — the 25-word mnemonic key (`algokey import -m`, `goal account import -m`)
//! pk        = encode(clamp(SHA-512(seed)[0..32]) · B)
//! addr32    = pk                                   // no hash: the address IS the key
//! display   = base32_nopad(addr32 || last4(SHA512_256(addr32)))
//! ```
//!
//! Unlike `f1`, there is no salt and no off-curve rule (a classic address is
//! on the curve by construction). `ed25519-dalek` implements the same RFC 8032
//! derivation; it is cross-checked against RFC 8032 test vectors here and
//! against Go's `crypto/ed25519` in `tests/classic_differential.rs`.

use ed25519_dalek::{Signer, SigningKey, Verifier};

use crate::address::Address;

/// Classic seed length (`crypto.Seed`, the mnemonic key).
pub const CLASSIC_SEED_SIZE: usize = 32;

/// A classic Algorand key pair derived from its 32-byte seed.
pub struct ClassicKeypair {
    signing: SigningKey,
}

impl ClassicKeypair {
    /// `crypto.GenerateSignatureSecrets(seed)`.
    pub fn from_seed(seed: &[u8; CLASSIC_SEED_SIZE]) -> Self {
        ClassicKeypair {
            signing: SigningKey::from_bytes(seed),
        }
    }

    /// The 32-byte Ed25519 public key.
    pub fn public_key(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// The account address (`basics.Address(pk)`).
    pub fn address(&self) -> Address {
        Address(self.public_key())
    }

    /// Sign `msg` and verify the signature against the public key.
    pub fn roundtrip(&self, msg: &[u8]) -> bool {
        let sig = self.signing.sign(msg);
        self.signing.verifying_key().verify(msg, &sig).is_ok()
    }
}

/// Seed → classic address (the public key).
pub fn classic_address(seed: &[u8; CLASSIC_SEED_SIZE]) -> Address {
    ClassicKeypair::from_seed(seed).address()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RFC 8032 §7.1 TEST 1–3: seed → public key.
    #[test]
    fn rfc8032_pubkeys() {
        let cases = [
            (
                "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
                "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            ),
            (
                "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
                "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            ),
            (
                "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
                "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            ),
        ];
        for (seed, pk) in cases {
            let seed: [u8; 32] = unhex(seed).try_into().unwrap();
            assert_eq!(ClassicKeypair::from_seed(&seed).public_key().to_vec(), unhex(pk));
        }
    }

    /// The display string is the standard 58-char checksummed base32 of pk.
    #[test]
    fn address_is_pubkey() {
        let seed = [7u8; 32];
        let kp = ClassicKeypair::from_seed(&seed);
        let addr = classic_address(&seed);
        assert_eq!(addr.as_bytes(), &kp.public_key());
        assert_eq!(addr.to_address_string().len(), 58);
        assert!(kp.roundtrip(b"classic"));
    }
}

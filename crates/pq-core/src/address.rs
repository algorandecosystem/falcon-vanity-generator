//! Byte-exact derivation of Algorand native post-quantum (`f1`) account
//! addresses, reproduced from go-algorand PR #6639 **as merged to master**
//! (pin 551fd57a). See `docs/DESIGN.md` for the source-of-truth mapping.
//!
//! ```text
//! seed32  = SHA512_256( "PQK" || "f1" || entropy(32 bytes) )         // keygen seed
//! addr32  = SHA512_256( "PQA" || "f1" || salt(1 byte) || pk )
//! display = base32_nopad( addr32 || last4(SHA512_256(addr32)) )      // 58 chars
//! compliant(addr32) = !IsEdwards25519Point(addr32)                   // off-curve
//! ```
//!
//! - `"PQK"` is `protocol.PostQuantumKey`; `entropy` is the 32-byte secret that
//!   encodes to the standard 25-word Algorand mnemonic (`algokey pq import -m`).
//! - `"PQA"` is `protocol.PostQuantumAddress` (a `HashID` domain separator).
//! - `"f1"` is `protocol.PQSchemeFalcon1024`.
//! - the hash is `crypto.Hash` = SHA-512/256.
//! - `IsEdwards25519Point` is `filippo.io/edwards25519`'s `Point.SetBytes`
//!   acceptance set (non-canonical encodings accepted, no subgroup check).

use data_encoding::BASE32_NOPAD;
use sha2::{Digest, Sha512_256};

/// `protocol.PostQuantumAddress` HashID separator.
pub const HASHID_PQ_ADDRESS: &[u8] = b"PQA";
/// `protocol.PostQuantumKey` HashID separator (entropy → keygen-seed hash).
pub const HASHID_PQ_KEY: &[u8] = b"PQK";
/// `protocol.PQSchemeFalcon1024` scheme tag (2-byte `PQScheme`).
pub const SCHEME_F1: &[u8; 2] = b"f1";
/// Mnemonic-sized entropy length (`crypto.Seed` — 25-word Algorand mnemonic).
pub const ENTROPY_SIZE: usize = 32;

const CHECKSUM_LEN: usize = 4;
const ADDR_LEN: usize = 32;

/// SHA-512/256 (`crypto.Hash` / `crypto.HashObj`'s underlying hash).
#[inline]
pub fn sha512_256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha512_256::new();
    h.update(data);
    h.finalize().into()
}

/// A 32-byte Algorand address (`addr32`), before checksum/base32 display.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address(pub [u8; ADDR_LEN]);

impl Address {
    #[inline]
    pub fn as_bytes(&self) -> &[u8; ADDR_LEN] {
        &self.0
    }

    /// The 4-byte checksum: last 4 bytes of `SHA512_256(addr32)`.
    #[inline]
    pub fn checksum(&self) -> [u8; CHECKSUM_LEN] {
        let h = sha512_256(&self.0);
        let mut c = [0u8; CHECKSUM_LEN];
        c.copy_from_slice(&h[h.len() - CHECKSUM_LEN..]);
        c
    }

    /// Human-readable checksummed address: `base32_nopad(addr32 || checksum4)`.
    /// This is `basics.Address.String()` — 58 base32 characters.
    pub fn to_address_string(&self) -> String {
        let mut buf = [0u8; ADDR_LEN + CHECKSUM_LEN];
        buf[..ADDR_LEN].copy_from_slice(&self.0);
        buf[ADDR_LEN..].copy_from_slice(&self.checksum());
        BASE32_NOPAD.encode(&buf)
    }

    /// `addr.IsPQCompliant()` — eligible for native PQ authorization iff the
    /// 32 bytes do NOT decompress to an Edwards25519 point.
    #[inline]
    pub fn is_pq_compliant(&self) -> bool {
        !is_edwards25519_point(&self.0)
    }

    /// The base32 display string of `addr32` alone (no checksum), used for
    /// prefix matching during the vanity search. The full displayed address
    /// shares this exact leading run because the checksum lives in the tail.
    ///
    /// Note: a vanity *prefix* targets the leading characters of the full
    /// displayed address, which equal the leading characters of this string
    /// for as long as the prefix stays within the first 51 chars (the first
    /// 32 bytes encode to 51.2 chars, so the 52nd char mixes addr+checksum).
    pub fn addr32_base32(&self) -> String {
        BASE32_NOPAD.encode(&self.0)
    }
}

impl core::fmt::Display for Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_address_string())
    }
}

impl core::fmt::Debug for Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Address({})", self.to_address_string())
    }
}

/// `IsEdwards25519Point`: reports whether `encoded` decompresses to an
/// Edwards25519 point under the `filippo.io/edwards25519` acceptance set.
///
/// We use `curve25519-dalek`'s `CompressedEdwardsY::decompress`. The two
/// libraries' acceptance sets are differential-tested in `tests/` against a
/// small Go oracle built on `filippo.io/edwards25519` directly.
#[inline]
pub fn is_edwards25519_point(encoded: &[u8; 32]) -> bool {
    curve25519_dalek::edwards::CompressedEdwardsY(*encoded)
        .decompress()
        .is_some()
}

/// `derivePQKeySeed` (cmd/algokey/pq_scheme.go): map 32-byte mnemonic entropy
/// to the Falcon-1024 keygen seed — `SHA512_256("PQK" || "f1" || entropy)`.
/// The result is the `crypto.FalconSeed` fed to `GenerateFalconSigner`.
pub fn keygen_seed_from_entropy(entropy: &[u8; ENTROPY_SIZE]) -> [u8; 32] {
    let mut h = Sha512_256::new();
    h.update(HASHID_PQ_KEY);
    h.update(SCHEME_F1);
    h.update(entropy);
    h.finalize().into()
}

/// Compute `addr32` for `(salt, pk)` under the Falcon-1024 (`f1`) scheme.
///
/// `pk` must be the 1793-byte Falcon-1024 public key. This is the consensus
/// `basics.PQAddress(PQSchemeFalcon1024, salt, pk)`.
pub fn pq_address(salt: u8, pk: &[u8]) -> Address {
    let mut h = Sha512_256::new();
    h.update(HASHID_PQ_ADDRESS);
    h.update(SCHEME_F1);
    h.update([salt]);
    h.update(pk);
    Address(h.finalize().into())
}

/// The exact preimage bytes hashed by `pq_address` (for tests / debugging):
/// `"PQA" || "f1" || salt || pk`.
pub fn pq_address_preimage(salt: u8, pk: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(HASHID_PQ_ADDRESS.len() + 2 + 1 + pk.len());
    v.extend_from_slice(HASHID_PQ_ADDRESS);
    v.extend_from_slice(SCHEME_F1);
    v.push(salt);
    v.extend_from_slice(pk);
    v
}

/// `basics.CanonicalPQAddressSalt`: the lowest salt in `0..=255` whose derived
/// address is PQ-compliant (off-curve), with that address. Tooling uses this as
/// the "default" address for a key; a vanity match may use any compliant salt.
pub fn canonical_pq_salt(pk: &[u8]) -> Option<(u8, Address)> {
    (0u16..=255).find_map(|salt| {
        let salt = salt as u8;
        let a = pq_address(salt, pk);
        a.is_pq_compliant().then_some((salt, a))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::{keygen_from_seed, SEED_SIZE};

    fn pk_for_seed_byte(b: u8) -> Vec<u8> {
        let mut seed = [0u8; SEED_SIZE];
        seed[0] = b;
        keygen_from_seed(&seed).unwrap().public_key.to_vec()
    }

    /// TestPQAddressPreimage: scheme f1, salt 0x7f, pk {ab,cd,ef}
    /// => "f1" 0x7f ab cd ef (the HashID "PQA" is prepended by HashObj).
    #[test]
    fn preimage_matches_kat() {
        let pre = pq_address_preimage(0x7f, &[0xab, 0xcd, 0xef]);
        assert_eq!(pre, b"PQAf1\x7f\xab\xcd\xef");
    }

    /// TestPQAddressKnownAnswers (data/basics/pq_address_test.go @ 551fd57a):
    /// (first-seed-byte, salt) → displayed address + compliance.
    #[test]
    fn known_answer_addresses() {
        let cases: &[(u8, u8, &str, bool)] = &[
            (3, 0, "KJGJA2DTCQH6LT2I2OH2YO5GIIBFC6JHX5O6UPA5ZZ5ZURFT3LHKMTRCEM", true),
            (1, 1, "GYBWVYVQIQF6CO7BUMG4UQ66DQYHASFOCA2P7PBYOIPKGWUZIBX4KA3TP4", true),
            (0, 255, "YJFADDEP6Z3WAWY6ZMLN6MF4T4NK3BXKCVLPCYB6C4SQHE76LLQSZ5JG7Q", true),
            (2, 2, "II4DO6IIP3EAEQMWJEOLOUU3VBRVCH3WF4MX6UCRUD36DOQJ3YSHA2DV5A", true),
            (255, 255, "3JXWI6BYYEO6WO6M7TC4SOZAZUWAD4RQO5GJ2ED6MYIEVVLOVJOETMGG4A", true),
            (1, 0, "FLX4VRWXQ65HD5G5BI2EPHJWMERHA2EBBQ7XMTZLATXH4XEOWPQSIYVIF4", false),
        ];
        for &(seed_byte, salt, expected, compliant) in cases {
            let pk = pk_for_seed_byte(seed_byte);
            let a = pq_address(salt, &pk);
            assert_eq!(a.to_address_string(), expected, "seed byte {seed_byte}, salt {salt}");
            assert_eq!(a.is_pq_compliant(), compliant, "seed byte {seed_byte}, salt {salt}");
        }
    }

    /// TestCanonicalPQAddressSalt: for seed-byte-1's key, salt 0 is NOT
    /// compliant and salt 1 is the canonical (lowest compliant) salt.
    #[test]
    fn canonical_salt_for_key1_is_one() {
        let pk1 = pk_for_seed_byte(1);
        assert!(!pq_address(0, &pk1).is_pq_compliant());
        let (salt, addr) = canonical_pq_salt(&pk1).unwrap();
        assert_eq!(salt, 1);
        assert_eq!(
            addr.to_address_string(),
            "GYBWVYVQIQF6CO7BUMG4UQ66DQYHASFOCA2P7PBYOIPKGWUZIBX4KA3TP4"
        );
    }

    /// TestPQGenerate (cmd/algokey/pq_test.go): full pipeline from mnemonic
    /// entropy {1,2,...,32} through the "PQK" seed hash, Falcon keygen, and the
    /// canonical-salt scan to the ground-truth address.
    #[test]
    fn entropy_to_canonical_address_kat() {
        let mut entropy = [0u8; ENTROPY_SIZE];
        for (i, b) in entropy.iter_mut().enumerate() {
            *b = (i + 1) as u8;
        }
        let seed = keygen_seed_from_entropy(&entropy);
        let kp = keygen_from_seed(&seed).unwrap();
        let (_, addr) = canonical_pq_salt(kp.public_key.as_ref()).unwrap();
        assert_eq!(
            addr.to_address_string(),
            "ZEJ4BLG3XWAUUZQGCEDJLYIC6D2NCWHRSX5DJMDPE54PXXR7G3PCQTARXU"
        );
    }

    /// TestDerivePQKeySeed: the seed hash preimage is "PQK" || "f1" || entropy.
    #[test]
    fn pqk_seed_preimage() {
        let mut entropy = [0u8; ENTROPY_SIZE];
        for (i, b) in entropy.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut pre = Vec::new();
        pre.extend_from_slice(HASHID_PQ_KEY);
        pre.extend_from_slice(SCHEME_F1);
        pre.extend_from_slice(&entropy);
        assert_eq!(keygen_seed_from_entropy(&entropy), sha512_256(&pre));
    }

    #[test]
    fn display_is_58_chars() {
        let pk0 = pk_for_seed_byte(0);
        assert_eq!(pq_address(0, &pk0).to_address_string().len(), 58);
    }
}

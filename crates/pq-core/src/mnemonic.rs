//! Algorand 25-word mnemonic codec, ported from go-algorand
//! `crypto/passphrase` (vendored at `vendor/go-algorand-ref/crypto/passphrase`).
//!
//! `algokey pq import -m <mnemonic>` consumes exactly this encoding of the
//! 32-byte entropy: 24 data words of 11 bits each (little-endian bit packing)
//! plus one checksum word — the first 11 bits of `SHA512_256(entropy)`.

use crate::address::sha512_256;

/// BIP-39 English wordlist (2048 words), byte-identical to go-algorand's
/// `crypto/passphrase/wordlist.go`.
static WORDLIST_RAW: &str = include_str!("wordlist_english.txt");

fn wordlist() -> &'static [&'static str] {
    use std::sync::OnceLock;
    static WORDS: OnceLock<Vec<&'static str>> = OnceLock::new();
    WORDS
        .get_or_init(|| {
            let w: Vec<&str> = WORDLIST_RAW.split_whitespace().collect();
            assert_eq!(w.len(), 2048, "embedded wordlist corrupted");
            w
        })
        .as_slice()
}

/// `toUint11Array`: little-endian bit packing of bytes into 11-bit values.
fn to_uint11_array(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len() * 8 / 11 + 1);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &b in bytes {
        buffer |= (b as u32) << bits;
        bits += 8;
        if bits >= 11 {
            out.push(buffer & 0x7ff);
            buffer >>= 11;
            bits -= 11;
        }
    }
    if bits != 0 {
        out.push(buffer & 0x7ff);
    }
    out
}

/// `toByteArray`: inverse packing (may emit one trailing padding byte).
fn to_byte_array(vals: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 11 / 8 + 1);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &v in vals {
        buffer |= v << bits;
        bits += 11;
        while bits >= 8 {
            out.push((buffer & 0xff) as u8);
            buffer >>= 8;
            bits -= 8;
        }
    }
    if bits != 0 {
        out.push(buffer as u8);
    }
    out
}

/// The checksum word: first 11 bits of `SHA512_256(data)`.
fn checksum_word(data: &[u8]) -> &'static str {
    let h = sha512_256(data);
    wordlist()[to_uint11_array(&h[0..2])[0] as usize]
}

/// `passphrase.KeyToMnemonic`: 32-byte entropy → 25-word mnemonic.
pub fn key_to_mnemonic(key: &[u8; 32]) -> String {
    let words = wordlist();
    let mut s = String::with_capacity(25 * 9);
    for v in to_uint11_array(key) {
        s.push_str(words[v as usize]);
        s.push(' ');
    }
    s.push_str(checksum_word(key));
    s
}

/// `passphrase.MnemonicToKey`: 25-word mnemonic → 32-byte entropy.
/// Rejects wrong word counts, unknown words, and checksum mismatches.
pub fn mnemonic_to_key(mnemonic: &str) -> Result<[u8; 32], String> {
    let words: Vec<&str> = mnemonic.split_whitespace().collect();
    if words.len() != 25 {
        return Err(format!("expected 25 words, got {}", words.len()));
    }
    let list = wordlist();
    let mut indices = Vec::with_capacity(24);
    for w in &words[..24] {
        match list.iter().position(|x| x == w) {
            Some(i) => indices.push(i as u32),
            None => return Err(format!("{w:?} is not in the words list")),
        }
    }
    let bytes = to_byte_array(&indices);
    // 24 * 11 = 264 bits = 33 bytes; the last byte is 3 bits of padding.
    if bytes.len() != 33 || bytes[32] != 0 {
        return Err("wrong checksum".to_string());
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes[..32]);
    if checksum_word(&key) != words[24] {
        return Err("wrong checksum".to_string());
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TestZeroVector (crypto/passphrase/passphrase_test.go): the all-zero key.
    #[test]
    fn zero_key_kat() {
        let mn = key_to_mnemonic(&[0u8; 32]);
        let expected = format!("{}invest", "abandon ".repeat(24));
        assert_eq!(mn, expected);
        assert_eq!(mnemonic_to_key(&mn).unwrap(), [0u8; 32]);
    }

    #[test]
    fn roundtrip_various_keys() {
        for i in 0..64u8 {
            let mut key = [0u8; 32];
            for (j, b) in key.iter_mut().enumerate() {
                *b = (j as u8).wrapping_mul(31).wrapping_add(i);
            }
            let mn = key_to_mnemonic(&key);
            assert_eq!(mn.split_whitespace().count(), 25);
            assert_eq!(mnemonic_to_key(&mn).unwrap(), key, "key #{i}");
        }
    }

    /// Wrong last data word (TestWrongChecksum-style) and unknown word.
    #[test]
    fn rejects_bad_input() {
        let mn = key_to_mnemonic(&[0u8; 32]);
        let mut words: Vec<&str> = mn.split_whitespace().collect();
        words[23] = "zoo";
        assert!(mnemonic_to_key(&words.join(" ")).is_err());
        words[23] = "zzz";
        assert!(mnemonic_to_key(&words.join(" ")).is_err());
        assert!(mnemonic_to_key("too few words").is_err());
    }

    /// The go wordlist self-check: checksum word of the raw list is "venue".
    #[test]
    fn wordlist_checksum_matches_go() {
        // go-algorand checksums `wordlistRaw` — newline-separated with a
        // trailing newline, exactly our embedded file.
        assert_eq!(checksum_word(WORDLIST_RAW.as_bytes()), "venue");
    }
}

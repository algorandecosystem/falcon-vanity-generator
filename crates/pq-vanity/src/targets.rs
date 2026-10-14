//! Search targets: one or many (prefix, suffix) patterns, shared by every
//! search path (CPU / GPU, f1 / ed25519).
//!
//! CLI forms (all comma-separated lists; a comma anywhere → multi search):
//! - `--prefix A,B`            any listed prefix
//! - `--suffix X,Y`            any listed suffix
//! - `--prefix A,B --suffix X,Y` every prefix × suffix combination
//! - `--pattern ABCD-WXYZ,EFGH..QRSA,ALGO-,-XYZA`  explicit prefix+suffix
//!   pairs (`-`, `..` or `…` separates; either side may be empty)
//!
//! The displayed address is `base32(addr32 || checksum4)`: 288 bits in 58
//! characters, so the LAST character carries 3 data bits + 2 zero padding
//! bits and can only be one of `AEIMQUY4`. A suffix costs `32^(S-1)·8` keys,
//! a prefix `32^P`.
//!
//! The checksum (`SHA512_256(addr32)[28..32]`) only affects the last ~6.4
//! characters, so prefix-only targets never need it: matchers compute it
//! lazily, only for a target with a suffix whose prefix already matched.
//!
//! Device form ([`Target::device_words`]): with `hi` = big-endian u64 of
//! `addr32[0..8]` and `lo` = big-endian u64 of `addr32[28..32] || checksum4`,
//! a target matches iff `hi & pmask == pval && lo & smask == sval`.

// Device-table and verification helpers are used only by the GPU paths.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::sync::atomic::{AtomicBool, Ordering};

use pq_core::Address;

use crate::search::BASE32_ALPHABET;

/// Device target-table capacity (`__constant__`, 32 bytes per target).
pub const MAX_TARGETS: usize = 1024;
/// GPU limits: the prefix must fit `hi` (5·12 = 60 bits), the suffix `lo`
/// (5·13 − 2 = 63 bits). Far beyond any feasible search anyway.
pub const GPU_MAX_PREFIX: usize = 12;
pub const GPU_MAX_SUFFIX: usize = 13;
/// The only characters that can end an Algorand address.
pub const LAST_CHARS: &str = "AEIMQUY4";
const DISPLAY_LEN: usize = 58;
/// Leading display chars determined by addr32 alone (32 bytes → 51.2 chars).
const ADDR_ONLY_CHARS: usize = 51;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub prefix: String,
    pub suffix: String,
}

impl Target {
    /// `ABCD…WXYZ`, `ABCD…`, `…WXYZ`.
    pub fn label(&self) -> String {
        match (self.prefix.is_empty(), self.suffix.is_empty()) {
            (false, true) => format!("{}…", self.prefix),
            (true, false) => format!("…{}", self.suffix),
            _ => format!("{}…{}", self.prefix, self.suffix),
        }
    }

    /// Expected keys (addresses) per hit for this target alone.
    pub fn addrs_per_hit(&self) -> f64 {
        let s = self.suffix.len() as i32;
        let suffix = if s == 0 { 1.0 } else { 32f64.powi(s - 1) * 8.0 };
        32f64.powi(self.prefix.len() as i32) * suffix
    }

    /// Reference matcher on the full display string (used to verify hits —
    /// deliberately independent of the bit-mask form).
    pub fn matches_display(&self, display: &str) -> bool {
        display.starts_with(&self.prefix) && display.ends_with(&self.suffix)
    }

    /// `[pmask, pval, smask, sval]` for the device (see module docs).
    /// Requires `prefix.len() <= GPU_MAX_PREFIX`, `suffix.len() <= GPU_MAX_SUFFIX`.
    pub fn device_words(&self) -> [u64; 4] {
        // u128: a 13-char suffix is 65 pattern bits before the padding drops.
        let bits = |s: &str| {
            s.bytes()
                .fold(0u128, |acc, c| (acc << 5) | BASE32_ALPHABET.find(c as char).unwrap() as u128)
        };
        let (mut pmask, mut pval, mut smask, mut sval) = (0, 0, 0, 0);
        let p = self.prefix.len() as u32;
        if p > 0 {
            pmask = u64::MAX << (64 - 5 * p);
            pval = (bits(&self.prefix) as u64) << (64 - 5 * p);
        }
        let s = self.suffix.len() as u32;
        if s > 0 {
            // The two padding bits of the last char are zero (validated).
            smask = (1u64 << (5 * s - 2)) - 1;
            sval = (bits(&self.suffix) >> 2) as u64;
        }
        [pmask, pval, smask, sval]
    }
}

fn split_list(arg: &str, what: &str) -> anyhow::Result<Vec<String>> {
    arg.split(',')
        .map(|s| {
            let s = s.trim().to_ascii_uppercase();
            anyhow::ensure!(!s.is_empty(), "empty entry in {what} list {arg:?}");
            Ok(s)
        })
        .collect()
}

fn check_chars(s: &str) -> anyhow::Result<()> {
    if let Some(c) = s.chars().find(|c| !BASE32_ALPHABET.contains(*c)) {
        anyhow::bail!(
            "pattern character {c:?} in {s:?} is not in the Algorand base32 alphabet ({BASE32_ALPHABET})"
        );
    }
    Ok(())
}

/// Split one `--pattern` entry into (prefix, suffix).
fn split_pair(entry: &str) -> anyhow::Result<(String, String)> {
    for sep in ["..", "…", "-"] {
        if let Some((p, s)) = entry.split_once(sep) {
            return Ok((p.to_string(), s.to_string()));
        }
    }
    anyhow::bail!("--pattern entry {entry:?} needs a separator: PREFIX-SUFFIX (or PREFIX..SUFFIX)")
}

/// The validated target list.
pub struct TargetSet {
    pub list: Vec<Target>,
    active: Vec<AtomicBool>,
    /// Retire a target after its first hit (multi search, unless --keep-found).
    retire: bool,
    /// Longest prefix among the targets (bytes of base32 needed).
    max_prefix: usize,
}

impl TargetSet {
    /// Parse the CLI options. `--pattern` conflicts with the others (clap).
    pub fn parse(
        prefix: Option<&str>,
        suffix: Option<&str>,
        pattern: Option<&str>,
        keep_found: bool,
    ) -> anyhow::Result<Self> {
        let mut list = Vec::new();
        if let Some(pat) = pattern {
            for e in split_list(pat, "--pattern")? {
                let (p, s) = split_pair(&e)?;
                list.push(Target { prefix: p, suffix: s });
            }
        } else {
            let ps = prefix.map(|p| split_list(p, "--prefix")).transpose()?;
            let ss = suffix.map(|s| split_list(s, "--suffix")).transpose()?;
            let ps = ps.unwrap_or_else(|| vec![String::new()]);
            let ss = ss.unwrap_or_else(|| vec![String::new()]);
            for p in &ps {
                for s in &ss {
                    list.push(Target { prefix: p.clone(), suffix: s.clone() });
                }
            }
        }
        let mut dedup: Vec<Target> = Vec::with_capacity(list.len());
        for t in list {
            check_chars(&t.prefix)?;
            check_chars(&t.suffix)?;
            anyhow::ensure!(
                !t.prefix.is_empty() || !t.suffix.is_empty(),
                "empty pattern (give a prefix, a suffix, or both)"
            );
            if let Some(last) = t.suffix.chars().last() {
                anyhow::ensure!(
                    LAST_CHARS.contains(last),
                    "suffix {:?} can never match: an Algorand address always ends in one of \
                     {LAST_CHARS} (the last base32 char holds 3 data bits + 2 zero padding bits)",
                    t.suffix
                );
            }
            anyhow::ensure!(
                t.prefix.len() + t.suffix.len() <= DISPLAY_LEN,
                "pattern {} is longer than an address ({DISPLAY_LEN} chars)",
                t.label()
            );
            if !dedup.contains(&t) {
                dedup.push(t);
            }
        }
        anyhow::ensure!(!dedup.is_empty(), "specify --prefix, --suffix, or --pattern");
        anyhow::ensure!(
            dedup.len() <= MAX_TARGETS,
            "{} targets (after prefix × suffix expansion) — at most {MAX_TARGETS} supported",
            dedup.len()
        );
        let max_prefix = dedup.iter().map(|t| t.prefix.len()).max().unwrap_or(0);
        let active = dedup.iter().map(|_| AtomicBool::new(true)).collect();
        let retire = dedup.len() > 1 && !keep_found;
        Ok(TargetSet { list: dedup, active, retire, max_prefix })
    }

    /// Default hit budget: one per target in multi mode, else 1.
    pub fn default_count(&self) -> usize {
        if self.retire { self.list.len() } else { 1 }
    }

    /// Enforce the device limits.
    pub fn check_gpu(&self) -> anyhow::Result<()> {
        for t in &self.list {
            anyhow::ensure!(
                t.prefix.len() <= GPU_MAX_PREFIX && t.suffix.len() <= GPU_MAX_SUFFIX,
                "GPU search supports prefixes up to {GPU_MAX_PREFIX} and suffixes up to \
                 {GPU_MAX_SUFFIX} chars ({} given)",
                t.label()
            );
        }
        Ok(())
    }

    pub fn is_active(&self, i: usize) -> bool {
        self.active[i].load(Ordering::Relaxed)
    }

    pub fn any_active(&self) -> bool {
        (0..self.list.len()).any(|i| self.is_active(i))
    }

    /// Claim target `i` for an emitted hit. In retiring mode only the first
    /// claim succeeds (races between CPU workers resolve here).
    pub fn claim(&self, i: usize) -> bool {
        if !self.retire {
            return true;
        }
        self.active[i]
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }

    /// Device table of the active targets.
    pub fn device_table(&self) -> Vec<[u64; 4]> {
        (0..self.list.len())
            .filter(|&i| self.is_active(i))
            .map(|i| self.list[i].device_words())
            .collect()
    }

    /// Hits per address, summed over active targets (overlap ignored).
    pub fn hit_rate(&self) -> f64 {
        (0..self.list.len())
            .filter(|&i| self.is_active(i))
            .map(|i| 1.0 / self.list[i].addrs_per_hit())
            .sum()
    }

    /// Hot-path matcher: first active target `addr` matches. The prefix is
    /// checked on the base32 of just the leading bytes; the checksummed
    /// display string is built only if a suffix target's prefix matched.
    pub fn first_match(&self, addr: &Address) -> Option<usize> {
        let head_bytes = (self.max_prefix.min(ADDR_ONLY_CHARS) * 5).div_ceil(8);
        let head = data_encoding::BASE32_NOPAD.encode(&addr.as_bytes()[..head_bytes]);
        let mut display: Option<String> = None;
        for (i, t) in self.list.iter().enumerate() {
            if !self.is_active(i) {
                continue;
            }
            let prefix_ok = if t.prefix.len() <= ADDR_ONLY_CHARS {
                head.starts_with(&t.prefix)
            } else {
                display
                    .get_or_insert_with(|| addr.to_address_string())
                    .starts_with(&t.prefix)
            };
            if !prefix_ok {
                continue;
            }
            if t.suffix.is_empty()
                || display
                    .get_or_insert_with(|| addr.to_address_string())
                    .ends_with(&t.suffix)
            {
                return Some(i);
            }
        }
        None
    }

    /// Verification matcher (full display string, independent of the above).
    pub fn first_match_display(&self, display: &str) -> Option<usize> {
        (0..self.list.len()).find(|&i| self.is_active(i) && self.list[i].matches_display(display))
    }

    /// One-line summary for the startup banner.
    pub fn describe(&self) -> String {
        const SHOW: usize = 8;
        let mut s: Vec<String> = self.list.iter().take(SHOW).map(Target::label).collect();
        if self.list.len() > SHOW {
            s.push(format!("… (+{} more)", self.list.len() - SHOW));
        }
        format!("{} target(s): {}", self.list.len(), s.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix(st: &mut u64) -> u64 {
        *st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Mirror of the device matcher (cuda/*_vanity.cu `*_match_targets`).
    fn device_match(addr: &Address, w: [u64; 4]) -> bool {
        let a = addr.as_bytes();
        let hi = u64::from_be_bytes(a[0..8].try_into().unwrap());
        let cs = addr.checksum();
        let mut lo8 = [0u8; 8];
        lo8[..4].copy_from_slice(&a[28..32]);
        lo8[4..].copy_from_slice(&cs);
        let lo = u64::from_be_bytes(lo8);
        hi & w[0] == w[1] && lo & w[2] == w[3]
    }

    /// The device bit-mask form agrees with string matching, for patterns cut
    /// from the address itself (positives) and mutated ones (negatives).
    #[test]
    fn device_words_match_display() {
        let mut st = 42u64;
        let mut pos = 0;
        for _ in 0..20_000 {
            let mut a = [0u8; 32];
            for c in a.chunks_exact_mut(8) {
                c.copy_from_slice(&splitmix(&mut st).to_le_bytes());
            }
            let addr = Address(a);
            let d = addr.to_address_string();
            let p = (splitmix(&mut st) % (GPU_MAX_PREFIX as u64 + 1)) as usize;
            let s = (splitmix(&mut st) % (GPU_MAX_SUFFIX as u64 + 1)) as usize;
            let mut t = Target { prefix: d[..p].to_string(), suffix: d[58 - s..].to_string() };
            if splitmix(&mut st) & 1 == 1 {
                // Flip one char of prefix or suffix to a different one.
                let flip = |x: &str, k: usize| {
                    let mut v: Vec<u8> = x.bytes().collect();
                    let cur = BASE32_ALPHABET.find(v[k] as char).unwrap();
                    v[k] = BASE32_ALPHABET.as_bytes()[(cur + 4) % 32];
                    String::from_utf8(v).unwrap()
                };
                if p > 0 && (s == 0 || splitmix(&mut st) & 1 == 0) {
                    let k = (splitmix(&mut st) % p as u64) as usize;
                    t.prefix = flip(&t.prefix, k);
                } else if s > 0 {
                    let k = (splitmix(&mut st) % s as u64) as usize;
                    t.suffix = flip(&t.suffix, k);
                }
            }
            let want = t.matches_display(&d);
            pos += want as u32;
            assert_eq!(device_match(&addr, t.device_words()), want, "{} vs {d}", t.label());
        }
        assert!(pos > 5_000);
    }

    #[test]
    fn parse_forms() {
        let t = TargetSet::parse(Some("ab,cd"), Some("xa"), None, false).unwrap();
        assert_eq!(t.list.len(), 2);
        assert_eq!(t.list[1], Target { prefix: "CD".into(), suffix: "XA".into() });
        assert!(t.list.len() == 2 && t.default_count() == 2);

        let t = TargetSet::parse(None, None, Some("ABCD-WXYA,EF..QE,ALGO-,-ZZ4"), false).unwrap();
        let labels: Vec<_> = t.list.iter().map(Target::label).collect();
        assert_eq!(labels, ["ABCD…WXYA", "EF…QE", "ALGO…", "…ZZ4"]);

        let t = TargetSet::parse(Some("ALGO"), None, None, false).unwrap();
        assert!(t.list.len() == 1 && t.default_count() == 1);

        // Impossible last char, bad alphabet, empty entries.
        assert!(TargetSet::parse(None, Some("WXYZ"), None, false).is_err());
        assert!(TargetSet::parse(Some("AB1"), None, None, false).is_err());
        assert!(TargetSet::parse(Some("A,,B"), None, None, false).is_err());
        assert!(TargetSet::parse(None, None, Some("ABCD"), false).is_err());
    }

    #[test]
    fn retire_claims_once() {
        let t = TargetSet::parse(Some("A,B"), None, None, false).unwrap();
        assert!(t.claim(0) && !t.claim(0) && t.any_active());
        assert!(t.claim(1) && !t.any_active());
        let k = TargetSet::parse(Some("A,B"), None, None, true).unwrap();
        assert!(k.claim(0) && k.claim(0) && k.default_count() == 1);
    }

    #[test]
    fn lazy_matcher_agrees_with_display() {
        let t = TargetSet::parse(None, None, Some("A-,B-A,-E,CD-4,-QA"), true).unwrap();
        let mut st = 7u64;
        for _ in 0..20_000 {
            let mut a = [0u8; 32];
            for c in a.chunks_exact_mut(8) {
                c.copy_from_slice(&splitmix(&mut st).to_le_bytes());
            }
            let addr = Address(a);
            assert_eq!(t.first_match(&addr), t.first_match_display(&addr.to_address_string()));
        }
    }
}

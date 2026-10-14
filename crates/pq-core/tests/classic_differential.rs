//! Differential test: `pq_core::classic` (ed25519-dalek) vs. Go's
//! `crypto/ed25519` + an independent Go `basics.Address.String()`, via the
//! oracle in `tools/classic-oracle` — seed → public key → displayed address.
//!
//! Ignored by default (needs the Go toolchain; stdlib only, no network).
//!     cargo test -p pq-core --test classic_differential -- --ignored --nocapture

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
#[ignore]
fn classic_matches_go() {
    // Edge seeds (all-zero, all-0xff, single bytes) plus a random mass.
    let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32], [0xffu8; 32]];
    for i in 0..32 {
        let mut s = [0u8; 32];
        s[i] = 1;
        seeds.push(s);
    }
    let mut st = 0x5eed_c1a5_51c0_u64;
    for _ in 0..20_000 {
        let mut s = [0u8; 32];
        for c in s.chunks_exact_mut(8) {
            c.copy_from_slice(&splitmix64(&mut st).to_le_bytes());
        }
        seeds.push(s);
    }

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/classic-oracle");
    let mut child = Command::new("go")
        .args(["run", "."])
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn `go run` (Go toolchain required)");
    let input: String = seeds.iter().map(|s| hex(s) + "\n").collect();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()).unwrap());
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(out.status.success(), "oracle failed");

    let lines: Vec<&str> = std::str::from_utf8(&out.stdout).unwrap().lines().collect();
    assert_eq!(lines.len(), seeds.len());
    for (seed, line) in seeds.iter().zip(lines) {
        let kp = pq_core::ClassicKeypair::from_seed(seed);
        let want = format!("{} {}", hex(&kp.public_key()), kp.address().to_address_string());
        assert_eq!(line, want, "seed {}", hex(seed));
    }
    eprintln!("classic: {} seeds agree with Go crypto/ed25519", seeds.len());
}

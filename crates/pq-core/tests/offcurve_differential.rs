//! Differential test: `pq_core::is_edwards25519_point` (curve25519-dalek
//! `decompress`) vs. the go-algorand ground truth `filippo.io/edwards25519`
//! `Point.SetBytes`, via the Go oracle in `tools/offcurve-oracle`.
//!
//! Off-curve compliance gates whether a found address is usable at the Algod
//! API boundary, so the two implementations must agree on the exact acceptance
//! set — including the non-canonical `y >= p` band and the `x == 0, sign == 1`
//! case. This test sweeps those edges plus a large random mass.
//!
//! Ignored by default (needs Go toolchain + network to fetch the dependency).
//! Run it to (re-)validate:
//!     cargo test -p pq-core --test offcurve_differential -- --ignored --nocapture

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn vectors() -> Vec<[u8; 32]> {
    let mut v: Vec<[u8; 32]> = Vec::new();

    // --- structured edges -------------------------------------------------
    // p = 2^255 - 19, little-endian.
    let mut p = [0xFFu8; 32];
    p[0] = 0xED;
    p[31] = 0x7F;

    let push_both_signs = |v: &mut Vec<[u8; 32]>, mut y: [u8; 32]| {
        y[31] &= 0x7F;
        v.push(y);
        let mut ys = y;
        ys[31] |= 0x80;
        v.push(ys);
    };

    // y = 0, 1, 2 (identity is y=1); the x==0,sign==1 case lives here.
    for n in 0u8..=3 {
        let mut y = [0u8; 32];
        y[0] = n;
        push_both_signs(&mut v, y);
    }
    // Walk the non-canonical band y in [p-4, p+260): reduces to small values,
    // exercising whether each library rejects y >= p or reduces it.
    for delta in 0u32..264 {
        // y = (p - 4) + delta, computed as a 256-bit little-endian add.
        let mut y = p;
        // subtract 4 then add delta == add (delta as i64 - 4)
        let add = delta as i64 - 4;
        let mut carry = add;
        for byte in y.iter_mut() {
            let cur = *byte as i64 + (carry & 0xFF);
            *byte = (cur & 0xFF) as u8;
            carry = (carry >> 8) + (cur >> 8);
            if carry == 0 {
                break;
            }
        }
        push_both_signs(&mut v, y);
    }
    // 2^255 - 1 (all 255 low bits set).
    let mut all = [0xFFu8; 32];
    all[31] = 0x7F;
    push_both_signs(&mut v, all);
    // all 0xFF (sign bit set, y = 2^255-1).
    v.push([0xFFu8; 32]);
    // The Ed25519 basepoint compressed encoding (Y = 4/5): must be on-curve.
    let mut bp = [0u8; 32];
    bp.copy_from_slice(&[
        0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66,
    ]);
    v.push(bp);

    // --- random mass ------------------------------------------------------
    let mut st = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..300_000 {
        let mut y = [0u8; 32];
        for chunk in y.chunks_mut(8) {
            chunk.copy_from_slice(&splitmix64(&mut st).to_le_bytes());
        }
        v.push(y);
    }
    v
}

#[test]
#[ignore = "requires Go toolchain + network (filippo.io/edwards25519)"]
fn dalek_matches_filippo() {
    let oracle_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/offcurve-oracle");

    let vecs = vectors();
    let mut input = String::with_capacity(vecs.len() * 65);
    for y in &vecs {
        input.push_str(&hex::encode(y));
        input.push('\n');
    }

    // Avoid pipe deadlock on large I/O: stage stdin in a temp file.
    let tmp = std::env::temp_dir().join(format!("offcurve_in_{}.hex", std::process::id()));
    std::fs::write(&tmp, input.as_bytes()).expect("write temp input");
    let stdin_file = std::fs::File::open(&tmp).expect("open temp input");

    let out = Command::new("go")
        .args(["run", "."])
        .current_dir(&oracle_dir)
        .stdin(stdin_file)
        .output()
        .expect("run go oracle (is the Go toolchain installed?)");
    let _ = std::fs::remove_file(&tmp);

    assert!(
        out.status.success(),
        "go oracle failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let oracle: Vec<&str> = std::str::from_utf8(&out.stdout).unwrap().lines().collect();
    assert_eq!(oracle.len(), vecs.len(), "oracle line count mismatch");

    let mut mismatches = 0usize;
    let mut on_curve = 0usize;
    for (y, &o) in vecs.iter().zip(oracle.iter()) {
        let filippo = match o {
            "1" => true,
            "0" => false,
            other => panic!("oracle error line {other:?} for {}", hex::encode(y)),
        };
        if filippo {
            on_curve += 1;
        }
        let dalek = pq_core::is_edwards25519_point(y);
        if dalek != filippo {
            if mismatches < 20 {
                eprintln!(
                    "MISMATCH {}: dalek={dalek} filippo={filippo}",
                    hex::encode(y)
                );
            }
            mismatches += 1;
        }
    }
    let _ = std::io::stderr().flush();
    eprintln!(
        "checked {} vectors; {} on-curve (filippo); {} mismatches",
        vecs.len(),
        on_curve,
        mismatches
    );
    assert_eq!(
        mismatches, 0,
        "dalek and filippo disagree on {mismatches} inputs"
    );
}

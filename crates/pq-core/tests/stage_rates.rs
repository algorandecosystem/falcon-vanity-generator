//! Measurement harness: first-sample keygen acceptance, stage by stage.
//! Run explicitly:
//!   cargo test -p pq-core --release --test stage_rates -- --ignored --nocapture
//!
//! Stages (reference keygen order, vendor/falcon/keygen.c):
//!   lim      |coef| <= 15                          (integer — GPU-exact)
//!   sqnorm   ||f||^2 + ||g||^2 < 16823             (integer — GPU-exact)
//!   invert   f invertible mod q                    (NTT     — GPU-exact)
//!   accept   + FP Gram-Schmidt bnorm + solve_NTRU  (CPU-only)
//!
//! `q = P(accept | lim & sqnorm & invert)` is the fraction of GPU-visible
//! candidates the CPU completion keeps — the efficiency ceiling of an
//! in-kernel retry loop that mirrors the integer/NTT checks only.

use pq_core::{complete_from_fg, keygen_seed_from_entropy, pubkey_from_fg, sample_fg};

/// The GPU-side bnorm reformulation, in f64 with direct O(n^2) evaluation:
/// by Parseval, the reference's coefficient-domain bnorm equals
///   (q^2 / n) * sum_k 1 / (|f^(z_k)|^2 + |g^(z_k)|^2)
/// over the n negacyclic bins z_k = exp(-i*pi*(2k+1)/n). Validate the formula
/// (and constant) against the FPEMU classifier before porting FP32 to CUDA.
#[test]
#[ignore = "measurement harness, run explicitly with --ignored --nocapture"]
fn bnorm_parseval_formula_matches_reference() {
    const N: usize = 1024;
    let bound = f64::from_bits(4670353323383631276); // fpr_bnorm_max = 16822.4121
    let (mut checked, mut agree, mut near) = (0u32, 0u32, 0u32);
    let mut max_margin_disagree = 0.0f64;
    for i in 0..1500u32 {
        let mut e = [0u8; 32];
        e[..4].copy_from_slice(&i.to_le_bytes());
        e[4] = 0x3C;
        let fg = sample_fg(&keygen_seed_from_entropy(&e));
        let lim_ok = fg.f.iter().chain(fg.g.iter()).all(|&c| c > -16 && c < 16);
        let sqn = |p: &[i8]| p.iter().map(|&c| (c as i32 * c as i32) as u32).sum::<u32>();
        if !lim_ok || sqn(&fg.f) + sqn(&fg.g) >= 16823 {
            continue; // classifier would reject earlier; bnorm never evaluated
        }
        checked += 1;
        // Reference decision: stage BNorm means bnorm rejected; any later
        // stage (or accept) means bnorm passed.
        let ref_pass = pq_core::first_sample_stage(&fg) != pq_core::SampleStage::BNorm;

        let mut inv_sum = 0.0f64;
        for k in 0..N {
            let ang = -std::f64::consts::PI * (2 * k + 1) as f64 / N as f64;
            let (zr, zi) = (ang.cos(), ang.sin());
            // Evaluate f and g at z via Horner-free incremental powers.
            let (mut pr, mut pi) = (1.0f64, 0.0f64); // z^0
            let (mut fr, mut fi, mut gr, mut gi) = (0.0f64, 0.0, 0.0, 0.0);
            for j in 0..N {
                fr += fg.f[j] as f64 * pr;
                fi += fg.f[j] as f64 * pi;
                gr += fg.g[j] as f64 * pr;
                gi += fg.g[j] as f64 * pi;
                let npr = pr * zr - pi * zi;
                pi = pr * zi + pi * zr;
                pr = npr;
            }
            inv_sum += 1.0 / (fr * fr + fi * fi + gr * gr + gi * gi);
        }
        let q = 12289.0f64;
        let bnorm = q * q / N as f64 * inv_sum;
        let my_pass = bnorm < bound;
        let margin = (bnorm - bound).abs();
        if my_pass == ref_pass {
            agree += 1;
        } else {
            if margin > max_margin_disagree {
                max_margin_disagree = margin;
            }
        }
        if margin < 1.0 {
            near += 1;
        }
    }
    println!("bnorm formula: checked={checked} agree={agree} near-bound(<1.0)={near}");
    println!("max |bnorm-bound| among disagreements: {max_margin_disagree:.6}");
    assert!(checked > 400, "not enough sqnorm-passing samples");
    // The formula must agree except possibly hair's-breadth borderline cases.
    assert!(
        (checked - agree) as f64 <= (checked as f64 * 0.002).max(1.0),
        "formula disagrees too often: {agree}/{checked}"
    );
}

#[test]
#[ignore = "measurement harness, run explicitly with --ignored --nocapture"]
fn first_sample_stage_rates() {
    const SEEDS: u32 = 4000;
    let (mut lim, mut sq, mut inv, mut acc) = (0u32, 0u32, 0u32, 0u32);
    for i in 0..SEEDS {
        let mut e = [0u8; 32];
        e[..4].copy_from_slice(&i.to_le_bytes());
        e[4] = 0xA5;
        let fg = sample_fg(&keygen_seed_from_entropy(&e));
        let lim_ok = fg.f.iter().chain(fg.g.iter()).all(|&c| c > -16 && c < 16);
        if !lim_ok {
            continue;
        }
        lim += 1;
        let sqn = |p: &[i8]| p.iter().map(|&c| (c as i32 * c as i32) as u32).sum::<u32>();
        if sqn(&fg.f) + sqn(&fg.g) >= 16823 {
            continue;
        }
        sq += 1;
        if pubkey_from_fg(&fg).is_none() {
            continue;
        }
        inv += 1;
        if complete_from_fg(&fg).is_some() {
            acc += 1;
        }
    }
    // Exact stage histogram (vendor classifier), same seeds.
    let mut hist = [0u32; 6];
    for i in 0..SEEDS {
        let mut e = [0u8; 32];
        e[..4].copy_from_slice(&i.to_le_bytes());
        e[4] = 0xA5;
        let fg = sample_fg(&keygen_seed_from_entropy(&e));
        hist[pq_core::first_sample_stage(&fg) as usize] += 1;
    }
    println!(
        "stage histogram: accepted={} lim={} sqnorm={} bnorm={} invert={} solve={}",
        hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
    );

    let pct = |a: u32, b: u32| 100.0 * a as f64 / b as f64;
    println!("seeds:                {SEEDS}");
    println!("lim pass:             {lim}  ({:.2}% of seeds)", pct(lim, SEEDS));
    println!("sqnorm pass:          {sq}  ({:.2}% | lim)", pct(sq, lim));
    println!("invertible:           {inv}  ({:.2}% | sqnorm)", pct(inv, sq));
    println!("accepted:             {acc}  ({:.2}% | invertible)  <- q", pct(acc, inv));
    println!("overall first-sample: {:.2}%", pct(acc, SEEDS));
    println!(
        "GPU-visible accept (lim&sqnorm&invert): {:.2}% of seeds",
        pct(inv, SEEDS)
    );
}

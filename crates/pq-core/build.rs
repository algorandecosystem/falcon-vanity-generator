//! Compiles the vendored Falcon `det1024` reference C (pinned commit in
//! `vendor/falcon/PINNED_COMMIT`) and links it into `pq-core`.
//!
//! Determinism is governed by `vendor/falcon/config.h`, which forces
//! `FALCON_FPEMU=1`, `FALCON_AVX2=0`, `FALCON_FMA=0`, and leaves
//! `FALCON_KG_CHACHA20` undefined. Those settings are what produced the
//! go-algorand KATs we validate against, so we MUST NOT override them here.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let falcon = manifest.join("../../vendor/falcon");

    // Standard Pornin reference translation units + the det1024 wrapper.
    // Order is irrelevant; every .c is a standalone TU.
    let srcs = [
        "codec.c",
        "common.c",
        "falcon.c",
        "deterministic.c",
        "fft.c",
        "fpr.c",
        "rng.c",
        "shake.c",
        "sign.c",
        "vrfy.c",
    ];

    // keygen.c is compiled via the shim (vendor-shim/pqv_shim.c, which
    // #includes it) so pq-vanity can reach its static helpers (mkgauss,
    // solve_NTRU). It must NOT also appear in `srcs`, or symbols would clash.
    let shim = manifest.join("../../vendor-shim/pqv_shim.c");

    let mut build = cc::Build::new();
    build.include(&falcon);
    for s in srcs {
        build.file(falcon.join(s));
    }
    build.file(&shim);
    build.opt_level(3);
    build.flag_if_supported("-fomit-frame-pointer");
    // The reference triggers a lot of pedantic warnings under modern clang/gcc;
    // they are noise here and config.h pins the security-relevant knobs.
    build.warnings(false);
    build.compile("falcon_det1024");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shim.display());
    println!(
        "cargo:rerun-if-changed={}",
        falcon.join("config.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        falcon.join("inner.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        falcon.join("falcon.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        falcon.join("deterministic.h").display()
    );
    for s in srcs {
        println!("cargo:rerun-if-changed={}", falcon.join(s).display());
    }
}

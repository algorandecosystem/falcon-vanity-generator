//! Build the CUDA kernel for the Blackwell targets, but only when the `cuda`
//! feature is enabled AND `nvcc` is available. Otherwise compile nothing and
//! let `src/lib.rs` fall back to a host-only stub. This keeps `cargo build`
//! green on machines without a CUDA toolkit (e.g. the dev box), while
//! `cargo build -p pq-cuda --features cuda` (or via `pq-vanity --features cuda`)
//! compiles the real kernel on a Blackwell box.
//!
//! We invoke `nvcc` directly (rather than via the `cc` crate) because `cc`
//! forwards GCC-only host flags like `-ffunction-sections` straight to nvcc,
//! which rejects them. Host flags must go through `-Xcompiler`.
//!
//! Target SMs (overridable via `PQ_CUDA_ARCHS`, comma-separated):
//!   - `120` → RTX PRO 6000 Blackwell  (`-gencode arch=compute_120,code=sm_120`)
//!   - `121` → DGX Spark GB10           (`-gencode arch=compute_121,code=sm_121`)
//!
//! Requested arches are filtered against what the local `nvcc` actually
//! supports (`sm_121a` exists only on some toolkits; CUDA 13.0 has `sm_121`).
//!
//! The kernel MUST stay FP64-free (Blackwell is ~1:64 FP64).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_built)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../cuda/pq_vanity.cu");
    println!("cargo:rerun-if-env-changed=PQ_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        println!("cargo:warning=pq-cuda: built without the `cuda` feature (host stub).");
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=pq-cuda: `cuda` feature enabled but nvcc not found; \
             building host stub. Install the CUDA toolkit or set CUDA_PATH."
        );
        return;
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cuda/pq_vanity.cu");
    let obj = out_dir.join("pq_vanity.o");
    let lib = out_dir.join("libpq_vanity_cuda.a");

    let archs = env::var("PQ_CUDA_ARCHS").unwrap_or_else(|_| "120,121".to_string());
    let supported = supported_gpu_codes(&nvcc);

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3")
        .arg("--std=c++17")
        .arg("-lineinfo")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-Xptxas")
        .arg("-v");

    let mut built_any = false;
    for arch in archs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let code = format!("sm_{arch}");
        if !supported.is_empty() && !supported.iter().any(|c| c == &code) {
            println!(
                "cargo:warning=pq-cuda: nvcc does not support {code}; skipping \
                 (supported: {}).",
                supported.join(" ")
            );
            continue;
        }
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{arch},code={code}"));
        built_any = true;
    }
    if !built_any {
        panic!(
            "pq-cuda: none of the requested arches ({archs}) are supported by nvcc \
             (supported: {}). Set PQ_CUDA_ARCHS.",
            supported.join(" ")
        );
    }

    cmd.arg("-c").arg(&kernel).arg("-o").arg(&obj);
    run(cmd, "nvcc compile");

    // Archive the object into a static lib for the Rust linker.
    let _ = std::fs::remove_file(&lib);
    run(
        {
            let mut ar = Command::new("ar");
            ar.arg("crs").arg(&lib).arg(&obj);
            ar
        },
        "ar archive",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pq_vanity_cuda");
    if let Some(libdir) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", libdir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    // nvcc-compiled host code pulls in the C++ runtime.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=cuda_built");
}

fn run(mut cmd: Command, what: &str) {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("pq-cuda: failed to spawn {what} ({cmd:?}): {e}"));
    // Surface nvcc's -Xptxas -v stats as cargo warnings.
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning=pq-cuda[{what}]: {line}");
        }
    }
    if !out.status.success() {
        // Known CUDA(<=13.1)/glibc(>=2.41) header clash: point at the fix.
        if stderr.contains("exception specification is incompatible") || stderr.contains("rsqrt") {
            println!(
                "cargo:warning=pq-cuda: this looks like the CUDA/glibc rsqrt header clash. \
                 Run `make patch-cuda-rsqrt` (or scripts/patch-cuda-rsqrt.sh) once, then rebuild."
            );
        }
        panic!("pq-cuda: {what} failed ({:?})", out.status);
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        let cand = PathBuf::from(p).join("bin/nvcc");
        if cand.exists() {
            return Some(cand);
        }
    }
    for base in ["/usr/local/cuda", "/opt/cuda"] {
        let cand = PathBuf::from(base).join("bin/nvcc");
        if cand.exists() {
            return Some(cand);
        }
    }
    if Command::new("nvcc")
        .arg("--version")
        .output()
        .map_or(false, |o| o.status.success())
    {
        return Some(PathBuf::from("nvcc"));
    }
    None
}

/// Parse `nvcc --list-gpu-code` into ["sm_75", ..., "sm_121"]. Empty on failure.
fn supported_gpu_codes(nvcc: &Path) -> Vec<String> {
    let Ok(out) = Command::new(nvcc).arg("--list-gpu-code").output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .filter(|s| s.starts_with("sm_"))
        .map(|s| s.to_string())
        .collect()
}

fn cuda_lib_dir() -> Option<PathBuf> {
    let bases: Vec<PathBuf> = env::var("CUDA_PATH")
        .map(|p| vec![PathBuf::from(p)])
        .unwrap_or_else(|_| vec![PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")]);
    for b in bases {
        for sub in [
            "lib64",
            "lib",
            "lib/x86_64-linux-gnu",
            "lib/aarch64-linux-gnu",
        ] {
            let d = b.join(sub);
            if d.join("libcudart.so").exists()
                || d.join("libcudart.so.13").exists()
                || d.join("libcudart.so.12").exists()
            {
                return Some(d);
            }
        }
    }
    None
}

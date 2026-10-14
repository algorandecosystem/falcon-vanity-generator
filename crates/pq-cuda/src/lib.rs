//! CUDA backend for `pq-vanity`.
//!
//! When built with the `cuda` feature on a machine with `nvcc`, this links the
//! device kernel in `cuda/pq_vanity.cu` and exposes [`search_batch`]. Otherwise
//! it compiles a host-only stub so the workspace still builds everywhere.
//!
//! The device kernel computes, per thread, the Falcon-1024 **public half**
//! (`h = g·f⁻¹ mod 12289`) from mnemonic entropy (PQK seed hash on-device), encodes `pk`, hashes the PQ preimage
//! with SHA-512/256 over a salt range, and matches `addr32` against a table of
//! prefix/suffix targets (the display checksum is hashed only when a suffix
//! target's prefix matched) — all integer-only (no FP64). Hits return the
//! `(entropy, salt)`; the host re-derives and completes the key on the CPU
//! (`pq-core`). See `cuda/KERNEL.md` for the kernel status and the bit-exact
//! validation protocol against the CPU oracle.

/// Mnemonic entropy length (matches `pq_core::ENTROPY_SIZE`); the kernel
/// derives the 32-byte Falcon seed on-device via the "PQK" SHA-512/256 hash.
pub const ENTROPY_SIZE: usize = 32;
/// Ring degree (length of the sampled polynomials f, g).
pub const N: usize = 1024;
/// Falcon-1024 public key size in bytes.
pub const PUBKEY_SIZE: usize = 1793;
/// PQ address size in bytes (addr32).
pub const ADDR_SIZE: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// The crate was built without the `cuda` feature / without nvcc.
    NotBuilt,
    /// A CUDA runtime call failed (negative code from the kernel shim).
    Runtime(i32),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NotBuilt => write!(
                f,
                "pq-cuda built without GPU support; rebuild with --features cuda on a CUDA box"
            ),
            CudaError::Runtime(c) => write!(f, "CUDA runtime error (code {c})"),
        }
    }
}

impl std::error::Error for CudaError {}

/// A single match found on the device. The host re-derives the full key from
/// `entropy` and applies `salt`.
#[derive(Debug, Clone)]
pub struct Hit {
    pub entropy: [u8; ENTROPY_SIZE],
    pub salt: u8,
}

/// Result of one device batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub hits: Vec<Hit>,
    /// True when the crate was built without nvcc (host stub) — i.e. `hits`
    /// are NOT real matches. The host must treat this as "not implemented".
    pub kernel_is_stub: bool,
}

/// Whether a real CUDA kernel is linked in (vs. the host stub).
pub const fn is_available() -> bool {
    cfg!(cuda_built)
}

#[cfg(cuda_built)]
mod ffi {
    use core::ffi::c_int;

    extern "C" {
        pub fn pq_cuda_device_count() -> c_int;
        pub fn pq_cuda_set_blocking_sync() -> c_int;
        pub fn pq_cuda_dbg_sample_fg(seeds: *const u8, n: u64, out_fg: *mut i8) -> c_int;
        pub fn pq_cuda_dbg_pubkey(
            seeds: *const u8,
            n: u64,
            out_pk: *mut u8,
            out_ok: *mut u8,
        ) -> c_int;
        pub fn pq_cuda_dbg_addr32(
            seeds: *const u8,
            n: u64,
            salt: u8,
            out_addr: *mut u8,
            out_ok: *mut u8,
        ) -> c_int;
        pub fn pq_cuda_dbg_accept(seeds: *const u8, n: u64, out_att: *mut i32) -> c_int;
        pub fn pq_cuda_search(
            base_entropy: *const u8, // [32]
            start_counter: u64,
            num_items: u64,
            max_salt: u32,
            targets: *const u64, // [n_targets * 4]
            n_targets: c_int,
            out_hits: *mut u8, // [max_hits * (ENTROPY_SIZE + 1)]
            max_hits: c_int,
            out_stub: *mut c_int,
        ) -> c_int;
        pub fn ed_cuda_dbg_pubkey(seeds: *const u8, n: u64, fast: c_int, out_pk: *mut u8) -> c_int;
        pub fn pq_cuda_dbg_match(
            addrs: *const u8,
            n: u64,
            targets: *const u64,
            n_targets: c_int,
            out: *mut u8,
        ) -> c_int;
        pub fn ed_cuda_dbg_match(
            addrs: *const u8,
            n: u64,
            targets: *const u64,
            n_targets: c_int,
            out: *mut u8,
        ) -> c_int;
        pub fn ed_cuda_search(
            base_seed: *const u8, // [32]
            start_counter: u64,
            num_items: u64,
            targets: *const u64, // [n_targets * 4]
            n_targets: c_int,
            out_hits: *mut u8, // [max_hits * ED_SEED_SIZE]
            max_hits: c_int,
        ) -> c_int;
    }
}

/// Debug/validation: run the device sampler for each seed and return a flat
/// buffer of `entropies.len() * 2 * N` int8 values — per seed, `f[0..N]` then
/// `g[0..N]`. Compared against `pq_core::fast::sample_fg` in `gpu-selftest`.
pub fn dbg_sample_fg(entropies: &[[u8; ENTROPY_SIZE]]) -> Result<Vec<i8>, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = entropies;
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        let n = entropies.len();
        let mut flat = Vec::with_capacity(n * ENTROPY_SIZE);
        for s in entropies {
            flat.extend_from_slice(s);
        }
        let mut out = vec![0i8; n * 2 * N];
        // SAFETY: flat is n*ENTROPY_SIZE bytes; out is n*2*N i8.
        let rc = unsafe { ffi::pq_cuda_dbg_sample_fg(flat.as_ptr(), n as u64, out.as_mut_ptr()) };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok(out)
    }
}

/// Debug/validation: run the device public-half (sample → compute_public →
/// modq_encode) for each seed. Returns `(pk_flat, ok)` where `pk_flat` is
/// `entropies.len() * PUBKEY_SIZE` bytes and `ok[i]` is 1 if seed i's f was
/// invertible. Compared against `pq_core::fast::pubkey_from_fg` in `gpu-selftest`.
pub fn dbg_pubkey(entropies: &[[u8; ENTROPY_SIZE]]) -> Result<(Vec<u8>, Vec<u8>), CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = entropies;
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        let n = entropies.len();
        let mut flat = Vec::with_capacity(n * ENTROPY_SIZE);
        for s in entropies {
            flat.extend_from_slice(s);
        }
        let mut pk = vec![0u8; n * PUBKEY_SIZE];
        let mut ok = vec![0u8; n];
        // SAFETY: flat is n*ENTROPY_SIZE; pk is n*PUBKEY_SIZE; ok is n.
        let rc = unsafe {
            ffi::pq_cuda_dbg_pubkey(flat.as_ptr(), n as u64, pk.as_mut_ptr(), ok.as_mut_ptr())
        };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok((pk, ok))
    }
}

/// Debug/validation: device addr32 = SHA512_256("PQA"||"f1"||salt||pk) for each
/// seed at the given salt. Returns `(addr_flat, ok)` (`entropies.len()*ADDR_SIZE`,
/// and `ok[i]`=1 if invertible). Compared against `pq_core::pq_address`.
pub fn dbg_addr32(entropies: &[[u8; ENTROPY_SIZE]], salt: u8) -> Result<(Vec<u8>, Vec<u8>), CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = (entropies, salt);
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        let n = entropies.len();
        let mut flat = Vec::with_capacity(n * ENTROPY_SIZE);
        for s in entropies {
            flat.extend_from_slice(s);
        }
        let mut addr = vec![0u8; n * ADDR_SIZE];
        let mut ok = vec![0u8; n];
        // SAFETY: flat n*ENTROPY_SIZE; addr n*ADDR_SIZE; ok n.
        let rc = unsafe {
            ffi::pq_cuda_dbg_addr32(
                flat.as_ptr(),
                n as u64,
                salt,
                addr.as_mut_ptr(),
                ok.as_mut_ptr(),
            )
        };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok((addr, ok))
    }
}

/// Debug/validation: the retry loop's 0-based accepted-attempt index for each
/// entropy (`-1` = capped, ~8e-6 of seeds). Compared against the CPU oracle
/// `pq_core::first_visible_accept` in `gpu-selftest`; agreement is exact except
/// for FP32-vs-FPEMU bnorm borderline samples.
pub fn dbg_accept(entropies: &[[u8; ENTROPY_SIZE]]) -> Result<Vec<i32>, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = entropies;
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        let n = entropies.len();
        let mut flat = Vec::with_capacity(n * ENTROPY_SIZE);
        for s in entropies {
            flat.extend_from_slice(s);
        }
        let mut att = vec![0i32; n];
        // SAFETY: flat is n*ENTROPY_SIZE bytes; att is n i32.
        let rc = unsafe { ffi::pq_cuda_dbg_accept(flat.as_ptr(), n as u64, att.as_mut_ptr()) };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok(att)
    }
}

/// Ask CUDA to put the host thread to sleep (not spin) while waiting on the GPU,
/// freeing the CPU core during the multi-second kernel. Call once before any
/// other CUDA use. No-op (returns 0) in the host stub. A nonzero return (e.g.
/// the context is already active) is non-fatal — the default spin policy stays.
pub fn set_blocking_sync() -> i32 {
    #[cfg(cuda_built)]
    // SAFETY: no pointers; sets a device flag.
    unsafe {
        ffi::pq_cuda_set_blocking_sync()
    }
    #[cfg(not(cuda_built))]
    {
        0
    }
}

/// Number of CUDA devices visible (0 when built as a stub).
pub fn device_count() -> i32 {
    #[cfg(cuda_built)]
    // SAFETY: pure CUDA runtime query, no pointers.
    unsafe {
        ffi::pq_cuda_device_count()
    }
    #[cfg(not(cuda_built))]
    {
        0
    }
}

/// Search target in device form: `[pmask, pval, smask, sval]` (see
/// `pq-vanity/src/targets.rs`). At most [`MAX_TARGETS`] per batch.
pub type DeviceTarget = [u64; 4];
/// Device target-table capacity (both kernels).
pub const MAX_TARGETS: usize = 1024;

/// Debug/validation: run a kernel's device target matcher on raw 32-byte
/// addresses (`ed = true` → the ed25519 TU's copy). Returns one 0/1 per address.
pub fn dbg_match(addrs: &[[u8; 32]], targets: &[DeviceTarget], ed: bool) -> Result<Vec<u8>, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = (addrs, targets, ed);
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        assert!(targets.len() <= MAX_TARGETS);
        let n = addrs.len();
        let flat: Vec<u8> = addrs.iter().flatten().copied().collect();
        let mut out = vec![0u8; n];
        let f = if ed { ffi::ed_cuda_dbg_match } else { ffi::pq_cuda_dbg_match };
        // SAFETY: flat is n*32 bytes, targets n_targets*4 u64, out n bytes.
        let rc = unsafe {
            f(
                flat.as_ptr(),
                n as u64,
                targets.as_ptr().cast(),
                targets.len() as core::ffi::c_int,
                out.as_mut_ptr(),
            )
        };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok(out)
    }
}

/// Launch one search batch of `num_items` entropies derived from `base_entropy` and
/// `start_counter`, matching `addr32` against `targets`, sweeping salts
/// `0..=max_salt` (first matching salt per key).
pub fn search_batch(
    base_entropy: &[u8; ENTROPY_SIZE],
    start_counter: u64,
    num_items: u64,
    max_salt: u32,
    targets: &[DeviceTarget],
    max_hits: usize,
) -> Result<BatchResult, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = (
            base_entropy,
            start_counter,
            num_items,
            max_salt,
            targets,
            max_hits,
        );
        Err(CudaError::NotBuilt)
    }

    #[cfg(cuda_built)]
    {
        assert!(targets.len() <= MAX_TARGETS);
        let stride = ENTROPY_SIZE + 1;
        let mut out = vec![0u8; max_hits * stride];
        let mut stub: core::ffi::c_int = 0;
        // SAFETY: all pointers are valid for the lengths passed; out is sized
        // max_hits*stride; prefix len matches the slice.
        let n = unsafe {
            ffi::pq_cuda_search(
                base_entropy.as_ptr(),
                start_counter,
                num_items,
                max_salt,
                targets.as_ptr().cast(),
                targets.len() as core::ffi::c_int,
                out.as_mut_ptr(),
                max_hits as core::ffi::c_int,
                &mut stub,
            )
        };
        if n < 0 {
            return Err(CudaError::Runtime(n));
        }
        let n = (n as usize).min(max_hits);
        let mut hits = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * stride;
            let mut entropy = [0u8; ENTROPY_SIZE];
            entropy.copy_from_slice(&out[base..base + ENTROPY_SIZE]);
            hits.push(Hit {
                entropy,
                salt: out[base + ENTROPY_SIZE],
            });
        }
        Ok(BatchResult {
            hits,
            kernel_is_stub: stub != 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Classic Algorand (Ed25519) accounts — cuda/ed25519_vanity.cu
// ---------------------------------------------------------------------------

/// Ed25519 seed length (the 32-byte mnemonic key of a classic account).
pub const ED_SEED_SIZE: usize = 32;
/// Ed25519 public key length (== the classic `addr32`).
pub const ED_PK_SIZE: usize = 32;

/// Debug/validation: device seed → public key for each seed. `fast = false`
/// runs the vendored constant-time reference path (`ge_scalarmult_base`),
/// `fast = true` the search path (variable-time table lookup + batched
/// inversion). Returns `seeds.len() * ED_PK_SIZE` bytes.
pub fn ed_dbg_pubkey(seeds: &[[u8; ED_SEED_SIZE]], fast: bool) -> Result<Vec<u8>, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = (seeds, fast);
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        let n = seeds.len();
        let flat: Vec<u8> = seeds.iter().flatten().copied().collect();
        let mut out = vec![0u8; n * ED_PK_SIZE];
        // SAFETY: flat is n*ED_SEED_SIZE bytes; out is n*ED_PK_SIZE bytes.
        let rc = unsafe {
            ffi::ed_cuda_dbg_pubkey(flat.as_ptr(), n as u64, fast as core::ffi::c_int, out.as_mut_ptr())
        };
        if rc < 0 {
            return Err(CudaError::Runtime(rc));
        }
        Ok(out)
    }
}

/// Launch one classic search batch: `num_items` seeds, each `base_seed` with
/// the 8-byte LE counter `start_counter + i` (wrapping) in bytes `[24..32]`,
/// matching the public key (== addr32) against `targets`. Returns up to
/// `max_hits` matching seeds.
pub fn ed_search_batch(
    base_seed: &[u8; ED_SEED_SIZE],
    start_counter: u64,
    num_items: u64,
    targets: &[DeviceTarget],
    max_hits: usize,
) -> Result<Vec<[u8; ED_SEED_SIZE]>, CudaError> {
    #[cfg(not(cuda_built))]
    {
        let _ = (base_seed, start_counter, num_items, targets, max_hits);
        Err(CudaError::NotBuilt)
    }
    #[cfg(cuda_built)]
    {
        assert!(targets.len() <= MAX_TARGETS);
        let mut out = vec![0u8; max_hits * ED_SEED_SIZE];
        // SAFETY: base_seed is 32 bytes; out is max_hits*ED_SEED_SIZE;
        // targets is n_targets*4 u64.
        let n = unsafe {
            ffi::ed_cuda_search(
                base_seed.as_ptr(),
                start_counter,
                num_items,
                targets.as_ptr().cast(),
                targets.len() as core::ffi::c_int,
                out.as_mut_ptr(),
                max_hits as core::ffi::c_int,
            )
        };
        if n < 0 {
            return Err(CudaError::Runtime(n));
        }
        Ok(out
            .chunks_exact(ED_SEED_SIZE)
            .take(n as usize)
            .map(|c| c.try_into().unwrap())
            .collect())
    }
}

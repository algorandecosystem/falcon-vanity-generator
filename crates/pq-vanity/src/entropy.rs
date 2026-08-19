//! Multi-source entropy mixing for search seeds (defense in depth).
//!
//! Every emitted 32-byte entropy is `SHA-512/256(tag ‖ sources ‖ counter)`.
//! Sources are gathered *before* hashing and none can observe the others, so
//! the output is unpredictable as long as ANY single source is — a broken or
//! even adversarial source cannot degrade the mix below the strongest input.
//!
//! | source                        | liveness    | freshness                     |
//! |-------------------------------|-------------|-------------------------------|
//! | OS CSPRNG (`getrandom`)       | MANDATORY   | fresh per output              |
//! | CPU DRNG (RDSEED / RNDR)      | best-effort | fresh per output              |
//! | timing jitter                 | always      | cached; renewed by `refresh()`|
//! | TPM 2.0 (`/dev/tpmrm0`)       | best-effort | cached; renewed by `refresh()`|
//! | `--extra-entropy` (user)      | optional    | static                        |
//! | process-local counter         | always      | monotonic (uniqueness)        |
//!
//! The cached sources exist because a TPM `GetRandom` costs 1–50 ms: the CPU
//! search calls `refresh()` only after each emitted hit (so consecutive hits
//! never share the cached contributions), the GPU path once per batch.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use pq_core::{sha512_256, ENTROPY_SIZE};

const MIX_TAG: &[u8; 24] = b"pq-vanity/entropy-mix/v1";

struct Cached {
    tpm: [u8; 32],    // zeros until the first successful TPM read
    jitter: [u8; 32],
}

pub struct EntropyMix {
    tpm: Option<Mutex<File>>,
    cached: RwLock<Cached>,
    extra: Option<[u8; 32]>,
    ctr: AtomicU64,
    desc: String,
}

impl EntropyMix {
    /// Probe all sources once. `extra` is user-supplied bytes (already read
    /// from a file or the command line); it is hashed, never used verbatim.
    pub fn new(extra: Option<&[u8]>) -> Self {
        let mut tpm_init = [0u8; 32];
        let tpm = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/tpmrm0")
            .ok()
            .and_then(|mut f| tpm2_get_random(&mut f, &mut tpm_init).ok().map(|()| f));

        let mut cpu_probe = [0u8; 32];
        let cpu_live = cpu_drng_fill(&mut cpu_probe);

        let mut desc = String::from("getrandom");
        if cpu_live {
            desc.push('+');
            desc.push_str(CPU_DRNG_NAME);
        }
        desc.push_str("+jitter");
        if tpm.is_some() {
            desc.push_str("+tpm2");
        }
        if extra.is_some() {
            desc.push_str("+extra");
        }

        EntropyMix {
            tpm: tpm.map(Mutex::new),
            cached: RwLock::new(Cached {
                tpm: tpm_init,
                jitter: jitter32(),
            }),
            extra: extra.map(sha512_256),
            ctr: AtomicU64::new(0),
            desc,
        }
    }

    /// The live source set, e.g. `getrandom+rdseed+jitter+tpm2`.
    pub fn sources(&self) -> &str {
        &self.desc
    }

    /// Renew the cached contributions (TPM + jitter). A TPM read error keeps
    /// the previous bytes — stale entropy is still entropy.
    pub fn refresh(&self) {
        let jitter = jitter32();
        let mut tpm_new = None;
        if let Some(t) = &self.tpm {
            let mut f = t.lock().unwrap();
            let mut b = [0u8; 32];
            if tpm2_get_random(&mut f, &mut b).is_ok() {
                tpm_new = Some(b);
            }
        }
        let mut c = self.cached.write().unwrap();
        c.jitter = jitter;
        if let Some(b) = tpm_new {
            c.tpm = b;
        }
    }

    /// One fresh 32-byte output. Fails only if the OS CSPRNG fails.
    pub fn next(&self) -> anyhow::Result<[u8; ENTROPY_SIZE]> {
        let mut buf = [0u8; 24 + 5 * 32 + 8];
        buf[..24].copy_from_slice(MIX_TAG);
        getrandom::getrandom(&mut buf[24..56])
            .map_err(|e| anyhow::anyhow!("getrandom failed: {e}"))?;
        let mut cpu = [0u8; 32];
        let _ = cpu_drng_fill(&mut cpu); // best-effort; zeros on failure
        buf[56..88].copy_from_slice(&cpu);
        {
            let c = self.cached.read().unwrap();
            buf[88..120].copy_from_slice(&c.tpm);
            buf[120..152].copy_from_slice(&c.jitter);
        }
        if let Some(x) = &self.extra {
            buf[152..184].copy_from_slice(x);
        }
        let ctr = self.ctr.fetch_add(1, Ordering::Relaxed);
        buf[184..192].copy_from_slice(&ctr.to_le_bytes());
        Ok(sha512_256(&buf))
    }
}

/// Raw `TPM2_GetRandom` over the kernel's resource-managed device — no TSS
/// stack needed. Each command must be one complete write; the response is one
/// read. Loops because a TPM may return fewer bytes than requested (its
/// digest size caps a single reply).
fn tpm2_get_random(f: &mut File, out: &mut [u8; 32]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let mut filled = 0usize;
    while filled < out.len() {
        let want = (out.len() - filled) as u16;
        #[rustfmt::skip]
        let cmd: [u8; 12] = [
            0x80, 0x01,                   // TPM_ST_NO_SESSIONS
            0x00, 0x00, 0x00, 0x0c,       // commandSize = 12
            0x00, 0x00, 0x01, 0x7b,       // TPM_CC_GetRandom
            (want >> 8) as u8, want as u8, // bytesRequested
        ];
        f.write_all(&cmd)?;
        let mut resp = [0u8; 4096];
        let n = f.read(&mut resp)?;
        if n < 12 {
            return Err(Error::new(ErrorKind::InvalidData, "short TPM response"));
        }
        let rc = u32::from_be_bytes(resp[6..10].try_into().unwrap());
        if rc != 0 {
            return Err(Error::new(ErrorKind::Other, format!("TPM rc {rc:#x}")));
        }
        let blen = u16::from_be_bytes(resp[10..12].try_into().unwrap()) as usize;
        if blen == 0 || 12 + blen > n {
            return Err(Error::new(ErrorKind::InvalidData, "bad TPM2B length"));
        }
        let take = blen.min(out.len() - filled);
        out[filled..filled + take].copy_from_slice(&resp[12..12 + take]);
        filled += take;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
const CPU_DRNG_NAME: &str = "rdseed";
#[cfg(target_arch = "aarch64")]
const CPU_DRNG_NAME: &str = "rndr";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const CPU_DRNG_NAME: &str = "cpu-drng";

/// Fill from the CPU's conditioned hardware DRNG. Returns false if the
/// instruction is unavailable; partial fills (transient depletion) leave the
/// remaining words zero, which the hash mix tolerates by construction.
#[cfg(target_arch = "x86_64")]
fn cpu_drng_fill(out: &mut [u8; 32]) -> bool {
    if !std::arch::is_x86_feature_detected!("rdseed") {
        return false;
    }
    let mut got = false;
    for chunk in out.chunks_exact_mut(8) {
        let mut v: u64 = 0;
        // RDSEED reads a shared conditioner that can be transiently empty.
        for _ in 0..64 {
            if unsafe { core::arch::x86_64::_rdseed64_step(&mut v) } == 1 {
                chunk.copy_from_slice(&v.to_le_bytes());
                got = true;
                break;
            }
            core::hint::spin_loop();
        }
    }
    got
}

#[cfg(target_arch = "aarch64")]
fn cpu_drng_fill(out: &mut [u8; 32]) -> bool {
    if !std::arch::is_aarch64_feature_detected!("rand") {
        return false;
    }
    let mut got = false;
    for chunk in out.chunks_exact_mut(8) {
        let v: u64;
        let ok: u64;
        // RNDR (FEAT_RNG): on failure sets Z and returns 0.
        unsafe {
            core::arch::asm!(
                "mrs {v}, S3_3_C2_C4_0",
                "cset {ok}, ne",
                v = out(reg) v,
                ok = out(reg) ok,
                options(nomem, nostack),
            );
        }
        if ok == 1 {
            chunk.copy_from_slice(&v.to_le_bytes());
            got = true;
        }
    }
    got
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn cpu_drng_fill(_out: &mut [u8; 32]) -> bool {
    false
}

/// Timing jitter: hash the nanosecond deltas of a memory-touching loop.
/// Auxiliary source only — its quality is never relied upon. ~100–300 µs.
fn jitter32() -> [u8; 32] {
    let mut samples = [0u8; 4096];
    let mut acc: u64 = 0x9e3779b97f4a7c15;
    let mut prev = std::time::Instant::now();
    for i in 0..samples.len() {
        // LCG-driven scattered writes induce cache/scheduler noise.
        acc = acc
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = (acc >> 33) as usize % samples.len();
        samples[idx] = samples[idx].wrapping_add(1);
        let now = std::time::Instant::now();
        let d = now.duration_since(prev).subsec_nanos();
        prev = now;
        samples[i] ^= (d as u8) ^ ((d >> 8) as u8);
    }
    sha512_256(&samples)
}

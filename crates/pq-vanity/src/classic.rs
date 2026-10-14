//! Classic Algorand (Ed25519) vanity search — `--scheme ed25519`.
//!
//! A classic address IS the Ed25519 public key, so a candidate costs one
//! SHA-512 plus one fixed-base scalar multiplication. The 32-byte seed is the
//! mnemonic key (`algokey import -m`, `goal account import -m`).
//!
//! Entropy hygiene is the same as the f1 path:
//! - every key derives from an `EntropyMix` output (getrandom + CPU DRNG +
//!   jitter + TPM2 + `--extra-entropy`, hashed together);
//! - keys are scanned in *runs* that share one mix output in bytes `[0..24]`
//!   and differ only in an 8-byte LE counter in `[24..32]` that starts at the
//!   output's own bytes (wrapping) — no zero-suffix fingerprint;
//! - at most one key is emitted per run (a GPU batch, or a CPU worker run), so
//!   reported keys never share base entropy; the cached mix contributions are
//!   refreshed after every CPU hit and every GPU batch.
//!
//! Unlike Falcon keygen, per-key mixing (getrandom + RDSEED per key) would be
//! the bottleneck here, hence the runs on the CPU path too.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use pq_core::{key_to_mnemonic, ClassicKeypair, CLASSIC_SEED_SIZE};

use crate::entropy::EntropyMix;
use crate::search::hex_lower;
use crate::targets::TargetSet;

/// Keys per CPU worker run (one mix output each). ~0.5–1 s of work per core.
const CPU_RUN: u64 = 1 << 15;

const ROUNDTRIP_MSG: &[u8] = b"pq-vanity-classic-emit-check";

/// `base` with the 8-byte LE `counter` in bytes `[24..32]` — the device's
/// `ed_derive_seed` layout.
fn seed_at(base: &[u8; CLASSIC_SEED_SIZE], counter: u64) -> [u8; CLASSIC_SEED_SIZE] {
    let mut s = *base;
    s[24..32].copy_from_slice(&counter.to_le_bytes());
    s
}

/// The counter value in a mix output's own bytes `[24..32]`, so the run's
/// first seed is the mix output verbatim.
fn run_start(base: &[u8; CLASSIC_SEED_SIZE]) -> u64 {
    u64::from_le_bytes(base[24..32].try_into().unwrap())
}

pub fn derive(seed: &[u8; CLASSIC_SEED_SIZE]) {
    let kp = ClassicKeypair::from_seed(seed);
    println!("scheme:   ed25519 (classic)");
    println!("mnemonic: {}", key_to_mnemonic(seed));
    println!("pubkey:   {}", B64.encode(kp.public_key()));
    println!("address:  {}", kp.address());
}

// ---------------------------------------------------------------------------
// CPU search
// ---------------------------------------------------------------------------

struct Shared {
    mix: Arc<EntropyMix>,
    targets: Arc<TargetSet>,
    count: u64,
    out_dir: PathBuf,
    keys: AtomicU64,
    hits: AtomicU64,
    stop: AtomicBool,
    io_lock: Mutex<()>,
}

pub fn cmd_search(
    a: crate::SearchArgs,
    targets: Arc<TargetSet>,
    count: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !a.allow_non_canonical,
        "--allow-non-canonical applies to f1 only (classic addresses have no salt)"
    );
    eprintln!(
        "searching (ed25519 / classic): {} threads={} count={count}",
        targets.describe(),
        a.threads
    );
    eprintln!("~{:.3e} keys per hit (expectation, any target)", 1.0 / targets.hit_rate());
    let mix = Arc::new(EntropyMix::new(
        crate::load_extra_entropy(&a.extra_entropy)?.as_deref(),
    ));
    eprintln!("entropy sources: {}", mix.sources());
    std::fs::create_dir_all(&a.out)?;

    let sh = Arc::new(Shared {
        mix,
        targets,
        count: count as u64,
        out_dir: a.out,
        keys: AtomicU64::new(0),
        hits: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        io_lock: Mutex::new(()),
    });
    let start = Instant::now();
    let handles: Vec<_> = (0..a.threads)
        .map(|_| {
            let sh = Arc::clone(&sh);
            std::thread::spawn(move || cpu_worker(&sh))
        })
        .collect();
    while !sh.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(1000));
        if a.quiet || sh.stop.load(Ordering::Relaxed) {
            continue;
        }
        let el = start.elapsed().as_secs_f64().max(1e-9);
        let keys = sh.keys.load(Ordering::Relaxed);
        eprintln!(
            "[{el:>6.0}s] keys={keys} ({:.3}M/s)  hits={}",
            keys as f64 / el / 1e6,
            sh.hits.load(Ordering::Relaxed)
        );
    }
    for h in handles {
        let _ = h.join();
    }
    eprintln!("done: {} hit(s)", sh.hits.load(Ordering::Relaxed));
    Ok(())
}

fn cpu_worker(sh: &Shared) {
    'runs: while !sh.stop.load(Ordering::Relaxed) {
        let Ok(base) = sh.mix.next() else {
            sh.stop.store(true, Ordering::Relaxed);
            return;
        };
        let c0 = run_start(&base);
        let mut done = 0u64;
        for i in 0..CPU_RUN {
            let seed = seed_at(&base, c0.wrapping_add(i));
            let kp = ClassicKeypair::from_seed(&seed);
            done += 1;
            // Prefix-only targets never build the checksum (see TargetSet).
            if let Some(t) = sh.targets.first_match(&kp.address()) {
                sh.keys.fetch_add(done, Ordering::Relaxed);
                record_cpu_hit(sh, &seed, &kp, t);
                continue 'runs; // one hit per run: next hit gets a fresh base
            }
            if done == 1024 {
                sh.keys.fetch_add(done, Ordering::Relaxed);
                done = 0;
                if sh.stop.load(Ordering::Relaxed) {
                    return;
                }
            }
        }
        sh.keys.fetch_add(done, Ordering::Relaxed);
    }
}

fn record_cpu_hit(sh: &Shared, seed: &[u8; CLASSIC_SEED_SIZE], kp: &ClassicKeypair, target: usize) {
    // Multi-target: the first hit retires its target (lost races drop out).
    if !sh.targets.claim(target) {
        return;
    }
    let n = sh.hits.fetch_add(1, Ordering::SeqCst);
    if n >= sh.count {
        sh.hits.fetch_sub(1, Ordering::SeqCst);
        sh.stop.store(true, Ordering::Relaxed);
        return;
    }
    let roundtrip = kp.roundtrip(ROUNDTRIP_MSG);
    let _guard = sh.io_lock.lock().unwrap();
    let path = write_hit_record(&sh.out_dir, seed, kp, sh.mix.sources());
    print_hit(n + 1, seed, kp, roundtrip, path.as_deref(), &sh.targets.list[target].label());
    if n + 1 >= sh.count || !sh.targets.any_active() {
        sh.stop.store(true, Ordering::Relaxed);
    } else {
        // Next hit never shares the cached TPM/jitter contributions.
        sh.mix.refresh();
    }
}

pub fn cmd_bench(seconds: u64, threads: usize) -> anyhow::Result<()> {
    eprintln!("benchmarking ed25519 (classic) keygen for {seconds}s on {threads} threads...");
    let stop = Arc::new(AtomicBool::new(false));
    let keys = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (stop, keys) = (Arc::clone(&stop), Arc::clone(&keys));
            std::thread::spawn(move || {
                let mut base = [0u8; CLASSIC_SEED_SIZE];
                base[0] = t as u8;
                let mut n = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..256 {
                        let kp = ClassicKeypair::from_seed(&seed_at(&base, n));
                        std::hint::black_box(kp.public_key());
                        n += 1;
                    }
                }
                keys.fetch_add(n, Ordering::Relaxed);
            })
        })
        .collect();
    std::thread::sleep(Duration::from_secs(seconds));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    let el = start.elapsed().as_secs_f64();
    let k = keys.load(Ordering::Relaxed);
    println!("elapsed:   {el:.2}s");
    println!("keygen:    {k} keys  ({:.3}M/s)", k as f64 / el / 1e6);
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU search
// ---------------------------------------------------------------------------

pub fn cmd_gpu(a: crate::GpuArgs, targets: TargetSet, count: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !a.allow_non_canonical && a.max_salt.is_none(),
        "--allow-non-canonical / --max-salt apply to f1 only (classic addresses have no salt)"
    );

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, targets, count);
        anyhow::bail!(
            "GPU support is not compiled in. Build on a CUDA box with:\n  \
             cargo build -p pq-vanity --release --features cuda"
        );
    }

    #[cfg(feature = "cuda")]
    {
        anyhow::ensure!(
            pq_cuda::is_available(),
            "nvcc was not found at build time, so the kernel is a host stub. \
             Install the CUDA toolkit (or set CUDA_PATH) and rebuild --features cuda."
        );
        let _ = pq_cuda::set_blocking_sync();
        anyhow::ensure!(pq_cuda::device_count() >= 1, "no CUDA device visible");
        std::fs::create_dir_all(&a.out)?;
        let items = a.items.unwrap_or(1 << 26);
        eprintln!(
            "GPU search (ed25519 / classic): {} items/batch={items} count={count} \
             (~{:.3e} keys per hit, any target; device: {} GPU)",
            targets.describe(),
            1.0 / targets.hit_rate(),
            pq_cuda::device_count()
        );
        let mix = EntropyMix::new(crate::load_extra_entropy(&a.extra_entropy)?.as_deref());
        eprintln!("entropy sources: {}", mix.sources());

        let start = Instant::now();
        let (mut found, mut raw_hits, mut keys_done, mut siblings) = (0usize, 0u64, 0u128, 0u64);
        for batch in 0u64.. {
            if batch > 0 {
                mix.refresh();
            }
            let base = mix.next()?;
            // Only still-active targets go to the device.
            let table = targets.device_table();
            let seeds = pq_cuda::ed_search_batch(&base, run_start(&base), items, &table, a.max_hits)?;
            raw_hits += seeds.len() as u64;
            keys_done += items as u128;

            let mut batch_emitted = 0usize;
            for seed in &seeds {
                // Re-derive on the CPU oracle and re-match on the full display
                // string; never trust the device alone.
                let kp = ClassicKeypair::from_seed(seed);
                let display = kp.address().to_address_string();
                let Some(t) = targets.first_match_display(&display) else {
                    // Either its target was retired earlier in this batch, or
                    // the device is wrong.
                    if !targets.list.iter().any(|x| x.matches_display(&display)) {
                        eprintln!("WARNING: device hit failed CPU re-verification (discarded)");
                    }
                    continue;
                };
                if !a.allow_sibling_hits && batch_emitted >= 1 {
                    siblings += 1;
                    continue;
                }
                targets.claim(t);
                let roundtrip = kp.roundtrip(ROUNDTRIP_MSG);
                let path = write_hit_record(&a.out, seed, &kp, mix.sources());
                found += 1;
                batch_emitted += 1;
                print_hit(found as u64, seed, &kp, roundtrip, path.as_deref(), &targets.list[t].label());
                if found >= count || !targets.any_active() {
                    break;
                }
            }

            let el = start.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "[batch {batch}] keys={keys_done} ({:.2}M/s)  raw_hits={raw_hits}  usable={found}",
                keys_done as f64 / el / 1e6
            );
            if found >= count || !targets.any_active() || (a.batches != 0 && batch + 1 >= a.batches)
            {
                break;
            }
        }
        if siblings > 0 {
            eprintln!(
                "note: discarded {siblings} additional same-batch hit(s) \
                 (entropy hygiene — pass --allow-sibling-hits to keep them)"
            );
        }
        eprintln!("done: {found} address(es) in {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

pub fn cmd_gpu_selftest(n: usize) -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = n;
        anyhow::bail!("GPU support is not compiled in. Build on a CUDA box with --features cuda.");
    }

    #[cfg(feature = "cuda")]
    {
        anyhow::ensure!(pq_cuda::is_available(), "CUDA kernel not linked (nvcc missing at build).");
        let _ = pq_cuda::set_blocking_sync();
        eprintln!("gpu-selftest (ed25519): validating device pipeline vs CPU oracle on {n} seeds...");
        let mut seeds = vec![[0u8; CLASSIC_SEED_SIZE]; n];
        for s in seeds.iter_mut() {
            getrandom::getrandom(s).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
        }
        // Edge seeds up front.
        if n >= 2 {
            seeds[0] = [0u8; 32];
            seeds[1] = [0xffu8; 32];
        }
        let cpu: Vec<[u8; 32]> = seeds
            .iter()
            .map(|s| ClassicKeypair::from_seed(s).public_key())
            .collect();

        // Stage 1 + 2: vendored constant-time reference path, then the search
        // path (variable-time lookup + batched inversion).
        for (fast, name) in [(false, "reference"), (true, "search")] {
            let dev = pq_cuda::ed_dbg_pubkey(&seeds, fast)?;
            let mism = cpu
                .iter()
                .enumerate()
                .filter(|(i, pk)| &dev[i * 32..(i + 1) * 32] != pk.as_slice())
                .count();
            if mism == 0 {
                println!("PASS: pubkey ({name} path) — device == ed25519-dalek on all {n} seeds");
            } else {
                anyhow::bail!("FAIL: pubkey ({name} path) — {mism}/{n} seeds mismatched");
            }
        }

        // Stage 3: the device target matcher on crafted addresses.
        crate::selftest_matcher(true)?;

        // Stage 4: the search kernel end to end — counter layout (incl. the
        // 2^64 wrap), multi-target prefix/suffix matching, hit collection.
        // The device must return exactly the seeds a CPU scan finds.
        let items = 1u64 << 16;
        let mut base = [0u8; 32];
        getrandom::getrandom(&mut base).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
        let c0 = u64::MAX - items / 2; // straddle the wrap
        let targets = TargetSet::parse(None, None, Some("K-,-E,AB-Q,7-4"), true)?;
        let mut hits =
            pq_cuda::ed_search_batch(&base, c0, items, &targets.device_table(), items as usize)?;
        let mut expect: Vec<[u8; 32]> = (0..items)
            .map(|i| seed_at(&base, c0.wrapping_add(i)))
            .filter(|s| {
                let d = ClassicKeypair::from_seed(s).address().to_address_string();
                targets.first_match_display(&d).is_some()
            })
            .collect();
        hits.sort();
        expect.sort();
        anyhow::ensure!(
            hits == expect,
            "FAIL: search — device found {} hits, CPU scan expects {}",
            hits.len(),
            expect.len()
        );
        println!(
            "PASS: search — {} hits for {} over {items} keys (across the 2^64 counter wrap) == CPU scan",
            hits.len(),
            targets.describe()
        );
        println!("ALL STAGES PASS: ed25519 device pipeline is bit-exact with the CPU oracle.");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hit output
// ---------------------------------------------------------------------------

/// Write the hit record file (`<ADDRESS>.edhit`) and return its path.
fn write_hit_record(
    out_dir: &Path,
    seed: &[u8; CLASSIC_SEED_SIZE],
    kp: &ClassicKeypair,
    entropy_sources: &str,
) -> Option<PathBuf> {
    let display = kp.address().to_address_string();
    let body = format!(
        "address:           {display}\n\
         scheme:            ed25519 (classic Algorand account)\n\
         pubkey (base64):   {pk}\n\
         seed (hex):        {seed}  # KEEP SECRET\n\
         mnemonic:          {mnemonic}  # KEEP SECRET\n\
         entropy sources:   {entropy_sources}\n\
         \n\
         # import: goal account import -m \"<mnemonic>\"\n\
         #     or: algokey import -m \"<mnemonic>\" -f <keyfile>\n",
        pk = B64.encode(kp.public_key()),
        seed = hex_lower(seed),
        mnemonic = key_to_mnemonic(seed),
    );
    let path = out_dir.join(format!("{display}.edhit"));
    match std::fs::write(&path, body.as_bytes()) {
        Ok(()) => Some(path),
        Err(e) => {
            eprintln!("WARNING: could not write hit file {}: {e}", path.display());
            None
        }
    }
}

fn print_hit(
    hit_no: u64,
    seed: &[u8; CLASSIC_SEED_SIZE],
    kp: &ClassicKeypair,
    roundtrip: bool,
    path: Option<&Path>,
    matched: &str,
) {
    println!("─────────────────────────────────────────────");
    println!("HIT #{hit_no}  address: {}", kp.address());
    println!("  matched:   {matched}");
    println!("  scheme:    ed25519 (classic)   roundtrip: {roundtrip}");
    println!("  mnemonic:  {}  (KEEP SECRET)", key_to_mnemonic(seed));
    println!("  seed:      {} (KEEP SECRET)", hex_lower(seed));
    if let Some(p) = path {
        println!("  hit file:  {}", p.display());
    }
    println!("  import:    goal account import -m \"<mnemonic>\"");
}

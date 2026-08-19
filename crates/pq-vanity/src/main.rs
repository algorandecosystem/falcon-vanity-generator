//! `pq-vanity` — vanity address generator for Algorand native post-quantum
//! (`f1` / Deterministic Falcon-1024) accounts.
//!
//! Subcommands:
//!   - `derive`  — derive the address for mnemonic entropy (KAT sanity check)
//!   - `search`  — CPU baseline vanity search (bit-exact oracle for `gpu`)
//!   - `bench`   — measure CPU keygen / derivation throughput
//!   - `gpu`     — CUDA throughput search
//!   - `gpu-selftest` — validate the device pipeline against the CPU oracle

mod entropy;
mod search;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use clap::{Parser, Subcommand};
use pq_core::{
    canonical_pq_salt, key_to_mnemonic, keygen_from_seed, keygen_seed_from_entropy, pq_address,
    ENTROPY_SIZE,
};

#[derive(Parser)]
#[command(
    name = "pq-vanity",
    version,
    about = "Vanity address generator for Algorand native post-quantum (f1 / Falcon-1024) accounts"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Derive the PQ address for 32-byte mnemonic entropy (KAT sanity check).
    Derive(DeriveArgs),
    /// Search for a vanity address on the CPU (slow baseline; use `gpu`).
    Search(SearchArgs),
    /// Measure CPU keygen / address-derivation throughput.
    Bench(BenchArgs),
    /// Run a CUDA search batch (requires `--features cuda` on a Blackwell box).
    Gpu(GpuArgs),
    /// Validate the CUDA device pipeline against the CPU oracle (bit-exact).
    GpuSelftest(GpuSelftestArgs),
}

#[derive(clap::Args)]
struct DeriveArgs {
    /// First entropy byte (rest zero) of the 32-byte mnemonic entropy.
    #[arg(long, default_value_t = 0, conflicts_with_all = ["entropy_hex", "mnemonic"])]
    entropy_byte: u8,
    /// Full 32-byte mnemonic entropy as hex.
    #[arg(long, conflicts_with = "mnemonic")]
    entropy_hex: Option<String>,
    /// 25-word Algorand mnemonic (as printed by search hits / algokey).
    #[arg(long)]
    mnemonic: Option<String>,
    /// Salt to derive with. Omit to show the canonical (lowest compliant) salt.
    #[arg(long)]
    salt: Option<u8>,
}

#[derive(clap::Args)]
struct SearchArgs {
    /// Match the start of the displayed address (base32: A-Z, 2-7).
    #[arg(long)]
    prefix: Option<String>,
    /// Match the end of the displayed address (base32: A-Z, 2-7).
    #[arg(long)]
    suffix: Option<String>,
    /// Sweep salts 0..=MAX per key (only with --allow-non-canonical; the
    /// default canonical mode checks exactly the canonical salt).
    #[arg(long, default_value_t = 255)]
    max_salt: u16,
    /// Also accept matches at non-canonical compliant salts (higher hit rate,
    /// but the address is NOT what `algokey pq import -m` shows — signing as
    /// it needs custom tooling). Default: canonical salt only.
    #[arg(long, default_value_t = false)]
    allow_non_canonical: bool,
    /// Worker threads (default: cores - 1).
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    /// Stop after this many hits.
    #[arg(long, default_value_t = 1)]
    count: usize,
    /// Directory for emitted hit record files (address, salt, mnemonic).
    #[arg(long, default_value = "./hits")]
    out: PathBuf,
    /// Extra entropy mixed into every generated seed (defense in depth): a
    /// file path if it names an existing file, otherwise the literal string.
    #[arg(long)]
    extra_entropy: Option<String>,
    /// Suppress the per-second progress line.
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Duration in seconds.
    #[arg(long, default_value_t = 5)]
    seconds: u64,
    /// Worker threads (default: cores - 1).
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

#[derive(clap::Args)]
struct GpuArgs {
    /// Match the start of the displayed address (base32: A-Z, 2-7).
    #[arg(long)]
    prefix: String,
    /// Entropies per GPU batch (one thread each).
    #[arg(long, default_value_t = 1 << 22)]
    items: u64,
    /// Sweep salts 0..=MAX per key on the device. High is good with
    /// --allow-non-canonical: the per-key cost is dominated by sampling+NTT,
    /// while each extra salt is a near-free SHA-512/256, so sweeping all 256
    /// salts amortizes the expensive part over 256 addresses.
    /// [default: 31 (canonical mode — the canonical salt exceeds 31 with
    /// probability ~2^-32, higher salts only add filtered-out hits), or 255
    /// with --allow-non-canonical]
    #[arg(long)]
    max_salt: Option<u32>,
    /// Also accept matches at non-canonical compliant salts (more candidate
    /// addresses per key, but the address is NOT what `algokey pq import -m`
    /// shows — signing as it needs custom tooling). Default: canonical salt
    /// only, so every hit works with stock algokey.
    #[arg(long, default_value_t = false)]
    allow_non_canonical: bool,
    /// Max raw device hits collected per batch.
    #[arg(long, default_value_t = 4096)]
    max_hits: usize,
    /// Stop after this many usable (completed + off-curve) addresses.
    #[arg(long, default_value_t = 1)]
    count: usize,
    /// Directory for emitted hit record files (address, salt, mnemonic).
    #[arg(long, default_value = "./hits")]
    out: PathBuf,
    /// Extra entropy mixed into every batch base seed (defense in depth): a
    /// file path if it names an existing file, otherwise the literal string.
    #[arg(long)]
    extra_entropy: Option<String>,
    /// Stop after this many GPU batches (0 = unlimited). Use for benchmarking.
    #[arg(long, default_value_t = 0)]
    batches: u64,
}

#[derive(clap::Args)]
struct GpuSelftestArgs {
    /// Number of random entropies to validate.
    #[arg(long, default_value_t = 4096)]
    items: usize,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Derive(a) => cmd_derive(a),
        Cmd::Search(a) => cmd_search(a),
        Cmd::Bench(a) => cmd_bench(a),
        Cmd::Gpu(a) => cmd_gpu(a),
        Cmd::GpuSelftest(a) => cmd_gpu_selftest(a),
    }
}

fn cmd_gpu_selftest(a: GpuSelftestArgs) -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = a;
        anyhow::bail!("GPU support is not compiled in. Build on a CUDA box with --features cuda.");
    }

    #[cfg(feature = "cuda")]
    {
        anyhow::ensure!(
            pq_cuda::is_available(),
            "CUDA kernel not linked (nvcc missing at build)."
        );
        let _ = pq_cuda::set_blocking_sync(); // sleep, don't spin, during GPU waits
        let n = a.items;
        eprintln!("gpu-selftest: validating device pipeline vs CPU oracle on {n} entropies...");
        let mut seeds = vec![[0u8; pq_cuda::ENTROPY_SIZE]; n];
        for s in seeds.iter_mut() {
            getrandom::getrandom(s).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
        }

        // Device sampler: flat n * 2N int8 (f then g per entropy). The device
        // applies the "PQK" seed hash internally; mirror it for the oracle.
        let dev = pq_cuda::dbg_sample_fg(&seeds)?;
        let nn = pq_cuda::N;

        let mut mismatches = 0usize;
        for (i, seed) in seeds.iter().enumerate() {
            let fg = pq_core::sample_fg(&pq_core::keygen_seed_from_entropy(seed));
            let base = i * 2 * nn;
            let dev_f = &dev[base..base + nn];
            let dev_g = &dev[base + nn..base + 2 * nn];
            if dev_f != &fg.f[..] || dev_g != &fg.g[..] {
                if mismatches < 5 {
                    let fi = dev_f.iter().zip(fg.f.iter()).position(|(a, b)| a != b);
                    let gi = dev_g.iter().zip(fg.g.iter()).position(|(a, b)| a != b);
                    eprintln!("  MISMATCH seed#{i}: first f diff @ {fi:?}, first g diff @ {gi:?}");
                }
                mismatches += 1;
            }
        }
        if mismatches == 0 {
            println!("PASS: sampler — device (f,g) == oracle on all {n} seeds");
        } else {
            anyhow::bail!("FAIL: sampler — {mismatches}/{n} seeds mismatched");
        }

        // Stage 2: public half (sample -> compute_public -> modq_encode -> pk).
        let (pk, ok) = pq_cuda::dbg_pubkey(&seeds)?;
        let ps = pq_cuda::PUBKEY_SIZE;
        let mut pk_mism = 0usize;
        let mut invertible = 0usize;
        for (i, seed) in seeds.iter().enumerate() {
            let fg = pq_core::sample_fg(&pq_core::keygen_seed_from_entropy(seed));
            let cpu = pq_core::pubkey_from_fg(&fg);
            let dev_ok = ok[i] != 0;
            match (dev_ok, cpu) {
                (true, Some(cpu_pk)) => {
                    invertible += 1;
                    let dev_pk = &pk[i * ps..(i + 1) * ps];
                    if dev_pk != cpu_pk.as_ref() {
                        if pk_mism < 5 {
                            let d = dev_pk.iter().zip(cpu_pk.iter()).position(|(a, b)| a != b);
                            eprintln!("  PK MISMATCH seed#{i}: first byte diff @ {d:?}");
                        }
                        pk_mism += 1;
                    }
                }
                (false, None) => {} // both reject f as singular — consistent
                (d, c) => {
                    if pk_mism < 5 {
                        eprintln!(
                            "  PK OK-FLAG MISMATCH seed#{i}: device_ok={d}, cpu_some={}",
                            c.is_some()
                        );
                    }
                    pk_mism += 1;
                }
            }
        }
        if pk_mism == 0 {
            println!(
                "PASS: pubkey — device pk == oracle on all {n} seeds ({invertible} invertible)"
            );
        } else {
            anyhow::bail!("FAIL: pubkey — {pk_mism}/{n} seeds mismatched");
        }

        // Stage 3: full address (adds SHA-512/256) at a few salts.
        for salt in [0u8, 7u8, 200u8] {
            let (addr, ok) = pq_cuda::dbg_addr32(&seeds, salt)?;
            let asz = pq_cuda::ADDR_SIZE;
            let mut amism = 0usize;
            for (i, seed) in seeds.iter().enumerate() {
                if ok[i] == 0 {
                    continue;
                }
                let fg = pq_core::sample_fg(&pq_core::keygen_seed_from_entropy(seed));
                let Some(cpu_pk) = pq_core::pubkey_from_fg(&fg) else {
                    continue;
                };
                let cpu_addr = pq_core::pq_address(salt, cpu_pk.as_ref());
                let dev_addr = &addr[i * asz..(i + 1) * asz];
                if dev_addr != &cpu_addr.as_bytes()[..] {
                    if amism < 5 {
                        eprintln!("  ADDR MISMATCH seed#{i} salt={salt}");
                    }
                    amism += 1;
                }
            }
            if amism == 0 {
                println!("PASS: addr32 — device == oracle on all {n} seeds (salt {salt})");
            } else {
                anyhow::bail!("FAIL: addr32 — {amism}/{n} mismatched (salt {salt})");
            }
        }
        // Stage 4: the in-kernel retry loop (continuing-stream resampling,
        // integer checks, FP32 bnorm, invertibility) vs the CPU oracle. The
        // FP32 bnorm may disagree with the FPEMU reference within a hair of
        // the bound, so allow a tiny mismatch rate (such a disagreement only
        // ever wastes a seed — the search re-verifies hits with full keygen).
        {
            let nr = n.min(2000);
            let subset = &seeds[..nr];
            let dev_att = pq_cuda::dbg_accept(subset)?;
            let mut cpu_att = vec![-2i32; nr];
            let workers = default_threads().min(nr).max(1);
            let chunk = nr.div_ceil(workers);
            std::thread::scope(|sc| {
                for (schunk, ochunk) in subset.chunks(chunk).zip(cpu_att.chunks_mut(chunk)) {
                    sc.spawn(move || {
                        for (e, slot) in schunk.iter().zip(ochunk.iter_mut()) {
                            let seed = pq_core::keygen_seed_from_entropy(e);
                            // 192 == the kernel's PQ_MAX_ATTEMPTS cap.
                            *slot = pq_core::first_visible_accept(&seed, 192)
                                .map_or(-1, |a| a as i32);
                        }
                    });
                }
            });
            let mism = dev_att.iter().zip(&cpu_att).filter(|(a, b)| a != b).count();
            let avg = cpu_att.iter().filter(|&&a| a >= 0).map(|&a| a as u64 + 1).sum::<u64>()
                as f64
                / nr as f64;
            if mism as f64 <= (nr as f64 * 0.005).max(4.0) {
                println!(
                    "PASS: retry — device accept-attempt == oracle on {}/{nr} seeds \
                     (avg {avg:.1} attempts/seed; {mism} FP32-borderline)",
                    nr - mism
                );
            } else {
                anyhow::bail!("FAIL: retry — {mism}/{nr} accept-attempt mismatches");
            }
        }
        println!("ALL STAGES PASS: device public pipeline is bit-exact with the CPU oracle.");
        Ok(())
    }
}

/// A raw device hit that survived CPU verification and is ready to emit.
#[cfg(feature = "cuda")]
struct VerifiedHit {
    entropy: [u8; ENTROPY_SIZE],
    salt: u8,
    addr: pq_core::Address,
    kp: pq_core::Keypair,
    roundtrip: bool,
}

/// Re-derive, complete, and check one raw device hit on the CPU.
#[cfg(feature = "cuda")]
fn verify_one(hit: &pq_cuda::Hit, prefix: &str, canonical_only: bool) -> Option<VerifiedHit> {
    // Single-attempt predicate, matching the kernel's first-sample scan: the
    // hit is usable iff the reference keygen accepts the FIRST (f,g) — then
    // `algokey pq import -m` derives this exact key. (Full keygen_from_seed
    // here would walk the whole retry chain at ~64ms per doomed raw hit;
    // complete_from_fg rejects those in a fraction of that.)
    let seed = keygen_seed_from_entropy(&hit.entropy);
    let fg = pq_core::sample_fg(&seed);
    let kp = pq_core::complete_from_fg(&fg)?;
    let addr = pq_core::pq_address(hit.salt, kp.public_key.as_ref());
    // Must actually match the requested pattern AND be off-curve.
    if !addr.to_address_string().starts_with(prefix) || !addr.is_pq_compliant() {
        return None;
    }
    // Canonical-only: the matched salt must be the lowest compliant one, so
    // algokey signs as this address without custom tooling.
    if canonical_only && canonical_pq_salt(kp.public_key.as_ref()).map(|(s, _)| s) != Some(hit.salt)
    {
        return None;
    }
    // Round-trip the key before emitting.
    let roundtrip = kp
        .sign(b"pq-vanity-gpu-emit")
        .and_then(|s| kp.verify(b"pq-vanity-gpu-emit", &s))
        .is_ok();
    Some(VerifiedHit {
        entropy: hit.entropy,
        salt: hit.salt,
        addr,
        kp,
        roundtrip,
    })
}

/// Verify a batch of raw hits across all cores, preserving order (None = culled).
#[cfg(feature = "cuda")]
fn verify_hits(hits: &[pq_cuda::Hit], prefix: &str, canonical_only: bool) -> Vec<Option<VerifiedHit>> {
    let n = hits.len();
    let mut out: Vec<Option<VerifiedHit>> = Vec::new();
    out.resize_with(n, || None);
    if n == 0 {
        return out;
    }
    let workers = default_threads().min(n);
    let chunk = n.div_ceil(workers);
    std::thread::scope(|s| {
        for (hchunk, ochunk) in hits.chunks(chunk).zip(out.chunks_mut(chunk)) {
            s.spawn(move || {
                for (h, slot) in hchunk.iter().zip(ochunk.iter_mut()) {
                    *slot = verify_one(h, prefix, canonical_only);
                }
            });
        }
    });
    out
}

fn cmd_gpu(a: GpuArgs) -> anyhow::Result<()> {
    // Validate/uppercase the pattern and map to 5-bit base32 indices (what the
    // kernel's base32_prefix_match compares against).
    let prefix = search::validate_pattern(&a.prefix)?;
    let indices: Vec<u8> = prefix
        .bytes()
        .map(|c| search::BASE32_ALPHABET.find(c as char).unwrap() as u8)
        .collect();

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, indices);
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
        // Sleep (don't spin) while the GPU runs — otherwise CUDA's default auto
        // policy pins a host core at 100% busy-polling for the whole kernel.
        let _ = pq_cuda::set_blocking_sync();
        anyhow::ensure!(pq_cuda::device_count() >= 1, "no CUDA device visible");
        std::fs::create_dir_all(&a.out)?;
        let canonical_only = !a.allow_non_canonical;
        // Canonical mode (default): only the lowest compliant salt per key
        // counts, and it exceeds 31 with probability ~2^-32, so a short sweep
        // loses nothing and skips device hits the host would only filter out.
        let max_salt = a.max_salt.unwrap_or(if canonical_only { 31 } else { 255 });
        eprintln!(
            "GPU search: prefix={prefix:?} items/batch={} max_salt={max_salt} canonical_only={canonical_only} count={} (device: {} GPU)",
            a.items,
            a.count,
            pq_cuda::device_count()
        );

        let mix = entropy::EntropyMix::new(load_extra_entropy(&a.extra_entropy)?.as_deref());
        eprintln!("entropy sources: {}", mix.sources());

        let start = std::time::Instant::now();
        let mut found = 0usize;
        let mut raw_hits = 0u64;
        let mut keys_done = 0u128;
        for batch in 0u64.. {
            if batch > 0 {
                // Fresh cached contributions (TPM + jitter) for every batch;
                // batches are seconds apart, so the ms-scale TPM read is free.
                mix.refresh();
            }
            let base_entropy = mix.next()?;
            let res =
                pq_cuda::search_batch(&base_entropy, 0, a.items, max_salt, &indices, a.max_hits)?;
            raw_hits += res.hits.len() as u64;
            keys_done += a.items as u128;

            // Verify raw hits on all cores: completing a key (NTRU solve) costs
            // ~2ms, and short prefixes can return thousands of raw hits per
            // batch — serial verification would throttle the (fast) kernel.
            for v in verify_hits(&res.hits, &prefix, canonical_only)
                .into_iter()
                .flatten()
            {
                let path = search::write_hit_record(
                    &a.out,
                    &v.entropy,
                    v.salt,
                    &v.addr,
                    &v.kp,
                    mix.sources(),
                );
                found += 1;
                search::print_hit(
                    found as u64,
                    &v.entropy,
                    v.salt,
                    &v.addr,
                    &v.kp,
                    v.roundtrip,
                    path.as_deref(),
                );
                if found >= a.count {
                    break;
                }
            }

            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            let addrs = keys_done * (max_salt as u128 + 1);
            eprintln!(
                "[batch {batch}] keys={keys_done} ({:.3}M/s)  addrs/s={:.1}M  raw_hits={raw_hits}  usable={found}",
                keys_done as f64 / elapsed / 1e6,
                addrs as f64 / elapsed / 1e6
            );
            if found >= a.count {
                break;
            }
            if a.batches != 0 && batch + 1 >= a.batches {
                eprintln!("(stopped after {} batch(es) — benchmark mode)", a.batches);
                break;
            }
        }
        eprintln!(
            "done: {found} usable address(es) in {:.1}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }
}

/// Resolve `--extra-entropy`: a path if it names an existing file, otherwise
/// the literal string bytes.
fn load_extra_entropy(arg: &Option<String>) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(v) = arg else { return Ok(None) };
    let p = std::path::Path::new(v);
    let bytes = if p.is_file() {
        std::fs::read(p).map_err(|e| anyhow::anyhow!("--extra-entropy file {v}: {e}"))?
    } else {
        v.clone().into_bytes()
    };
    anyhow::ensure!(!bytes.is_empty(), "--extra-entropy is empty");
    Ok(Some(bytes))
}

fn parse_hex(s: &str) -> anyhow::Result<Vec<u8>> {
    let s = s.trim();
    anyhow::ensure!(s.len() % 2 == 0, "hex string must have even length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| anyhow::anyhow!("bad hex: {e}")))
        .collect()
}

fn cmd_derive(a: DeriveArgs) -> anyhow::Result<()> {
    let mut entropy = [0u8; ENTROPY_SIZE];
    if let Some(m) = &a.mnemonic {
        entropy = pq_core::mnemonic_to_key(m).map_err(|e| anyhow::anyhow!("bad mnemonic: {e}"))?;
    } else if let Some(h) = &a.entropy_hex {
        let bytes = parse_hex(h)?;
        anyhow::ensure!(
            bytes.len() == ENTROPY_SIZE,
            "--entropy-hex must be {ENTROPY_SIZE} bytes ({} given)",
            bytes.len()
        );
        entropy.copy_from_slice(&bytes);
    } else {
        entropy[0] = a.entropy_byte;
    }
    let seed = keygen_seed_from_entropy(&entropy);
    let kp = keygen_from_seed(&seed)?;
    let pk = kp.public_key.as_ref();
    println!("scheme:   f1 (Falcon-1024)");
    println!("mnemonic: {}", key_to_mnemonic(&entropy));
    println!("pubkey:   {}", B64.encode(pk));

    match a.salt {
        Some(salt) => {
            let addr = pq_address(salt, pk);
            println!("salt:    {salt}");
            println!("address: {addr}");
            println!("compliant: {}", addr.is_pq_compliant());
        }
        None => match canonical_pq_salt(pk) {
            Some((salt, addr)) => {
                println!("salt:    {salt}  (canonical / lowest compliant)");
                println!("address: {addr}");
                println!("compliant: {}", addr.is_pq_compliant());
            }
            None => println!("no compliant salt in 0..=255 (probability ~2^-256)"),
        },
    }
    Ok(())
}

fn cmd_search(a: SearchArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        a.prefix.is_some() || a.suffix.is_some(),
        "specify at least one of --prefix / --suffix"
    );
    anyhow::ensure!(a.threads >= 1, "--threads must be >= 1");
    anyhow::ensure!(a.count >= 1, "--count must be >= 1");

    let prefix = a
        .prefix
        .as_deref()
        .map(search::validate_pattern)
        .transpose()?;
    let suffix = a
        .suffix
        .as_deref()
        .map(search::validate_pattern)
        .transpose()?;

    let canonical_only = !a.allow_non_canonical;
    let pat_len = prefix.as_ref().map_or(0, |p| p.len()) + suffix.as_ref().map_or(0, |s| s.len());
    eprintln!(
        "searching: prefix={:?} suffix={:?} threads={} count={} canonical_only={canonical_only}",
        prefix, suffix, a.threads, a.count
    );
    if canonical_only {
        // One canonical (always-compliant) address per key.
        eprintln!(
            "pattern length {pat_len} → ~{:.3e} keys per hit (expectation)",
            32f64.powi(pat_len as i32)
        );
    } else {
        // Each tested address matches with probability ~32^-pat_len; we
        // additionally require off-curve compliance (~1/2).
        eprintln!(
            "pattern length {pat_len} → ~{:.3e} addresses per hit (expectation)",
            2.0f64 * 32f64.powi(pat_len as i32)
        );
    }

    let mix = Arc::new(entropy::EntropyMix::new(
        load_extra_entropy(&a.extra_entropy)?.as_deref(),
    ));
    eprintln!("entropy sources: {}", mix.sources());

    let cfg = search::SearchConfig {
        mix,
        prefix,
        suffix,
        max_salt: a.max_salt,
        canonical_only,
        threads: a.threads,
        count: a.count,
        out_dir: a.out,
        progress: !a.quiet,
    };
    let hits = search::run(cfg)?;
    eprintln!("done: {hits} hit(s)");
    Ok(())
}

fn cmd_bench(a: BenchArgs) -> anyhow::Result<()> {
    eprintln!(
        "benchmarking keygen + 256-salt derivation for {}s on {} threads...",
        a.seconds, a.threads
    );
    let stop = Arc::new(AtomicBool::new(false));
    let keys = Arc::new(AtomicU64::new(0));
    let addrs = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..a.threads {
        let stop = Arc::clone(&stop);
        let keys = Arc::clone(&keys);
        let addrs = Arc::clone(&addrs);
        handles.push(std::thread::spawn(move || {
            let mut entropy = [0u8; ENTROPY_SIZE];
            let mut local_keys = 0u64;
            let mut local_addrs = 0u64;
            while !stop.load(Ordering::Relaxed) {
                if getrandom::getrandom(&mut entropy).is_err() {
                    break;
                }
                let seed = keygen_seed_from_entropy(&entropy);
                if let Ok(kp) = keygen_from_seed(&seed) {
                    let pk = kp.public_key.as_ref();
                    for salt in 0u16..=255 {
                        let addr = pq_address(salt as u8, pk);
                        // Touch compliance so the bench reflects real per-address cost.
                        std::hint::black_box(addr.is_pq_compliant());
                        local_addrs += 1;
                    }
                    local_keys += 1;
                }
            }
            keys.fetch_add(local_keys, Ordering::Relaxed);
            addrs.fetch_add(local_addrs, Ordering::Relaxed);
        }));
    }

    std::thread::sleep(Duration::from_secs(a.seconds));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let k = keys.load(Ordering::Relaxed);
    let ad = addrs.load(Ordering::Relaxed);
    println!("elapsed:   {elapsed:.2}s");
    println!("keygen:    {k} keys  ({:.0}/s)", k as f64 / elapsed);
    println!(
        "derive:    {ad} addresses  ({:.2}M/s)",
        ad as f64 / elapsed / 1e6
    );
    println!(
        "note: keygen dominates CPU cost; the GPU fast path computes only the\n      public half (h = g·f^-1) and defers the NTRU solve to hits."
    );
    Ok(())
}

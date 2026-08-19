//! CPU baseline vanity search.
//!
//! This is the **correct-but-slow** reference path: every attempt runs the full
//! Falcon reference keygen (including the expensive NTRU solve), then sweeps
//! salts and checks the displayed address against the target pattern, requiring
//! off-curve compliance. It exists to validate the end-to-end pipeline and to
//! serve as the bit-exact oracle for the CUDA kernel. The `gpu`
//! subcommand is the throughput path.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use pq_core::{
    canonical_pq_salt, key_to_mnemonic, keygen_from_seed, keygen_seed_from_entropy, pq_address,
    Address, Keypair, ENTROPY_SIZE,
};

/// Base32 alphabet used by Algorand addresses (RFC-4648, no 0/1/8/9).
pub const BASE32_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

#[derive(Clone)]
pub struct SearchConfig {
    /// Uppercased, validated prefix to match against the displayed address.
    pub prefix: Option<String>,
    /// Uppercased, validated suffix.
    pub suffix: Option<String>,
    /// Sweep salts `0..=max_salt` per key (ignored when `canonical_only`).
    pub max_salt: u16,
    /// Require the match to be the *canonical* (lowest compliant) salt, so the
    /// address shows by default in `algokey pq` without `--salt`.
    pub canonical_only: bool,
    pub threads: usize,
    /// Stop after this many hits.
    pub count: usize,
    pub out_dir: PathBuf,
    /// Print a throughput line roughly every second.
    pub progress: bool,
}

/// Validate that `pat` uses only the base32 alphabet; returns the uppercased form.
pub fn validate_pattern(pat: &str) -> anyhow::Result<String> {
    let up = pat.to_ascii_uppercase();
    for c in up.chars() {
        if !BASE32_ALPHABET.contains(c) {
            anyhow::bail!(
                "pattern character {c:?} is not in the Algorand base32 alphabet ({BASE32_ALPHABET})"
            );
        }
    }
    Ok(up)
}

#[inline]
fn matches(display: &str, cfg: &SearchConfig) -> bool {
    if let Some(p) = &cfg.prefix {
        if !display.starts_with(p.as_str()) {
            return false;
        }
    }
    if let Some(s) = &cfg.suffix {
        if !display.ends_with(s.as_str()) {
            return false;
        }
    }
    true
}

struct Counters {
    keys: AtomicU64,
    addrs: AtomicU64,
    hits: AtomicU64,
    stop: AtomicBool,
}

/// Run the search until `count` hits are recorded. Returns the number of hits.
pub fn run(cfg: SearchConfig) -> anyhow::Result<u64> {
    std::fs::create_dir_all(&cfg.out_dir)?;
    let counters = Arc::new(Counters {
        keys: AtomicU64::new(0),
        addrs: AtomicU64::new(0),
        hits: AtomicU64::new(0),
        stop: AtomicBool::new(false),
    });
    let io_lock = Arc::new(Mutex::new(()));
    let cfg = Arc::new(cfg);

    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..cfg.threads {
        let counters = Arc::clone(&counters);
        let io_lock = Arc::clone(&io_lock);
        let cfg = Arc::clone(&cfg);
        handles.push(std::thread::spawn(move || {
            worker(&cfg, &counters, &io_lock)
        }));
    }

    // Progress reporter on the main thread.
    if cfg.progress {
        while !counters.stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(1000));
            if counters.stop.load(Ordering::Relaxed) {
                break;
            }
            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            let keys = counters.keys.load(Ordering::Relaxed);
            let addrs = counters.addrs.load(Ordering::Relaxed);
            let hits = counters.hits.load(Ordering::Relaxed);
            eprintln!(
                "[{:>6.0}s] keys={keys} ({:.0}/s)  addrs={addrs} ({:.2}M/s)  hits={hits}",
                elapsed,
                keys as f64 / elapsed,
                addrs as f64 / elapsed / 1e6,
            );
        }
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(counters.hits.load(Ordering::Relaxed))
}

fn worker(cfg: &SearchConfig, counters: &Counters, io_lock: &Mutex<()>) {
    let mut entropy = [0u8; ENTROPY_SIZE];
    while !counters.stop.load(Ordering::Relaxed) {
        if getrandom::getrandom(&mut entropy).is_err() {
            counters.stop.store(true, Ordering::Relaxed);
            return;
        }
        let seed = keygen_seed_from_entropy(&entropy);
        let kp = match keygen_from_seed(&seed) {
            Ok(kp) => kp,
            Err(_) => continue, // extremely rare keygen failure; try another entropy
        };

        if cfg.canonical_only {
            if let Some((salt, addr)) = canonical_pq_salt(kp.public_key.as_ref()) {
                counters.addrs.fetch_add(1, Ordering::Relaxed);
                if matches(&addr.to_address_string(), cfg) {
                    record_hit(cfg, counters, io_lock, &entropy, salt, &addr, &kp);
                }
            }
        } else {
            for salt in 0..=cfg.max_salt {
                let salt = salt as u8;
                let addr = pq_address(salt, kp.public_key.as_ref());
                counters.addrs.fetch_add(1, Ordering::Relaxed);
                // Cheap string match first; compliance (curve decompress) only on a
                // candidate. A usable address must be off-curve compliant.
                if matches(&addr.to_address_string(), cfg) && addr.is_pq_compliant() {
                    record_hit(cfg, counters, io_lock, &entropy, salt, &addr, &kp);
                    break; // one usable salt per key is enough
                }
            }
        }
        counters.keys.fetch_add(1, Ordering::Relaxed);
    }
}

fn record_hit(
    cfg: &SearchConfig,
    counters: &Counters,
    io_lock: &Mutex<()>,
    entropy: &[u8; ENTROPY_SIZE],
    salt: u8,
    addr: &Address,
    kp: &Keypair,
) {
    let n = counters.hits.fetch_add(1, Ordering::SeqCst);
    if n >= cfg.count as u64 {
        // Already have enough; undo the over-count and stop.
        counters.hits.fetch_sub(1, Ordering::SeqCst);
        counters.stop.store(true, Ordering::Relaxed);
        return;
    }

    // Defence in depth: re-verify the key actually round-trips before emitting.
    let roundtrip = kp
        .sign(b"pq-vanity-emit-check")
        .and_then(|sig| kp.verify(b"pq-vanity-emit-check", &sig))
        .is_ok();

    let _guard = io_lock.lock().unwrap();
    let path = write_hit_record(&cfg.out_dir, entropy, salt, addr, kp);
    print_hit(n + 1, entropy, salt, addr, kp, roundtrip, path.as_deref());
    if n + 1 >= cfg.count as u64 {
        counters.stop.store(true, Ordering::Relaxed);
    }
}

/// Write the hit record file (`<ADDRESS>.pqhit`) and return its path.
/// The mnemonic is everything needed to use the account:
/// `algokey pq import -m "<mnemonic>" -k <keyfile>` re-derives the key pair.
pub fn write_hit_record(
    out_dir: &std::path::Path,
    entropy: &[u8; ENTROPY_SIZE],
    salt: u8,
    addr: &Address,
    kp: &Keypair,
) -> Option<PathBuf> {
    let display = addr.to_address_string();
    let mnemonic = key_to_mnemonic(entropy);
    let canonical = canonical_pq_salt(kp.public_key.as_ref());
    let (canon_salt, canon_addr) = match &canonical {
        Some((s, a)) => (s.to_string(), a.to_address_string()),
        None => ("-".into(), "-".into()),
    };
    let body = format!(
        "address:           {display}\n\
         scheme:            f1 (Falcon-1024, deterministic signing profile)\n\
         salt:              {salt}\n\
         canonical salt:    {canon_salt}\n\
         canonical address: {canon_addr}\n\
         pubkey (base64):   {pk}\n\
         entropy (hex):     {ent}  # KEEP SECRET\n\
         mnemonic:          {mnemonic}  # KEEP SECRET\n\
         \n\
         # import: algokey pq import -m \"<mnemonic>\" -k <keyfile>\n\
         # note: algokey signs with the canonical salt; a non-canonical salt\n\
         # is consensus-valid but needs custom signing tooling.\n",
        pk = B64.encode(kp.public_key.as_ref()),
        ent = hex_lower(entropy),
    );
    let path = out_dir.join(format!("{display}.pqhit"));
    match std::fs::write(&path, body.as_bytes()) {
        Ok(()) => Some(path),
        Err(e) => {
            eprintln!("WARNING: could not write hit file {}: {e}", path.display());
            None
        }
    }
}

pub fn print_hit(
    hit_no: u64,
    entropy: &[u8; ENTROPY_SIZE],
    salt: u8,
    addr: &Address,
    kp: &Keypair,
    roundtrip: bool,
    path: Option<&std::path::Path>,
) {
    let display = addr.to_address_string();
    let canonical = canonical_pq_salt(kp.public_key.as_ref());
    println!("─────────────────────────────────────────────");
    println!("HIT #{hit_no}  address: {display}");
    println!("  scheme:    f1 (Falcon-1024)");
    match canonical {
        Some((cs, ca)) if cs == salt => {
            println!("  salt:      {salt} (canonical — algokey shows this address by default)");
            debug_assert_eq!(ca.as_bytes(), addr.as_bytes());
        }
        Some((cs, ca)) => {
            println!("  salt:      {salt} (NON-canonical — needs custom signing tooling;");
            println!("             algokey uses salt {cs} → {})", ca.to_address_string());
        }
        None => println!("  salt:      {salt} (no canonical salt found?!)"),
    }
    println!("  compliant: {}   roundtrip: {roundtrip}", addr.is_pq_compliant());
    println!("  mnemonic:  {}  (KEEP SECRET)", pq_core::key_to_mnemonic(entropy));
    println!("  entropy:   {} (KEEP SECRET)", hex_lower(entropy));
    if let Some(p) = path {
        println!("  hit file:  {}", p.display());
    }
    println!("  import:    algokey pq import -m \"<mnemonic>\" -k <keyfile>");
}

pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

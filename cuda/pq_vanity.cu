// pq_vanity.cu — CUDA backend for the f1 (Deterministic Falcon-1024) PQ address
// vanity search. Targets RTX PRO 6000 Blackwell (sm_120) and DGX Spark GB10
// (sm_121a). MUST stay integer-only — no FP64 (Blackwell is ~1:64 FP64).
//
// Per-thread device pipeline (see docs/DESIGN.md §6 and cuda/KERNEL.md):
//   entropy = base_entropy with bytes[24..32] = (start_counter + global_id) LE
//   seed = SHA512_256("PQK" || "f1" || entropy)      [go-algorand derivePQKeySeed]
//   SHAKE256(seed) -> mkgauss -> (f, g)              [int8 polynomials, n=1024]
//   compute_public: h = g * f^-1 mod 12289 (skip if f not invertible)
//   modq_encode(h) -> pk[1793]
//   for salt in 0..=max_salt:
//       addr32 = SHA512_256("PQA" || "f1" || salt || pk)
//       compare base32(addr32) leading chars to target prefix
//   on match: append (entropy, salt) to the device hit buffer (atomic)
//
// The CPU host (pq-core) re-derives (f,g) from a hit's entropy, runs solve_NTRU
// + the norm checks, assembles + round-trips the key, and emits it (the entropy
// is what encodes to the 25-word mnemonic for `algokey pq import`). The address
// depends only on h, so GPU and CPU agree for the same (f,g) regardless of the
// CPU-only acceptance checks. First-sample scanning is deliberate: see
// dev_retry_sample for why in-seed retrying loses in a thread-per-key kernel.
//
// STATUS: fully implemented and validated bit-exact against the CPU oracle
// (`pq-vanity gpu-selftest`). If nvcc is absent at build time the crate falls
// back to a host stub and `pq_cuda_search` reports out_stub=1 with no hits.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "falcon_ntt_tables.h" // PQ_GMB[1024], PQ_IGMB[1024] (Montgomery NTT roots)

#define PQ_N            1024
#define PQ_Q            12289
#define PQ_ENTROPY_SIZE 32                  // mnemonic entropy (crypto.Seed)
#define PQ_SEED_SIZE    32                  // Falcon keygen seed (FalconSeedSize)
#define PQ_PUBKEY_SIZE  1793
#define PQ_ADDR_SIZE    32
#define PQ_HIT_STRIDE   (PQ_ENTROPY_SIZE + 1)  // entropy[32] + salt[1]

// Falcon keygen Gaussian table for n=1024 (vendor/falcon/keygen.c:2266).
// Ready for the mkgauss port; constant memory keeps it cached for all threads.
__device__ __constant__ uint64_t GAUSS_1024_12289[27] = {
    1283868770400643928ull,  6416574995475331444ull,  4078260278032692663ull,
    2353523259288686585ull,  1227179971273316331ull,   575931623374121527ull,
     242543240509105209ull,    91437049221049666ull,    30799446349977173ull,
       9255276791179340ull,     2478152334826140ull,      590642893610164ull,
        125206034929641ull,       23590435911403ull,        3948334035941ull,
           586753615614ull,          77391054539ull,           9056793210ull,
              940121950ull,             86539696ull,              7062824ull,
                 510971ull,                32764ull,                 1862ull,
                     94ull,                    4ull,                    0ull
};

// base32 alphabet (RFC-4648), matching Go base32.StdEncoding.
__device__ __constant__ char B32_ALPHABET[32] =
    {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
     'Q','R','S','T','U','V','W','X','Y','Z','2','3','4','5','6','7'};

// PQ address preimage prefix: "PQA" || "f1" (HashID || scheme). The salt byte
// and pk follow at hash time.
__device__ __constant__ uint8_t PQ_PREFIX[5] = {'P','Q','A','f','1'};

// PQ keygen-seed preimage prefix: "PQK" || "f1" (protocol.PostQuantumKey).
__device__ __constant__ uint8_t PQK_PREFIX[5] = {'P','Q','K','f','1'};

// ===========================================================================
// Device public-half pipeline, validated bit-exact vs the pq-core CPU oracle
// (see `pq-vanity gpu-selftest` + cuda/KERNEL.md): SHAKE256 + Gaussian sampler,
// NTT mod 12289 + modq_encode, and SHA-512/256 — all defined below.
// ===========================================================================

// Derive this thread's 32-byte mnemonic entropy into `entropy` (base_entropy
// with an 8-byte LE counter spliced into bytes [24..32]).
__device__ __forceinline__ void derive_entropy(
    const uint8_t* __restrict__ base_entropy, uint64_t counter, uint8_t* __restrict__ entropy)
{
    #pragma unroll
    for (int i = 0; i < PQ_ENTROPY_SIZE; ++i) entropy[i] = base_entropy[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) entropy[24 + i] = (uint8_t)(counter >> (8 * i));
}

// entropy -> Falcon keygen seed: SHA512_256("PQK" || "f1" || entropy).
// Defined after the SHA-512/256 section below.
__device__ void dev_seed_from_entropy(const uint8_t* entropy, uint8_t* seed);

// Compare the base32 of addr32's leading bytes to `prefix` (prefix_len chars,
// each already an index 0..31 OR an ASCII base32 char — see host). Returns 1 on
// match. Valid for prefixes up to 51 chars (first 32 bytes -> 51.2 base32 chars).
__device__ int base32_prefix_match(
    const uint8_t* addr32, const uint8_t* prefix, int prefix_len)
{
    // 5 bits per output char, big-endian bit order (RFC-4648).
    int bitpos = 0;
    for (int i = 0; i < prefix_len; ++i) {
        int byte_index = bitpos >> 3;
        int bit_index = bitpos & 7;
        // Gather 5 bits starting at (byte_index, bit_index), MSB-first.
        uint32_t window = ((uint32_t)addr32[byte_index] << 8) |
                          (uint32_t)(byte_index + 1 < PQ_ADDR_SIZE ? addr32[byte_index + 1] : 0);
        uint32_t v = (window >> (11 - bit_index)) & 0x1F;
        if ((uint8_t)v != prefix[i]) return 0;
        bitpos += 5;
    }
    return 1;
}

// ===========================================================================
// Device SHAKE256 (Keccak-f[1600], rate 136) — ported from vendor/falcon/shake.c,
// specialized to the sampler's use (absorb exactly one 32-byte seed, squeeze LE
// u64s). Fully unrolled with constexpr index tables so the 25-lane state stays
// in REGISTERS — the rolled/runtime-indexed version kept it in local memory and
// made the sampler ~92% of kernel time. Squeezed u64s go through a 17-lane pool
// so the only runtime-indexed array is 136 bytes (L1), never the state itself.
// Byte stream is identical to the generic byte-granular SHAKE256.
// ===========================================================================

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ __forceinline__ void keccakf(uint64_t st[25]) {
    // constexpr (not __constant__) so unrolled st[] indices are compile-time
    // constants — that is what lets ptxas keep st in registers.
    constexpr int ROTC[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                              27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    constexpr int PILN[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                              15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    for (int round = 0; round < 24; ++round) {
        uint64_t bc[5], t;
        // Theta
        #pragma unroll
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // Rho + Pi
        t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[PILN[i]];
            st[PILN[i]] = rotl64(t, ROTC[i]);
            t = tmp;
        }
        // Chi
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            uint64_t b0 = st[j], b1 = st[j + 1], b2 = st[j + 2];
            uint64_t b3 = st[j + 3], b4 = st[j + 4];
            st[j]     ^= (~b1) & b2;
            st[j + 1] ^= (~b2) & b3;
            st[j + 2] ^= (~b3) & b4;
            st[j + 3] ^= (~b4) & b0;
            st[j + 4] ^= (~b0) & b1;
        }
        // Iota (value lookup by loop index is fine — st index is static)
        st[0] ^= KECCAK_RC[round];
    }
}

struct Shake {
    uint64_t a[25];    // registerized Keccak state
    uint64_t pool[17]; // squeezed rate lanes (the only runtime-indexed part)
    int idx;           // next pool u64; 17 = exhausted
};

// Absorb exactly one 32-byte seed and finalize: lanes 0..3 = seed (LE), SHAKE
// domain pad 0x1F at byte 32 (lane 4), final bit 0x80 at byte 135 (lane 16).
// Bit-identical to shake_absorb(seed,32) + shake_finalize on a fresh state.
__device__ __forceinline__ void shake_init_seed32(Shake& s, const uint8_t* seed) {
    #pragma unroll
    for (int i = 0; i < 25; ++i) s.a[i] = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v |= (uint64_t)seed[i * 8 + j] << (8 * j);
        s.a[i] = v;
    }
    s.a[4] = 0x1F;
    s.a[16] = (uint64_t)0x80 << 56;
    s.idx = 17;
}

__device__ __forceinline__ uint64_t shake_squeeze_u64(Shake& s) {
    if (s.idx == 17) {
        keccakf(s.a);
        #pragma unroll
        for (int i = 0; i < 17; ++i) s.pool[i] = s.a[i];
        s.idx = 0;
    }
    return s.pool[s.idx++];
}

// One Gaussian sample for n=1024 (g=1), per vendor/falcon/keygen.c mkgauss.
__device__ int dev_mkgauss(Shake& s) {
    uint64_t r;
    uint32_t f, v, k, neg;

    r = shake_squeeze_u64(s);
    neg = (uint32_t)(r >> 63);
    r &= ~((uint64_t)1 << 63);
    f = (uint32_t)((r - GAUSS_1024_12289[0]) >> 63);

    v = 0;
    r = shake_squeeze_u64(s);
    r &= ~((uint64_t)1 << 63);
    #pragma unroll
    for (k = 1; k < 27; ++k) {
        uint32_t t = (uint32_t)((r - GAUSS_1024_12289[k]) >> 63) ^ 1u;
        v |= k & (0u - (t & (f ^ 1u)));
        f |= t;
    }
    v = (v ^ (0u - neg)) + neg;
    return (int)(int32_t)v;
}

// Fill a length-1024 small polynomial (poly_small_mkgauss): reject |s|>127,
// and force an odd coefficient sum (parity fix on the last coefficient).
__device__ void dev_poly_small_mkgauss(Shake& s, int8_t* f) {
    unsigned mod2 = 0;
    for (int u = 0; u < PQ_N; ++u) {
        const bool last = (u == PQ_N - 1);
        int sv;
        do {
            sv = dev_mkgauss(s);
        } while (sv < -127 || sv > 127 || (last && ((mod2 ^ (unsigned)(sv & 1)) == 0)));
        if (!last) mod2 ^= (unsigned)(sv & 1);
        f[u] = (int8_t)sv;
    }
}

// seed -> (f,g): SHAKE256-PRNG(seed) then two poly_small_mkgauss. Matches
// pq-core::fast::sample_fg (and keygen's first attempt) bit-for-bit. The seed
// is always the 32-byte keygen seed (the shake is specialized to that length).
__device__ void dev_sample_fg(const uint8_t* seed, int seed_len, int8_t* f, int8_t* g) {
    (void)seed_len; // always PQ_SEED_SIZE
    Shake s;
    shake_init_seed32(s, seed);
    dev_poly_small_mkgauss(s, f);
    dev_poly_small_mkgauss(s, g);
}

// Debug kernel: emit (f||g) for each entropy (1024 + 1024 int8 each). Applies
// the "PQK" seed hash first; validated against sample_fg(pqk_seed(entropy)).
extern "C" __global__ void pq_dbg_sample_fg_kernel(
    const uint8_t* __restrict__ entropies, uint64_t n, int8_t* __restrict__ out)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint8_t seed[PQ_SEED_SIZE];
    dev_seed_from_entropy(entropies + gid * PQ_ENTROPY_SIZE, seed);
    int8_t* f = out + gid * (2 * PQ_N);
    int8_t* g = f + PQ_N;
    dev_sample_fg(seed, PQ_SEED_SIZE, f, g);
}

extern "C" int pq_cuda_dbg_sample_fg(const uint8_t* seeds, uint64_t n, int8_t* out_fg)
{
    uint8_t* d_seeds = nullptr;
    int8_t* d_out = nullptr;
    int rc = 0;
    #define CKD(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKD(cudaMalloc(&d_seeds, n * PQ_ENTROPY_SIZE));
    CKD(cudaMalloc(&d_out, n * (2 * PQ_N)));
    CKD(cudaMemcpy(d_seeds, seeds, n * PQ_ENTROPY_SIZE, cudaMemcpyHostToDevice));
    {
        const int block = 128;
        unsigned long long blocks = (n + block - 1) / block;
        if (blocks == 0) blocks = 1;
        pq_dbg_sample_fg_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, d_out);
        CKD(cudaGetLastError());
        CKD(cudaDeviceSynchronize());
    }
    CKD(cudaMemcpy(out_fg, d_out, n * (2 * PQ_N), cudaMemcpyDeviceToHost));
cleanup:
    if (d_seeds) cudaFree(d_seeds);
    if (d_out) cudaFree(d_out);
    #undef CKD
    return rc;
}

// ===========================================================================
// mod-q (q=12289) arithmetic + NTT — ported from vendor/falcon/vrfy.c.
// Integer-only (Montgomery). compute_public: h = g·f^-1 mod q.
// ===========================================================================

#define PQ_Q0I 12287u
#define PQ_R 4091u
#define PQ_R2 10952u

__device__ __forceinline__ uint32_t mq_conv_small(int x) {
    uint32_t y = (uint32_t)x;
    y += PQ_Q & (0u - (y >> 31));
    return y;
}
__device__ __forceinline__ uint32_t mq_add(uint32_t x, uint32_t y) {
    uint32_t d = x + y - PQ_Q;
    d += PQ_Q & (0u - (d >> 31));
    return d;
}
__device__ __forceinline__ uint32_t mq_sub(uint32_t x, uint32_t y) {
    uint32_t d = x - y;
    d += PQ_Q & (0u - (d >> 31));
    return d;
}
__device__ __forceinline__ uint32_t mq_rshift1(uint32_t x) {
    x += PQ_Q & (0u - (x & 1));
    return x >> 1;
}
__device__ __forceinline__ uint32_t mq_montymul(uint32_t x, uint32_t y) {
    uint32_t z = x * y;
    uint32_t w = ((z * PQ_Q0I) & 0xFFFF) * PQ_Q;
    z = (z + w) >> 16;
    z -= PQ_Q;
    z += PQ_Q & (0u - (z >> 31));
    return z;
}
__device__ __forceinline__ uint32_t mq_montysqr(uint32_t x) { return mq_montymul(x, x); }

// x / y mod q via y^(q-2); addition chain from vrfy.c mq_div_12289.
__device__ uint32_t mq_div_12289(uint32_t x, uint32_t y) {
    uint32_t y0 = mq_montymul(y, PQ_R2);
    uint32_t y1 = mq_montysqr(y0);
    uint32_t y2 = mq_montymul(y1, y0);
    uint32_t y3 = mq_montymul(y2, y1);
    uint32_t y4 = mq_montysqr(y3);
    uint32_t y5 = mq_montysqr(y4);
    uint32_t y6 = mq_montysqr(y5);
    uint32_t y7 = mq_montysqr(y6);
    uint32_t y8 = mq_montysqr(y7);
    uint32_t y9 = mq_montymul(y8, y2);
    uint32_t y10 = mq_montymul(y9, y8);
    uint32_t y11 = mq_montysqr(y10);
    uint32_t y12 = mq_montysqr(y11);
    uint32_t y13 = mq_montymul(y12, y9);
    uint32_t y14 = mq_montysqr(y13);
    uint32_t y15 = mq_montysqr(y14);
    uint32_t y16 = mq_montymul(y15, y10);
    uint32_t y17 = mq_montysqr(y16);
    uint32_t y18 = mq_montymul(y17, y0);
    return mq_montymul(y18, x);
}

__device__ void mq_NTT(uint16_t* a, unsigned logn) {
    size_t n = (size_t)1 << logn;
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1) {
        size_t ht = t >> 1, j1 = 0;
        for (size_t i = 0; i < m; ++i, j1 += t) {
            uint32_t s = PQ_GMB[m + i];
            size_t j2 = j1 + ht;
            for (size_t j = j1; j < j2; ++j) {
                uint32_t u = a[j];
                uint32_t v = mq_montymul(a[j + ht], s);
                a[j] = (uint16_t)mq_add(u, v);
                a[j + ht] = (uint16_t)mq_sub(u, v);
            }
        }
        t = ht;
    }
}

__device__ void mq_iNTT(uint16_t* a, unsigned logn) {
    size_t n = (size_t)1 << logn;
    size_t t = 1, m = n;
    while (m > 1) {
        size_t hm = m >> 1, dt = t << 1, j1 = 0;
        for (size_t i = 0; i < hm; ++i, j1 += dt) {
            uint32_t s = PQ_IGMB[hm + i];
            size_t j2 = j1 + t;
            for (size_t j = j1; j < j2; ++j) {
                uint32_t u = a[j];
                uint32_t v = a[j + t];
                a[j] = (uint16_t)mq_add(u, v);
                uint32_t w = mq_sub(u, v);
                a[j + t] = (uint16_t)mq_montymul(w, s);
            }
        }
        t = dt;
        m = hm;
    }
    uint32_t ni = PQ_R;
    for (size_t mm = n; mm > 1; mm >>= 1) ni = mq_rshift1(ni);
    for (size_t mm = 0; mm < n; ++mm) a[mm] = (uint16_t)mq_montymul(a[mm], ni);
}

// NTT(f) into tt; returns 0 if f is not invertible mod q (some coeff is 0).
// Split out so the retry loop can reject before touching g's half.
__device__ int dev_ntt_f_check(const int8_t* f, uint16_t* tt) {
    for (int u = 0; u < PQ_N; ++u) tt[u] = (uint16_t)mq_conv_small(f[u]);
    mq_NTT(tt, 10);
    int ok = 1;
    for (int u = 0; u < PQ_N; ++u) ok &= (tt[u] != 0);
    return ok;
}

// h = g·f^-1 mod q, given tt = NTT(f) already checked nonzero everywhere.
__device__ void dev_pub_from_ntt_f(const int8_t* g, const uint16_t* tt, uint16_t* h) {
    for (int u = 0; u < PQ_N; ++u) h[u] = (uint16_t)mq_conv_small(g[u]);
    mq_NTT(h, 10);
    for (int u = 0; u < PQ_N; ++u) h[u] = (uint16_t)mq_div_12289(h[u], tt[u]);
    mq_iNTT(h, 10);
}

// h = g·f^-1 mod q. Returns 0 if f is not invertible (some NTT coeff is 0).
// `h` and `tt` are caller-provided scratch of PQ_N uint16 each.
__device__ int dev_compute_public(const int8_t* f, const int8_t* g, uint16_t* h, uint16_t* tt) {
    if (!dev_ntt_f_check(f, tt)) return 0;
    dev_pub_from_ntt_f(g, tt, h);
    return 1;
}

// ===========================================================================
// FP32 bnorm filter — keygen's orthogonalized-vector norm bound, reformulated.
//
// The reference computes (in FPEMU FP64):
//   bnorm = ||q·adj(f)/(f·adj f + g·adj g)||^2 + ||q·adj(g)/(...)||^2 < 16822.4121
// in the coefficient domain. By Parseval over the n negacyclic bins z_k =
// exp(-i·pi·(2k+1)/n) this equals (q^2/n)·sum_k 1/(|f^(z_k)|^2 + |g^(z_k)|^2)
// — forward transforms only (validated exactly vs the FPEMU classifier in
// pq-core tests/stage_rates.rs). We pack both real polys into ONE complex FFT
// (z_j = (f_j + i·g_j)·W2048[j]); for a conjugate bin pair the algebra
// collapses to |f^|^2+|g^|^2 = (|Z_b|^2 + |Z_b'|^2)/2, and with a DIF FFT
// (bit-reversed output) pairs sit at (b, 1023-b) in output order too.
//
// FP32 (never FP64 — Blackwell is ~1:64 FP64) differs from FPEMU only within
// ~1e-4 of the bound; a borderline disagreement merely wastes one seed, since
// the host re-verifies every hit with the reference keygen.
// ===========================================================================

// W2048[t] = exp(-i·pi·t/1024); filled once from host doubles (constant memory:
// all lanes of a warp read the same twiddle -> broadcast).
__device__ __constant__ float2 PQ_W2048[2048];

__device__ int dev_bnorm_ok(const int8_t* f, const int8_t* g, float2* z /* PQ_N scratch */) {
    // Twist + pack: z_j = (f_j + i·g_j) · W2048[j].
    for (int j = 0; j < PQ_N; ++j) {
        float2 w = PQ_W2048[j];
        float a = (float)f[j], b = (float)g[j];
        z[j].x = a * w.x - b * w.y;
        z[j].y = a * w.y + b * w.x;
    }
    // Iterative radix-2 DIF FFT, natural input, bit-reversed output (output
    // order is irrelevant: we only sum over conjugate bin pairs).
    for (int len = PQ_N; len >= 2; len >>= 1) {
        int half = len >> 1;
        int tstep = 2048 / len; // W_len^j = W2048[j·2048/len]
        for (int s = 0; s < PQ_N; s += len) {
            for (int j = 0; j < half; ++j) {
                float2 u = z[s + j], v = z[s + j + half];
                float dx = u.x - v.x, dy = u.y - v.y;
                z[s + j].x = u.x + v.x;
                z[s + j].y = u.y + v.y;
                float2 w = PQ_W2048[j * tstep];
                z[s + j + half].x = dx * w.x - dy * w.y;
                z[s + j + half].y = dx * w.y + dy * w.x;
            }
        }
    }
    // bnorm = (q^2/n)·sum_{1024 bins} 1/D = (q^2/256)·sum_{b<512} 1/(|Z_b|^2+|Z_{1023-b}|^2).
    float inv_sum = 0.0f;
    for (int b = 0; b < 512; ++b) {
        float2 p = z[b], r = z[1023 - b];
        float d = p.x * p.x + p.y * p.y + r.x * r.x + r.y * r.y;
        inv_sum += 1.0f / d;
    }
    const float bnorm = (12289.0f * 12289.0f / 256.0f) * inv_sum;
    return bnorm < 16822.4121f;
}

// Coefficient-limit (|c| <= 15) and squared-norm (< 16823) checks — integer,
// exactly the reference keygen's first two rejections (order-equivalent).
__device__ int dev_fg_int_ok(const int8_t* f, const int8_t* g) {
    uint32_t nf = 0, ng = 0;
    int bad = 0;
    for (int u = 0; u < PQ_N; ++u) {
        int a = f[u], b = g[u];
        bad |= (a >= 16) | (a <= -16) | (b >= 16) | (b <= -16);
        nf += (uint32_t)(a * a);
        ng += (uint32_t)(b * b);
    }
    return !bad && (nf + ng) < 16823;
}

// Retry cap: P(no visible accept in 192 attempts) ~ 0.941^192 ~ 8e-6 per seed;
// a capped seed is skipped (never a wrong emission — the host re-verifies).
#define PQ_MAX_ATTEMPTS 192

// Encode h into the 1793-byte public key: pk[0]=0x00|logn=0x0A, then 14-bit
// big-endian packed coefficients (codec.c modq_encode). 1024*14=14336 bits =
// 1792 bytes exactly (no tail).
__device__ void dev_modq_encode(const uint16_t* x, uint8_t* pk) {
    pk[0] = (uint8_t)(0x00u + 10u);
    uint8_t* buf = pk + 1;
    uint32_t acc = 0;
    int acc_len = 0;
    for (int u = 0; u < PQ_N; ++u) {
        acc = (acc << 14) | x[u];
        acc_len += 14;
        while (acc_len >= 8) {
            acc_len -= 8;
            *buf++ = (uint8_t)(acc >> acc_len);
        }
    }
}

// seed -> pk (public half): sample (f,g), compute h, encode. Returns 0 if f is
// singular (skip). Matches pq-core::fast::pubkey_from_fg byte-for-byte.
__device__ int dev_pubkey_from_seed(const uint8_t* seed, int seed_len, uint8_t* pk) {
    int8_t f[PQ_N], g[PQ_N];
    uint16_t h[PQ_N], tt[PQ_N];
    dev_sample_fg(seed, seed_len, f, g);
    if (!dev_compute_public(f, g, h, tt)) return 0;
    dev_modq_encode(h, pk);
    return 1;
}

// Debug kernel: emit pk[1793] + ok flag per entropy ("PQK" seed hash applied),
// to validate the device public half against the CPU oracle.
extern "C" __global__ void pq_dbg_pubkey_kernel(
    const uint8_t* __restrict__ entropies, uint64_t n, uint8_t* __restrict__ out_pk,
    uint8_t* __restrict__ out_ok)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint8_t seed[PQ_SEED_SIZE];
    dev_seed_from_entropy(entropies + gid * PQ_ENTROPY_SIZE, seed);
    uint8_t* pk = out_pk + gid * PQ_PUBKEY_SIZE;
    out_ok[gid] = (uint8_t)dev_pubkey_from_seed(seed, PQ_SEED_SIZE, pk);
}

extern "C" int pq_cuda_dbg_pubkey(
    const uint8_t* seeds, uint64_t n, uint8_t* out_pk, uint8_t* out_ok)
{
    uint8_t* d_seeds = nullptr;
    uint8_t* d_pk = nullptr;
    uint8_t* d_ok = nullptr;
    int rc = 0;
    #define CKP(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKP(cudaMalloc(&d_seeds, n * PQ_ENTROPY_SIZE));
    CKP(cudaMalloc(&d_pk, n * PQ_PUBKEY_SIZE));
    CKP(cudaMalloc(&d_ok, n));
    CKP(cudaMemcpy(d_seeds, seeds, n * PQ_ENTROPY_SIZE, cudaMemcpyHostToDevice));
    {
        const int block = 64;
        unsigned long long blocks = (n + block - 1) / block;
        if (blocks == 0) blocks = 1;
        pq_dbg_pubkey_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, d_pk, d_ok);
        CKP(cudaGetLastError());
        CKP(cudaDeviceSynchronize());
    }
    CKP(cudaMemcpy(out_pk, d_pk, n * PQ_PUBKEY_SIZE, cudaMemcpyDeviceToHost));
    CKP(cudaMemcpy(out_ok, d_ok, n, cudaMemcpyDeviceToHost));
cleanup:
    if (d_seeds) cudaFree(d_seeds);
    if (d_pk) cudaFree(d_pk);
    if (d_ok) cudaFree(d_ok);
    #undef CKP
    return rc;
}

// ===========================================================================
// Device SHA-512/256 (FIPS 180-4) for the PQ address hash. Integer-only.
// addr32 = SHA512_256("PQA" || "f1" || salt || pk).
// ===========================================================================

__device__ __constant__ uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
    0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
    0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
    0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
    0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
    0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
    0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
    0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
    0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
    0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
    0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
    0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
    0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
    0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
    0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
    0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
    0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

struct Sha512 {
    uint64_t h[8];
    uint64_t total; // bytes absorbed
    uint8_t buf[128];
    int buflen;
};

__device__ void sha512_256_init(Sha512& s) {
    s.h[0] = 0x22312194FC2BF72CULL;
    s.h[1] = 0x9F555FA3C84C64C2ULL;
    s.h[2] = 0x2393B86B6F53B151ULL;
    s.h[3] = 0x963877195940EABDULL;
    s.h[4] = 0x96283EE2A88EFFE3ULL;
    s.h[5] = 0xBE5E1E2553863992ULL;
    s.h[6] = 0x2B0199FC2C85B8AAULL;
    s.h[7] = 0x0EB72DDC81C52CA2ULL;
    s.total = 0;
    s.buflen = 0;
}

// Rolling 16-word message schedule, fully unrolled: all w[] indices are
// compile-time constants, so the 128-byte schedule lives in registers (the
// classic w[80] form is 640B of runtime-indexed local memory).
__device__ void sha512_compress(uint64_t h[8], const uint8_t* p) {
    uint64_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int b = i * 8;
        w[i] = ((uint64_t)p[b] << 56) | ((uint64_t)p[b + 1] << 48) |
               ((uint64_t)p[b + 2] << 40) | ((uint64_t)p[b + 3] << 32) |
               ((uint64_t)p[b + 4] << 24) | ((uint64_t)p[b + 5] << 16) |
               ((uint64_t)p[b + 6] << 8) | ((uint64_t)p[b + 7]);
    }
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];
    #pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint64_t wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint64_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            uint64_t s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7);
            uint64_t s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
            wi = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
            w[i & 15] = wi;
        }
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t t1 = hh + S1 + ch + SHA512_K[i] + wi;
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

__device__ void sha512_update(Sha512& s, const uint8_t* data, int len) {
    s.total += (uint64_t)len;
    if (s.buflen > 0) {
        int need = 128 - s.buflen;
        int take = (len < need) ? len : need;
        for (int i = 0; i < take; ++i) s.buf[s.buflen + i] = data[i];
        s.buflen += take;
        data += take;
        len -= take;
        if (s.buflen == 128) {
            sha512_compress(s.h, s.buf);
            s.buflen = 0;
        }
        // Everything fit in the partial buffer — do NOT fall through to the
        // tail copy, whose `s.buflen = len` would clobber the buffered count.
        if (len == 0) return;
    }
    while (len >= 128) {
        sha512_compress(s.h, data);
        data += 128;
        len -= 128;
    }
    for (int i = 0; i < len; ++i) s.buf[i] = data[i];
    s.buflen = len;
}

// Finalize and write the first 32 bytes (SHA-512/256 truncation) to out.
__device__ void sha512_256_final(Sha512& s, uint8_t* out) {
    uint64_t bitlen = s.total * 8;
    // Append 0x80, then zeros until offset 112 mod 128, then the 16-byte
    // big-endian bit length (high 64 bits are 0 for our message sizes).
    s.buf[s.buflen++] = 0x80;
    if (s.buflen > 112) {
        while (s.buflen < 128) s.buf[s.buflen++] = 0;
        sha512_compress(s.h, s.buf);
        s.buflen = 0;
    }
    while (s.buflen < 112) s.buf[s.buflen++] = 0;
    for (int i = 0; i < 8; ++i) s.buf[112 + i] = 0;
    for (int i = 0; i < 8; ++i) s.buf[120 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    sha512_compress(s.h, s.buf);
    // Output h[0..3] big-endian = 32 bytes.
    for (int i = 0; i < 4; ++i) {
        uint64_t v = s.h[i];
        for (int j = 0; j < 8; ++j) out[i * 8 + j] = (uint8_t)(v >> (56 - 8 * j));
    }
}

// addr32 = SHA512_256("PQA" || "f1" || salt || pk).
__device__ void dev_pq_addr32(const uint8_t* pk, uint8_t salt, uint8_t* addr32) {
    Sha512 s;
    sha512_256_init(s);
    uint8_t pre[6] = {'P', 'Q', 'A', 'f', '1', salt};
    sha512_update(s, pre, 6);
    sha512_update(s, pk, PQ_PUBKEY_SIZE);
    sha512_256_final(s, addr32);
}

// entropy -> Falcon keygen seed: SHA512_256("PQK" || "f1" || entropy). Mirrors
// go-algorand derivePQKeySeed; forward-declared near the top of the file.
__device__ void dev_seed_from_entropy(const uint8_t* entropy, uint8_t* seed) {
    Sha512 s;
    sha512_256_init(s);
    uint8_t pre[5];
    #pragma unroll
    for (int i = 0; i < 5; ++i) pre[i] = PQK_PREFIX[i];
    sha512_update(s, pre, 5);
    sha512_update(s, entropy, PQ_ENTROPY_SIZE);
    sha512_256_final(s, seed);
}

// entropy + salt -> addr32 (full public pipeline). Returns 0 if f is singular.
__device__ int dev_addr_from_entropy(const uint8_t* entropy, uint8_t salt, uint8_t* addr32) {
    uint8_t seed[PQ_SEED_SIZE];
    dev_seed_from_entropy(entropy, seed);
    uint8_t pk[PQ_PUBKEY_SIZE];
    if (!dev_pubkey_from_seed(seed, PQ_SEED_SIZE, pk)) return 0;
    dev_pq_addr32(pk, salt, addr32);
    return 1;
}

extern "C" __global__ void pq_dbg_addr32_kernel(
    const uint8_t* __restrict__ entropies, uint64_t n, uint8_t salt,
    uint8_t* __restrict__ out_addr, uint8_t* __restrict__ out_ok)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out_ok[gid] = (uint8_t)dev_addr_from_entropy(
        entropies + gid * PQ_ENTROPY_SIZE, salt, out_addr + gid * PQ_ADDR_SIZE);
}

extern "C" int pq_cuda_dbg_addr32(
    const uint8_t* seeds, uint64_t n, uint8_t salt, uint8_t* out_addr, uint8_t* out_ok)
{
    uint8_t* d_seeds = nullptr;
    uint8_t* d_addr = nullptr;
    uint8_t* d_ok = nullptr;
    int rc = 0;
    #define CKA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKA(cudaMalloc(&d_seeds, n * PQ_ENTROPY_SIZE));
    CKA(cudaMalloc(&d_addr, n * PQ_ADDR_SIZE));
    CKA(cudaMalloc(&d_ok, n));
    CKA(cudaMemcpy(d_seeds, seeds, n * PQ_ENTROPY_SIZE, cudaMemcpyHostToDevice));
    {
        const int block = 64;
        unsigned long long blocks = (n + block - 1) / block;
        if (blocks == 0) blocks = 1;
        pq_dbg_addr32_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, salt, d_addr, d_ok);
        CKA(cudaGetLastError());
        CKA(cudaDeviceSynchronize());
    }
    CKA(cudaMemcpy(out_addr, d_addr, n * PQ_ADDR_SIZE, cudaMemcpyDeviceToHost));
    CKA(cudaMemcpy(out_ok, d_ok, n, cudaMemcpyDeviceToHost));
cleanup:
    if (d_seeds) cudaFree(d_seeds);
    if (d_addr) cudaFree(d_addr);
    if (d_ok) cudaFree(d_ok);
    #undef CKA
    return rc;
}

// ===========================================================================
// Kernel + host harness (complete; exercises the full launch/copy pipeline).
// ===========================================================================

// In-seed retry mirroring the reference keygen loop for every check except
// solve_NTRU (host-verified): sample (f,g) on the CONTINUING SHAKE stream
// until coef-limit, sqnorm, f-invertibility, and bnorm (FP32) all pass.
//
// NOT used by the search kernel — measured 3x WORSE per usable key than
// first-sample scanning (RTX 5070: 51us vs 17.7us). With p ~ 5.9% per-attempt
// acceptance, a thread-per-key warp waits for its slowest lane: E[max of 32
// geometric(p)] ~ 67-80 attempts vs E[mean] = 17, so lockstep burns ~4x the
// mean work, while fresh seeds are free (first-sample scanning pays ~1/p seeds
// of one attempt each — cheaper than one seed of E[max] attempts). Kept (with
// the FP32 bnorm above, both oracle-validated in gpu-selftest) as groundwork
// for a warp-cooperative kernel, where per-lane tails do not lockstep-stall.
// (do/while, not for(;;)+continue — nvcc 13 miscompiles the latter here.)
__device__ int dev_retry_sample(
    const uint8_t* seed, int8_t* f, int8_t* g, uint16_t* tt, float2* z)
{
    Shake s;
    shake_init_seed32(s, seed);
    int att = 0;
    int ok = 0;
    do {
        dev_poly_small_mkgauss(s, f);
        dev_poly_small_mkgauss(s, g);
        ++att;
        // Same accept set as the reference (minus solve): the integer checks
        // and invertibility are bit-exact; bnorm is the FP32 filter above.
        ok = dev_fg_int_ok(f, g) && dev_ntt_f_check(f, tt) && dev_bnorm_ok(f, g, z);
    } while (!ok && att < PQ_MAX_ATTEMPTS);
    return ok ? att : 0;
}

// Note: forcing higher occupancy via __launch_bounds__ (capping registers)
// measured slower — the per-key sampler/NTT benefits more from the registers
// (ILP / fewer local round-trips) than from extra resident warps.
extern "C" __global__ void pq_search_kernel(
    const uint8_t* __restrict__ base_entropy,
    uint64_t start_counter,
    uint64_t num_items,
    uint32_t max_salt,
    const uint8_t* __restrict__ prefix,
    int prefix_len,
    uint8_t* __restrict__ out_hits,
    int max_hits,
    unsigned int* __restrict__ hit_count)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= num_items) return;

    uint8_t entropy[PQ_ENTROPY_SIZE];
    derive_entropy(base_entropy, start_counter + gid, entropy);
    uint8_t seed[PQ_SEED_SIZE];
    dev_seed_from_entropy(entropy, seed);

    // Public half of the FIRST sample only: candidate keys are seeds whose
    // first (f,g) the reference keygen accepts (~5.2%); the host discards the
    // rest at hit time. Scanning fresh seeds beats retrying in-seed here: see
    // dev_retry_sample's comment for the warp-tail measurement, and note the
    // sweep below runs on keygen-doomed keys because skipping a lane cannot
    // shorten its lockstep warp (a compacted two-kernel split measured <=1.3x
    // in the model — not worth it until the kernel goes warp-cooperative).
    uint8_t pk[PQ_PUBKEY_SIZE];
    if (!dev_pubkey_from_seed(seed, PQ_SEED_SIZE, pk)) return;

    uint8_t addr32[PQ_ADDR_SIZE];
    for (uint32_t salt = 0; salt <= max_salt; ++salt) {
        dev_pq_addr32(pk, (uint8_t)salt, addr32);
        if (base32_prefix_match(addr32, prefix, prefix_len)) {
            unsigned int slot = atomicAdd(hit_count, 1u);
            if ((int)slot < max_hits) {
                uint8_t* dst = out_hits + (size_t)slot * PQ_HIT_STRIDE;
                #pragma unroll
                for (int i = 0; i < PQ_ENTROPY_SIZE; ++i) dst[i] = entropy[i];
                dst[PQ_ENTROPY_SIZE] = (uint8_t)salt;
            }
            break; // one usable salt per key is enough
        }
    }
}

// Debug kernel: 0-based accepted-attempt index of the retry loop per entropy
// (-1 = capped). Validated against the CPU oracle pqv_first_visible_accept.
extern "C" __global__ void pq_dbg_accept_kernel(
    const uint8_t* __restrict__ entropies, uint64_t n, int32_t* __restrict__ out_att)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint8_t seed[PQ_SEED_SIZE];
    dev_seed_from_entropy(entropies + gid * PQ_ENTROPY_SIZE, seed);
    int8_t f[PQ_N], g[PQ_N];
    uint16_t tt[PQ_N];
    float2 z[PQ_N];
    out_att[gid] = dev_retry_sample(seed, f, g, tt, z) - 1;
}

// Fill the FP32 twiddle table once (host doubles -> device floats).
static int pq_w2048_ready = 0;
extern "C" int pq_cuda_init_tables(void)
{
    if (pq_w2048_ready) return 0;
    static float2 tab[2048];
    for (int t = 0; t < 2048; ++t) {
        double a = -M_PI * (double)t / 1024.0;
        tab[t].x = (float)cos(a);
        tab[t].y = (float)sin(a);
    }
    cudaError_t e = cudaMemcpyToSymbol(PQ_W2048, tab, sizeof(tab));
    if (e != cudaSuccess) return -(int)e;
    pq_w2048_ready = 1;
    return 0;
}

extern "C" int pq_cuda_dbg_accept(const uint8_t* seeds, uint64_t n, int32_t* out_att)
{
    uint8_t* d_seeds = nullptr;
    int32_t* d_att = nullptr;
    int rc = pq_cuda_init_tables();
    if (rc != 0) return rc;
    #define CKR(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKR(cudaMalloc(&d_seeds, n * PQ_ENTROPY_SIZE));
    CKR(cudaMalloc(&d_att, n * sizeof(int32_t)));
    CKR(cudaMemcpy(d_seeds, seeds, n * PQ_ENTROPY_SIZE, cudaMemcpyHostToDevice));
    {
        const int block = 128;
        unsigned long long blocks = (n + block - 1) / block;
        if (blocks == 0) blocks = 1;
        pq_dbg_accept_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, d_att);
        CKR(cudaGetLastError());
        CKR(cudaDeviceSynchronize());
    }
    CKR(cudaMemcpy(out_att, d_att, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
cleanup:
    if (d_seeds) cudaFree(d_seeds);
    if (d_att) cudaFree(d_att);
    #undef CKR
    return rc;
}

extern "C" int pq_cuda_device_count()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

// Make the host thread SLEEP (not spin) while waiting on the GPU. CUDA's default
// auto policy spins on one core when contexts < CPUs, pinning a core at 100% for
// the whole (multi-second) kernel. Blocking sync frees that core; the wakeup
// latency is negligible vs our batch time. Must be called before the device's
// primary context is initialized (i.e. before the first malloc/launch).
// Returns the cudaError_t (0 = ok; nonzero, e.g. SetOnActiveProcess, is ignored
// by the caller and we simply keep the default spin policy).
extern "C" int pq_cuda_set_blocking_sync()
{
    return (int)cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
}

// Launch one batch. Returns hit count (>=0) or a negative CUDA error code.
// Sets *out_stub = 1 while the device math is unimplemented.
extern "C" int pq_cuda_search(
    const uint8_t* base_entropy,
    uint64_t start_counter,
    uint64_t num_items,
    uint32_t max_salt,
    const uint8_t* prefix,
    int prefix_len,
    uint8_t* out_hits,
    int max_hits,
    int* out_stub)
{
    // Device math is implemented and validated bit-exact vs the CPU oracle.
    if (out_stub) *out_stub = 0;

    {
        int ir = pq_cuda_init_tables();
        if (ir != 0) return ir;
    }

    uint8_t* d_seed = nullptr;
    uint8_t* d_prefix = nullptr;
    uint8_t* d_hits = nullptr;
    unsigned int* d_count = nullptr;
    unsigned int count = 0;   // declared before any `goto cleanup` (no init-crossing)
    int rc = 0;

    #define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)

    CK(cudaMalloc(&d_seed, PQ_ENTROPY_SIZE));
    CK(cudaMalloc(&d_prefix, prefix_len > 0 ? (size_t)prefix_len : 1));
    CK(cudaMalloc(&d_hits, (size_t)max_hits * PQ_HIT_STRIDE));
    CK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CK(cudaMemcpy(d_seed, base_entropy, PQ_ENTROPY_SIZE, cudaMemcpyHostToDevice));
    if (prefix_len > 0)
        CK(cudaMemcpy(d_prefix, prefix, prefix_len, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    {
        // Block size is tunable per GPU via PQ_GPU_BLOCK (default 256).
        int block = 256;
        if (const char* e = getenv("PQ_GPU_BLOCK")) {
            int b = atoi(e);
            if (b >= 32 && b <= 1024) block = b;
        }
        unsigned long long blocks = (num_items + block - 1) / block;
        if (blocks == 0) blocks = 1;
        pq_search_kernel<<<(unsigned int)blocks, block>>>(
            d_seed, start_counter, num_items, max_salt,
            d_prefix, prefix_len, d_hits, max_hits, d_count);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }

    CK(cudaMemcpy(&count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if ((int)count > max_hits) count = (unsigned int)max_hits;
    if (count > 0)
        CK(cudaMemcpy(out_hits, d_hits, (size_t)count * PQ_HIT_STRIDE, cudaMemcpyDeviceToHost));
    rc = (int)count;

cleanup:
    if (d_seed)   cudaFree(d_seed);
    if (d_prefix) cudaFree(d_prefix);
    if (d_hits)   cudaFree(d_hits);
    if (d_count)  cudaFree(d_count);
    #undef CK
    return rc;
}

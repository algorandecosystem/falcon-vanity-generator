// ed25519_vanity.cu — CUDA search for classic Algorand (Ed25519) vanity
// addresses. Separate translation unit from pq_vanity.cu.
//
// Classic account derivation (go-algorand crypto.GenerateSignatureSecrets →
// libsodium crypto_sign_ed25519_seed_keypair, i.e. RFC 8032):
//
//   seed[32]   — the 25-word mnemonic key (algokey/goal import -m)
//   a          = clamp(SHA-512(seed)[0..32])
//   pk         = encode(a·B)
//   addr32     = pk                       (no hash — the address IS the key)
//   display    = base32(pk || last4(SHA512_256(pk)))
//
// Field / group / SHA-512 arithmetic is the CUDA port of orlp/ed25519 from
// solana-perf-libs, vendored UNMODIFIED in vendor/ed25519-cuda (zlib +
// Apache-2.0; pinned in vendor/ed25519-cuda/PINNED_COMMIT). This file only
// adds the search driver and a variable-time base-point table lookup (see
// ed_scalarmult_base_vartime) — the vendored ge_scalarmult_base is
// constant-time, which costs 8 table scans per window and buys nothing here:
// every candidate seed is public to this process anyway, and hits are
// re-derived on the CPU before they are reported.
//
// Validated bit-exact against ed25519-dalek by `pq-vanity gpu-selftest
// --scheme ed25519`.

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"

#define ED_SEED_SIZE   32
#define ED_PK_SIZE     32
#define ED_HIT_STRIDE  ED_SEED_SIZE   // seed[32]

// Keys per thread. Their compressions share ONE field inversion (Montgomery
// batch trick): fe_invert is ~265 field ops — about a third of a per-key
// scalar multiplication — so amortizing it is the largest single win.
#ifndef ED_KEYS_PER_THREAD
#define ED_KEYS_PER_THREAD 8
#endif

// Search targets (see crates/pq-vanity/src/targets.rs): with hi = BE u64 of
// addr32[0..8] and lo = BE u64 of addr32[28..32] || checksum4, a target
// matches iff (hi & pmask) == pval && (lo & smask) == sval. Uniform loop over
// __constant__ memory → every lane reads the same entry (broadcast).
#define ED_MAX_TARGETS 1024
struct EdTarget { uint64_t pmask, pval, smask, sval; };
__device__ __constant__ EdTarget ED_TARGETS[ED_MAX_TARGETS];

__device__ __forceinline__ uint64_t ed_be64(const uint8_t* p) {
    uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

// The address checksum: SHA512_256(addr32)[28..32] as a BE u32. SHA-512/256
// is SHA-512 with its own IV, truncated — reuse the vendored SHA-512.
__device__ uint32_t ed_checksum32(const uint8_t* addr32)
{
    sha512_context md;
    sha512_init(&md);
    md.state[0] = 0x22312194FC2BF72CULL; md.state[1] = 0x9F555FA3C84C64C2ULL;
    md.state[2] = 0x2393B86B6F53B151ULL; md.state[3] = 0x963877195940EABDULL;
    md.state[4] = 0x96283EE2A88EFFE3ULL; md.state[5] = 0xBE5E1E2553863992ULL;
    md.state[6] = 0x2B0199FC2C85B8AAULL; md.state[7] = 0x0EB72DDC81C52CA2ULL;
    sha512_update(&md, addr32, 32);
    uint8_t out[64];
    sha512_final(&md, out);
    return ((uint32_t)out[28] << 24) | ((uint32_t)out[29] << 16) |
           ((uint32_t)out[30] << 8) | (uint32_t)out[31];
}

// 1 if any of the n_targets matches. The checksum is computed lazily — only
// for a suffix target whose prefix already matched — so prefix-only searches
// never hash.
__device__ int ed_match_targets(const uint8_t* addr32, int n_targets)
{
    const uint64_t hi = ed_be64(addr32);
    uint64_t lo = 0;
    int have_lo = 0;
    for (int t = 0; t < n_targets; ++t) {
        const EdTarget T = ED_TARGETS[t];
        if ((hi & T.pmask) != T.pval) continue;
        if (T.smask == 0) return 1;
        if (!have_lo) {
            lo = ((uint64_t)(uint32_t)(ed_be64(addr32 + 24)) << 32) | ed_checksum32(addr32);
            have_lo = 1;
        }
        if ((lo & T.smask) == T.sval) return 1;
    }
    return 0;
}

// seed with an 8-byte LE counter spliced into bytes [24..32] — the same
// layout as the Falcon kernel's derive_entropy, so the host hygiene (random
// wrapping counter start, one hit per batch) carries over unchanged.
__device__ __forceinline__ void ed_derive_seed(
    const uint8_t* __restrict__ base_seed, uint64_t counter, uint8_t* __restrict__ seed)
{
    #pragma unroll
    for (int i = 0; i < 24; ++i) seed[i] = base_seed[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) seed[24 + i] = (uint8_t)(counter >> (8 * i));
}

// seed -> clamped scalar a (first half of SHA-512(seed)).
__device__ __forceinline__ void ed_seed_to_scalar(const uint8_t* seed, uint8_t* a)
{
    uint8_t h[64];
    sha512(seed, ED_SEED_SIZE, h);
    #pragma unroll
    for (int i = 0; i < 32; ++i) a[i] = h[i];
    a[0] &= 248;
    a[31] &= 63;
    a[31] |= 64;
}

// t = b·256^pos·B from the vendored table, variable-time (b in -8..8).
__device__ __forceinline__ void ed_select_vartime(ge_precomp* t, int pos, signed char b)
{
    if (b == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
        return;
    }
    const int babs = b < 0 ? -b : b;
    const ge_precomp* u = &base[pos][babs - 1];
    if (b > 0) {
        fe_copy(t->yplusx, u->yplusx);
        fe_copy(t->yminusx, u->yminusx);
        fe_copy(t->xy2d, u->xy2d);
    } else {
        // -P in Duif form: swap (y+x, y-x), negate 2dxy.
        fe_copy(t->yplusx, u->yminusx);
        fe_copy(t->yminusx, u->yplusx);
        fe_neg(t->xy2d, u->xy2d);
    }
}

// h = a·B. Same signed-radix-16 schedule as the vendored ge_scalarmult_base
// (ref10), with ed_select_vartime in place of the constant-time select.
__device__ void ed_scalarmult_base_vartime(ge_p3* h, const uint8_t* a)
{
    signed char e[64];
    for (int i = 0; i < 32; ++i) {
        e[2 * i + 0] = (a[i] >> 0) & 15;
        e[2 * i + 1] = (a[i] >> 4) & 15;
    }
    signed char carry = 0;
    for (int i = 0; i < 63; ++i) {
        e[i] += carry;
        carry = e[i] + 8;
        carry >>= 4;
        e[i] -= carry << 4;
    }
    e[63] += carry;

    ge_p1p1 r;
    ge_p2 s;
    ge_precomp t;
    ge_p3_0(h);
    for (int i = 1; i < 64; i += 2) {
        ed_select_vartime(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
    ge_p3_dbl(&r, h);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p3(h, &r);
    for (int i = 0; i < 64; i += 2) {
        ed_select_vartime(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

// Encode (X:Y:Z) given recip = 1/Z — ge_p3_tobytes minus its inversion.
__device__ __forceinline__ void ed_encode_with_recip(uint8_t* pk, const ge_p3* p, const fe recip)
{
    fe x, y;
    fe_mul(x, p->X, recip);
    fe_mul(y, p->Y, recip);
    fe_tobytes(pk, y);
    pk[31] ^= fe_isnegative(x) << 7;
}

// Debug kernel: seed -> pk via the vendored, constant-time reference path
// (ge_scalarmult_base + ge_p3_tobytes). Cross-checked against BOTH the CPU
// oracle and the search path (ed_dbg_pubkey_fast_kernel).
extern "C" __global__ void ed_dbg_pubkey_ref_kernel(
    const uint8_t* __restrict__ seeds, uint64_t n, uint8_t* __restrict__ out_pk)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    uint8_t a[32];
    ed_seed_to_scalar(seeds + gid * ED_SEED_SIZE, a);
    ge_p3 A;
    ge_scalarmult_base(&A, a);
    ge_p3_tobytes(out_pk + gid * ED_PK_SIZE, &A);
}

// Shared core of the search and its debug twin: ED_KEYS_PER_THREAD
// consecutive seeds, one batched inversion, pk[k] written to `pks`.
__device__ void ed_pubkeys_batch(const uint8_t (*seeds)[ED_SEED_SIZE], int nk,
                                 uint8_t (*pks)[ED_PK_SIZE])
{
    ge_p3 P[ED_KEYS_PER_THREAD];
    fe acc[ED_KEYS_PER_THREAD];   // acc[k] = Z_0 · … · Z_k
    for (int k = 0; k < nk; ++k) {
        uint8_t a[32];
        ed_seed_to_scalar(seeds[k], a);
        ed_scalarmult_base_vartime(&P[k], a);
        if (k == 0) fe_copy(acc[0], P[0].Z);
        else        fe_mul(acc[k], acc[k - 1], P[k].Z);
    }
    fe inv;                        // (Z_0 · … · Z_{k})^-1, peeled from the top
    fe_invert(inv, acc[nk - 1]);
    for (int k = nk - 1; k >= 0; --k) {
        fe recip;
        if (k > 0) {
            fe_mul(recip, inv, acc[k - 1]);   // 1/Z_k
            fe_mul(inv, inv, P[k].Z);         // drop Z_k from the running inverse
        } else {
            fe_copy(recip, inv);
        }
        ed_encode_with_recip(pks[k], &P[k], recip);
    }
}

// Debug kernel: the SEARCH path's pubkeys (vartime select + batch inversion).
// Each thread takes ED_KEYS_PER_THREAD consecutive seeds from `seeds`.
extern "C" __global__ void ed_dbg_pubkey_fast_kernel(
    const uint8_t* __restrict__ seeds, uint64_t n, uint8_t* __restrict__ out_pk)
{
    uint64_t first = (blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) * ED_KEYS_PER_THREAD;
    if (first >= n) return;
    int nk = (int)((n - first) < ED_KEYS_PER_THREAD ? (n - first) : ED_KEYS_PER_THREAD);
    uint8_t s[ED_KEYS_PER_THREAD][ED_SEED_SIZE];
    uint8_t pk[ED_KEYS_PER_THREAD][ED_PK_SIZE];
    for (int k = 0; k < nk; ++k)
        for (int i = 0; i < ED_SEED_SIZE; ++i) s[k][i] = seeds[(first + k) * ED_SEED_SIZE + i];
    ed_pubkeys_batch(s, nk, pk);
    for (int k = 0; k < nk; ++k)
        for (int i = 0; i < ED_PK_SIZE; ++i) out_pk[(first + k) * ED_PK_SIZE + i] = pk[k][i];
}

extern "C" __global__ void ed_search_kernel(
    const uint8_t* __restrict__ base_seed,
    uint64_t start_counter,
    uint64_t num_items,
    int n_targets,
    uint8_t* __restrict__ out_hits,
    int max_hits,
    unsigned int* __restrict__ hit_count)
{
    uint64_t first = (blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) * ED_KEYS_PER_THREAD;
    if (first >= num_items) return;
    int nk = (int)((num_items - first) < ED_KEYS_PER_THREAD ? (num_items - first)
                                                           : ED_KEYS_PER_THREAD);
    uint8_t s[ED_KEYS_PER_THREAD][ED_SEED_SIZE];
    uint8_t pk[ED_KEYS_PER_THREAD][ED_PK_SIZE];
    for (int k = 0; k < nk; ++k) ed_derive_seed(base_seed, start_counter + first + k, s[k]);
    ed_pubkeys_batch(s, nk, pk);
    for (int k = 0; k < nk; ++k) {
        if (ed_match_targets(pk[k], n_targets)) {
            unsigned int slot = atomicAdd(hit_count, 1u);
            if ((int)slot < max_hits) {
                uint8_t* dst = out_hits + (size_t)slot * ED_HIT_STRIDE;
                for (int i = 0; i < ED_SEED_SIZE; ++i) dst[i] = s[k][i];
            }
        }
    }
}

// Block size, tunable via PQ_GPU_BLOCK like the Falcon kernel (default 128:
// the batched path holds ED_KEYS_PER_THREAD points per thread).
static int ed_block_size()
{
    int block = 128;
    if (const char* e = getenv("PQ_GPU_BLOCK")) {
        int b = atoi(e);
        if (b >= 32 && b <= 1024) block = b;
    }
    return block;
}

// Debug: out[n] = ed_match_targets(addrs[i]) for n raw 32-byte addresses
// against the given target table — validates the matcher (incl. the device
// checksum) on crafted addresses independent of keygen.
extern "C" __global__ void ed_dbg_match_kernel(
    const uint8_t* __restrict__ addrs, uint64_t n, int n_targets, uint8_t* __restrict__ out)
{
    uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = (uint8_t)ed_match_targets(addrs + gid * 32, n_targets);
}

static int ed_upload_targets(const uint64_t* targets, int n_targets)
{
    if (n_targets < 0 || n_targets > ED_MAX_TARGETS) return -1000;
    if (n_targets == 0) return 0;
    cudaError_t e = cudaMemcpyToSymbol(ED_TARGETS, targets, (size_t)n_targets * sizeof(EdTarget));
    return e == cudaSuccess ? 0 : -(int)e;
}

extern "C" int ed_cuda_dbg_match(const uint8_t* addrs, uint64_t n,
                                 const uint64_t* targets, int n_targets, uint8_t* out)
{
    uint8_t* d_addrs = nullptr;
    uint8_t* d_out = nullptr;
    int rc = ed_upload_targets(targets, n_targets);
    if (rc != 0) return rc;
    #define CKM(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKM(cudaMalloc(&d_addrs, n * 32));
    CKM(cudaMalloc(&d_out, n));
    CKM(cudaMemcpy(d_addrs, addrs, n * 32, cudaMemcpyHostToDevice));
    {
        unsigned long long blocks = (n + 127) / 128;
        if (blocks == 0) blocks = 1;
        ed_dbg_match_kernel<<<(unsigned int)blocks, 128>>>(d_addrs, n, n_targets, d_out);
        CKM(cudaGetLastError());
        CKM(cudaDeviceSynchronize());
    }
    CKM(cudaMemcpy(out, d_out, n, cudaMemcpyDeviceToHost));
cleanup:
    if (d_addrs) cudaFree(d_addrs);
    if (d_out) cudaFree(d_out);
    #undef CKM
    return rc;
}

// Debug: out_pk[n*32] for n seeds. fast=0 runs the vendored constant-time
// reference path, fast=1 the search path. Returns 0 or a negative CUDA error.
extern "C" int ed_cuda_dbg_pubkey(const uint8_t* seeds, uint64_t n, int fast, uint8_t* out_pk)
{
    uint8_t* d_seeds = nullptr;
    uint8_t* d_pk = nullptr;
    int rc = 0;
    #define CKE(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKE(cudaMalloc(&d_seeds, n * ED_SEED_SIZE));
    CKE(cudaMalloc(&d_pk, n * ED_PK_SIZE));
    CKE(cudaMemcpy(d_seeds, seeds, n * ED_SEED_SIZE, cudaMemcpyHostToDevice));
    {
        const int block = ed_block_size();
        uint64_t threads = fast ? (n + ED_KEYS_PER_THREAD - 1) / ED_KEYS_PER_THREAD : n;
        unsigned long long blocks = (threads + block - 1) / block;
        if (blocks == 0) blocks = 1;
        if (fast)
            ed_dbg_pubkey_fast_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, d_pk);
        else
            ed_dbg_pubkey_ref_kernel<<<(unsigned int)blocks, block>>>(d_seeds, n, d_pk);
        CKE(cudaGetLastError());
        CKE(cudaDeviceSynchronize());
    }
    CKE(cudaMemcpy(out_pk, d_pk, n * ED_PK_SIZE, cudaMemcpyDeviceToHost));
cleanup:
    if (d_seeds) cudaFree(d_seeds);
    if (d_pk) cudaFree(d_pk);
    #undef CKE
    return rc;
}

// Launch one batch of num_items seeds (base_seed with counters
// start_counter..start_counter+num_items, wrapping) against n_targets targets
// (4 u64 each: pmask, pval, smask, sval). Returns the number of hit seeds
// copied (<= max_hits) or a negative error.
extern "C" int ed_cuda_search(
    const uint8_t* base_seed,
    uint64_t start_counter,
    uint64_t num_items,
    const uint64_t* targets,
    int n_targets,
    uint8_t* out_hits,
    int max_hits)
{
    uint8_t* d_seed = nullptr;
    uint8_t* d_hits = nullptr;
    unsigned int* d_count = nullptr;
    unsigned int count = 0;
    int rc = ed_upload_targets(targets, n_targets);
    if (rc != 0) return rc;
    #define CKS(call) do { cudaError_t e = (call); if (e != cudaSuccess) { rc = -(int)e; goto cleanup; } } while (0)
    CKS(cudaMalloc(&d_seed, ED_SEED_SIZE));
    CKS(cudaMalloc(&d_hits, (size_t)(max_hits > 0 ? max_hits : 1) * ED_HIT_STRIDE));
    CKS(cudaMalloc(&d_count, sizeof(unsigned int)));
    CKS(cudaMemcpy(d_seed, base_seed, ED_SEED_SIZE, cudaMemcpyHostToDevice));
    CKS(cudaMemset(d_count, 0, sizeof(unsigned int)));
    {
        const int block = ed_block_size();
        uint64_t threads = (num_items + ED_KEYS_PER_THREAD - 1) / ED_KEYS_PER_THREAD;
        unsigned long long blocks = (threads + block - 1) / block;
        if (blocks == 0) blocks = 1;
        ed_search_kernel<<<(unsigned int)blocks, block>>>(
            d_seed, start_counter, num_items, n_targets, d_hits, max_hits, d_count);
        CKS(cudaGetLastError());
        CKS(cudaDeviceSynchronize());
    }
    CKS(cudaMemcpy(&count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    {
        unsigned int copy = count > (unsigned int)max_hits ? (unsigned int)max_hits : count;
        if (copy > 0)
            CKS(cudaMemcpy(out_hits, d_hits, (size_t)copy * ED_HIT_STRIDE, cudaMemcpyDeviceToHost));
    }
    rc = (int)(count > (unsigned int)max_hits ? (unsigned int)max_hits : count);
cleanup:
    if (d_seed)   cudaFree(d_seed);
    if (d_hits)   cudaFree(d_hits);
    if (d_count)  cudaFree(d_count);
    #undef CKS
    return rc;
}

/*
 * pq-vanity shim over the vendored Falcon reference keygen.
 *
 * It #includes keygen.c to gain access to its *static* helpers
 * (poly_small_mkgauss, poly_small_sqnorm, poly_small_to_fp, solve_NTRU) and the
 * file-local fpr constants (fpr_q, fpr_bnorm_max). Exported helpers
 * (Zf(compute_public), Zf(modq_encode), Zf(trim_i8_encode), the FFT routines,
 * the max_*_bits tables) and shake256_init_prng_from_seed() come from the other
 * translation units at link time.
 *
 * IMPORTANT: because this file includes keygen.c, the build (crates/pq-core/
 * build.rs) compiles THIS file INSTEAD of keygen.c, to avoid duplicate symbols.
 *
 * The three entry points split keygen's per-attempt work so the GPU can compute
 * only the public half while the CPU completes the key at a hit:
 *   - pqv_sample_fg         : seed -> (f,g)         [keygen's FIRST attempt]
 *   - pqv_pubkey_from_fg     : (f,g) -> pk           [public half only: what the GPU mirrors]
 *   - pqv_complete_from_fg   : (f,g) -> (pk, sk)      [full acceptance + NTRU solve]
 *
 * Consistency: pk depends only on h = g/f mod q, which pqv_pubkey_from_fg and
 * pqv_complete_from_fg compute identically; the extra acceptance checks only
 * decide whether (f,g) yields a usable key.
 */

#include <stdint.h>
#include <string.h>

#include "falcon.h"
#include "keygen.c" /* brings in inner.h + all of keygen.c's statics */

#define PQV_LOGN 10u
#define PQV_N 1024u

/* shake256_init_prng_from_seed lives in falcon.c (declared in falcon.h). */

/*
 * Sample (f,g) from a seed exactly as the first iteration of Zf(keygen):
 * SHAKE256-PRNG(seed) -> poly_small_mkgauss(f) -> poly_small_mkgauss(g).
 * Returns 1 (always; rejection happens later, in completion).
 */
int pqv_sample_fg(const void *seed, size_t seed_len, int8_t *f, int8_t *g)
{
	shake256_context sc;
	shake256_init_prng_from_seed(&sc, seed, seed_len);
	inner_shake256_context *rng = (inner_shake256_context *)&sc;
	poly_small_mkgauss(rng, f, PQV_LOGN);
	poly_small_mkgauss(rng, g, PQV_LOGN);
	return 1;
}

/*
 * Public key for (f,g): pk[0] = 0x00|logn, then modq_encode(h = g/f mod q).
 * Returns 0 if f is not invertible mod q (skip) or encoding fails.
 * This is exactly what the GPU hot loop must reproduce, byte-for-byte.
 */
int pqv_pubkey_from_fg(const int8_t *f, const int8_t *g, uint8_t *pk)
{
	uint16_t h[PQV_N];
	uint64_t tmp64[(FALCON_TMPSIZE_MAKEPUB(PQV_LOGN) + 7) / 8];
	uint8_t *tmp = (uint8_t *)tmp64;

	if (!Zf(compute_public)(h, f, g, PQV_LOGN, tmp)) {
		return 0;
	}
	size_t pk_len = FALCON_PUBKEY_SIZE(PQV_LOGN);
	pk[0] = (uint8_t)(0x00u + PQV_LOGN);
	size_t v = Zf(modq_encode)(pk + 1, pk_len - 1, h, PQV_LOGN);
	return v == (pk_len - 1);
}

/*
 * Full completion for (f,g): replicates a single iteration of Zf(keygen)'s
 * acceptance (coefficient bounds, squared-norm bound, orthogonalized-vector
 * norm) then solve_NTRU, and encodes the standard Falcon sk (0x50|logn ||
 * trim_i8(f) || trim_i8(g) || trim_i8(F)) and pk (0x00|logn || modq_encode(h)).
 *
 * Returns 1 on success, 0 if (f,g) is rejected at any step (the caller then
 * discards the hit and keeps searching).
 */
int pqv_complete_from_fg(const int8_t *fin, const int8_t *gin, uint8_t *pk, uint8_t *sk)
{
	int8_t f[PQV_N], g[PQV_N], F[PQV_N], G[PQV_N];
	uint16_t h[PQV_N];
	uint64_t tmp64[(FALCON_TMPSIZE_KEYGEN(PQV_LOGN) + 7) / 8];
	uint8_t *tmp = (uint8_t *)tmp64;
	size_t u, v;
	int lim;

	memcpy(f, fin, PQV_N);
	memcpy(g, gin, PQV_N);

	/* Coefficient bounds (encodability with FALCON_COMP_TRIM). */
	lim = 1 << (Zf(max_fg_bits)[PQV_LOGN] - 1);
	for (u = 0; u < PQV_N; u++) {
		if (f[u] >= lim || f[u] <= -lim || g[u] >= lim || g[u] <= -lim) {
			return 0;
		}
	}

	/* Squared-norm bound: ||(g,-f)||^2 < 16823. */
	{
		uint32_t normf = poly_small_sqnorm(f, PQV_LOGN);
		uint32_t normg = poly_small_sqnorm(g, PQV_LOGN);
		uint32_t norm = (normf + normg) | -((normf | normg) >> 31);
		if (norm >= 16823) {
			return 0;
		}
	}

	/* Orthogonalized-vector norm bound (FFT; integer-emulated via FPEMU). */
	{
		fpr *rt1 = (fpr *)tmp;
		fpr *rt2 = rt1 + PQV_N;
		fpr *rt3 = rt2 + PQV_N;
		fpr bnorm;
		poly_small_to_fp(rt1, f, PQV_LOGN);
		poly_small_to_fp(rt2, g, PQV_LOGN);
		Zf(FFT)(rt1, PQV_LOGN);
		Zf(FFT)(rt2, PQV_LOGN);
		Zf(poly_invnorm2_fft)(rt3, rt1, rt2, PQV_LOGN);
		Zf(poly_adj_fft)(rt1, PQV_LOGN);
		Zf(poly_adj_fft)(rt2, PQV_LOGN);
		Zf(poly_mulconst)(rt1, fpr_q, PQV_LOGN);
		Zf(poly_mulconst)(rt2, fpr_q, PQV_LOGN);
		Zf(poly_mul_autoadj_fft)(rt1, rt3, PQV_LOGN);
		Zf(poly_mul_autoadj_fft)(rt2, rt3, PQV_LOGN);
		Zf(iFFT)(rt1, PQV_LOGN);
		Zf(iFFT)(rt2, PQV_LOGN);
		bnorm = fpr_zero;
		for (u = 0; u < PQV_N; u++) {
			bnorm = fpr_add(bnorm, fpr_sqr(rt1[u]));
			bnorm = fpr_add(bnorm, fpr_sqr(rt2[u]));
		}
		if (!fpr_lt(bnorm, fpr_bnorm_max)) {
			return 0;
		}
	}

	/* Public key h = g/f mod q (also rejects non-invertible f). */
	if (!Zf(compute_public)(h, f, g, PQV_LOGN, tmp)) {
		return 0;
	}

	/* Solve NTRU for (F,G). */
	lim = (1 << (Zf(max_FG_bits)[PQV_LOGN] - 1)) - 1;
	if (!solve_NTRU(PQV_LOGN, F, G, f, g, lim, (uint32_t *)tmp)) {
		return 0;
	}

	/* Encode private key: 0x50|logn || trim_i8(f) || trim_i8(g) || trim_i8(F). */
	size_t sk_len = FALCON_PRIVKEY_SIZE(PQV_LOGN);
	sk[0] = (uint8_t)(0x50u + PQV_LOGN);
	u = 1;
	v = Zf(trim_i8_encode)(sk + u, sk_len - u, f, PQV_LOGN, Zf(max_fg_bits)[PQV_LOGN]);
	if (v == 0) {
		return 0;
	}
	u += v;
	v = Zf(trim_i8_encode)(sk + u, sk_len - u, g, PQV_LOGN, Zf(max_fg_bits)[PQV_LOGN]);
	if (v == 0) {
		return 0;
	}
	u += v;
	v = Zf(trim_i8_encode)(sk + u, sk_len - u, F, PQV_LOGN, Zf(max_FG_bits)[PQV_LOGN]);
	if (v == 0) {
		return 0;
	}
	u += v;
	if (u != sk_len) {
		return 0;
	}

	/* Encode public key: 0x00|logn || modq_encode(h). */
	size_t pk_len = FALCON_PUBKEY_SIZE(PQV_LOGN);
	pk[0] = (uint8_t)(0x00u + PQV_LOGN);
	v = Zf(modq_encode)(pk + 1, pk_len - 1, h, PQV_LOGN);
	if (v != (pk_len - 1)) {
		return 0;
	}

	(void)G; /* G is recomputed at sign time; not stored in the sk. */
	return 1;
}

/*
 * Diagnostic: which keygen acceptance stage rejects (f,g)? Mirrors
 * pqv_complete_from_fg's order exactly (== the reference keygen loop).
 *   0 = accepted   1 = coef limit   2 = sqnorm   3 = bnorm (FP)
 *   4 = f not invertible mod q      5 = solve_NTRU failed
 */
int pqv_first_sample_stage(const int8_t *fin, const int8_t *gin)
{
	int8_t f[PQV_N], g[PQV_N], F[PQV_N], G[PQV_N];
	uint16_t h[PQV_N];
	uint64_t tmp64[(FALCON_TMPSIZE_KEYGEN(PQV_LOGN) + 7) / 8];
	uint8_t *tmp = (uint8_t *)tmp64;
	size_t u;
	int lim;

	memcpy(f, fin, PQV_N);
	memcpy(g, gin, PQV_N);

	lim = 1 << (Zf(max_fg_bits)[PQV_LOGN] - 1);
	for (u = 0; u < PQV_N; u++) {
		if (f[u] >= lim || f[u] <= -lim || g[u] >= lim || g[u] <= -lim) {
			return 1;
		}
	}
	{
		uint32_t normf = poly_small_sqnorm(f, PQV_LOGN);
		uint32_t normg = poly_small_sqnorm(g, PQV_LOGN);
		uint32_t norm = (normf + normg) | -((normf | normg) >> 31);
		if (norm >= 16823) {
			return 2;
		}
	}
	{
		fpr *rt1 = (fpr *)tmp;
		fpr *rt2 = rt1 + PQV_N;
		fpr *rt3 = rt2 + PQV_N;
		fpr bnorm;
		poly_small_to_fp(rt1, f, PQV_LOGN);
		poly_small_to_fp(rt2, g, PQV_LOGN);
		Zf(FFT)(rt1, PQV_LOGN);
		Zf(FFT)(rt2, PQV_LOGN);
		Zf(poly_invnorm2_fft)(rt3, rt1, rt2, PQV_LOGN);
		Zf(poly_adj_fft)(rt1, PQV_LOGN);
		Zf(poly_adj_fft)(rt2, PQV_LOGN);
		Zf(poly_mulconst)(rt1, fpr_q, PQV_LOGN);
		Zf(poly_mulconst)(rt2, fpr_q, PQV_LOGN);
		Zf(poly_mul_autoadj_fft)(rt1, rt3, PQV_LOGN);
		Zf(poly_mul_autoadj_fft)(rt2, rt3, PQV_LOGN);
		Zf(iFFT)(rt1, PQV_LOGN);
		Zf(iFFT)(rt2, PQV_LOGN);
		bnorm = fpr_zero;
		for (u = 0; u < PQV_N; u++) {
			bnorm = fpr_add(bnorm, fpr_sqr(rt1[u]));
			bnorm = fpr_add(bnorm, fpr_sqr(rt2[u]));
		}
		if (!fpr_lt(bnorm, fpr_bnorm_max)) {
			return 3;
		}
	}
	if (!Zf(compute_public)(h, f, g, PQV_LOGN, tmp)) {
		return 4;
	}
	lim = (1 << (Zf(max_FG_bits)[PQV_LOGN] - 1)) - 1;
	if (!solve_NTRU(PQV_LOGN, F, G, f, g, lim, (uint32_t *)tmp)) {
		return 5;
	}
	(void)G;
	return 0;
}

/*
 * The n-th (0-based) Gaussian attempt for a seed, with the SHAKE stream
 * CONTINUING across attempts exactly as in the reference keygen loop. n=0
 * equals pqv_sample_fg. Oracle for the GPU's in-kernel retry loop.
 */
int pqv_sample_fg_nth(const void *seed, size_t seed_len, uint32_t n,
	int8_t *f, int8_t *g)
{
	shake256_context sc;
	shake256_init_prng_from_seed(&sc, seed, seed_len);
	inner_shake256_context *rng = (inner_shake256_context *)&sc;
	uint32_t i;

	for (i = 0; i <= n; i++) {
		poly_small_mkgauss(rng, f, PQV_LOGN);
		poly_small_mkgauss(rng, g, PQV_LOGN);
	}
	return 1;
}

/*
 * Oracle for the GPU's in-kernel retry loop: the first attempt (0-based, with
 * the SHAKE stream CONTINUING across attempts, as in the reference keygen
 * loop) whose sample passes the DEVICE-visible predicate — coefficient limit,
 * sqnorm, bnorm (FPEMU-exact), f invertible mod q. solve_NTRU is NOT consulted
 * (the device cannot predict it; the host re-verifies hits with full keygen).
 * Returns the attempt index, or -1 if none within max_attempts.
 */
int pqv_first_visible_accept(const void *seed, size_t seed_len, uint32_t max_attempts)
{
	shake256_context sc;
	inner_shake256_context *rng;
	int8_t f[PQV_N], g[PQV_N];
	uint16_t h[PQV_N];
	uint64_t tmp64[(FALCON_TMPSIZE_KEYGEN(PQV_LOGN) + 7) / 8];
	uint8_t *tmp = (uint8_t *)tmp64;
	uint32_t att;

	shake256_init_prng_from_seed(&sc, seed, seed_len);
	rng = (inner_shake256_context *)&sc;

	for (att = 0; att < max_attempts; att++) {
		size_t u;
		int lim, bad;

		poly_small_mkgauss(rng, f, PQV_LOGN);
		poly_small_mkgauss(rng, g, PQV_LOGN);

		lim = 1 << (Zf(max_fg_bits)[PQV_LOGN] - 1);
		bad = 0;
		for (u = 0; u < PQV_N; u++) {
			if (f[u] >= lim || f[u] <= -lim || g[u] >= lim || g[u] <= -lim) {
				bad = 1;
				break;
			}
		}
		if (bad) {
			continue;
		}
		{
			uint32_t normf = poly_small_sqnorm(f, PQV_LOGN);
			uint32_t normg = poly_small_sqnorm(g, PQV_LOGN);
			uint32_t norm = (normf + normg) | -((normf | normg) >> 31);
			if (norm >= 16823) {
				continue;
			}
		}
		{
			fpr *rt1 = (fpr *)tmp;
			fpr *rt2 = rt1 + PQV_N;
			fpr *rt3 = rt2 + PQV_N;
			fpr bnorm;
			poly_small_to_fp(rt1, f, PQV_LOGN);
			poly_small_to_fp(rt2, g, PQV_LOGN);
			Zf(FFT)(rt1, PQV_LOGN);
			Zf(FFT)(rt2, PQV_LOGN);
			Zf(poly_invnorm2_fft)(rt3, rt1, rt2, PQV_LOGN);
			Zf(poly_adj_fft)(rt1, PQV_LOGN);
			Zf(poly_adj_fft)(rt2, PQV_LOGN);
			Zf(poly_mulconst)(rt1, fpr_q, PQV_LOGN);
			Zf(poly_mulconst)(rt2, fpr_q, PQV_LOGN);
			Zf(poly_mul_autoadj_fft)(rt1, rt3, PQV_LOGN);
			Zf(poly_mul_autoadj_fft)(rt2, rt3, PQV_LOGN);
			Zf(iFFT)(rt1, PQV_LOGN);
			Zf(iFFT)(rt2, PQV_LOGN);
			bnorm = fpr_zero;
			for (u = 0; u < PQV_N; u++) {
				bnorm = fpr_add(bnorm, fpr_sqr(rt1[u]));
				bnorm = fpr_add(bnorm, fpr_sqr(rt2[u]));
			}
			if (!fpr_lt(bnorm, fpr_bnorm_max)) {
				continue;
			}
		}
		if (!Zf(compute_public)(h, f, g, PQV_LOGN, tmp)) {
			continue;
		}
		return (int)att;
	}
	return -1;
}

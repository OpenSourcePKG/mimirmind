// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Chunked GatedDeltaNet prefill — stage K1: per-chunk ungated triangular
// inverse A0. Direct port of compute::deltanetKktSolveInverse.
//
//   lt[a,m] = beta_a (k_a . k_m)   for m < a   (strict-lower gated-free Gram)
//   L = I + strictLower(lt);  A0 = L^-1 is unit lower-triangular:
//     A0[a,a] = 1;  A0[a,m>a] = 0;
//     A0[a,m] = -sum_{p=m..a-1} lt[a,p] A0[p,m]   (a > m)
//
//   k [T,H,S]; beta [T,H]; a0 [nChunks,H,C,C] row-major.
//
// One block per (chunk, head): grid = nChunks*H (bid = c*H + h). Block = C
// threads. lt is built into shared (thread = row a, dot over S), then the
// unit-lower inverse is solved with thread = column m: each column is an
// independent serial forward-substitution over rows a (A0[a,m] needs
// A0[p<a,m], same thread → no cross-thread hazard). Correctness-first; the
// per-row Gram dot and the serial substitution are the O(C^2) work.
//
// Assumes C <= MAX_C. Block dim = C (<= MAX_C = __launch_bounds__).

#include <cuda_runtime.h>

#ifndef DELTANET_KKT_MAX_C
#define DELTANET_KKT_MAX_C 64
#endif

extern "C" __global__ __launch_bounds__(DELTANET_KKT_MAX_C)
void deltanet_kkt_solve(
    const float* __restrict__ k,
    const float* __restrict__ beta,
    float*       __restrict__ a0,
    const int T, const int H, const int S, const int C)
{
    const int nChunks = (T + C - 1) / C;
    const int bid = blockIdx.x;                 // c*H + h
    if (bid >= nChunks * H) return;
    const int c = bid / H;
    const int h = bid % H;
    const int c0 = c * C;
    int cs = C;
    if (c0 + cs > T) cs = T - c0;

    const int t = threadIdx.x;                  // row a (phase 1) / col m (phase 2)
    float* a0c = a0 + (static_cast<size_t>(c) * H + h) * C * C;

    __shared__ float lt[DELTANET_KKT_MAX_C * DELTANET_KKT_MAX_C];

    // Zero this chunk's a0 block and lt.
    for (int idx = t; idx < C * C; idx += blockDim.x) {
        a0c[idx] = 0.0f;
        lt[idx]  = 0.0f;
    }
    __syncthreads();

    // Phase 1: strict-lower Gram, thread = row a. Non-FMA (__fmul_rn/__fadd_rn)
    // so the accumulation rounds bit-for-bit like the CPU reference — the
    // triangular inverse below amplifies tiny rounding differences into the
    // large A0 entries, and a fused-vs-separate mul-add there breaks the
    // absolute 2e-3 parity tolerance at S=128.
    if (t < cs) {
        const float* ka = k + (static_cast<size_t>(c0 + t) * H + h) * S;
        const float  ba = beta[(c0 + t) * H + h];
        for (int m = 0; m < t; ++m) {
            const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
            float kk = 0.0f;
            for (int i = 0; i < S; ++i) kk = __fadd_rn(kk, __fmul_rn(ka[i], km[i]));
            lt[t * C + m] = __fmul_rn(ba, kk);
        }
    }
    __syncthreads();

    // Phase 2: unit-lower inverse, thread = column m (independent per column).
    if (t < cs) {
        a0c[t * C + t] = 1.0f;
        for (int a = t + 1; a < cs; ++a) {
            float acc = 0.0f;
            for (int p = t; p < a; ++p)
                acc = __fadd_rn(acc, __fmul_rn(lt[a * C + p], a0c[p * C + t]));
            a0c[a * C + t] = -acc;
        }
    }
}

// 5.21.9 — batched/ragged K1 for the serving prefill path. grid =
// dim3(maxChunks*H, nSeq) with maxChunks = ceil(T/C), T = maxSeqT. a0 uses a
// COMPACT chunk layout [totalChunks, H, C, C]: sequence seq's chunk c lives
// at block index chunkBase(seq) + c, where chunkBase = sum over s<seq of
// ceil(seqT[s]/C) (computed in-kernel from seqT — an O(nSeq) scalar loop,
// nSeq <= 64). This keeps the a0 allocation proportional to the ACTUAL chunk
// count (sum(ceil(seqT/C))) instead of nSeq*maxChunks, which explodes on
// mixed decode+prefill forwards where most slots carry one token.
// seqT/seqOff/activeMask all nullptr => uniform T at k/beta stride seq*T and
// chunkBase = seq*maxChunks, math per (seq, chunk, head) bit-identical to
// nSeq single deltanet_kkt_solve calls. Blocks for chunks beyond a
// sequence's own ceil(Tseq/C) return (their a0 blocks are never read by
// K2); frozen slots (activeMask 0) return untouched.
extern "C" __global__ __launch_bounds__(256)
void deltanet_kkt_solve_batched(
    const float* __restrict__ kIn,
    const float* __restrict__ betaIn,
    float*       __restrict__ a0,
    const int T, const int H, const int S, const int C,
    const unsigned char* __restrict__ activeMask,
    const int* __restrict__ seqT,
    const int* __restrict__ seqOff)
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;
    const int maxChunks = (T + C - 1) / C;
    const int bid = blockIdx.x;                 // c*H + h (grid-side numbering)
    if (bid >= maxChunks * H) return;
    const int c = bid / H;
    const int h = bid % H;

    const int Tseq = (seqT != nullptr) ? seqT[seq] : T;
    const int nChunks = (Tseq + C - 1) / C;
    if (c >= nChunks) return;
    const int c0 = c * C;
    int cs = C;
    if (c0 + cs > Tseq) cs = Tseq - c0;

    // v4 fast path: a 1-token chunk's inverse is the 1x1 identity, and K2
    // only ever reads a0c[0] for it — skip the C*C zeroing (the DECODE slots
    // of a mixed serving forward all hit this).
    if (cs == 1) {
        if (threadIdx.x == 0) {
            size_t cb;
            if (seqT != nullptr) {
                cb = 0;
                for (int s = 0; s < seq; ++s) cb += (size_t)(seqT[s] + C - 1) / C;
            } else {
                cb = (size_t)seq * maxChunks;
            }
            a0[(cb + c) * H * C * C + (size_t)h * C * C] = 1.0f;
        }
        return;
    }

    const size_t tokBase = (seqOff != nullptr) ? (size_t)seqOff[seq]
                                               : (size_t)seq * (size_t)T;
    const float* __restrict__ k    = kIn    + tokBase * (size_t)H * S;
    const float* __restrict__ beta = betaIn + tokBase * H;

    size_t chunkBase;
    if (seqT != nullptr) {
        chunkBase = 0;
        for (int s = 0; s < seq; ++s) chunkBase += (size_t)(seqT[s] + C - 1) / C;
    } else {
        chunkBase = (size_t)seq * maxChunks;
    }

    const int t = threadIdx.x;
    float* a0c = a0 + (chunkBase + c) * H * C * C + (size_t)h * C * C;

    __shared__ float lt[DELTANET_KKT_MAX_C * DELTANET_KKT_MAX_C];

    for (int idx = t; idx < C * C; idx += blockDim.x) {
        a0c[idx] = 0.0f;
        lt[idx]  = 0.0f;
    }
    __syncthreads();

    // Phase 1: strict-lower Gram — v4 (5.21.9): the (a, m) entries are
    // independent, so distribute the PAIRS across the whole block instead of
    // one serial row per thread (the old form left thread a computing a dots
    // serially — the 2026-09-05 sub-split profile showed K1 at 9% of the
    // whole prefill, 2x the entire TC chunk forward). Each dot keeps its
    // serial ascending-i non-FMA order, so every entry is BIT-IDENTICAL to
    // the previous form — only the thread assignment changes.
    for (int p = t; p < C * C; p += blockDim.x) {
        const int a = p / C;
        const int m = p % C;
        if (a >= cs || m >= a) continue;
        const float* ka = k + (static_cast<size_t>(c0 + a) * H + h) * S;
        const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
        const float  ba = beta[(c0 + a) * H + h];
        float kk = 0.0f;
        for (int i = 0; i < S; ++i) kk = __fadd_rn(kk, __fmul_rn(ka[i], km[i]));
        lt[a * C + m] = __fmul_rn(ba, kk);
    }
    __syncthreads();

    // Phase 2: unit-lower inverse, thread = column m (unchanged).
    if (t < cs) {
        a0c[t * C + t] = 1.0f;
        for (int a = t + 1; a < cs; ++a) {
            float acc = 0.0f;
            for (int p = t; p < a; ++p)
                acc = __fadd_rn(acc, __fmul_rn(lt[a * C + p], a0c[p * C + t]));
            a0c[a * C + t] = -acc;
        }
    }
}

// ---------------------------------------------------------------------------
// 5.21.9 v5 — TENSOR-CORE K1 for the TC chunk pipeline (S=128, C=64 only;
// dispatched together with deltanet_chunk_forward_batched_tc via
// MIMIRMIND_GDN_CHUNK_TC — the scalar pipeline keeps the exact kernel above).
//
// The 2026-09-05 sub-split profile showed K1 at 9% of the WHOLE serving
// prefill — 2x the entire TC chunk forward — bound by the serial 64-step
// phase-2 substitution and uncoalesced Gram reads. This variant:
//   * Gram lt = tril(diag(b) K K^T, -1) as TF32 wmma GEMMs (m16n16k8) over
//     an smem-staged K chunk (TF32's 10-bit mantissa chosen over BF16
//     because the triangular inverse AMPLIFIES operand rounding).
//   * BLOCKED unit-lower inverse (4x4 blocks of 16): the four diagonal
//     16x16 inverses run in PARALLEL (16-step chains instead of one
//     64-step chain), off-diagonal blocks come from block substitution
//     X_ij = -X_ii (sum_k L_ik X_kj) as TF32 mmas.
// Result is tolerance-equal (TF32 rounding + reassociation), consumed by a
// pipeline that rounds a0 to BF16 anyway (the mB mirror in K2-TC).
//
// Launch: grid = (maxChunks*H, nSeq), block = 256, dyn smem = C*S*4 (kSh).
// ---------------------------------------------------------------------------

#include <mma.h>
using namespace nvcuda;

#define KKT_TC_S 128
#define KKT_TC_C 64
#define KKT_TC_B 16          // block edge

extern "C" __global__ __launch_bounds__(256)
void deltanet_kkt_solve_batched_tc(
    const float* __restrict__ kIn,
    const float* __restrict__ betaIn,
    float*       __restrict__ a0,
    const int T, const int H, const int S, const int C,
    const unsigned char* __restrict__ activeMask,
    const int* __restrict__ seqT,
    const int* __restrict__ seqOff)
{
    constexpr int BS = KKT_TC_B;
    if (S != KKT_TC_S || C != KKT_TC_C) return;   // host guards

    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;
    const int maxChunks = (T + C - 1) / C;
    const int bid = blockIdx.x;
    if (bid >= maxChunks * H) return;
    const int c = bid / H;
    const int h = bid % H;

    const int Tseq = (seqT != nullptr) ? seqT[seq] : T;
    const int nChunks = (Tseq + C - 1) / C;
    if (c >= nChunks) return;
    const int c0 = c * C;
    int cs = C;
    if (c0 + cs > Tseq) cs = Tseq - c0;

    const size_t tokBase = (seqOff != nullptr) ? (size_t)seqOff[seq]
                                               : (size_t)seq * (size_t)T;
    const float* __restrict__ k    = kIn    + tokBase * (size_t)H * S;
    const float* __restrict__ beta = betaIn + tokBase * H;

    size_t chunkBase;
    if (seqT != nullptr) {
        chunkBase = 0;
        for (int s = 0; s < seq; ++s) chunkBase += (size_t)(seqT[s] + C - 1) / C;
    } else {
        chunkBase = (size_t)seq * maxChunks;
    }
    float* a0c = a0 + (chunkBase + c) * H * C * C + (size_t)h * C * C;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;

    if (cs == 1) {
        if (tid == 0) a0c[0] = 1.0f;
        return;
    }

    extern __shared__ float kSh[];                 // [C, S] fp32 K chunk
    __shared__ float lt[KKT_TC_C * KKT_TC_C];      // Gram (then L)
    __shared__ float X [KKT_TC_C * KKT_TC_C];      // the inverse
    __shared__ float sStg[8][KKT_TC_B * KKT_TC_B]; // per-warp staging tile

    // Phase 0: stage the K chunk (zero-pad beyond cs) + zero X.
    for (int i = tid; i < C * S; i += 256) {
        const int m = i / S;
        kSh[i] = (m < cs) ? k[((size_t)(c0 + m) * H + h) * S + i % S] : 0.0f;
    }
    for (int i = tid; i < C * C; i += 256) {
        X[i] = 0.0f;
    }
    __syncthreads();

    // Phase A: Gram tiles (block-lower incl. diagonal) as TF32 mma.
    // 10 lower tiles (i >= j) over 8 warps.
    {
        constexpr int kTiles = KKT_TC_S / 8;       // 16 k-steps of 8
        for (int t = warp; t < 10; t += 8) {
            // Enumerate lower tiles: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),...
            int ti = 0, tj = 0, cnt = 0;
            for (int i = 0; i < 4 && cnt <= t; ++i) {
                for (int j = 0; j <= i && cnt <= t; ++j) {
                    ti = i; tj = j; ++cnt;
                }
            }
            wmma::fragment<wmma::accumulator, BS, BS, 8, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < kTiles; ++kk) {
                wmma::fragment<wmma::matrix_a, BS, BS, 8,
                               wmma::precision::tf32, wmma::row_major> aF;
                wmma::fragment<wmma::matrix_b, BS, BS, 8,
                               wmma::precision::tf32, wmma::col_major> bF;
                wmma::load_matrix_sync(
                    aF, kSh + (size_t)(ti * BS) * S + kk * 8, S);
                wmma::load_matrix_sync(
                    bF, kSh + (size_t)(tj * BS) * S + kk * 8, S);
#pragma unroll
                for (int e = 0; e < aF.num_elements; ++e)
                    aF.x[e] = wmma::__float_to_tf32(aF.x[e]);
#pragma unroll
                for (int e = 0; e < bF.num_elements; ++e)
                    bF.x[e] = wmma::__float_to_tf32(bF.x[e]);
                wmma::mma_sync(acc, aF, bF, acc);
            }
            wmma::store_matrix_sync(lt + (size_t)(ti * BS) * C + tj * BS,
                                    acc, C, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Phase A2: apply beta + strict-lower mask (upper tiles were never
    // written — mask covers them via the m<a condition only inside lower
    // tiles; explicitly zero everything not strict-lower-in-range).
    for (int i = tid; i < C * C; i += 256) {
        const int a = i / C, m = i % C;
        if (a < cs && m < a) {
            lt[i] *= beta[(c0 + a) * H + h];
        } else {
            lt[i] = 0.0f;
        }
    }
    __syncthreads();

    // Phase B: the four diagonal 16x16 unit-lower inverses IN PARALLEL
    // (warps 0..3, lane = column within the block).
    if (warp < 4 && lane < BS) {
        const int b0r = warp * BS;                 // block row/col base
        const int m = lane;
        X[(size_t)(b0r + m) * C + b0r + m] = 1.0f;
        for (int a = m + 1; a < BS; ++a) {
            float acc = 0.0f;
            for (int p = m; p < a; ++p) {
                acc += lt[(size_t)(b0r + a) * C + b0r + p]
                     * X[(size_t)(b0r + p) * C + b0r + m];
            }
            X[(size_t)(b0r + a) * C + b0r + m] = -acc;
        }
    }
    __syncthreads();

    // Phase C: block substitution, one warp per block column j (j = 0..2);
    // within a column the i-chain is serial (at most 3 steps).
    if (warp < 3) {
        const int j = warp;
        for (int i = j + 1; i < 4; ++i) {
            // S16 = sum_{k=j..i-1} L[i][k] @ X[k][j]  (TF32 mma, fp32 acc)
            wmma::fragment<wmma::accumulator, BS, BS, 8, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kb = j; kb < i; ++kb) {
                for (int half = 0; half < 2; ++half) {
                    wmma::fragment<wmma::matrix_a, BS, BS, 8,
                                   wmma::precision::tf32, wmma::row_major> aF;
                    wmma::fragment<wmma::matrix_b, BS, BS, 8,
                                   wmma::precision::tf32, wmma::row_major> bF;
                    wmma::load_matrix_sync(
                        aF, lt + (size_t)(i * BS) * C + kb * BS + half * 8, C);
                    wmma::load_matrix_sync(
                        bF, X + (size_t)(kb * BS + half * 8) * C + j * BS, C);
#pragma unroll
                    for (int e = 0; e < aF.num_elements; ++e)
                        aF.x[e] = wmma::__float_to_tf32(aF.x[e]);
#pragma unroll
                    for (int e = 0; e < bF.num_elements; ++e)
                        bF.x[e] = wmma::__float_to_tf32(bF.x[e]);
                    wmma::mma_sync(acc, aF, bF, acc);
                }
            }
            // Stage -S16, then X[i][j] = X[i][i] @ (-S16).
            wmma::store_matrix_sync(sStg[warp], acc, BS, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < BS * BS; e += 32) {
                sStg[warp][e] = -sStg[warp][e];
            }
            __syncwarp();
            wmma::fragment<wmma::accumulator, BS, BS, 8, float> acc2;
            wmma::fill_fragment(acc2, 0.0f);
            for (int half = 0; half < 2; ++half) {
                wmma::fragment<wmma::matrix_a, BS, BS, 8,
                               wmma::precision::tf32, wmma::row_major> aF;
                wmma::fragment<wmma::matrix_b, BS, BS, 8,
                               wmma::precision::tf32, wmma::row_major> bF;
                wmma::load_matrix_sync(
                    aF, X + (size_t)(i * BS) * C + i * BS + half * 8, C);
                wmma::load_matrix_sync(
                    bF, sStg[warp] + (size_t)half * 8 * BS, BS);
#pragma unroll
                for (int e = 0; e < aF.num_elements; ++e)
                    aF.x[e] = wmma::__float_to_tf32(aF.x[e]);
#pragma unroll
                for (int e = 0; e < bF.num_elements; ++e)
                    bF.x[e] = wmma::__float_to_tf32(bF.x[e]);
                wmma::mma_sync(acc2, aF, bF, acc2);
            }
            wmma::store_matrix_sync(X + (size_t)(i * BS) * C + j * BS,
                                    acc2, C, wmma::mem_row_major);
            __syncwarp();
        }
    }
    __syncthreads();

    // Phase D: publish. Rows/cols beyond cs must behave like the exact
    // kernel (identity diagonal, zero elsewhere) — with the zero-padded K
    // the Gram rows beyond cs are 0, so X already carries exactly that.
    for (int i = tid; i < C * C; i += 256) {
        a0c[i] = X[i];
    }
}
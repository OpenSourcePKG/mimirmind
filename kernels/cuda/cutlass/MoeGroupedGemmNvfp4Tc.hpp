// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MoeGroup Sub-Step E-d.3 — CUTLASS block-scaled NVFP4 *grouped* GEMM
// on Blackwell sm_120/sm_121 (GB10), the FP4-tensor-core path that makes the
// grouped MoE GEMM compute-dense enough to beat the batched fused-K kernel.
//
// One expert == one CUTLASS group. Per group: A = [M_e, K] NVFP4 (E2M1 nibbles,
// row-major, produced by moe_act_quant_nvfp4), B = [N, K] NVFP4 weight (E2M1,
// column-major), both with swizzled UE4M3 block scales (SFA from the act-quant
// kernel, SFB from swizzleMoeBlockScales). Output D = [M_e, N] BF16, with a
// per-group scalar alpha = 1/(act_gscale * weight_gscale) folding both global
// scales back in. Mirrors vLLM's run_fp4_blockwise_scaled_group_mm_sm120
// (CUTLASS v4.4.2), adapted to raw device pointers and mimirmind's idiom.
//
// This header is deliberately CUTLASS-free (plain C++/CUDA types) so it can be
// included from the rest of the engine without dragging CUTLASS headers — the
// heavy instantiation lives in MoeGroupedGemmNvfp4Tc.cu, compiled as real
// sm_121a SASS in its own static library.

#pragma once

#include <cstdint>

// Forward-declared so callers need no CUDA runtime include beyond the stream.
struct CUstream_st;

namespace mimirmind::kernels::cutlassmoe {

/// Whether this build linked the CUTLASS grouped-NVFP4-TC kernel at all
/// (third_party/cutlass present at configure time). When false, callers must
/// fall back to the scalar grouped / batched path.
[[nodiscard]] bool nvfp4TcAvailable() noexcept;

/**
 * Host-known-shapes grouped NVFP4 tensor-core GEMM (one expert per group),
 * BF16 output. The per-group problem sizes are supplied on the host (mHost[e],
 * with a shared N and K), so this entry is used by the parity test and the
 * host-driven runtime path; the fully device-driven builder (no D2H) is a
 * later sub-step.
 *
 * All `d*` arrays are device arrays of length `groups`:
 *   dPtrA[e]     -> M_e*K NVFP4 nibbles (E2M1), row-major   (const uint8)
 *   dPtrSFA[e]   -> swizzled UE4M3 block scales for A        (const uint8)
 *   dPtrB[e]     -> N*K NVFP4 nibbles (E2M1), column-major   (const uint8)
 *   dPtrSFB[e]   -> swizzled UE4M3 block scales for B        (const uint8)
 *   dPtrAlpha[e] -> one F32 alpha for the group              (const float*)
 *   dPtrD[e]     -> M_e*N BF16 output                        (uint16 / bf16)
 *
 * `mHost`/`nHost`/`kHost` are host arrays of length `groups` (N,K equal across
 * experts in practice, but kept per-group for generality). K and N must be
 * multiples of 16 (NVFP4 group) and meet the 32-element FP4 alignment.
 *
 * Returns 0 on success, non-zero on a CUTLASS error (can_implement / run).
 * Blocks: runs on `stream` and does not synchronize; the caller syncs.
 */
[[nodiscard]] int runGroupedNvfp4TcBf16(
    int                  groups,
    const int*           mHost,
    const int*           nHost,
    const int*           kHost,
    const void* const*   dPtrA,
    const void* const*   dPtrSFA,
    const void* const*   dPtrB,
    const void* const*   dPtrSFB,
    const float* const*  dPtrAlpha,
    void* const*         dPtrD,
    CUstream_st*         stream);

/**
 * Fully device-driven variant (E-d.3b) — the runtime path. The per-group
 * problem sizes come from a device expOffset (M_e = expOffset[e+1] -
 * expOffset[e]); the per-group problem_sizes, strides, and SF layouts are BUILT
 * ON DEVICE by a small builder kernel, so nothing crosses to the host (no M
 * D2H, the Option-1 host-sync killer that made the scalar grouped path lose to
 * batched on GB10). N and K are shared across all groups.
 *
 * The operand pointer arrays are device arrays of device pointers, exactly as
 * in runGroupedNvfp4TcBf16 (in the runtime they are produced by the gather +
 * grouped act-quant kernels; SFA is one tensor per expert). `dExpOffset` has
 * `groups+1` entries. Runs on `stream`. Returns 0 on success.
 */
[[nodiscard]] int runGroupedNvfp4TcBf16DeviceDriven(
    int                  groups,
    int                  N,
    int                  K,
    const int*           dExpOffset,
    const void* const*   dPtrA,
    const void* const*   dPtrSFA,
    const void* const*   dPtrB,
    const void* const*   dPtrSFB,
    const float* const*  dPtrAlpha,
    void* const*         dPtrD,
    CUstream_st*         stream);

/**
 * Banks variant (E-d.4) — the runtime entry the MoE backend calls. All operands
 * are contiguous device banks; the per-group pointers are computed ON DEVICE
 * from two device offset arrays, so nothing crosses to the host:
 *   - `dExpOffset[groups+1]`  real cumulative token counts -> problem M_e.
 *   - `dPadOffset[groups+1]`  padded cumulative row starts (each expert padded
 *                             to a multiple of 128 rows so its SFA sub-tensor is
 *                             tile-aligned). Row offset for A / SFA / D.
 * The activation banks come from ONE act-quant over the padded gathered buffer
 * (`aBank` [totalPad, K/2] nibbles, `sfaBank` its swizzled SFA for M=totalPad).
 * The weight banks are the E-d.2b load-time side banks (`bBank` [groups,N,K/2]
 * nibbles, `sfbBank` swizzled SFB, `globalsBank` [groups] F32 -> per-group
 * alpha). `dBank` is [totalPad, N] bf16 output. N,K shared across groups.
 * Runs on `stream`. Returns 0 on success.
 */
[[nodiscard]] int runGroupedNvfp4TcBf16Banks(
    int                  groups,
    int                  N,
    int                  K,
    const int*           dExpOffset,
    const int*           dPadOffset,
    const void*          aBank,
    const void*          sfaBank,
    const void*          bBank,
    const void*          sfbBank,
    const float*         globalsBank,
    void*                dBank,
    CUstream_st*         stream);

} // namespace mimirmind::kernels::cutlassmoe

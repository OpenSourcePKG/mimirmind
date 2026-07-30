// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE gate+up projection (blocked-NVFP4) — NVFP4 analogue of
// moe_gate_up_fused_k_q4k_batched. Keeps the routed experts native 4-bit
// (E2M1) and LOSSLESS (repackaged from the checkpoint's NVFP4, not re-quantised
// like the Q4_K path). Same warp layout (one warp per (k,f) output over
// kActive*nFf, 32 lanes span a 32-element super), silu(gate)*up.
//
// Weight bank row: nSuper = dModel/32 super-blocks of 20 bytes each
//   fp16 s0 | fp16 s1 | 16 packed E2M1 bytes  (2 NVFP4 blocks = 32 elems)
//   value[e in super] = (e<16 ? s0 : s1) * e2m1(nibble_e)
//
// Launch: grid = dim3(ceil(kActive*nFf / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_GU_LOCAL.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MOE_GU_LOCAL
#define MOE_GU_LOCAL 128
#endif

#define MOE_GU_WARPS             (MOE_GU_LOCAL / 32)
#define MOE_GU_OUTPUTS_PER_GROUP MOE_GU_WARPS

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define X_TILE_ELEMENTS      1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// E2M1 magnitude decode without a runtime-indexed local LUT. A `float mag[8]`
// indexed by a per-lane (divergent) value cannot be kept in registers, so it
// spills to local memory — a latency load on every dequant. Instead build the
// fp32 bit pattern arithmetically. Bit-exact to the table
// {0,0.5,1,1.5,2,3,4,6}: mm = [E1 E0 M]; the normal path (E>0) is
// 2^(E-1)*(1+0.5*M) => exponent (E-1)+127 = (mm>>1)+126, mantissa top bit = M.
// The two subnormals (mm<2) are 0.0 and 0.5.
__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const unsigned mm   = nib & 0x7u;
    const unsigned bits = (((mm >> 1) + 126u) << 23) | ((mm & 1u) << 22);
    float v = __uint_as_float(bits);
    v = (mm < 2u) ? (0.5f * static_cast<float>(mm)) : v;
    return (nib & 0x8u) ? -v : v;
}

// Contribution of element `l` (lane) of one 32-element super to the dot product.
__device__ __forceinline__ float nvblkSuperDot(
    const unsigned char* super, const float* xBase, int l) {
    // Each lane owns exactly one element of the super, so it needs only its own
    // scale (s0 for lanes 0-15, s1 for 16-31). Read and convert just that one
    // half instead of converting both per lane (32 lanes previously did 64
    // half->float conversions per super for 2 distinct values).
    const __half hs =
        *reinterpret_cast<const __half*>(super + ((l < 16) ? 0 : 2));
    const float scale = __half2float(hs);
    const unsigned char byte = super[4 + (l >> 1)];
    const unsigned nib = (l & 1) ? (byte >> 4) : (byte & 0x0F);
    return xBase[l] * (scale * dq_e2m1(nib));
}

} // namespace

extern "C" __global__ __launch_bounds__(MOE_GU_LOCAL)
void moe_gate_up_fused_k_nvfp4blk_batched(
    const float*         __restrict__ X,           // [nSeq, dModel]
    const unsigned char* __restrict__ Wg,          // gate expert bank (shared)
    const unsigned char* __restrict__ Wu,          // up expert bank (shared)
    const int*           __restrict__ expIdx,      // [nSeq, kActive]
          float*         __restrict__ gateActOut,  // [nSeq, kActive, nFf]
    const int                         dModel,
    const int                         nFf,
    const int                         kActive,
    const int                         expertBytesGate,
    const int                         expertBytesUp)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* Xseq          = X          + static_cast<size_t>(seq) * dModel;
    const int*   expIdxSeq     = expIdx     + static_cast<size_t>(seq) * kActive;
    float*       gateActOutSeq = gateActOut
        + static_cast<size_t>(seq) * kActive * nFf;

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int o       = wg * MOE_GU_OUTPUTS_PER_GROUP + warpId;
    const bool active = (o < kActive * nFf);
    const int k       = active ? (o / nFf) : 0;
    const int f       = active ? (o % nFf) : 0;
    const int nSuper  = dModel / NVBLK_SUPER_ELEMENTS;
    const int rowBytes = nSuper * NVBLK_SUPER_BYTES;

    const unsigned char* gateRow = nullptr;
    const unsigned char* upRow   = nullptr;
    if (active) {
        const int e = expIdxSeq[k];
        gateRow = Wg + static_cast<size_t>(e) * expertBytesGate
                     + static_cast<size_t>(f) * rowBytes;
        upRow   = Wu + static_cast<size_t>(e) * expertBytesUp
                     + static_cast<size_t>(f) * rowBytes;
    }

    float gsum = 0.0f;
    float usum = 0.0f;

    for (int tile = 0; tile < dModel; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < dModel - tile)
                            ? X_TILE_ELEMENTS : (dModel - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xseq[tile + i];
        }
        __syncthreads();

        if (active) {
            const int superStart   = tile / NVBLK_SUPER_ELEMENTS;
            const int supersInTile  = X_TILE_ELEMENTS / NVBLK_SUPER_ELEMENTS;
            const int superEnd     = (superStart + supersInTile < nSuper)
                                       ? (superStart + supersInTile) : nSuper;
            for (int sp = superStart; sp < superEnd; ++sp) {
                const float* xBase = xTile + (sp - superStart) * NVBLK_SUPER_ELEMENTS;
                gsum += nvblkSuperDot(gateRow + sp * NVBLK_SUPER_BYTES, xBase, laneId);
                usum += nvblkSuperDot(upRow   + sp * NVBLK_SUPER_BYTES, xBase, laneId);
            }
        }

        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        const float g = gsum;
        const float silu = g / (1.0f + __expf(-g));
        gateActOutSeq[static_cast<size_t>(k) * nFf + f] = silu * usum;
    }
}

// FP32 → Q8_0 KV write kernel. Consumes an fp32 workspace holding
// `T` fresh rows of K or V (one per new token, kvDim elements each,
// post-rmsnorm + post-RoPE), splits every row into 32-element blocks
// and writes them into the Q8_0 KV cache slot starting at row
// `curLenPtr[0]`.
//
// Block layout (34 B, matches ggml block_q8_0):
//   [0..1]   fp16 scale = absmax / 127
//   [2..33]  int8 quants = round(srcBlock[i] / scale) clamped to [-127, 127]
//
// Zero-input block: scale=0 stored and all quants=0 — matches the CPU
// reference at src/compute/quant/Q8_0.cpp:79. Dequant round-trips to
// zero, which is the correct behaviour for an all-zero KV row.
//
// The per-block absmax reduction uses SLM (32 f32 slots per WG,
// 128 B). A subgroup-based reduction would work on Xe-LPG with
// intel_reqd_sub_group_size=16 but would still require a cross-SG
// merge step; SLM keeps this portable across future SG-width tuning.
//
// M-CLR compatibility: `curLenPtr` is a __global int-slot that the
// host writes before every dispatch (immediate mode) or between
// replays (recording mode), analogous to rope_inplace / rmsnorm_qkv.
//
// Launch geometry (host emits):
//   local  = (32, 1, 1)
//   groups = (T, nBlocksPerRow, 1)
//
// Reference CPU implementation: `compute::quant::Q8_0::quantizeRow`
// (byte-exact parity target for `kv_quant_commit_q8_0_parity` in
// tests/gpu_tests.cpp).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef KV_QUANT_COMMIT_LOCAL
#define KV_QUANT_COMMIT_LOCAL 32
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(KV_QUANT_COMMIT_LOCAL, 1, 1)))
__kernel void kv_quant_commit_q8_0(
    __global const float* xSrc,        // [T, kvDim] fp32 workspace
    __global       uchar* kvDst,       // Q8_0 cache slot base
    const int             kvDim,       // must be multiple of 32
    __global const int*   curLenPtr)   // startPos (= cache.length())
{
    const int t   = (int)get_group_id(0);
    const int blk = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    __global const float* srcRow =
        xSrc + (size_t)t * (size_t)kvDim;
    const float xv = srcRow[blk * Q8_0_BLOCK_ELEMENTS + lid];

    // Per-block absmax reduction. Tree in SLM — every thread writes
    // its |x|, then log2(32) = 5 rounds of pairwise fmax.
    __local float scratch[KV_QUANT_COMMIT_LOCAL];
    scratch[lid] = fabs(xv);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = KV_QUANT_COMMIT_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float absMax   = scratch[0];
    const float scale    = (absMax > 0.0f) ? (absMax * (1.0f / 127.0f)) : 0.0f;
    const float invScale = (absMax > 0.0f) ? (127.0f / absMax) : 0.0f;

    // Byte offset of this block inside the cache slot. curLenPtr[0]
    // gives the current logical KV-cache length; the T new rows land
    // at [curLen, curLen+T).
    const int startPos      = curLenPtr[0];
    const int nBlocksPerRow = kvDim / Q8_0_BLOCK_ELEMENTS;
    const size_t rowBase =
        (size_t)(startPos + t) * (size_t)nBlocksPerRow *
        (size_t)Q8_0_BLOCK_BYTES;
    const size_t blkBase = rowBase + (size_t)blk * (size_t)Q8_0_BLOCK_BYTES;
    __global uchar* blkPtr = kvDst + blkBase;

    // Lane 0 writes the fp16 scale into the 2-byte header. vstore_half
    // uses IEEE-754 round-to-nearest-even for the fp32→fp16 conversion,
    // matching the CPU reference's floatToHalf helper.
    if (lid == 0) {
        vstore_half(scale, 0, (__global half*)blkPtr);
    }

    // All 32 lanes write their own int8 quant. OpenCL `round()` is
    // round-half-away-from-zero, matching `std::lround` used by
    // Q8_0::quantizeRow.
    const float qf = round(xv * invScale);
    const int   qi = (int)clamp(qf, -127.0f, 127.0f);
    ((__global char*)blkPtr)[2 + lid] = (char)qi;
}
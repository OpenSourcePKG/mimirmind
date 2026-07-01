// Batched matrix-matrix multiply with Q6_K weights, on-the-fly dequant.
// M-way generalisation of matmul_q6k_vec: processes M_TILE token rows
// per workgroup so a single W dequant amortises across M_TILE mads.
//
//   Y[m, n] = sum_{k=0..K-1} X[m, k] * dequant_q6k(W, n, k)
//
//   X:  [M, K]         F32 row-major
//   W:  [N, K/256]     Q6_K super-blocks (210 B each)
//   Y:  [M, N]         F32 row-major
//
// Launch geometry:
//   local_size_x          = MATMUL_Q6K_LOCAL   (64)
//   sub_group_size        = MATMUL_Q6K_SG      (16)
//   outputs per workgroup = MATMUL_Q6K_LOCAL / MATMUL_Q6K_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//   global_size_y         = ceil(M / MATMUL_Q6K_GEMM_M_TILE)
//
// See ADR "GEMM statt matvec für Prefill" (M5i) for the design context.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q6K_LOCAL
#define MATMUL_Q6K_LOCAL 64
#endif

#ifndef MATMUL_Q6K_SG
#define MATMUL_Q6K_SG 16
#endif

#ifndef MATMUL_Q6K_GEMM_M_TILE
#define MATMUL_Q6K_GEMM_M_TILE 8
#endif

#define MATMUL_Q6K_OUTPUTS_PER_GROUP (MATMUL_Q6K_LOCAL / MATMUL_Q6K_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// 1024 elements = 4 super-blocks per K-tile.
// SLM footprint: MATMUL_Q6K_GEMM_M_TILE * X_TILE_ELEMENTS * 4 B
//              = 8 * 1024 * 4 = 32 KiB per workgroup.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q6K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q6K_SG)))
__kernel void matmul_q6k_gemm(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N,
    const int             M)
{
    __local float xTile[MATMUL_Q6K_GEMM_M_TILE][X_TILE_ELEMENTS];

    const int  wgN     = (int)get_group_id(0);
    const int  wgM     = (int)get_group_id(1);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wgN * MATMUL_Q6K_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q6K_GEMM_M_TILE;
    const bool nActive = (n < N);
    const int  nSuper  = K / Q6K_BLOCK_ELEMENTS;

    // M_TILE-fold accumulator + Kahan compensation, one pair per m slot.
    // volatile keeps the compiler from cancelling out the Kahan
    // correction (see matmul_q6k_vec for the single-m variant).
    float          sum[MATMUL_Q6K_GEMM_M_TILE];
    volatile float kc [MATMUL_Q6K_GEMM_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q6K_GEMM_M_TILE; ++mm) {
        sum[mm] = 0.0f;
        kc [mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        // Cooperative load of all M_TILE rows for this K-tile. Rows
        // past M get zero-filled so their MACs contribute nothing;
        // guards at the write-back step drop the garbage anyway.
        const int loadTotal = MATMUL_Q6K_GEMM_M_TILE * X_TILE_ELEMENTS;
        for (int idx = tid; idx < loadTotal; idx += lsize) {
            const int  mSlot = idx / X_TILE_ELEMENTS;
            const int  iSlot = idx - mSlot * X_TILE_ELEMENTS;
            const int  mAct  = mBase + mSlot;
            const bool valid = (mAct < M) && (iSlot < tileK);
            xTile[mSlot][iSlot] =
                valid ? X[(size_t)mAct * (size_t)K + tile + iSlot] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (nActive) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nSuper * Q6K_BLOCK_BYTES;

            const int sbStart  = tile / Q6K_BLOCK_ELEMENTS;
            const int sbInTile = X_TILE_ELEMENTS / Q6K_BLOCK_ELEMENTS;
            const int sbEnd    = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

                __global const uchar* ql = block;              // 128 B
                __global const uchar* qh = block + 128;        // 64 B
                __global const char*  sc =
                    (__global const char*)(block + 192);       // 16 signed
                const float d =
                    vload_half(0, (__global const half*)(block + 208));

                const int xLocalBase = (sb - sbStart) * Q6K_BLOCK_ELEMENTS;

                // Two 128-element halves per super-block.
                for (int hIdx = 0; hIdx < 2; ++hIdx) {
                    const int xHalfBase = xLocalBase + hIdx * 128;
                    __global const uchar* qlp = ql + hIdx * 64;
                    __global const uchar* qhp = qh + hIdx * 32;
                    __global const char*  scp = sc + hIdx * 8;

                    // 16 subgroup lanes share the 32-element inner span.
                    // Each lane dequantises 4 W values then MACs them
                    // into all M_TILE accumulators.
                    for (int l = sgLocal; l < 32; l += MATMUL_Q6K_SG) {
                        const int is = l / 16;

                        const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                               (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                        const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                               (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                        const char q3 = (char)((qlp[l +  0] >> 4) |
                                               (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                        const char q4 = (char)((qlp[l + 32] >> 4) |
                                               (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                        const float w0 = d * (float)scp[is + 0] * (float)q1;
                        const float w1 = d * (float)scp[is + 2] * (float)q2;
                        const float w2 = d * (float)scp[is + 4] * (float)q3;
                        const float w3 = d * (float)scp[is + 6] * (float)q4;

                        // Kahan-compensated MAC into every m slot. The
                        // dequant work (q1..q4, w0..w3) is done once
                        // and reused M_TILE-fold — that's the amortising
                        // win over the matvec variant.
                        #define KAHAN_ADD(mm, term)                              \
                            do {                                                  \
                                const float _y = (term) - kc[mm];                 \
                                const float _t = sum[mm] + _y;                    \
                                kc[mm]  = (_t - sum[mm]) - _y;                    \
                                sum[mm] = _t;                                     \
                            } while (0)

                        #pragma unroll
                        for (int mm = 0; mm < MATMUL_Q6K_GEMM_M_TILE; ++mm) {
                            KAHAN_ADD(mm, xTile[mm][xHalfBase + l +  0] * w0);
                            KAHAN_ADD(mm, xTile[mm][xHalfBase + l + 32] * w1);
                            KAHAN_ADD(mm, xTile[mm][xHalfBase + l + 64] * w2);
                            KAHAN_ADD(mm, xTile[mm][xHalfBase + l + 96] * w3);
                        }

                        #undef KAHAN_ADD
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fold Kahan residuals into the running sums, then subgroup-reduce
    // and write.
    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q6K_GEMM_M_TILE; ++mm) {
        float s = sum[mm] + kc[mm];
        s = sub_group_reduce_add(s);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[(size_t)mAct * (size_t)N + n] = s;
            }
        }
    }
}
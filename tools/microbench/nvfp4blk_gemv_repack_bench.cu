// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// A/B micro-bench for the dense NVFP4 decode GEMV LOAD PIPELINE — roadmap
// 5.18.3(a) / 5.18.5. Built on the 5.18.6 harness method (CUDA-event timing,
// no ncu). It answers ONE question the 5.18.6 sweep raised: the prod kernel
// `matmul_nvfp4blk_vec` runs at ~60-75% of the 273 GB/s GB10 peak on the wide
// dense projections — is the gap the WEIGHT-LOAD LAYOUT (Marlin's real lever,
// not tensor cores at M=1) or inherent latency?
//
// The prod kernel reads each 20-byte super-block IN PLACE: two fp16 scales
// (offset +0,+2) interleaved with 16 nibble bytes (offset +4), at a
// non-16-aligned, 20-strided offset. This bench A/Bs it against repacked
// variants that split the weight into two 16-byte-aligned streams:
//   - Q stream: [N][nSuper][16] nibble bytes (each super = one 128-bit uint4)
//   - S stream: [N][nSuper]     __half2 scale pair (s0,s1)
// Same logical values, same math, same total bytes moved — only the memory
// layout differs, so higher GB/s = better DRAM sector utilisation.
//
//   v0  = prod matmul_nvfp4blk_vec (20-byte in-place, reference + oracle)
//   v1  = repacked, same per-lane scalar byte load (isolates ALIGNMENT)
//   v2  = repacked, one 128-bit uint4 load/super broadcast via shfl (VECTORISE)
//
// Correctness: v1/v2 outputs are compared against v0 (max abs diff must be ~0).
//
// Build:  cmake --build build-cuda --target microbench_nvfp4blk_repack
// Run:    ./microbench_nvfp4blk_repack            # built-in dense sweep
//         ./microbench_nvfp4blk_repack <K> <N> <iters>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The exact prod kernel (kernels/cuda/common/matmul_nvfp4blk_vec.cu), linked in.
extern "C" __global__ void matmul_nvfp4blk_vec(const float*         X,
                                               const unsigned char* W,
                                               float*               Y,
                                               const int            K,
                                               const int            N);

static constexpr int kBlock           = 128;
static constexpr int kOutputsPerGroup = 4;    // 4 warps -> 4 output rows/group
static constexpr int kSuperElements   = 32;
static constexpr int kSuperBytes      = 20;   // prod in-place layout
static constexpr int kSuperQBytes     = 16;   // repacked nibble stream / super
static constexpr int kXTile           = 1024;

__device__ __forceinline__ float dq_e2m1_b(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

__device__ __forceinline__ float warpReduce(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// v1 — repacked, aligned per-lane scalar byte load. Same access shape as prod
// but the nibble stream is 16-byte aligned and 16-strided (not +4 / 20).
extern "C" __global__ __launch_bounds__(kBlock)
void matmul_nvfp4blk_repack_v1(const float*         __restrict__ X,
                               const unsigned char* __restrict__ Q,   // [N][nSuper][16]
                               const __half2*       __restrict__ S,   // [N][nSuper]
                                     float*         __restrict__ Y,
                               const int                          K,
                               const int                          N) {
    __shared__ float xTile[kXTile];
    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * kOutputsPerGroup + warpId;
    const bool active = (n < N);
    const int nSuper = K / kSuperElements;

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += kXTile) {
        const int tileK = min(kXTile, K - tile);
        for (int i = tid; i < tileK; i += lsize) xTile[i] = X[tile + i];
        __syncthreads();
        if (active) {
            const int superStart   = tile / kSuperElements;
            const int supersInTile = kXTile / kSuperElements;
            const int superEnd     = min(superStart + supersInTile, nSuper);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const size_t sidx = static_cast<size_t>(n) * nSuper + sp;
                const unsigned char* qblk = Q + sidx * kSuperQBytes;
                const __half2 sc = S[sidx];
                const float s0 = __low2float(sc);
                const float s1 = __high2float(sc);
                const unsigned char byte = qblk[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scale = (laneId < 16) ? s0 : s1;
                const int xLocalBase = (sp - superStart) * kSuperElements;
                const float xv = xTile[xLocalBase + laneId];
                sum = __fmaf_rn(xv, scale * dq_e2m1_b(nib), sum);
            }
        }
        __syncthreads();
    }
    sum = warpReduce(sum);
    if (active && laneId == 0) Y[n] = sum;
}

// v2 — repacked, one 128-bit uint4 load per super (lanes 0-3 hold the 4 words),
// broadcast across the warp via shfl, then extract the per-lane nibble byte.
extern "C" __global__ __launch_bounds__(kBlock)
void matmul_nvfp4blk_repack_v2(const float*         __restrict__ X,
                               const unsigned char* __restrict__ Q,
                               const __half2*       __restrict__ S,
                                     float*         __restrict__ Y,
                               const int                          K,
                               const int                          N) {
    __shared__ float xTile[kXTile];
    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * kOutputsPerGroup + warpId;
    const bool active = (n < N);
    const int nSuper = K / kSuperElements;

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += kXTile) {
        const int tileK = min(kXTile, K - tile);
        for (int i = tid; i < tileK; i += lsize) xTile[i] = X[tile + i];
        __syncthreads();
        if (active) {
            const int superStart   = tile / kSuperElements;
            const int supersInTile = kXTile / kSuperElements;
            const int superEnd     = min(superStart + supersInTile, nSuper);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const size_t sidx = static_cast<size_t>(n) * nSuper + sp;
                const uint4* q4 = reinterpret_cast<const uint4*>(Q + sidx * kSuperQBytes);
                // One aligned 128-bit load by lane 0, broadcast the 4 words.
                uint4 v = (laneId == 0) ? __ldg(q4) : make_uint4(0, 0, 0, 0);
                unsigned w0 = __shfl_sync(0xffffffffu, v.x, 0);
                unsigned w1 = __shfl_sync(0xffffffffu, v.y, 0);
                unsigned w2 = __shfl_sync(0xffffffffu, v.z, 0);
                unsigned w3 = __shfl_sync(0xffffffffu, v.w, 0);
                const int bi = laneId >> 1;              // byte 0..15
                unsigned word = (bi < 4) ? w0 : (bi < 8) ? w1 : (bi < 12) ? w2 : w3;
                const unsigned byte = (word >> ((bi & 3) * 8)) & 0xFFu;
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const __half2 sc = S[sidx];
                const float scale = (laneId < 16) ? __low2float(sc) : __high2float(sc);
                const int xLocalBase = (sp - superStart) * kSuperElements;
                const float xv = xTile[xLocalBase + laneId];
                sum = __fmaf_rn(xv, scale * dq_e2m1_b(nib), sum);
            }
        }
        __syncthreads();
    }
    sum = warpReduce(sum);
    if (active && laneId == 0) Y[n] = sum;
}

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Buffers {
    float*         dX = nullptr;
    unsigned char* dW = nullptr;   // 20-byte in-place (v0)
    unsigned char* dQ = nullptr;   // repacked nibble stream
    __half2*       dS = nullptr;   // repacked scale stream
    float*         dY = nullptr;
};

double timeKernel(void (*launch)(const Buffers&, int, int, cudaStream_t),
                  const Buffers& b, int K, int N, int iters) {
    for (int i = 0; i < 10; ++i) launch(b, K, N, 0);
    ck(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "ev0");
    ck(cudaEventCreate(&t1), "ev1");
    ck(cudaEventRecord(t0), "rec0");
    for (int i = 0; i < iters; ++i) launch(b, K, N, 0);
    ck(cudaEventRecord(t1), "rec1");
    ck(cudaEventSynchronize(t1), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return (ms * 1000.0) / iters;   // us/call
}

unsigned int nGroups(int N) {
    return static_cast<unsigned int>((N + kOutputsPerGroup - 1) / kOutputsPerGroup);
}

void launchV0(const Buffers& b, int K, int N, cudaStream_t s) {
    matmul_nvfp4blk_vec<<<nGroups(N), kBlock, 0, s>>>(b.dX, b.dW, b.dY, K, N);
}
void launchV1(const Buffers& b, int K, int N, cudaStream_t s) {
    matmul_nvfp4blk_repack_v1<<<nGroups(N), kBlock, 0, s>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
}
void launchV2(const Buffers& b, int K, int N, cudaStream_t s) {
    matmul_nvfp4blk_repack_v2<<<nGroups(N), kBlock, 0, s>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
}

// Read Y back to host.
std::vector<float> readY(const Buffers& b, int N) {
    std::vector<float> y(N);
    ck(cudaMemcpy(y.data(), b.dY, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H Y");
    return y;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& c) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - c[i]));
    return m;
}

void benchShape(const char* label, int K, int N, int iters) {
    if ((K % kSuperElements) != 0) {
        std::fprintf(stderr, "skip %s: K=%d not mult of %d\n", label, K, kSuperElements);
        return;
    }
    const int nSuper = K / kSuperElements;
    const size_t rowSupers = static_cast<size_t>(nSuper);
    const size_t wBytes = static_cast<size_t>(N) * rowSupers * kSuperBytes;
    const size_t qBytes = static_cast<size_t>(N) * rowSupers * kSuperQBytes;
    const size_t sCount = static_cast<size_t>(N) * rowSupers;   // half2 entries
    const size_t xBytes = static_cast<size_t>(K) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(N) * sizeof(float);

    // Ground-truth host fill in the 20-byte layout, then transcode to repacked.
    std::vector<float>         hX(static_cast<size_t>(K));
    std::vector<unsigned char> hW(wBytes);
    std::vector<unsigned char> hQ(qBytes);
    std::vector<__half2>       hS(sCount);
    for (int i = 0; i < K; ++i) hX[i] = 1.0f / static_cast<float>((i % 97) + 1);
    for (size_t sp = 0; sp < sCount; ++sp) {
        // Deterministic non-trivial scales so a layout bug shows in the diff.
        const __half s0 = __float2half(0.5f + 0.01f * static_cast<float>(sp % 7));
        const __half s1 = __float2half(0.25f + 0.02f * static_cast<float>(sp % 5));
        unsigned char* blk = hW.data() + sp * kSuperBytes;
        *reinterpret_cast<__half*>(blk)     = s0;
        *reinterpret_cast<__half*>(blk + 2) = s1;
        unsigned char* qs = blk + 4;
        unsigned char* qd = hQ.data() + sp * kSuperQBytes;
        for (int b = 0; b < kSuperQBytes; ++b) {
            const unsigned char val =
                static_cast<unsigned char>((sp * 131u + b * 17u + 3u) & 0xFFu);
            qs[b] = val;   // 20-byte layout
            qd[b] = val;   // repacked stream (identical values)
        }
        hS[sp] = __halves2half2(s0, s1);
    }

    Buffers b;
    ck(cudaMalloc(&b.dX, xBytes), "malloc X");
    ck(cudaMalloc(&b.dW, wBytes), "malloc W");
    ck(cudaMalloc(&b.dQ, qBytes), "malloc Q");
    ck(cudaMalloc(&b.dS, sCount * sizeof(__half2)), "malloc S");
    ck(cudaMalloc(&b.dY, yBytes), "malloc Y");
    ck(cudaMemcpy(b.dX, hX.data(), xBytes, cudaMemcpyHostToDevice), "H2D X");
    ck(cudaMemcpy(b.dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "H2D W");
    ck(cudaMemcpy(b.dQ, hQ.data(), qBytes, cudaMemcpyHostToDevice), "H2D Q");
    ck(cudaMemcpy(b.dS, hS.data(), sCount * sizeof(__half2), cudaMemcpyHostToDevice), "H2D S");

    // Correctness: v1/v2 must match v0.
    launchV0(b, K, N, 0); ck(cudaDeviceSynchronize(), "v0"); const auto y0 = readY(b, N);
    launchV1(b, K, N, 0); ck(cudaDeviceSynchronize(), "v1"); const auto y1 = readY(b, N);
    launchV2(b, K, N, 0); ck(cudaDeviceSynchronize(), "v2"); const auto y2 = readY(b, N);
    const float d1 = maxAbsDiff(y0, y1);
    const float d2 = maxAbsDiff(y0, y2);

    const double us0 = timeKernel(launchV0, b, K, N, iters);
    const double us1 = timeKernel(launchV1, b, K, N, iters);
    const double us2 = timeKernel(launchV2, b, K, N, iters);
    const double bytes = static_cast<double>(wBytes + xBytes + yBytes);
    auto gbps = [&](double us) { return bytes / (us * 1e-6) / 1e9; };
    auto pct  = [&](double us) { return gbps(us) / 273.0 * 100.0; };

    std::printf("| %-9s | %6d | %6d "
                "| %7.2f %5.1f%% | %7.2f %5.1f%% %+5.1f%% | %7.2f %5.1f%% %+5.1f%% "
                "| %.1e %.1e |\n",
                label, K, N,
                us0, pct(us0),
                us1, pct(us1), (us0 - us1) / us0 * 100.0,
                us2, pct(us2), (us0 - us2) / us0 * 100.0,
                d1, d2);

    cudaFree(b.dX); cudaFree(b.dW); cudaFree(b.dQ); cudaFree(b.dS); cudaFree(b.dY);
}

} // namespace

int main(int argc, char** argv) {
    ck(cudaSetDevice(0), "setDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "getProps");
    std::printf("# nvfp4blk_gemv REPACK A/B — %s (sm_%d%d), M=1 dense NVFP4 GEMV\n",
                prop.name, prop.major, prop.minor);
    std::printf("# v0=prod(20B in-place)  v1=repacked-aligned  v2=repacked-uint4  "
                "(speedup vs v0; d1/d2 = max|Y-Yv0|)\n");
    std::printf("| shape     |      K |      N "
                "|      v0  %%pk |      v1  %%pk   spd |      v2  %%pk   spd | d1 d2 |\n");

    if (argc >= 3) {
        const int K = std::atoi(argv[1]);
        const int N = std::atoi(argv[2]);
        const int iters = argc >= 4 ? std::atoi(argv[3]) : 200;
        benchShape("custom", K, N, iters);
        return 0;
    }
    const int iters = 200;
    benchShape("o-proj",   2048,   2048, iters);
    benchShape("qkv",      2048,   6144, iters);
    benchShape("ffn-shex", 2048,    512, iters);
    benchShape("ffn-up",   2048,  17408, iters);
    benchShape("lm_head",  2048, 152064, iters);
    return 0;
}

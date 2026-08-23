// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Occupancy / latency-hiding A/B for the dense NVFP4 decode GEMV — roadmap
// 5.18.3(a) brick 2, on the 5.18.6 harness method. Brick 1
// (nvfp4blk_gemv_repack_bench) showed the 16-byte-aligned repacked layout wins
// +3-7% on the WIDE shapes but leaves narrow-N (ffn-shex) at 29% and does not
// reach Marlin's ~90%. The residual is two separate axes; this bench bounds
// both, all on the repacked-aligned layout (the brick-1 winner):
//
//   v0    = prod matmul_nvfp4blk_vec (20-byte in-place) — reference + oracle
//   vRPW  = repacked-aligned + 4 output rows per warp (memory-level parallelism
//           to hide load latency — targets the WIDE-shape ceiling)
//   vSK   = repacked-aligned + split-K=4 with atomic accumulate (fills the grid
//           for narrow N — targets ffn-shex occupancy)
//
// Correctness: vRPW/vSK are compared against v0 (max abs diff must be ~0).
//
// Build:  cmake --build build-cuda --target microbench_nvfp4blk_occ
// Run:    ./microbench_nvfp4blk_occ            # built-in dense sweep

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void matmul_nvfp4blk_vec(const float*, const unsigned char*,
                                               float*, const int, const int);

static constexpr int kBlock         = 128;
static constexpr int kWarps         = kBlock / 32;   // 4
static constexpr int kSuperElements = 32;
static constexpr int kSuperBytes    = 20;
static constexpr int kSuperQBytes   = 16;
static constexpr int kXTile         = 1024;
static constexpr int kRowsPerWarp   = 4;             // vRPW
static constexpr int kSplitK        = 4;             // vSK

__device__ __forceinline__ float dq_e2m1_o(unsigned nib) {
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

// vRPW — each warp computes kRowsPerWarp consecutive output rows, issuing that
// many independent weight loads per super to raise memory-level parallelism.
extern "C" __global__ __launch_bounds__(kBlock)
void matmul_nvfp4blk_rpw(const float*         __restrict__ X,
                         const unsigned char* __restrict__ Q,
                         const __half2*       __restrict__ S,
                               float*         __restrict__ Y,
                         const int K, const int N) {
    __shared__ float xTile[kXTile];
    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nSuper = K / kSuperElements;
    const int rowBase = (blockIdx.x * kWarps + warpId) * kRowsPerWarp;

    float sum[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) sum[r] = 0.0f;

    for (int tile = 0; tile < K; tile += kXTile) {
        const int tileK = min(kXTile, K - tile);
        for (int i = tid; i < tileK; i += blockDim.x) xTile[i] = X[tile + i];
        __syncthreads();
        const int superStart   = tile / kSuperElements;
        const int supersInTile = kXTile / kSuperElements;
        const int superEnd     = min(superStart + supersInTile, nSuper);
        for (int sp = superStart; sp < superEnd; ++sp) {
            const int xLocalBase = (sp - superStart) * kSuperElements;
            const float xv = xTile[xLocalBase + laneId];
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const int n = rowBase + r;
                if (n >= N) continue;
                const size_t sidx = static_cast<size_t>(n) * nSuper + sp;
                const unsigned char* qblk = Q + sidx * kSuperQBytes;
                const __half2 sc = S[sidx];
                const float scale = (laneId < 16) ? __low2float(sc) : __high2float(sc);
                const unsigned char byte = qblk[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                sum[r] = __fmaf_rn(xv, scale * dq_e2m1_o(nib), sum[r]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int n = rowBase + r;
        const float s = warpReduce(sum[r]);
        if (n < N && laneId == 0) Y[n] = s;
    }
}

// vSK — split-K: gridDim.y = kSplitK, each block does a contiguous K-slice for
// its 4 rows and atomically accumulates into Y (Y pre-zeroed by the host).
extern "C" __global__ __launch_bounds__(kBlock)
void matmul_nvfp4blk_splitk(const float*         __restrict__ X,
                            const unsigned char* __restrict__ Q,
                            const __half2*       __restrict__ S,
                                  float*         __restrict__ Y,
                            const int K, const int N) {
    __shared__ float xTile[kXTile];
    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = blockIdx.x * kWarps + warpId;
    const bool active = (n < N);
    const int nSuper = K / kSuperElements;

    // K-slice for this block, aligned to super boundaries.
    const int supersPerSplit = (nSuper + kSplitK - 1) / kSplitK;
    const int spLo = blockIdx.y * supersPerSplit;
    const int spHi = min(spLo + supersPerSplit, nSuper);
    const int kLo  = spLo * kSuperElements;
    const int kHi  = spHi * kSuperElements;

    float sum = 0.0f;
    for (int tile = kLo; tile < kHi; tile += kXTile) {
        const int tileK = min(kXTile, kHi - tile);
        for (int i = tid; i < tileK; i += blockDim.x) xTile[i] = X[tile + i];
        __syncthreads();
        if (active) {
            const int superStart = tile / kSuperElements;
            const int superEnd   = min(superStart + kXTile / kSuperElements, spHi);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const size_t sidx = static_cast<size_t>(n) * nSuper + sp;
                const unsigned char* qblk = Q + sidx * kSuperQBytes;
                const __half2 sc = S[sidx];
                const float scale = (laneId < 16) ? __low2float(sc) : __high2float(sc);
                const unsigned char byte = qblk[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const int xLocalBase = (sp - superStart) * kSuperElements;
                const float xv = xTile[xLocalBase + laneId];
                sum = __fmaf_rn(xv, scale * dq_e2m1_o(nib), sum);
            }
        }
        __syncthreads();
    }
    sum = warpReduce(sum);
    if (active && laneId == 0) atomicAdd(&Y[n], sum);
}

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Buf {
    float* dX = nullptr; unsigned char* dW = nullptr; unsigned char* dQ = nullptr;
    __half2* dS = nullptr; float* dY = nullptr;
};

unsigned int gN(int N) { return static_cast<unsigned int>((N + kWarps - 1) / kWarps); }
unsigned int gRPW(int N) {
    return static_cast<unsigned int>((N + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp));
}

std::vector<float> readY(const Buf& b, int N) {
    std::vector<float> y(N);
    ck(cudaMemcpy(y.data(), b.dY, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H Y");
    return y;
}
float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& c) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - c[i]));
    return m;
}

double timeV0(const Buf& b, int K, int N, int iters) {
    for (int i = 0; i < 10; ++i) matmul_nvfp4blk_vec<<<gN(N), kBlock>>>(b.dX, b.dW, b.dY, K, N);
    ck(cudaDeviceSynchronize(), "wu0");
    cudaEvent_t t0, t1; ck(cudaEventCreate(&t0), "e"); ck(cudaEventCreate(&t1), "e");
    ck(cudaEventRecord(t0), "r");
    for (int i = 0; i < iters; ++i) matmul_nvfp4blk_vec<<<gN(N), kBlock>>>(b.dX, b.dW, b.dY, K, N);
    ck(cudaEventRecord(t1), "r"); ck(cudaEventSynchronize(t1), "s");
    float ms = 0; ck(cudaEventElapsedTime(&ms, t0, t1), "el");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (ms * 1000.0) / iters;
}
double timeRPW(const Buf& b, int K, int N, int iters) {
    for (int i = 0; i < 10; ++i) matmul_nvfp4blk_rpw<<<gRPW(N), kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
    ck(cudaDeviceSynchronize(), "wu");
    cudaEvent_t t0, t1; ck(cudaEventCreate(&t0), "e"); ck(cudaEventCreate(&t1), "e");
    ck(cudaEventRecord(t0), "r");
    for (int i = 0; i < iters; ++i) matmul_nvfp4blk_rpw<<<gRPW(N), kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
    ck(cudaEventRecord(t1), "r"); ck(cudaEventSynchronize(t1), "s");
    float ms = 0; ck(cudaEventElapsedTime(&ms, t0, t1), "el");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (ms * 1000.0) / iters;
}
double timeSK(const Buf& b, int K, int N, int iters) {
    dim3 grid(gN(N), kSplitK);
    for (int i = 0; i < 10; ++i) matmul_nvfp4blk_splitk<<<grid, kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
    ck(cudaDeviceSynchronize(), "wu");
    cudaEvent_t t0, t1; ck(cudaEventCreate(&t0), "e"); ck(cudaEventCreate(&t1), "e");
    ck(cudaEventRecord(t0), "r");
    for (int i = 0; i < iters; ++i) matmul_nvfp4blk_splitk<<<grid, kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
    ck(cudaEventRecord(t1), "r"); ck(cudaEventSynchronize(t1), "s");
    float ms = 0; ck(cudaEventElapsedTime(&ms, t0, t1), "el");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (ms * 1000.0) / iters;
}

void benchShape(const char* label, int K, int N, int iters) {
    if ((K % kSuperElements) != 0) return;
    const int nSuper = K / kSuperElements;
    const size_t rs = static_cast<size_t>(nSuper);
    const size_t wBytes = static_cast<size_t>(N) * rs * kSuperBytes;
    const size_t qBytes = static_cast<size_t>(N) * rs * kSuperQBytes;
    const size_t sCount = static_cast<size_t>(N) * rs;
    const size_t xBytes = static_cast<size_t>(K) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> hX(static_cast<size_t>(K));
    std::vector<unsigned char> hW(wBytes), hQ(qBytes);
    std::vector<__half2> hS(sCount);
    for (int i = 0; i < K; ++i) hX[i] = 1.0f / static_cast<float>((i % 97) + 1);
    for (size_t sp = 0; sp < sCount; ++sp) {
        const __half s0 = __float2half(0.5f + 0.01f * static_cast<float>(sp % 7));
        const __half s1 = __float2half(0.25f + 0.02f * static_cast<float>(sp % 5));
        unsigned char* blk = hW.data() + sp * kSuperBytes;
        *reinterpret_cast<__half*>(blk)     = s0;
        *reinterpret_cast<__half*>(blk + 2) = s1;
        unsigned char* qs = blk + 4;
        unsigned char* qd = hQ.data() + sp * kSuperQBytes;
        for (int b = 0; b < kSuperQBytes; ++b) {
            const unsigned char v = static_cast<unsigned char>((sp * 131u + b * 17u + 3u) & 0xFFu);
            qs[b] = v; qd[b] = v;
        }
        hS[sp] = __halves2half2(s0, s1);
    }

    Buf b;
    ck(cudaMalloc(&b.dX, xBytes), "m"); ck(cudaMalloc(&b.dW, wBytes), "m");
    ck(cudaMalloc(&b.dQ, qBytes), "m"); ck(cudaMalloc(&b.dS, sCount * sizeof(__half2)), "m");
    ck(cudaMalloc(&b.dY, yBytes), "m");
    ck(cudaMemcpy(b.dX, hX.data(), xBytes, cudaMemcpyHostToDevice), "h");
    ck(cudaMemcpy(b.dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "h");
    ck(cudaMemcpy(b.dQ, hQ.data(), qBytes, cudaMemcpyHostToDevice), "h");
    ck(cudaMemcpy(b.dS, hS.data(), sCount * sizeof(__half2), cudaMemcpyHostToDevice), "h");

    // Correctness.
    matmul_nvfp4blk_vec<<<gN(N), kBlock>>>(b.dX, b.dW, b.dY, K, N);
    ck(cudaDeviceSynchronize(), "v0"); const auto y0 = readY(b, N);
    matmul_nvfp4blk_rpw<<<gRPW(N), kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N);
    ck(cudaDeviceSynchronize(), "rpw"); const auto yr = readY(b, N);
    ck(cudaMemset(b.dY, 0, yBytes), "z");
    { dim3 g(gN(N), kSplitK); matmul_nvfp4blk_splitk<<<g, kBlock>>>(b.dX, b.dQ, b.dS, b.dY, K, N); }
    ck(cudaDeviceSynchronize(), "sk"); const auto ys = readY(b, N);
    const float dr = maxAbsDiff(y0, yr);
    const float ds = maxAbsDiff(y0, ys);

    const double us0 = timeV0(b, K, N, iters);
    const double ur  = timeRPW(b, K, N, iters);
    ck(cudaMemset(b.dY, 0, yBytes), "z");
    const double us  = timeSK(b, K, N, iters);
    const double bytes = static_cast<double>(wBytes + xBytes + yBytes);
    auto pct = [&](double u) { return bytes / (u * 1e-6) / 1e9 / 273.0 * 100.0; };

    std::printf("| %-9s | %6d | %6d | %8.2f %5.1f%% | %8.2f %5.1f%% %+5.1f%% "
                "| %8.2f %5.1f%% %+5.1f%% | %.0e %.0e |\n",
                label, K, N,
                us0, pct(us0),
                ur, pct(ur), (us0 - ur) / us0 * 100.0,
                us, pct(us), (us0 - us) / us0 * 100.0, dr, ds);

    cudaFree(b.dX); cudaFree(b.dW); cudaFree(b.dQ); cudaFree(b.dS); cudaFree(b.dY);
}

} // namespace

int main(int argc, char** argv) {
    ck(cudaSetDevice(0), "dev");
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("# nvfp4blk_gemv OCCUPANCY A/B — %s (sm_%d%d), %d SMs, M=1 dense NVFP4 GEMV\n",
                p.name, p.major, p.minor, p.multiProcessorCount);
    std::printf("# v0=prod  vRPW=aligned+%drows/warp  vSK=aligned+splitK%d  "
                "(spd vs v0; dr/ds=max|Y-Yv0|)\n", kRowsPerWarp, kSplitK);
    std::printf("| shape     |      K |      N |       v0  %%pk |     vRPW  %%pk   spd "
                "|      vSK  %%pk   spd | dr ds |\n");
    if (argc >= 3) {
        const int K = std::atoi(argv[1]), N = std::atoi(argv[2]);
        benchShape("custom", K, N, argc >= 4 ? std::atoi(argv[3]) : 200);
        return 0;
    }
    const int it = 200;
    benchShape("o-proj",   2048,   2048, it);
    benchShape("qkv",      2048,   6144, it);
    benchShape("ffn-shex", 2048,    512, it);
    benchShape("ffn-up",   2048,  17408, it);
    benchShape("lm_head",  2048, 152064, it);
    return 0;
}

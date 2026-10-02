// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 5.18.30.6 — isolated FP8 (E4M3) dense GEMV microbench at the qwen3.6 gdn.proj
// decode (M=1) shapes. The engine runs these through cuBLASLt-FP8, which the
// decode profile shows at ~2.6x the FP8 weight-read floor (gdn.proj 9.4 ms/step
// vs a ~3.7 ms roofline). This bench measures how close a hand FP8 GEMV can get
// to the 273 GB/s DRAM roofline at M=1, to decide whether beating cuBLASLt-FP8
// is worth wiring into GpuMatmul (the real 5.18 dense-FP8 kernel work).
//
// At M=1 every output is a matrix-VECTOR dot with zero weight reuse, so the
// kernel is purely weight-bandwidth-bound (tensor cores cannot fill — the
// m16n8k64 FP8 mma is 1/16 filled). The lever is coalesced wide loads + enough
// resident warps to hide DRAM latency, NOT TC math.
//
// Per-tensor E4M3: y[n] = wscale * sum_k X[k] * e4m3_to_float(W[n,k]). X is F32.
// Build: see CMakeLists (microbench_fp8_dense). Run: ./microbench_fp8_dense
//   [K N iters]  — no args runs the built-in gdn.proj shape sweep.

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int   kBlock     = 256;          // 8 warps / block
constexpr int   kWarps     = kBlock / 32;
constexpr float kPeakGBs   = 273.0f;       // GB10 LPDDR5x

__device__ __forceinline__ float e4m3f(unsigned char b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&hr));
}

__device__ __forceinline__ float warpReduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// --- baseline: one warp per output, lanes stride K by 32, scalar fp8 loads ---
__global__ __launch_bounds__(kBlock) void fp8_gemv_naive(
        const float* __restrict__ X, const unsigned char* __restrict__ W,
        float* __restrict__ Y, int K, int N, float wscale) {
    const int warp = (blockIdx.x * kWarps) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const unsigned char* wrow = W + static_cast<size_t>(warp) * K;
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) sum = __fmaf_rn(X[k], e4m3f(wrow[k]), sum);
    sum = warpReduce(sum);
    if (lane == 0) Y[warp] = sum * wscale;
}

// --- candidate: one warp per output, each lane reads a uint4 (16 fp8 bytes)
// per step = 512 contiguous weight bytes/warp/step; X staged in shared memory
// once per block and reused across all 8 warps' outputs. -------------------
extern __shared__ float s_x[];
__global__ __launch_bounds__(kBlock) void fp8_gemv_vec(
        const float* __restrict__ X, const unsigned char* __restrict__ W,
        float* __restrict__ Y, int K, int N, float wscale) {
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = X[i];
    __syncthreads();
    const int warp = (blockIdx.x * kWarps) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool active = warp < N;
    const unsigned char* wrow = W + static_cast<size_t>(warp) * K;
    float sum = 0.0f;
    // 32 lanes * 16 bytes = 512 weight bytes per step.
    for (int base = lane * 16; base < K; base += 32 * 16) {
        if (active && base + 16 <= K) {
            const uint4 w4 = *reinterpret_cast<const uint4*>(wrow + base);
            const unsigned char* wb = reinterpret_cast<const unsigned char*>(&w4);
            #pragma unroll
            for (int j = 0; j < 16; ++j) sum = __fmaf_rn(s_x[base + j], e4m3f(wb[j]), sum);
        } else if (active) {
            for (int j = 0; base + j < K; ++j) sum = __fmaf_rn(s_x[base + j], e4m3f(wrow[base + j]), sum);
        }
    }
    sum = warpReduce(sum);
    if (active && lane == 0) Y[warp] = sum * wscale;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

unsigned gN(int N) { return static_cast<unsigned>((N + kWarps - 1) / kWarps); }

// A RING of distinct weight buffers whose aggregate exceeds L2, so each timed
// iteration reads a buffer that was evicted since its last use — i.e. COLD DRAM,
// matching the real decode (30 layers, each weight read once/token, no reuse).
// A single reused buffer would sit in GB10's ~24-32 MB L2 and report >100%peak.
struct Buf { float* dX = nullptr; float* dY = nullptr; std::vector<unsigned char*> dW; };

double timeK(void (*k)(const float*, const unsigned char*, float*, int, int, float),
             const Buf& b, int K, int N, float ws, int iters, size_t smem) {
    const int nBuf = static_cast<int>(b.dW.size());
    for (int i = 0; i < 20; ++i) k<<<gN(N), kBlock, smem>>>(b.dX, b.dW[i % nBuf], b.dY, K, N, ws);
    ck(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t t0, t1; ck(cudaEventCreate(&t0), "e"); ck(cudaEventCreate(&t1), "e");
    ck(cudaEventRecord(t0), "r");
    for (int i = 0; i < iters; ++i) k<<<gN(N), kBlock, smem>>>(b.dX, b.dW[i % nBuf], b.dY, K, N, ws);
    ck(cudaEventRecord(t1), "r"); ck(cudaEventSynchronize(t1), "s");
    float ms = 0; ck(cudaEventElapsedTime(&ms, t0, t1), "el");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return static_cast<double>(ms) / iters;   // ms per launch
}

std::vector<float> readY(const Buf& b, int N) {
    std::vector<float> y(N); ck(cudaMemcpy(y.data(), b.dY, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H"); return y;
}

void runShape(int K, int N, int iters) {
    Buf b;
    std::vector<float> hX(K);
    std::vector<unsigned char> hW(static_cast<size_t>(N) * K);
    for (int k = 0; k < K; ++k) hX[k] = 0.01f * static_cast<float>((k % 17) - 8);
    for (size_t i = 0; i < hW.size(); ++i) {
        // small E4M3-representable values; byte pattern is fine for a bw bench.
        hW[i] = static_cast<unsigned char>((i * 7 + 3) & 0x3F);
    }
    const float ws = 0.5f;
    // Ring of distinct weight buffers whose total exceeds L2 (defeat caching).
    const size_t wBytes = hW.size();
    const int    nBuf   = static_cast<int>((size_t{160} * 1024 * 1024 + wBytes - 1) / wBytes);
    const int    nBufC  = nBuf < 2 ? 2 : nBuf;
    ck(cudaMalloc(&b.dX, K * sizeof(float)), "mX");
    ck(cudaMalloc(&b.dY, N * sizeof(float)), "mY");
    ck(cudaMemcpy(b.dX, hX.data(), K * sizeof(float), cudaMemcpyHostToDevice), "cX");
    for (int i = 0; i < nBufC; ++i) {
        unsigned char* w = nullptr;
        ck(cudaMalloc(&w, wBytes), "mW");
        ck(cudaMemcpy(w, hW.data(), wBytes, cudaMemcpyHostToDevice), "cW");
        b.dW.push_back(w);
    }

    const size_t smem = static_cast<size_t>(K) * sizeof(float);
    const double bytes = static_cast<double>(N) * K;   // FP8 weight = 1 byte/elem (dominant)

    const double msNaive = timeK(fp8_gemv_naive, b, K, N, ws, iters, 0);
    auto yNaive = readY(b, N);
    const double msVec = (smem <= 48 * 1024)
        ? timeK(fp8_gemv_vec, b, K, N, ws, iters, smem) : -1.0;
    auto yVec = (msVec > 0) ? readY(b, N) : std::vector<float>{};

    // correctness: vec vs naive
    float md = 0.0f;
    for (int n = 0; msVec > 0 && n < N; ++n) md = fmaxf(md, fabsf(yVec[n] - yNaive[n]));

    auto gbs = [&](double ms) { return bytes / (ms * 1e-3) / 1e9; };
    auto pct = [&](double ms) { return 100.0 * gbs(ms) / kPeakGBs; };
    char vecCol[64];
    if (msVec > 0) {
        std::snprintf(vecCol, sizeof(vecCol), "vec %.4f ms %6.1f GB/s %4.1f%%peak",
                      msVec, gbs(msVec), pct(msVec));
    } else {
        std::snprintf(vecCol, sizeof(vecCol), "vec (smem %zuk > 48k, skipped)", smem / 1024);
    }
    std::printf("K=%-5d N=%-6d | naive %.4f ms %6.1f GB/s %4.1f%%peak | %s | max|dVec-naive|=%.3g\n",
                K, N, msNaive, gbs(msNaive), pct(msNaive), vecCol, md);

    cudaFree(b.dX); cudaFree(b.dY);
    for (unsigned char* w : b.dW) cudaFree(w);
}

} // namespace

int main(int argc, char** argv) {
    int dev = 0; ck(cudaSetDevice(dev), "setdev");
    cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, dev), "props");
    std::printf("# FP8 E4M3 dense GEMV (M=1) microbench — %s, SMs=%d, peak=%.0f GB/s\n",
                p.name, p.multiProcessorCount, kPeakGBs);
    std::printf("# baseline cuBLASLt-FP8 in-engine ~= 38%%peak at these shapes (gdn.proj 9.4ms = 2.6x floor)\n");
    if (argc >= 3) { runShape(std::atoi(argv[1]), std::atoi(argv[2]), argc >= 4 ? std::atoi(argv[3]) : 300); return 0; }
    // qwen3.6 gdn.proj decode shapes (K, N):
    runShape(2048, 12288, 300);  // fused in_proj_qkv(8192)+in_proj_z(4096)
    runShape(2048,  8192, 300);  // in_proj_qkv alone
    runShape(2048,  4096, 300);  // in_proj_z alone
    runShape(4096,  2048, 300);  // out_proj
    return 0;
}

// moe_gemv_m1_bench — isolated M=1 routed-MoE decode-GEMV bandwidth probe (roadmap 5.18.5).
//
// Measures the REAL prod decode kernel `moe_grouped_gemm_nvfp4blk_m1reg`
// (kernels/cuda/llm/moe_grouped_gemm_nvfp4blk.cu) in isolation on the routed-MoE
// decode shape, to settle the crux question the 5.15 / 5.18.3a notes left open:
//
//   The dense NVFP4 GEMV hits 60-75% of peak (5.18.6), but the roofline note puts
//   moe.gemm at only ~22% peak in real decode. Is the 22% a KERNEL inefficiency
//   (fixable) or a SMALL-per-expert-GEMV / scheduling effect (the lever is grouping
//   = a batch lever that does NOT transfer to single-user M=1)?
//
// This bench times the exact real-decode launch (E experts, grid.y=E, one row each,
// top-K all active) and reports GB/s + %peak vs the 273 GB/s GB10 ceiling, for both
// MoE projection shapes:
//   up/gate:  N=512,  K=2048   (per expert-proj weight = N*(K/32)*20 = 640 KiB)
//   down:     N=2048, K=512
//
// No prod swap, no ncu (banned). CUDA-event timing only. Build: microbench_moe_gemv_m1.
//
// Interpretation:
//   - If isolated %peak >> 22% (approaches the dense 60-75%): the real-decode 22% is
//     small-GEMV / launch / routing-tail overhead, NOT the kernel -> the single-user
//     lever is grouping (does not apply at M=1) -> 5.18.5 is a PLATEAU.
//   - If isolated %peak ~= 22%: the kernel itself is bandwidth-inefficient on this
//     shape -> a load-layout/prefetch variant is worth building (brick 2).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// The prod kernel under test (extern-C, compiled in via CMake second source).
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m1reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows,
    int K, int N);

static constexpr int   kSuperElems = 32;
static constexpr int   kSuperBytes = 20;
static constexpr int   kWarps      = 4;    // MATMUL_NVBLK_GEMM_WARPS
static constexpr int   kLocal      = 128;  // MATMUL_NVBLK_GEMM_LOCAL
static constexpr float kPeakGBs    = 273.0f;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fill a [E][N][K] blocked NVFP4 bank with deterministic supers:
//   [+0] fp16 s0 = 1.0, [+2] fp16 s1 = 1.0, [+4..+19] 16 nibble bytes (varying).
static void fillBank(std::vector<unsigned char>& W, int E, int N, int K) {
    const int nSuper = K / kSuperElems;
    const __half one = __float2half(1.0f);
    unsigned short oneBits;
    std::memcpy(&oneBits, &one, sizeof(oneBits));
    for (size_t i = 0; i < W.size(); i += kSuperBytes) {
        W[i + 0] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 1] = static_cast<unsigned char>(oneBits >> 8);
        W[i + 2] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 3] = static_cast<unsigned char>(oneBits >> 8);
        for (int b = 0; b < 16; ++b) {
            W[i + 4 + b] = static_cast<unsigned char>((i / kSuperBytes + b) & 0xFF);
        }
    }
    (void)E; (void)N; (void)nSuper;
}

// Time the real-decode launch: E experts, grid.y=E, one active row per expert.
static float benchShape(int E, int N, int K, int iters) {
    const int    nSuper = K / kSuperElems;
    const size_t wBytes = static_cast<size_t>(E) * N * nSuper * kSuperBytes;
    const int    rows   = E;   // one activation row per expert-tile (M=1 each)

    std::vector<unsigned char> hW(wBytes);
    fillBank(hW, E, N, K);
    std::vector<float> hX(static_cast<size_t>(rows) * K, 0.5f);
    std::vector<int> hTileExpert(E), hTileRow0(E), hTileRows(E, 1);
    for (int t = 0; t < E; ++t) { hTileExpert[t] = t; hTileRow0[t] = t; }

    unsigned char* dW = nullptr; float* dX = nullptr; float* dY = nullptr;
    int *dTE = nullptr, *dTR0 = nullptr, *dTR1 = nullptr;
    ck(cudaMalloc(&dW, wBytes), "malloc W");
    ck(cudaMalloc(&dX, hX.size() * sizeof(float)), "malloc X");
    ck(cudaMalloc(&dY, static_cast<size_t>(rows) * N * sizeof(float)), "malloc Y");
    ck(cudaMalloc(&dTE, E * sizeof(int)), "malloc TE");
    ck(cudaMalloc(&dTR0, E * sizeof(int)), "malloc TR0");
    ck(cudaMalloc(&dTR1, E * sizeof(int)), "malloc TR1");
    ck(cudaMemcpy(dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "cpy W");
    ck(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy X");
    ck(cudaMemcpy(dTE, hTileExpert.data(), E * sizeof(int), cudaMemcpyHostToDevice), "cpy TE");
    ck(cudaMemcpy(dTR0, hTileRow0.data(), E * sizeof(int), cudaMemcpyHostToDevice), "cpy TR0");
    ck(cudaMemcpy(dTR1, hTileRows.data(), E * sizeof(int), cudaMemcpyHostToDevice), "cpy TR1");

    dim3 grid((N + kWarps - 1) / kWarps, E, 1);
    dim3 block(kLocal, 1, 1);
    auto launch = [&]() {
        moe_grouped_gemm_nvfp4blk_m1reg<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N);
    };

    for (int i = 0; i < 10; ++i) launch();
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    ck(cudaEventSynchronize(b), "event sync");
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);

    const double us   = static_cast<double>(ms) * 1000.0 / iters;
    const double bytes = static_cast<double>(wBytes)
                       + static_cast<double>(hX.size()) * sizeof(float)
                       + static_cast<double>(rows) * N * sizeof(float);
    const double gbps = bytes / (us * 1e-6) / 1e9;
    const double pct  = gbps / kPeakGBs * 100.0;

    std::printf("| E=%-3d N=%-5d K=%-5d | %8.2f us | %6.1f GB/s | %5.1f%% peak | %6.0f KiB/exp |\n",
                E, N, K, us, gbps, pct, wBytes / 1024.0 / E);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    cudaFree(dTE); cudaFree(dTR0); cudaFree(dTR1);
    return static_cast<float>(pct);
}

int main(int argc, char** argv) {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("# moe_gemv_m1_bench on %s (sm_%d%d), peak=%.0f GB/s\n",
                prop.name, prop.major, prop.minor, kPeakGBs);
    std::printf("# real-decode launch: E experts, grid.y=E, one M=1 row each\n");
    std::printf("| shape | us/call | GB/s | %%peak | weight/expert |\n");
    std::printf("|---|---|---|---|---|\n");

    const int iters = (argc > 1) ? std::atoi(argv[1]) : 300;
    // Qwen3.6-35B-A3B: top-8 experts, hidden=2048, moe_ff=512.
    // up/gate per expert: N=512, K=2048 ; down per expert: N=2048, K=512.
    benchShape(8, 512, 2048, iters);   // up / gate (top-8)
    benchShape(8, 2048, 512, iters);   // down (top-8)
    // reference: a single BIG expert-proj in isolation (E=1) to expose the
    // small-per-expert-GEMV / grid-fill effect vs the 8-expert real launch.
    std::printf("# isolation reference (E=1, exposes small-GEMV / grid-fill effect):\n");
    benchShape(1, 512, 2048, iters);
    benchShape(1, 2048, 512, iters);
    // and a wide single GEMV that fully fills the SMs (large N), like the dense bench:
    benchShape(1, 8192, 2048, iters);
    return 0;
}

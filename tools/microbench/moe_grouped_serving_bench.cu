// moe_grouped_serving_bench — isolated SERVING-decode grouped-MoE GEMM roofline probe
// (roadmap 5.18.9, Inc-2). Unlike moe_gemv_m1_bench (5.18.5, single-user E=8/M=1), this
// drives the REAL conc64 decode kernel `moe_grouped_gemm_nvfp4blk_m4` at the SERVING
// shape: all E=256 experts active, M rows per expert (M = batch tokens routed to that
// expert). At conc64/top-8 the ~512 assignments hit ~all 256 experts with M~=2 each, so
// the decode MoE reads ~all expert weights once per step (weight-bandwidth bound).
//
// The prod DECODE_PROFILE put moe.gemm at ~58% of a ~310ms/step conc64 decode, implying
// the grouped GEMM runs ~2.7x above the 273 GB/s weight-read roofline. This bench settles
// whether that is a KERNEL memory-efficiency problem (fixable) at the real serving shape.
//
// Reports achieved GB/s + %peak (273 GB/s GB10 ceiling) reading all E experts once, plus
// us-per-token (us / (E*M)) so the M-sweep shows the batch amortization directly.
//
// No prod swap needed (bare microbench, ~200MB, ~seconds); no ncu (banned on-box).
// Build: microbench_moe_serving.  Run: ./microbench_moe_serving [iters]

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

// Prod kernels under test (extern-C, compiled in via CMake second source).
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m4(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m1reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m2reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m4reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);

// Kernel selector for the bench/parity harness.
enum Kern { K_M4 = 0, K_M1REG, K_M2REG, K_M4REG };
static const char* kernName(Kern k) {
    switch (k) { case K_M4: return "m4"; case K_M1REG: return "m1reg";
                 case K_M2REG: return "m2reg"; case K_M4REG: return "m4reg"; }
    return "?";
}

static constexpr int   kSuperElems = 32;
static constexpr int   kSuperBytes = 20;
static constexpr int   kWarps      = 4;    // MATMUL_NVBLK_GEMM_WARPS (LOCAL/32)
static constexpr int   kLocal      = 128;  // MATMUL_NVBLK_GEMM_LOCAL
static constexpr float kPeakGBs    = 273.0f;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fill a [E][N][K] blocked NVFP4 bank with deterministic supers (s0=s1=1.0, varying nibbles).
static void fillBank(std::vector<unsigned char>& W) {
    const __half one = __float2half(1.0f);
    unsigned short oneBits;
    std::memcpy(&oneBits, &one, sizeof(oneBits));
    for (size_t i = 0; i < W.size(); i += kSuperBytes) {
        W[i + 0] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 1] = static_cast<unsigned char>(oneBits >> 8);
        W[i + 2] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 3] = static_cast<unsigned char>(oneBits >> 8);
        for (int b = 0; b < 16; ++b)
            W[i + 4 + b] = static_cast<unsigned char>((i / kSuperBytes + b) & 0xFF);
    }
}

// Launch the selected kernel (all share the same signature/launch geometry).
static void launchKern(Kern kern, dim3 grid, dim3 block,
                       const float* dX, const unsigned char* dW, float* dY,
                       const int* dTE, const int* dTR0, const int* dTR1, int K, int N) {
    switch (kern) {
        case K_M4:    moe_grouped_gemm_nvfp4blk_m4   <<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N); break;
        case K_M1REG: moe_grouped_gemm_nvfp4blk_m1reg<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N); break;
        case K_M2REG: moe_grouped_gemm_nvfp4blk_m2reg<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N); break;
        case K_M4REG: moe_grouped_gemm_nvfp4blk_m4reg<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N); break;
    }
}

// Serving-decode launch: E expert-tiles, grid.y=E, M rows per tile (all experts active).
static void benchShape(Kern kern, int E, int N, int K, int M, int iters) {
    const int    nSuper = K / kSuperElems;
    const size_t wBytes = static_cast<size_t>(E) * N * nSuper * kSuperBytes;
    const int    rows   = E * M;

    std::vector<unsigned char> hW(wBytes);
    fillBank(hW);
    std::vector<float> hX(static_cast<size_t>(rows) * K, 0.5f);
    std::vector<int> hTileExpert(E), hTileRow0(E), hTileRows(E, M);
    for (int t = 0; t < E; ++t) { hTileExpert[t] = t; hTileRow0[t] = t * M; }

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
        launchKern(kern, grid, block, dX, dW, dY, dTE, dTR0, dTR1, K, N);
    };

    for (int i = 0; i < 10; ++i) launch();
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    ck(cudaEventSynchronize(b), "event sync");
    float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);

    const double us    = static_cast<double>(ms) * 1000.0 / iters;
    const double bytes = static_cast<double>(wBytes)
                       + static_cast<double>(hX.size()) * sizeof(float)
                       + static_cast<double>(rows) * N * sizeof(float);
    const double gbps      = bytes / (us * 1e-6) / 1e9;
    const double pct       = gbps / kPeakGBs * 100.0;
    const double usPerTok  = us / static_cast<double>(E * M);
    const double idealUs   = static_cast<double>(wBytes) / (kPeakGBs * 1e9) * 1e6; // weight-read floor

    std::printf("| %-6s E=%-3d M=%d N=%-5d K=%-5d | %8.2f us | %6.1f GB/s | %5.1f%% | %7.3f us/tok | roof %6.1fus |\n",
                kernName(kern), E, M, N, K, us, gbps, pct, usPerTok, idealUs);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    cudaFree(dTE); cudaFree(dTR0); cudaFree(dTR1);
}

// Parity: run the reference m4 and a candidate kernel on identical inputs and
// report max abs diff over Y. The register-staged variants must be bit-identical
// (same math, same per-k accumulation order) -> expect exactly 0.
static double parityCheck(Kern cand, int E, int N, int K, int M) {
    const int    nSuper = K / kSuperElems;
    const size_t wBytes = static_cast<size_t>(E) * N * nSuper * kSuperBytes;
    const int    rows   = E * M;

    std::vector<unsigned char> hW(wBytes); fillBank(hW);
    std::vector<float> hX(static_cast<size_t>(rows) * K);
    for (size_t i = 0; i < hX.size(); ++i) hX[i] = 0.25f + 0.5f * ((i % 7) / 7.0f);
    std::vector<int> hTE(E), hTR0(E), hTR1(E, M);
    for (int t = 0; t < E; ++t) { hTE[t] = t; hTR0[t] = t * M; }

    unsigned char* dW = nullptr; float* dX = nullptr; float* dYr = nullptr; float* dYc = nullptr;
    int *dTE = nullptr, *dTR0 = nullptr, *dTR1 = nullptr;
    ck(cudaMalloc(&dW, wBytes), "pW"); ck(cudaMalloc(&dX, hX.size()*sizeof(float)), "pX");
    ck(cudaMalloc(&dYr, (size_t)rows*N*sizeof(float)), "pYr");
    ck(cudaMalloc(&dYc, (size_t)rows*N*sizeof(float)), "pYc");
    ck(cudaMalloc(&dTE, E*sizeof(int)), "pTE"); ck(cudaMalloc(&dTR0, E*sizeof(int)), "pTR0");
    ck(cudaMalloc(&dTR1, E*sizeof(int)), "pTR1");
    ck(cudaMemcpy(dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "cW");
    ck(cudaMemcpy(dX, hX.data(), hX.size()*sizeof(float), cudaMemcpyHostToDevice), "cX");
    ck(cudaMemcpy(dTE, hTE.data(), E*sizeof(int), cudaMemcpyHostToDevice), "cTE");
    ck(cudaMemcpy(dTR0, hTR0.data(), E*sizeof(int), cudaMemcpyHostToDevice), "cTR0");
    ck(cudaMemcpy(dTR1, hTR1.data(), E*sizeof(int), cudaMemcpyHostToDevice), "cTR1");

    dim3 grid((N + kWarps - 1) / kWarps, E, 1); dim3 block(kLocal, 1, 1);
    launchKern(K_M4, grid, block, dX, dW, dYr, dTE, dTR0, dTR1, K, N);
    launchKern(cand,  grid, block, dX, dW, dYc, dTE, dTR0, dTR1, K, N);
    ck(cudaDeviceSynchronize(), "parity sync");

    std::vector<float> hr((size_t)rows*N), hc((size_t)rows*N);
    ck(cudaMemcpy(hr.data(), dYr, hr.size()*sizeof(float), cudaMemcpyDeviceToHost), "gYr");
    ck(cudaMemcpy(hc.data(), dYc, hc.size()*sizeof(float), cudaMemcpyDeviceToHost), "gYc");
    double maxd = 0.0;
    for (size_t i = 0; i < hr.size(); ++i) {
        const double d = std::abs(static_cast<double>(hr[i]) - static_cast<double>(hc[i]));
        if (d > maxd) maxd = d;
    }
    cudaFree(dW); cudaFree(dX); cudaFree(dYr); cudaFree(dYc);
    cudaFree(dTE); cudaFree(dTR0); cudaFree(dTR1);
    std::printf("# parity %-6s vs m4  E=%d M=%d N=%d K=%d : max|dY|=%.3e  %s\n",
                kernName(cand), E, M, N, K, maxd, maxd == 0.0 ? "BIT-IDENTICAL" : (maxd < 1e-3 ? "OK" : "MISMATCH"));
    return maxd;
}

int main(int argc, char** argv) {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("# moe_grouped_serving_bench on %s (sm_%d%d), peak=%.0f GB/s\n",
                prop.name, prop.major, prop.minor, static_cast<double>(kPeakGBs));
    std::printf("# SERVING decode shape: E=256 experts all active, M rows/expert (conc64 top-8 ~= M2)\n");
    std::printf("# Qwen3.6-35B-A3B: gate/up N=512 K=2048 ; down N=2048 K=512\n\n");

    const int iters = (argc > 1) ? std::atoi(argv[1]) : 300;
    const int E = 256;

    std::printf("## parity (register-staged variants must match the shared m4 core)\n");
    parityCheck(K_M2REG, 64, 512, 2048, 2);
    parityCheck(K_M4REG, 64, 512, 2048, 4);
    parityCheck(K_M4REG, 64, 2048, 512, 3);   // ragged M<MAX_M
    parityCheck(K_M2REG, 64, 2048, 512, 1);   // M<MAX_M
    std::printf("\n");

    std::printf("## perf (%%peak reading all 256 experts once = weight-BW efficiency)\n");
    std::printf("| kernel shape                    | us/call  | GB/s   | %%peak | us/token   | ideal      |\n");
    std::printf("|---|---|---|---|---|---|\n");
    // m4reg is the only register-staged kernel SAFE for the tileM=4 decode schedule
    // (m2reg caps at 2 rows). Bench m4reg across M1/M2/M4 (the real mixed decode
    // distribution) to predict the drop-in win vs the current shared m4.
    std::printf("# --- gate/up projection (N=512, K=2048) ---\n");
    benchShape(K_M4,    E, 512, 2048, 1, iters);
    benchShape(K_M4REG, E, 512, 2048, 1, iters);
    benchShape(K_M4,    E, 512, 2048, 2, iters);
    benchShape(K_M2REG, E, 512, 2048, 2, iters);   // tileM=2-only ceiling (48%)
    benchShape(K_M4REG, E, 512, 2048, 2, iters);   // <-- Inc-4 drop-in candidate @M2
    benchShape(K_M4,    E, 512, 2048, 4, iters);
    benchShape(K_M4REG, E, 512, 2048, 4, iters);
    std::printf("# --- down projection (N=2048, K=512) ---\n");
    benchShape(K_M4,    E, 2048, 512, 1, iters);
    benchShape(K_M4REG, E, 2048, 512, 1, iters);
    benchShape(K_M4,    E, 2048, 512, 2, iters);
    benchShape(K_M2REG, E, 2048, 512, 2, iters);
    benchShape(K_M4REG, E, 2048, 512, 2, iters);
    benchShape(K_M4,    E, 2048, 512, 4, iters);
    benchShape(K_M4REG, E, 2048, 512, 4, iters);

    std::printf("# Target: m2reg@M2 / m4reg@M4 recover from m4's ~34%%/24%% toward the m1reg ~48%% ceiling.\n");
    return 0;
}

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
#include <vector>

// Prod kernels under test (extern-C, compiled in via CMake second source).
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m4(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m1reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);

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

// Serving-decode launch: E expert-tiles, grid.y=E, M rows per tile (all experts active).
// useM1reg selects the single-user register-staged kernel (only valid for M==1).
static void benchShape(const char* tag, int E, int N, int K, int M, int iters, bool useM1reg) {
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
        if (useM1reg)
            moe_grouped_gemm_nvfp4blk_m1reg<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N);
        else
            moe_grouped_gemm_nvfp4blk_m4<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N);
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

    std::printf("| %-8s E=%-3d M=%d N=%-5d K=%-5d | %8.2f us | %6.1f GB/s | %5.1f%% | %7.3f us/tok | roof %6.1fus |\n",
                tag, E, M, N, K, us, gbps, pct, usPerTok, idealUs);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    cudaFree(dTE); cudaFree(dTR0); cudaFree(dTR1);
}

int main(int argc, char** argv) {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("# moe_grouped_serving_bench on %s (sm_%d%d), peak=%.0f GB/s\n",
                prop.name, prop.major, prop.minor, kPeakGBs);
    std::printf("# SERVING decode shape: E=256 experts all active, M rows/expert (conc64 top-8 ~= M2)\n");
    std::printf("# Qwen3.6-35B-A3B: gate/up N=512 K=2048 ; down N=2048 K=512\n");
    std::printf("| kernel   shape                  | us/call  | GB/s   | %%peak | us/token   | ideal      |\n");
    std::printf("|---|---|---|---|---|---|\n");

    const int iters = (argc > 1) ? std::atoi(argv[1]) : 300;
    const int E = 256;

    std::printf("# --- gate/up projection (N=512, K=2048) ---\n");
    benchShape("m1reg", E, 512, 2048, 1, iters, /*useM1reg=*/true);
    benchShape("m4",    E, 512, 2048, 1, iters, false);
    benchShape("m4",    E, 512, 2048, 2, iters, false);
    benchShape("m4",    E, 512, 2048, 4, iters, false);

    std::printf("# --- down projection (N=2048, K=512) ---\n");
    benchShape("m1reg", E, 2048, 512, 1, iters, /*useM1reg=*/true);
    benchShape("m4",    E, 2048, 512, 1, iters, false);
    benchShape("m4",    E, 2048, 512, 2, iters, false);
    benchShape("m4",    E, 2048, 512, 4, iters, false);

    std::printf("# Interpretation: %%peak reading all 256 experts once = kernel weight-BW efficiency.\n");
    std::printf("# us/token should DROP ~linearly M=1->2->4 (batch amortizes the weight read) while\n");
    std::printf("# GB/s stays flat. If %%peak ~35-40%% -> the m4 kernel is the ~2.7x lever (fixable).\n");
    return 0;
}

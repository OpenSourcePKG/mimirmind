// moe_gemm_widem_bench — isolated WIDE-M routed-MoE PREFILL-GEMM roofline probe
// (roadmap 5.21.5 / Option 1 gate).
//
// Measures the REAL prod prefill kernel `moe_grouped_gemm_nvfp4blk` (MAX_M=16,
// kernels/cuda/llm/moe_grouped_gemm_nvfp4blk.cu) in isolation across an M sweep
// (tiles of 16 rows each), to settle the go/no-go question for a wide-M FP4
// tensor-core prefill GEMM:
//
//   At M=16 the grouped GEMM reads each expert weight row ONCE and reuses it
//   across up to 16 activation rows via acc[m] on CUDA-core __fmaf_rn (NO tensor
//   cores). The mma m16n8k64 shape wants exactly 16 M-rows -> at prefill the MMA
//   would be FULLY filled (vs 1/16 at decode). So: is the current M16 CUDA-core
//   kernel COMPUTE-bound (near the fp32 CUDA-core peak -> an FP4-TC kernel that
//   does the same MACs at ~8-16x wins big -> GO) or BANDWIDTH-bound (near the
//   273 GB/s DRAM peak -> TC reads the same bytes, cannot help -> NO-GO) or
//   LATENCY/occupancy-bound (far below both -> a different fix)?
//
// No prod swap, no ncu (banned). CUDA-event timing only.
// Build: microbench_moe_gemm_widem.
//
// Interpretation (printed per row):
//   - %BW  = effective DRAM GB/s / 273
//   - %FMA = effective fp32 GFLOP/s / (device fp32 CUDA-core peak)
//   Verdict: BW-bound if %BW high; COMPUTE-bound (TC candidate) if %FMA high &
//   %BW low; LATENCY-bound if both low.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// Real prod prefill kernel (MAX_M=16) + decode reference (M=1), compiled in via CMake.
extern "C" __global__ void moe_grouped_gemm_nvfp4blk(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_m1reg(
    const float* X, const unsigned char* W, float* Y,
    const int* tileExpert, const int* tileRow0, const int* tileRows, int K, int N);

static constexpr int   kSuperElems = 32;
static constexpr int   kSuperBytes = 20;
static constexpr int   kWarps      = 4;     // MATMUL_NVBLK_GEMM_OUTPUTS_PER_GROUP
static constexpr int   kLocal      = 128;   // MATMUL_NVBLK_GEMM_LOCAL
static constexpr int   kTileM      = 16;    // GEMM_MAX_M
static constexpr float kPeakGBs    = 273.0f;

static double g_fp32PeakGflops = 0.0;       // filled from device props in main

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fill an [E][N][K] blocked NVFP4 bank with deterministic supers (s0=s1=1.0).
static void fillBank(std::vector<unsigned char>& W) {
    const __half one = __float2half(1.0f);
    unsigned short oneBits; std::memcpy(&oneBits, &one, sizeof(oneBits));
    for (size_t i = 0; i < W.size(); i += kSuperBytes) {
        W[i + 0] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 1] = static_cast<unsigned char>(oneBits >> 8);
        W[i + 2] = static_cast<unsigned char>(oneBits & 0xFF);
        W[i + 3] = static_cast<unsigned char>(oneBits >> 8);
        for (int b = 0; b < 16; ++b)
            W[i + 4 + b] = static_cast<unsigned char>((i / kSuperBytes + b) & 0xFF);
    }
}

// One wide-M GEMM launch: `tiles` tiles of kTileM rows.
//   distinctExperts=true  -> tile t uses expert t (no cross-tile L2 weight reuse;
//                            worst case, realistic when many experts are hit).
//   distinctExperts=false -> all tiles use expert 0 (best-case L2 reuse).
static void benchWideM(const char* tag, int N, int K, int tiles, bool distinctExperts,
                       int iters) {
    const int    nSuper = K / kSuperElems;
    const int    E      = distinctExperts ? tiles : 1;
    const size_t wBytes = static_cast<size_t>(E) * N * nSuper * kSuperBytes;
    const int    rows   = tiles * kTileM;

    std::vector<unsigned char> hW(wBytes);
    fillBank(hW);
    std::vector<float> hX(static_cast<size_t>(rows) * K, 0.5f);
    std::vector<int> hTE(tiles), hTR0(tiles), hTR1(tiles, kTileM);
    for (int t = 0; t < tiles; ++t) {
        hTE[t]  = distinctExperts ? t : 0;
        hTR0[t] = t * kTileM;
    }

    unsigned char* dW = nullptr; float* dX = nullptr; float* dY = nullptr;
    int *dTE = nullptr, *dTR0 = nullptr, *dTR1 = nullptr;
    ck(cudaMalloc(&dW, wBytes), "malloc W");
    ck(cudaMalloc(&dX, hX.size() * sizeof(float)), "malloc X");
    ck(cudaMalloc(&dY, static_cast<size_t>(rows) * N * sizeof(float)), "malloc Y");
    ck(cudaMalloc(&dTE, tiles * sizeof(int)), "malloc TE");
    ck(cudaMalloc(&dTR0, tiles * sizeof(int)), "malloc TR0");
    ck(cudaMalloc(&dTR1, tiles * sizeof(int)), "malloc TR1");
    ck(cudaMemcpy(dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "cpy W");
    ck(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy X");
    ck(cudaMemcpy(dTE, hTE.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cpy TE");
    ck(cudaMemcpy(dTR0, hTR0.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cpy TR0");
    ck(cudaMemcpy(dTR1, hTR1.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cpy TR1");

    dim3 grid((N + kWarps - 1) / kWarps, tiles, 1);
    dim3 block(kLocal, 1, 1);
    auto launch = [&]() {
        moe_grouped_gemm_nvfp4blk<<<grid, block>>>(dX, dW, dY, dTE, dTR0, dTR1, K, N);
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
    // DRAM bytes: weights (read once per distinct expert) + X read + Y write.
    const double bytes = static_cast<double>(wBytes)
                       + static_cast<double>(hX.size()) * sizeof(float)
                       + static_cast<double>(rows) * N * sizeof(float);
    const double gbps = bytes / (us * 1e-6) / 1e9;
    const double pctBW = gbps / kPeakGBs * 100.0;
    // Compute: 2*M*N*K MACs (fp32 FMA on CUDA cores).
    const double flops   = 2.0 * static_cast<double>(rows) * N * K;
    const double gflops  = flops / (us * 1e-6) / 1e9;
    const double pctFMA  = g_fp32PeakGflops > 0 ? gflops / g_fp32PeakGflops * 100.0 : 0.0;
    const double aiFlopB = flops / bytes;   // arithmetic intensity FLOP/byte

    std::printf("| %-9s M=%-4d %s | N=%-5d K=%-5d | %8.2f us | %6.1f GB/s %4.1f%%BW | "
                "%7.0f GF/s %4.1f%%FMA | AI=%5.1f |\n",
                tag, rows, distinctExperts ? "distinctE" : "sameE    ",
                N, K, us, gbps, pctBW, gflops, pctFMA, aiFlopB);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    cudaFree(dTE); cudaFree(dTR0); cudaFree(dTR1);
}

int main(int argc, char** argv) {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    // fp32 CUDA-core peak = SMs * 128 fp32 cores/SM * 2 (FMA) * clockGHz.
    // cudaDeviceProp::clockRate was removed in CUDA 13 — query the attribute.
    int clkKHz = 0;
    cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, dev);
    const double clkGHz = clkKHz / 1e6;                    // attribute is kHz
    const int    fp32PerSM = 128;                          // consumer Blackwell
    g_fp32PeakGflops = prop.multiProcessorCount * fp32PerSM * 2.0 * clkGHz;

    std::printf("# moe_gemm_widem_bench on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("# DRAM peak=%.0f GB/s ; fp32 CUDA-core peak~=%.0f GFLOP/s "
                "(%d SM x %d x 2 x %.2f GHz)\n",
                kPeakGBs, g_fp32PeakGflops, prop.multiProcessorCount, fp32PerSM, clkGHz);
    std::printf("# kernel: moe_grouped_gemm_nvfp4blk (MAX_M=16, CUDA-core __fmaf_rn, "
                "weight reused across the tile's <=16 rows)\n");
    std::printf("# ridge point (fp32-CUDA-core) = %.0f FLOP/byte ; ridge (fp4-TC ~500 TFLOP) "
                "= ~1830 FLOP/byte\n", g_fp32PeakGflops / kPeakGBs);
    std::printf("| tag  M      experts | shape        | us/call | DRAM         | compute        | AI |\n");
    std::printf("|---|---|---|---|---|---|\n");

    const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    // Qwen3.6-35B-A3B routed-MoE per-expert shapes (same as moe_gemv_m1_bench):
    //   gate/up per expert: N=512, K=2048 ; down per expert: N=2048, K=512.
    // Sweep M via tile count (16 rows/tile): 1,2,4,8,16,32,64 tiles = M 16..1024.
    const int tileCounts[] = {1, 2, 4, 8, 16, 32, 64};
    std::printf("# --- gate/up (N=512, K=2048) ---\n");
    for (int t : tileCounts) benchWideM("gate/up", 512, 2048, t, true,  iters);
    for (int t : tileCounts) benchWideM("gate/up", 512, 2048, t, false, iters);
    std::printf("# --- down (N=2048, K=512) ---\n");
    for (int t : tileCounts) benchWideM("down",    2048, 512, t, true,  iters);
    for (int t : tileCounts) benchWideM("down",    2048, 512, t, false, iters);
    return 0;
}

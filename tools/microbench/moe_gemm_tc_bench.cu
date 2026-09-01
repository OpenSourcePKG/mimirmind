// moe_gemm_tc_bench — parity + perf for the wide-M FP16 tensor-core prefill
// MoE-GEMM (roadmap 5.21.6 Increment I). Links BOTH the prod CUDA-core kernel
// `moe_grouped_gemm_nvfp4blk` (MAX_M=16, the numerical oracle) and the new
// `moe_grouped_gemm_nvfp4blk_tc` (wmma FP16), runs them on IDENTICAL random
// inputs, and reports (1) max abs / rel error vs the oracle and (2) the perf
// speedup across an M sweep. No prod swap, no ncu. CUDA-event timing.
//   Build: microbench_moe_gemm_tc ; run: ./microbench_moe_gemm_tc [iters]

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" __global__ void moe_grouped_gemm_nvfp4blk(
    const float*, const unsigned char*, float*,
    const int*, const int*, const int*, int, int);
extern "C" __global__ void moe_grouped_gemm_nvfp4blk_tc(
    const float*, const unsigned char*, float*,
    const int*, const int*, const int*, int, int);

static constexpr int   kSuperElems = 32;
static constexpr int   kSuperBytes = 20;
static constexpr float kPeakGBs    = 273.0f;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static uint32_t s_rng = 0x1234567u;
static float frand(float lo, float hi) {
    s_rng = s_rng * 1664525u + 1013904223u;
    return lo + (hi - lo) * (static_cast<float>(s_rng >> 8) / 16777216.0f);
}

// Random blocked-NVFP4 bank: per super two fp16 scales in [0.02,0.15] + 16 random
// nibble bytes. [E][N][K].
static void fillBankRandom(std::vector<unsigned char>& W) {
    for (size_t i = 0; i < W.size(); i += kSuperBytes) {
        const __half s0 = __float2half(frand(0.02f, 0.15f));
        const __half s1 = __float2half(frand(0.02f, 0.15f));
        unsigned short b0, b1;
        std::memcpy(&b0, &s0, 2); std::memcpy(&b1, &s1, 2);
        W[i + 0] = b0 & 0xFF; W[i + 1] = b0 >> 8;
        W[i + 2] = b1 & 0xFF; W[i + 3] = b1 >> 8;
        for (int b = 0; b < 16; ++b) {
            s_rng = s_rng * 1664525u + 1013904223u;
            W[i + 4 + b] = static_cast<unsigned char>(s_rng >> 16);
        }
    }
}

struct Dev {
    unsigned char* W = nullptr; float* X = nullptr; float* Y = nullptr;
    int *TE = nullptr, *TR0 = nullptr, *TR1 = nullptr;
    int rows = 0, N = 0;
};

// Build device buffers for a schedule of `tiles` tiles (rowsPerTile, distinct
// experts). Returns host X copy for reference.
static Dev makeProblem(int N, int K, int tiles, int rowsPerTile) {
    Dev d; d.N = N; d.rows = tiles * rowsPerTile;
    const int nSuper = K / kSuperElems;
    const int E = tiles;
    const size_t wBytes = static_cast<size_t>(E) * N * nSuper * kSuperBytes;

    std::vector<unsigned char> hW(wBytes); fillBankRandom(hW);
    std::vector<float> hX(static_cast<size_t>(d.rows) * K);
    for (auto& v : hX) v = frand(-1.0f, 1.0f);
    std::vector<int> hTE(tiles), hTR0(tiles), hTR1(tiles, rowsPerTile);
    for (int t = 0; t < tiles; ++t) { hTE[t] = t; hTR0[t] = t * rowsPerTile; }

    ck(cudaMalloc(&d.W, wBytes), "W");
    ck(cudaMalloc(&d.X, hX.size() * sizeof(float)), "X");
    ck(cudaMalloc(&d.Y, static_cast<size_t>(d.rows) * N * sizeof(float)), "Y");
    ck(cudaMalloc(&d.TE, tiles * sizeof(int)), "TE");
    ck(cudaMalloc(&d.TR0, tiles * sizeof(int)), "TR0");
    ck(cudaMalloc(&d.TR1, tiles * sizeof(int)), "TR1");
    ck(cudaMemcpy(d.W, hW.data(), wBytes, cudaMemcpyHostToDevice), "cW");
    ck(cudaMemcpy(d.X, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice), "cX");
    ck(cudaMemcpy(d.TE, hTE.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cTE");
    ck(cudaMemcpy(d.TR0, hTR0.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cTR0");
    ck(cudaMemcpy(d.TR1, hTR1.data(), tiles * sizeof(int), cudaMemcpyHostToDevice), "cTR1");
    return d;
}

static void freeProblem(Dev& d) {
    cudaFree(d.W); cudaFree(d.X); cudaFree(d.Y);
    cudaFree(d.TE); cudaFree(d.TR0); cudaFree(d.TR1);
}

// Parity: run both kernels on one problem, compare Y.
static void parity(const char* tag, int N, int K, int tiles, int rowsPerTile) {
    Dev d = makeProblem(N, K, tiles, rowsPerTile);
    const size_t yElems = static_cast<size_t>(d.rows) * N;

    dim3 gCore((N + 3) / 4, tiles, 1), bCore(128, 1, 1);
    const int TCW = 4;  /* must match TC_WARPS in the kernel */
    dim3 gTc((N + 16 * TCW - 1) / (16 * TCW), tiles, 1), bTc(32 * TCW, 1, 1);

    ck(cudaMemset(d.Y, 0, yElems * sizeof(float)), "memsetY");
    moe_grouped_gemm_nvfp4blk<<<gCore, bCore>>>(d.X, d.W, d.Y, d.TE, d.TR0, d.TR1, K, N);
    ck(cudaDeviceSynchronize(), "core sync");
    std::vector<float> yCore(yElems);
    ck(cudaMemcpy(yCore.data(), d.Y, yElems * sizeof(float), cudaMemcpyDeviceToHost), "dY core");

    ck(cudaMemset(d.Y, 0, yElems * sizeof(float)), "memsetY2");
    moe_grouped_gemm_nvfp4blk_tc<<<gTc, bTc>>>(d.X, d.W, d.Y, d.TE, d.TR0, d.TR1, K, N);
    ck(cudaDeviceSynchronize(), "tc sync");
    std::vector<float> yTc(yElems);
    ck(cudaMemcpy(yTc.data(), d.Y, yElems * sizeof(float), cudaMemcpyDeviceToHost), "dY tc");

    // Proper GEMM parity: L2 relative error ||tc-core||/||core|| and max-abs
    // normalised by the reference max magnitude (near-zero-output cancellation
    // makes per-element relative error meaningless).
    double sqErr = 0, sqRef = 0, maxAbs = 0, maxRef = 0;
    for (size_t i = 0; i < yElems; ++i) {
        const double a = yCore[i], b = yTc[i];
        const double d = a - b;
        sqErr += d * d; sqRef += a * a;
        maxAbs = std::max(maxAbs, std::fabs(d));
        maxRef = std::max(maxRef, std::fabs(a));
    }
    const double l2rel  = std::sqrt(sqErr / (sqRef + 1e-30));
    const double absRel = maxAbs / (maxRef + 1e-30);
    std::printf("| %-22s | N=%-4d K=%-4d tiles=%d rows/tile=%-2d | L2rel=%.3g "
                "maxAbs/maxRef=%.3g (maxAbs=%.3g maxRef=%.3g) %s |\n",
                tag, N, K, tiles, rowsPerTile, l2rel, absRel, maxAbs, maxRef,
                (l2rel < 1e-2) ? "PASS" : "FAIL");
    freeProblem(d);
}

static void perf(int N, int K, int tiles, int rowsPerTile, int iters) {
    Dev d = makeProblem(N, K, tiles, rowsPerTile);
    dim3 gCore((N + 3) / 4, tiles, 1), bCore(128, 1, 1);
    const int TCW = 4;  /* must match TC_WARPS in the kernel */
    dim3 gTc((N + 16 * TCW - 1) / (16 * TCW), tiles, 1), bTc(32 * TCW, 1, 1);

    auto timeit = [&](bool tc) -> double {
        auto launch = [&]() {
            if (tc) moe_grouped_gemm_nvfp4blk_tc<<<gTc, bTc>>>(d.X, d.W, d.Y, d.TE, d.TR0, d.TR1, K, N);
            else    moe_grouped_gemm_nvfp4blk   <<<gCore, bCore>>>(d.X, d.W, d.Y, d.TE, d.TR0, d.TR1, K, N);
        };
        for (int i = 0; i < 10; ++i) launch();
        ck(cudaDeviceSynchronize(), "warm");
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(b); ck(cudaEventSynchronize(b), "ev");
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        return static_cast<double>(ms) * 1000.0 / iters;   // us
    };

    const double usCore = timeit(false);
    const double usTc   = timeit(true);
    const int M = tiles * rowsPerTile;
    const double flops = 2.0 * M * N * K;
    const double gfCore = flops / (usCore * 1e-6) / 1e9;
    const double gfTc   = flops / (usTc   * 1e-6) / 1e9;
    std::printf("| M=%-4d N=%-4d K=%-4d | core %8.2f us %6.0f GF/s | tc %8.2f us %7.0f GF/s | "
                "%5.2fx |\n", M, N, K, usCore, gfCore, usTc, gfTc, usCore / usTc);
    freeProblem(d);
}

int main(int argc, char** argv) {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("# moe_gemm_tc_bench on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::printf("## PARITY (tc wmma-fp16 vs core fp32 oracle, random weights)\n");
    parity("full M16 N512",   512,  2048, 1, 16);
    parity("partial M13 N520", 520, 2048, 1, 13);   // exercises M + N masking
    parity("down full M16",   2048, 512,  1, 16);
    parity("multi-tile 8x16",  512, 2048, 8, 16);
    parity("multi-tile M7",    512, 2048, 4, 7);

    const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    std::printf("## PERF (M-sweep, distinct experts)\n");
    std::printf("# --- gate/up N=512 K=2048 ---\n");
    for (int t : {1, 2, 4, 8, 16, 32, 64}) perf(512, 2048, t, 16, iters);
    std::printf("# --- down N=2048 K=512 ---\n");
    for (int t : {1, 2, 4, 8, 16, 32, 64}) perf(2048, 512, t, 16, iters);
    (void)kPeakGBs;
    return 0;
}

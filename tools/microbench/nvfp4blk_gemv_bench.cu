// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Isolated single-kernel micro-bench for the dense NVFP4 decode GEMV
// (`matmul_nvfp4blk_vec`) — M-Munin/Bragi roadmap 5.18.6, the sanctioned,
// zero-reboot-risk replacement for an "ncu probe" (see doc/perf-notes-gb10.md).
//
// It does the minimum: CUDA init -> deterministic buffers -> launch ONE kernel
// with the exact prod grid/block/dtype/layout -> CUDA-event timing -> effective
// GB/s -> exit. No HTTP / threads / model loader / KV / graph / serving. This
// is the baseline harness for the 5.18.3 TC-dense work: A/B the current
// CUDA-core GEMV against a future tensor-core dense kernel in isolation, on a
// one-GPU container, WITHOUT the full server and WITHOUT ncu.
//
// Values are irrelevant (this is a bandwidth bench, not a correctness test):
// the weight bytes moved per launch are the same regardless of content, and
// the M=1 GEMV is weight-bandwidth/latency bound (measured ~60 GB/s = 22% of
// the 273 GB/s GB10 peak — see the roofline note). The kernel is the exact
// prod source, compiled straight into this harness.
//
// Build:  cmake --build build-cuda --target microbench_nvfp4blk
// Run:    ./microbench_nvfp4blk               # built-in dense-projection sweep
//         ./microbench_nvfp4blk <K> <N> <iters>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The exact prod kernel (kernels/cuda/common/matmul_nvfp4blk_vec.cu) is
// compiled into this target; declare its launch prototype.
extern "C" __global__ void matmul_nvfp4blk_vec(const float*         X,
                                               const unsigned char* W,
                                               float*               Y,
                                               const int            K,
                                               const int            N);

// Must match the kernel's launch geometry (kernels/.../matmul_nvfp4blk_vec.cu:
// MATMUL_NVBLK_LOCAL=128, 4 warps = 4 outputs/group; super = 32 elems / 20 B).
static constexpr int kBlock            = 128;
static constexpr int kOutputsPerGroup  = 4;
static constexpr int kSuperElements    = 32;
static constexpr int kSuperBytes       = 20;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// One (K, N) shape: bandwidth of the M=1 dense NVFP4 GEMV.
void benchShape(const char* label, int K, int N, int iters) {
    if ((K % kSuperElements) != 0) {
        std::fprintf(stderr, "skip %s: K=%d not a multiple of %d\n",
                     label, K, kSuperElements);
        return;
    }
    const std::size_t nSuperPerRow = static_cast<std::size_t>(K) / kSuperElements;
    const std::size_t wBytes = static_cast<std::size_t>(N) * nSuperPerRow * kSuperBytes;
    const std::size_t xBytes = static_cast<std::size_t>(K) * sizeof(float);
    const std::size_t yBytes = static_cast<std::size_t>(N) * sizeof(float);

    // Deterministic host fill (content irrelevant for a bandwidth bench; set
    // the 2x fp16 block-scale of each super-block to 1.0 so nothing is NaN).
    std::vector<float>         hX(static_cast<std::size_t>(K));
    std::vector<unsigned char> hW(wBytes);
    for (int i = 0; i < K; ++i) hX[i] = 1.0f / static_cast<float>(i + 1);
    for (std::size_t s = 0; s < wBytes; s += kSuperBytes) {
        // bytes [0..3] = two fp16 scales (0x3C00 = 1.0), [4..19] = 16 nibbles.
        hW[s + 0] = 0x00; hW[s + 1] = 0x3C;
        hW[s + 2] = 0x00; hW[s + 3] = 0x3C;
        for (int b = 4; b < kSuperBytes; ++b) {
            hW[s + b] = static_cast<unsigned char>((s + b) * 131u + 7u);
        }
    }

    float*         dX = nullptr;
    unsigned char* dW = nullptr;
    float*         dY = nullptr;
    ck(cudaMalloc(&dX, xBytes), "malloc X");
    ck(cudaMalloc(&dW, wBytes), "malloc W");
    ck(cudaMalloc(&dY, yBytes), "malloc Y");
    ck(cudaMemcpy(dX, hX.data(), xBytes, cudaMemcpyHostToDevice), "H2D X");
    ck(cudaMemcpy(dW, hW.data(), wBytes, cudaMemcpyHostToDevice), "H2D W");

    const unsigned int nGroups =
        static_cast<unsigned int>((N + kOutputsPerGroup - 1) / kOutputsPerGroup);

    auto launch = [&] {
        matmul_nvfp4blk_vec<<<nGroups, kBlock>>>(dX, dW, dY, K, N);
    };

    // Warmup.
    for (int i = 0; i < 10; ++i) launch();
    ck(cudaDeviceSynchronize(), "warmup sync");
    ck(cudaGetLastError(), "warmup launch");

    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "event0");
    ck(cudaEventCreate(&t1), "event1");
    ck(cudaEventRecord(t0), "record0");
    for (int i = 0; i < iters; ++i) launch();
    ck(cudaEventRecord(t1), "record1");
    ck(cudaEventSynchronize(t1), "sync t1");

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    const double usPer = (ms * 1000.0) / iters;
    const double bytes = static_cast<double>(wBytes + xBytes + yBytes);
    const double gbps  = bytes / (usPer * 1e-6) / 1e9;
    const double pctPeak = gbps / 273.0 * 100.0;

    std::printf("| %-10s | %6d | %6d | %8.2f | %6.0f | %5.1f%% | %6.1f |\n",
                label, K, N, usPer, gbps, pctPeak,
                static_cast<double>(wBytes) / 1e6);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(dX);
    cudaFree(dW);
    cudaFree(dY);
}

} // namespace

int main(int argc, char** argv) {
    int dev = 0;
    ck(cudaSetDevice(dev), "setDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "getProps");
    std::printf("# nvfp4blk_gemv micro-bench — %s (sm_%d%d), M=1 dense NVFP4 GEMV\n",
                prop.name, prop.major, prop.minor);
    std::printf("| shape      |      K |      N |  us/call |  GB/s | %%peak | W(MB) |\n");
    std::printf("|------------|--------|--------|----------|-------|-------|-------|\n");

    if (argc >= 3) {
        const int K     = std::atoi(argv[1]);
        const int N     = std::atoi(argv[2]);
        const int iters = argc >= 4 ? std::atoi(argv[3]) : 200;
        benchShape("custom", K, N, iters);
        return 0;
    }

    // Built-in sweep: Qwen3.6-35B-class dense-projection shapes (hidden 2048).
    const int iters = 200;
    benchShape("o-proj",  2048,   2048, iters);   // attn output / gdn out proj
    benchShape("qkv",     2048,   6144, iters);   // fused QKV-ish
    benchShape("ffn-shex",2048,    512, iters);   // shared-expert gate/up
    benchShape("ffn-up",  2048,  17408, iters);   // dense FFN width (qwen3.8)
    benchShape("lm_head", 2048, 152064, iters);   // vocab projection
    return 0;
}

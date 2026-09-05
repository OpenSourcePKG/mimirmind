// gdn_chunk_tc_bench — isolated chunked-GDN prefill K2 probe (5.21.9 v4).
// Links the REAL scalar and tensor-core batched chunk-forward kernels and
// drives them at the prod serving-prefill shape (H=32, S=128, C=64, uniform
// T, seqT=nullptr) so the two can be timed head-to-head in isolation and the
// TC kernel can be ncu-profiled SAFELY per the ncu-canary runbook (single
// kernel, disposable container, prod stopped).
//
// a0 content is random smallish values — TIMING is shape-driven, not
// value-driven (correctness lives in cuda_parity_tests).
//
// Run: ./microbench_gdn_chunk_tc [nSeq=8] [T=512] [iters=50]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

extern "C" __global__ void deltanet_chunk_forward_batched(
    const float* q, const float* k, const float* v, const float* gCum,
    const float* beta, const float* a0, float* state, float* out,
    float* scratch, int T, int H, int nSeq, int C,
    const unsigned char* activeMask, const int* seqT, const int* seqOff);

extern "C" __global__ void deltanet_chunk_forward_batched_tc(
    const float* q, const float* k, const float* v, const float* gCum,
    const float* beta, const float* a0, float* state, float* out,
    float* scratch, int T, int H, int nSeq, int C,
    const unsigned char* activeMask, const int* seqT, const int* seqOff);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what,
                     cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const int nSeq  = (argc > 1) ? std::atoi(argv[1]) : 8;
    const int T     = (argc > 2) ? std::atoi(argv[2]) : 512;
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 50;
    const int H = 32, S = 128, C = 64;
    const int maxChunks = (T + C - 1) / C;
    const int G = 96 < nSeq * H ? 96 : nSeq * H;

    const std::size_t actElems   = (std::size_t)nSeq * T * H * S;
    const std::size_t gateElems  = (std::size_t)nSeq * T * H;
    const std::size_t stateElems = (std::size_t)nSeq * H * S * S;
    const std::size_t a0Elems    = (std::size_t)nSeq * maxChunks * H * C * C;
    const std::size_t scrElems   = (std::size_t)96 * 7 * C * S;

    std::vector<float> h(actElems);
    unsigned s = 0x1234u;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (float)((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    };
    for (auto& x : h) x = rnd();

    float *dQ, *dK, *dV, *dG, *dB, *dA0, *dSt, *dOut, *dScr;
    ck(cudaMalloc(&dQ, actElems * 4), "q");
    ck(cudaMalloc(&dK, actElems * 4), "k");
    ck(cudaMalloc(&dV, actElems * 4), "v");
    ck(cudaMalloc(&dG, gateElems * 4), "g");
    ck(cudaMalloc(&dB, gateElems * 4), "b");
    ck(cudaMalloc(&dA0, a0Elems * 4), "a0");
    ck(cudaMalloc(&dSt, stateElems * 4), "st");
    ck(cudaMalloc(&dOut, actElems * 4), "out");
    ck(cudaMalloc(&dScr, scrElems * 4), "scr");
    ck(cudaMemcpy(dQ, h.data(), actElems * 4, cudaMemcpyHostToDevice), "cq");
    ck(cudaMemcpy(dK, h.data(), actElems * 4, cudaMemcpyHostToDevice), "ck");
    ck(cudaMemcpy(dV, h.data(), actElems * 4, cudaMemcpyHostToDevice), "cv");
    // gLog <= 0 (decay), beta in (0, 0.5), a0/state smallish.
    {
        std::vector<float> hg(gateElems), hb(gateElems);
        for (std::size_t i = 0; i < gateElems; ++i) {
            hg[i] = -std::abs(rnd());
            hb[i] = 0.25f * (rnd() + 0.5f);
        }
        ck(cudaMemcpy(dG, hg.data(), gateElems * 4, cudaMemcpyHostToDevice), "cg");
        ck(cudaMemcpy(dB, hb.data(), gateElems * 4, cudaMemcpyHostToDevice), "cb");
        std::vector<float> ha(a0Elems);
        for (auto& x : ha) x = 0.1f * rnd();
        ck(cudaMemcpy(dA0, ha.data(), a0Elems * 4, cudaMemcpyHostToDevice), "ca");
        std::vector<float> hs(stateElems, 0.01f);
        ck(cudaMemcpy(dSt, hs.data(), stateElems * 4, cudaMemcpyHostToDevice), "cs");
    }

    const std::size_t smem = (std::size_t)S * S * 4;
    ck(cudaFuncSetAttribute(deltanet_chunk_forward_batched,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "smem-sc");
    ck(cudaFuncSetAttribute(deltanet_chunk_forward_batched_tc,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "smem-tc");

    auto bench = [&](const char* name, bool tc) {
        auto launch = [&]() {
            if (tc) {
                deltanet_chunk_forward_batched_tc<<<G, 256, smem>>>(
                    dQ, dK, dV, dG, dB, dA0, dSt, dOut, dScr,
                    T, H, nSeq, C, nullptr, nullptr, nullptr);
            } else {
                deltanet_chunk_forward_batched<<<G, S, smem>>>(
                    dQ, dK, dV, dG, dB, dA0, dSt, dOut, dScr,
                    T, H, nSeq, C, nullptr, nullptr, nullptr);
            }
        };
        for (int i = 0; i < 5; ++i) launch();
        ck(cudaDeviceSynchronize(), "warmup");
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(b);
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("| %-6s | nSeq=%-3d T=%-5d | %9.1f us/launch | %6.2f us/tok |\n",
                    name, nSeq, T, us, us / ((double)nSeq * T));
        cudaEventDestroy(a); cudaEventDestroy(b);
    };

    std::printf("# gdn_chunk_tc_bench H=%d S=%d C=%d G=%d iters=%d\n",
                H, S, C, G, iters);
    bench("scalar", false);
    bench("tc",     true);
    return 0;
}

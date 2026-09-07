// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// I-6 parity: verify the moe_topk e512 kernel variant (MOE_TOPK_MAX_EXPERTS=512,
// PER=16 experts/lane) computes top-10-of-512 routing bit-for-bit vs a host
// reference — the exact routing qwen4_exp uses and the ONLY MoE piece the
// standard cuda_parity_tests (small nExperts) never exercise.
//
// Reference math (Qwen4ExpTextTopKRouter, norm_topk_prob=True):
//   p    = softmax(logits)              over ALL 512 experts
//   keep = top_k(p, 10)                 descending, ties -> low index
//   w    = p[keep] / sum(p[keep])       renormalised over the 10 kept
//
// Build (on box, in the cuda builder container):
//   nvcc -O2 -DMOE_TOPK_MAX_EXPERTS=512 -arch=native \
//        tools/microbench/q4e_moe_route_parity.cu -o /tmp/q4e_route && /tmp/q4e_route
//
// The kernel source is compiled straight in with the 512 define, so this
// exercises the identical selection/renorm logic the engine's e512 PTX runs.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// Pull in the kernel under test with the 512-expert define.
#include "../../kernels/cuda/llm/moe_topk.cu"

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            std::printf("CUDA error %s @ %s:%d\n", cudaGetErrorString(e_),       \
                        __FILE__, __LINE__);                                     \
            return 2;                                                            \
        }                                                                        \
    } while (0)

// Host reference — mirrors the kernel's exp(logit-max) selection so ties break
// the same way (highest prob, lowest index).
static void refRoute(const float* logits, int nE, int K, float wScale,
                     std::vector<int>& idx, std::vector<float>& w) {
    float m = -3.402823466e+38f;
    for (int e = 0; e < nE; ++e) m = std::max(m, logits[e]);
    std::vector<float> ex(nE);
    for (int e = 0; e < nE; ++e) ex[e] = std::exp(logits[e] - m);
    std::vector<char> taken(nE, 0);
    idx.assign(K, 0);
    std::vector<float> selEx(K, 0.0f);
    float keptSum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float lb = -1.0f;
        int lbi = 0x7fffffff;
        for (int e = 0; e < nE; ++e) {
            if (!taken[e] && ex[e] > lb) {   // strict > -> lowest index on ties
                lb = ex[e];
                lbi = e;
            }
        }
        taken[lbi] = 1;
        idx[k] = lbi;
        selEx[k] = lb;
        keptSum += lb;
    }
    w.assign(K, 0.0f);
    float inv = keptSum > 0.0f ? 1.0f / keptSum : 1.0f;
    for (int k = 0; k < K; ++k) w[k] = selEx[k] * inv * wScale;
}

int main() {
    const int T = 16, nE = 512, K = 10;
    const float wScale = 1.0f;

    std::mt19937 rng(1234);                 // fixed seed (no Math.random in scope)
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> logits(static_cast<size_t>(T) * nE);
    for (auto& v : logits) v = nd(rng);

    // A few tokens with deliberate ties (equal logits) to stress tie-break.
    for (int e = 100; e < 110; ++e) logits[static_cast<size_t>(3) * nE + e] = 5.0f;

    float* dL = nullptr;
    int* dI = nullptr;
    float* dW = nullptr;
    CK(cudaMalloc(&dL, logits.size() * sizeof(float)));
    CK(cudaMalloc(&dI, static_cast<size_t>(T) * K * sizeof(int)));
    CK(cudaMalloc(&dW, static_cast<size_t>(T) * K * sizeof(float)));
    CK(cudaMemcpy(dL, logits.data(), logits.size() * sizeof(float),
                  cudaMemcpyHostToDevice));

    moe_topk<<<T, 32>>>(dL, dI, dW, nE, K, wScale);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<int> hI(static_cast<size_t>(T) * K);
    std::vector<float> hW(static_cast<size_t>(T) * K);
    CK(cudaMemcpy(hI.data(), dI, hI.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hW.data(), dW, hW.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int idxMismatch = 0;
    float wMaxAbs = 0.0f;
    for (int t = 0; t < T; ++t) {
        std::vector<int> ri;
        std::vector<float> rw;
        refRoute(logits.data() + static_cast<size_t>(t) * nE, nE, K, wScale, ri, rw);
        for (int k = 0; k < K; ++k) {
            if (hI[static_cast<size_t>(t) * K + k] != ri[k]) {
                if (idxMismatch < 8)
                    std::printf("  t=%d k=%d idx eng=%d ref=%d\n", t, k,
                                hI[static_cast<size_t>(t) * K + k], ri[k]);
                ++idxMismatch;
            }
            wMaxAbs = std::max(wMaxAbs,
                               std::fabs(hW[static_cast<size_t>(t) * K + k] - rw[k]));
        }
    }

    std::printf("[q4e-moe-route] T=%d nE=%d K=%d idxMismatch=%d wMaxAbs=%.3e -> %s\n",
                T, nE, K, idxMismatch, wMaxAbs,
                (idxMismatch == 0 && wMaxAbs < 1e-6f) ? "PASS" : "FAIL");

    cudaFree(dL);
    cudaFree(dI);
    cudaFree(dW);
    return (idxMismatch == 0 && wMaxAbs < 1e-6f) ? 0 : 1;
}

// paged_prefill_attn_bench — Option-A gate: (paged-KV gather + cuDNN SDPA) vs the
// CUDA-core paged prefill-causal kernel, at identical single-seq paged geometry
// (roadmap prefill-attention lever). Answers: is gather+cuDNN materially faster
// than moe... the attn.paged wall, and how big is the gather overhead?
//
// Links the REAL kernels: paged_attention_prefill_causal (F32, the serving-prefill
// wall) + CudnnSdpaPrefill (the exact wrapper the integration would call). Pool
// layout [num_blocks, block_size, num_kv_heads, head_size]; block table maps
// position p -> (blk=bt[p/bs], slot=p%bs). Query/out [T_q, num_heads, head_size].
// Shape = Qwen3.6: num_heads=16, num_kv_heads=2, head_size=256, block_size=16.
// No prod swap; CUDA-event timing.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/cuda/CudnnSdpaPrefill.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <functional>

using mimirmind::compute::cuda::CudnnSdpaPrefill;

// The real serving-prefill wall kernel (F32 variant; key/value are void*).
extern "C" __global__ void paged_attention_prefill_causal(
    float* out, const float* query, const void* key_cache, const void* value_cache,
    const int* block_tables, const int* seqT, const int* queryOff, const int* startPos,
    int num_seqs, int num_heads, int num_kv_heads, int head_size, int block_size,
    int max_num_blocks_per_seq, float scale, float softcap);

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA err (%s): %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// Gather paged K or V pool -> contiguous [T_kv, nKvHeads, headSize] (cuDNN layout).
__global__ void gatherKv(const float* __restrict__ pool, const int* __restrict__ bt,
                         float* __restrict__ dst, int Tkv, int nKvHeads,
                         int headSize, int blockSize) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(Tkv) * nKvHeads * headSize;
    if (i >= total) return;
    const int d   = i % headSize;
    const int hkv = (i / headSize) % nKvHeads;
    const int p   = i / (static_cast<long>(headSize) * nKvHeads);
    const int blk = bt[p / blockSize];
    const int slot = p % blockSize;
    const long src = (static_cast<long>(blk * blockSize + slot) * nKvHeads + hkv)
                     * headSize + d;
    dst[i] = pool[src];
}

static float timeKernel(const std::function<void()>& fn, int iters) {
    for (int i = 0; i < 10; ++i) fn();
    ck(cudaDeviceSynchronize(), "warm");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b); ck(cudaEventSynchronize(b), "ev");
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

int main(int argc, char** argv) {
    const int nHeads = 16, nKvHeads = 2, headSize = 256, blockSize = 16;
    const int Tq = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 100;
    const float scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    std::printf("# paged_prefill_attn_bench on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("# shape: nHeads=%d nKvHeads=%d headSize=%d blockSize=%d Tq=%d\n",
                nHeads, nKvHeads, headSize, blockSize, Tq);
    std::printf("| Tkv | paged-CUDA-core | gather | cuDNN | gather+cuDNN | speedup | maxRel |\n");
    std::printf("|---|---|---|---|---|---|---|\n");

    CudnnSdpaPrefill cudnn;
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);

    for (int Tkv : {512, 1024, 1536, 2048}) {
        if (Tkv < Tq) continue;
        const int nBlocks = (Tkv + blockSize - 1) / blockSize;
        const int startPos = Tkv - Tq;
        const size_t poolElems = static_cast<size_t>(nBlocks) * blockSize * nKvHeads * headSize;
        const size_t qElems = static_cast<size_t>(Tq) * nHeads * headSize;
        const size_t kvcElems = static_cast<size_t>(Tkv) * nKvHeads * headSize;

        std::vector<float> hKpool(poolElems), hVpool(poolElems), hQ(qElems);
        for (auto& v : hKpool) v = nd(rng);
        for (auto& v : hVpool) v = nd(rng);
        for (auto& v : hQ) v = nd(rng);
        std::vector<int> hBt(nBlocks); for (int i = 0; i < nBlocks; ++i) hBt[i] = i;  // contiguous
        const int hSeqT = Tq, hQoff = 0, hStart = startPos;

        float *dKpool, *dVpool, *dQ, *dOutB, *dOutA, *dKc, *dVc;
        int *dBt, *dSeqT, *dQoff, *dStart;
        ck(cudaMalloc(&dKpool, poolElems*4), "kp"); ck(cudaMalloc(&dVpool, poolElems*4), "vp");
        ck(cudaMalloc(&dQ, qElems*4), "q"); ck(cudaMalloc(&dOutB, qElems*4), "ob");
        ck(cudaMalloc(&dOutA, qElems*4), "oa");
        ck(cudaMalloc(&dKc, kvcElems*4), "kc"); ck(cudaMalloc(&dVc, kvcElems*4), "vc");
        ck(cudaMalloc(&dBt, nBlocks*4), "bt"); ck(cudaMalloc(&dSeqT, 4), "st");
        ck(cudaMalloc(&dQoff, 4), "qo"); ck(cudaMalloc(&dStart, 4), "sp");
        ck(cudaMemcpy(dKpool, hKpool.data(), poolElems*4, cudaMemcpyHostToDevice), "ckp");
        ck(cudaMemcpy(dVpool, hVpool.data(), poolElems*4, cudaMemcpyHostToDevice), "cvp");
        ck(cudaMemcpy(dQ, hQ.data(), qElems*4, cudaMemcpyHostToDevice), "cq");
        ck(cudaMemcpy(dBt, hBt.data(), nBlocks*4, cudaMemcpyHostToDevice), "cbt");
        ck(cudaMemcpy(dSeqT, &hSeqT, 4, cudaMemcpyHostToDevice), "cst");
        ck(cudaMemcpy(dQoff, &hQoff, 4, cudaMemcpyHostToDevice), "cqo");
        ck(cudaMemcpy(dStart, &hStart, 4, cudaMemcpyHostToDevice), "csp");

        // Path B: CUDA-core paged prefill causal.
        const unsigned smem = (2*headSize + 128) * sizeof(float);
        auto runB = [&]() {
            dim3 grid(nHeads, 1, Tq), blk(128);
            paged_attention_prefill_causal<<<grid, blk, smem>>>(
                dOutB, dQ, dKpool, dVpool, dBt, dSeqT, dQoff, dStart,
                1, nHeads, nKvHeads, headSize, blockSize, nBlocks, scale, 0.0f);
        };
        // Path A: gather + cuDNN.
        const int gthr = 256; const long gtot = kvcElems;
        auto runGather = [&]() {
            gatherKv<<<(gtot+gthr-1)/gthr, gthr>>>(dKpool, dBt, dKc, Tkv, nKvHeads, headSize, blockSize);
            gatherKv<<<(gtot+gthr-1)/gthr, gthr>>>(dVpool, dBt, dVc, Tkv, nKvHeads, headSize, blockSize);
        };
        auto runCudnn = [&]() {
            cudnn.runF32Causal(nullptr, dQ, dKc, dVc, dOutA, Tq, Tkv, nHeads, nKvHeads, headSize, scale);
        };

        runB(); runGather(); const bool cudnnOk = cudnn.runF32Causal(nullptr, dQ, dKc, dVc, dOutA, Tq, Tkv, nHeads, nKvHeads, headSize, scale);
        ck(cudaDeviceSynchronize(), "prime");
        if (!cudnnOk) { std::printf("| %d | cuDNN runF32Causal returned FALSE (unsupported shape?) |\n", Tkv); continue; }

        const float msB = timeKernel(runB, iters);
        const float msG = timeKernel(runGather, iters);
        const float msC = timeKernel(runCudnn, iters);
        const float msA = msG + msC;

        // Parity: outB (f32 kernel) vs outA (cuDNN bf16).
        std::vector<float> oB(qElems), oA(qElems);
        ck(cudaMemcpy(oB.data(), dOutB, qElems*4, cudaMemcpyDeviceToHost), "doB");
        ck(cudaMemcpy(oA.data(), dOutA, qElems*4, cudaMemcpyDeviceToHost), "doA");
        double sqErr=0, sqRef=0; for (size_t i=0;i<qElems;++i){ double d=oB[i]-oA[i]; sqErr+=d*d; sqRef+=(double)oB[i]*oB[i]; }
        const double l2rel = std::sqrt(sqErr/(sqRef+1e-30));

        std::printf("| %-4d | %7.4f ms | %6.4f ms | %6.4f ms | %7.4f ms | %5.2fx | L2rel=%.3g |\n",
                    Tkv, msB, msG, msC, msA, msB/msA, l2rel);

        cudaFree(dKpool); cudaFree(dVpool); cudaFree(dQ); cudaFree(dOutB); cudaFree(dOutA);
        cudaFree(dKc); cudaFree(dVc); cudaFree(dBt); cudaFree(dSeqT); cudaFree(dQoff); cudaFree(dStart);
    }
    return 0;
}

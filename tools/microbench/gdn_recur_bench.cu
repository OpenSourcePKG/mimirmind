// gdn_recur_bench — isolated GatedDeltaNet DECODE recurrence roofline probe
// (roadmap 5.18.10 Inc-1). Drives the REAL prod decode kernel
// `gated_deltanet_ar_batched_v3_gatefused` (kernels/cuda/llm/gated_deltanet_ar_batched_v3.cu)
// at the serving-decode shape (T=1, one timestep) to settle whether gdn.recur
// (~16% of the conc64 decode step, the #2 term after MoE) is at the SSM-state
// bandwidth floor (=> the only lever is shrinking state bytes, e.g. BF16 state)
// or has kernel headroom.
//
// At T=1 the kernel reads its [S,S] state global->smem, runs one timestep, and
// writes [S,S] back — one full state read + write per (seq, head). That mandatory
// traffic (S*S*hV*4 bytes/seq, R+W) dominates; q/k/v/out are ~tens of KiB. So the
// achieved GB/s vs the 273 GB/s GB10 ceiling is the gdn.recur efficiency.
//
// No prod swap (bare microbench); no ncu. Build: microbench_gdn_recur.
// Run: ./microbench_gdn_recur [iters]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" __global__ void gated_deltanet_ar_batched_v3_gatefused(
    const float* q, const float* k, const float* v, const float* alpha,
    const float* betaRaw, const float* ssmA, const float* ssmDt,
    float* state, float* out, int T, int H, int S,
    const unsigned char* activeMask, const int* seqT, const int* seqOff);

static constexpr float kPeakGBs = 273.0f;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Bench the recurrence at (nSeq, H, S, T). bytesState = 2 * nSeq*H*S*S*4 (one
// read + one write of the whole state, held in smem across all T steps); io =
// T rows of q/k/v/out + alpha/beta. T=1 = decode; T>1 = the PREFILL chunk
// shape (5.21.8 floor-check): the SAME kernel runs T sequential steps per
// block, so state traffic amortises over T while FLOPs grow ~T — at prefill T
// the kernel leaves the bandwidth roof and the question becomes whether it
// reaches the COMPUTE roof or is serial-latency-bound (headroom for a
// chunked-parallel rewrite).
static void benchShape(int nSeq, int H, int S, int iters, int T = 1) {
    const std::size_t stateElems = (std::size_t)nSeq * H * S * S;
    const std::size_t ioSH       = (std::size_t)nSeq * T * H * S;   // q/k/v/out each
    const std::size_t gate       = (std::size_t)nSeq * T * H;       // alpha/beta each

    std::vector<float> hState(stateElems, 0.01f), hIO(ioSH, 0.1f),
        hGate(gate, 0.2f), hHead(H, -0.5f);

    float *dState=nullptr,*dQ=nullptr,*dK=nullptr,*dV=nullptr,*dA=nullptr,
          *dB=nullptr,*dSsmA=nullptr,*dSsmDt=nullptr,*dOut=nullptr;
    ck(cudaMalloc(&dState, stateElems*sizeof(float)), "state");
    ck(cudaMalloc(&dQ, ioSH*sizeof(float)), "q"); ck(cudaMalloc(&dK, ioSH*sizeof(float)), "k");
    ck(cudaMalloc(&dV, ioSH*sizeof(float)), "v"); ck(cudaMalloc(&dOut, ioSH*sizeof(float)), "out");
    ck(cudaMalloc(&dA, gate*sizeof(float)), "a"); ck(cudaMalloc(&dB, gate*sizeof(float)), "b");
    ck(cudaMalloc(&dSsmA, H*sizeof(float)), "ssmA"); ck(cudaMalloc(&dSsmDt, H*sizeof(float)), "ssmDt");
    ck(cudaMemcpy(dState, hState.data(), stateElems*sizeof(float), cudaMemcpyHostToDevice), "cS");
    ck(cudaMemcpy(dQ, hIO.data(), ioSH*sizeof(float), cudaMemcpyHostToDevice), "cQ");
    ck(cudaMemcpy(dK, hIO.data(), ioSH*sizeof(float), cudaMemcpyHostToDevice), "cK");
    ck(cudaMemcpy(dV, hIO.data(), ioSH*sizeof(float), cudaMemcpyHostToDevice), "cV");
    ck(cudaMemcpy(dA, hGate.data(), gate*sizeof(float), cudaMemcpyHostToDevice), "cA");
    ck(cudaMemcpy(dB, hGate.data(), gate*sizeof(float), cudaMemcpyHostToDevice), "cB");
    ck(cudaMemcpy(dSsmA, hHead.data(), H*sizeof(float), cudaMemcpyHostToDevice), "cSsmA");
    ck(cudaMemcpy(dSsmDt, hHead.data(), H*sizeof(float), cudaMemcpyHostToDevice), "cSsmDt");

    const std::size_t smemBytes = (std::size_t)S * S * sizeof(float);
    dim3 grid((unsigned)H, (unsigned)nSeq, 1);
    dim3 block((unsigned)S, 1, 1);
    auto launch = [&]() {
        gated_deltanet_ar_batched_v3_gatefused<<<grid, block, smemBytes>>>(
            dQ, dK, dV, dA, dB, dSsmA, dSsmDt, dState, dOut, T, H, S,
            nullptr, nullptr, nullptr);
    };
    for (int i = 0; i < 10; ++i) launch();
    ck(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    ck(cudaEventSynchronize(b), "sync");
    float ms=0.0f; cudaEventElapsedTime(&ms, a, b);

    const double us       = (double)ms * 1000.0 / iters;
    const double stateBytes = 2.0 * (double)stateElems * sizeof(float);   // R + W
    const double ioBytes    = (4.0*(double)ioSH + 2.0*(double)gate + 2.0*H) * sizeof(float);
    const double bytes    = stateBytes + ioBytes;
    const double gbps     = bytes / (us*1e-6) / 1e9;
    const double pct      = gbps / kPeakGBs * 100.0;
    const double idealUs  = bytes / (kPeakGBs*1e9) * 1e6;
    if (T == 1) {
        std::printf("| nSeq=%-3d H=%d S=%d | %8.2f us | %6.1f GB/s | %5.1f%% | state=%5.0f MiB R+W | roof %6.1fus |\n",
                    nSeq, H, S, us, gbps, pct, stateBytes/1048576.0, idealUs);
    } else {
        // Prefill row: report both roofs plus the launch's parallelism.
        // FLOP model per (head, step): ~8*S^2 (delta-rule update + readout),
        // matching the two S-loops in the kernel body.
        cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
        int clockKHz = 0;   // cudaDeviceProp::clockRate was removed in CUDA 13
        cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
        const double flops   = (double)nSeq * H * T * 8.0 * S * S;
        const double gflops  = flops / (us*1e-6) / 1e9;
        // fp32 peak approx: SMs * 128 lanes * 2 (FMA) * clock.
        const double peakGf  = (double)prop.multiProcessorCount * 128.0 * 2.0
                             * ((double)clockKHz * 1e3) / 1e9;
        const int    blocks  = H * nSeq;
        std::printf("| nSeq=%-3d T=%-5d | %9.1f us | %6.1f GB/s (%4.1f%%BW) | %7.1f GF/s (%4.1f%%fp32) | %4d blk/%d SM | %6.2f us/tok |\n",
                    nSeq, T, us, gbps, pct, gflops, 100.0*gflops/peakGf,
                    blocks, prop.multiProcessorCount,
                    us / ((double)nSeq * T));
    }

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dState); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
    cudaFree(dA); cudaFree(dB); cudaFree(dSsmA); cudaFree(dSsmDt);
}

int main(int argc, char** argv) {
    int dev=0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    const int H=32, S=128;                       // qwen3.6-35b-a3b: hV=32, headDim=128
    const std::size_t smemBytes=(std::size_t)S*S*sizeof(float);   // 64 KiB
    // >48 KiB dynamic smem requires the opt-in.
    ck(cudaFuncSetAttribute(gated_deltanet_ar_batched_v3_gatefused,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes), "smem opt-in");
    std::printf("# gdn_recur_bench on %s (sm_%d%d), peak=%.0f GB/s\n",
                prop.name, prop.major, prop.minor, (double)kPeakGBs);
    std::printf("# v3_gatefused, decode T=1, prod dims H=%d S=%d; state=S*S*H*4=%.1f MiB/seq\n",
                H, S, (double)S*S*H*4/1048576.0);
    std::printf("| shape | us/call | GB/s | %%peak | state traffic | ideal |\n");
    std::printf("|---|---|---|---|---|---|\n");
    const int iters = (argc>1) ? std::atoi(argv[1]) : 500;
    // Small-nSeq rows (added 5.18.10.1): single/low-user decode. Measured
    // 2026-09-03: ALREADY at/above the DRAM floor (nSeq=1 ~130% of peak via L2
    // residency; 2-8 at 81-118%) — there is NO small-nSeq occupancy lever; a
    // column-split variant (grid.z over state columns, bit-identical) was
    // built, measured SLOWER at nSeq=1 (15.3 vs 12.0 us) and reverted. The
    // only remaining recur lever at any nSeq is BF16 state (halves R+W).
    // NOTE: co-resident prod suppresses these numbers ~25% — solo-GPU runs are
    // the comparable ones for the 16-64 rows (73-75% solo vs 51-61% co-resident).
    for (int nSeq : {1, 2, 4, 8}) benchShape(nSeq, H, S, iters);
    benchShape(16, H, S, iters);
    benchShape(32, H, S, iters);
    benchShape(64, H, S, iters);
    std::printf("# %%peak reading+writing the SSM state once = recur BW efficiency. If ~high (>70%%),\n");
    std::printf("# gdn.recur is at the F32 state-bandwidth floor -> the lever is BF16 state (halves R+W).\n");

    // PREFILL rows (5.21.8 floor-check): same kernel, T = chunk tokens. State
    // traffic amortises over T, so the bandwidth roof is irrelevant here; the
    // question is achieved FLOP/s vs the fp32 roof. Far from BOTH roofs =>
    // the T-sequential chain (2 __syncthreads per token, H*nSeq blocks of S
    // threads) is latency-bound => a chunked-parallel rewrite has headroom.
    std::printf("\n# PREFILL shapes (T>1, uniform seqT): serving multi-slot chunked prefill.\n");
    std::printf("| shape | us/call | traffic | compute | parallelism | per-token |\n");
    std::printf("|---|---|---|---|---|---|\n");
    const int pIters = (argc>2) ? std::atoi(argv[2]) : 20;
    for (int T : {128, 256, 512, 1024, 2048}) benchShape(1, H, S, pIters, T);
    for (int T : {128, 256, 512})             benchShape(4, H, S, pIters, T);
    for (int T : {128, 256, 512})             benchShape(16, H, S, pIters, T);
    std::printf("# If %%fp32 is low AND %%BW is low, gdn.recur prefill is serial/latency-bound\n");
    std::printf("# -> chunked delta-rule (intra-chunk parallel, FLA-style) is the lever, NOT BF16 state.\n");
    return 0;
}

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

// Bench the recurrence at (nSeq, H, S), T=1 decode. bytesState = 2 * nSeq*H*S*S*4
// (one read + one write of the whole state); io = q/k/v/out + alpha/beta.
static void benchShape(int nSeq, int H, int S, int iters, std::size_t smemBytes) {
    const std::size_t stateElems = (std::size_t)nSeq * H * S * S;
    const std::size_t ioSH       = (std::size_t)nSeq * H * S;   // q/k/v/out each
    const std::size_t gate       = (std::size_t)nSeq * H;       // alpha/beta each

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

    dim3 grid((unsigned)H, (unsigned)nSeq, 1);
    dim3 block((unsigned)S, 1, 1);
    auto launch = [&]() {
        gated_deltanet_ar_batched_v3_gatefused<<<grid, block, smemBytes>>>(
            dQ, dK, dV, dA, dB, dSsmA, dSsmDt, dState, dOut, 1, H, S,
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
    std::printf("| nSeq=%-3d H=%d S=%d | %8.2f us | %6.1f GB/s | %5.1f%% | state=%5.0f MiB R+W | roof %6.1fus |\n",
                nSeq, H, S, us, gbps, pct, stateBytes/1048576.0, idealUs);

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
    benchShape(16, H, S, iters, smemBytes);
    benchShape(32, H, S, iters, smemBytes);
    benchShape(64, H, S, iters, smemBytes);
    std::printf("# %%peak reading+writing the SSM state once = recur BW efficiency. If ~high (>70%%),\n");
    std::printf("# gdn.recur is at the F32 state-bandwidth floor -> the lever is BF16 state (halves R+W).\n");
    return 0;
}

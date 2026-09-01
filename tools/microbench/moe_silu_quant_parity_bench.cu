// moe_silu_quant_parity_bench — parity + perf for the fused silu-mul+act-quant
// MoE-plumbing kernel (roadmap 5.21.8). Compares:
//   REF   : silu_mul (whole maxPad intermediate) + moe_act_quant_nvfp4_rows
//   FUSED : moe_silu_mul_quant_nvfp4_rows (one pass, real rows only)
// Byte-compares nibbles + swizzled SF (expected BIT-IDENTICAL) and times both.
// Links the REAL kernels. No prod swap; CUDA-event timing.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <functional>

extern "C" __global__ void silu_mul(float*, const float*, int);
extern "C" __global__ void moe_act_quant_nvfp4_rows(
    const float*, unsigned char*, unsigned char*, float, const int*, int, int);
extern "C" __global__ void moe_silu_mul_quant_nvfp4_rows(
    const float*, const float*, unsigned char*, unsigned char*, float,
    const int*, int, int);

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA err %s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
static float timeit(const std::function<void()>& fn, int it) {
    for (int i=0;i<10;++i) fn(); ck(cudaDeviceSynchronize(),"warm");
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int i=0;i<it;++i) fn(); cudaEventRecord(b); ck(cudaEventSynchronize(b),"ev");
    float ms=0; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}

int main(int argc, char** argv) {
    const int nRows  = (argc>1)?std::atoi(argv[1]):512;   // real rows (R)
    const int K      = (argc>2)?std::atoi(argv[2]):768;   // n_ff_exp (mult 16)
    const int iters  = (argc>3)?std::atoi(argv[3]):200;
    // Engine pads EVERY expert to 128 rows: maxPad = R + nExperts*128. The 2-pass
    // siluMul runs over the whole maxPad (padding included); the fused runs over R.
    const int nExperts = (argc>4)?std::atoi(argv[4]):256;
    const int maxPad = nRows + nExperts*128;             // realistic padded rows
    const int nBlk   = K/16;
    const int rowsTiles = (maxPad+127)/128, kTiles=(nBlk+3)/4;
    const size_t sfBytes = (size_t)rowsTiles*kTiles*512;
    const size_t elems = (size_t)maxPad*K, outBytes=(size_t)maxPad*K/2;
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    std::printf("# moe_silu_quant_parity on %s: nRows=%d K=%d maxPad=%d\n", p.name, nRows, K, maxPad);

    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1.5f);
    std::vector<float> hG(elems), hU(elems);
    for (auto& v:hG) v=nd(rng); for (auto& v:hU) v=nd(rng);
    std::vector<int> hMap(nRows); for (int i=0;i<nRows;++i) hMap[i]=i;  // real rows 0..nRows-1

    float *dG,*dU; unsigned char *dOut,*dSF; int* dMap;
    ck(cudaMalloc(&dG,elems*4),"g"); ck(cudaMalloc(&dU,elems*4),"u");
    ck(cudaMalloc(&dOut,outBytes),"o"); ck(cudaMalloc(&dSF,sfBytes),"sf");
    ck(cudaMalloc(&dMap,nRows*4),"m");
    ck(cudaMemcpy(dU,hU.data(),elems*4,cudaMemcpyHostToDevice),"cu");
    ck(cudaMemcpy(dMap,hMap.data(),nRows*4,cudaMemcpyHostToDevice),"cm");

    dim3 qgrid(nRows,(nBlk+255)/256), qblk(256);
    auto resetG = [&]{ ck(cudaMemcpy(dG,hG.data(),elems*4,cudaMemcpyHostToDevice),"cg"); };

    // ---- FUSED first (non-destructive read of dG) ----
    resetG();
    ck(cudaMemset(dOut,0,outBytes),"z"); ck(cudaMemset(dSF,0,sfBytes),"zsf");
    moe_silu_mul_quant_nvfp4_rows<<<qgrid,qblk>>>(dG,dU,dOut,dSF,1.0f,dMap,nRows,K);
    ck(cudaDeviceSynchronize(),"fused");
    std::vector<unsigned char> oF(outBytes),sF(sfBytes);
    ck(cudaMemcpy(oF.data(),dOut,outBytes,cudaMemcpyDeviceToHost),"df");
    ck(cudaMemcpy(sF.data(),dSF,sfBytes,cudaMemcpyDeviceToHost),"dsf");

    // ---- REF: silu_mul (destroys dG) + act_quant_rows ----
    resetG();
    ck(cudaMemset(dOut,0,outBytes),"z2"); ck(cudaMemset(dSF,0,sfBytes),"zsf2");
    silu_mul<<<(unsigned)((elems+255)/256),256>>>(dG,dU,(int)elems);
    moe_act_quant_nvfp4_rows<<<qgrid,qblk>>>(dG,dOut,dSF,1.0f,dMap,nRows,K);
    ck(cudaDeviceSynchronize(),"ref");
    std::vector<unsigned char> oR(outBytes),sR(sfBytes);
    ck(cudaMemcpy(oR.data(),dOut,outBytes,cudaMemcpyDeviceToHost),"dr");
    ck(cudaMemcpy(sR.data(),dSF,sfBytes,cudaMemcpyDeviceToHost),"dsr");

    size_t nibDiff=0,sfDiff=0;
    for (size_t i=0;i<outBytes;++i) if (oF[i]!=oR[i]) ++nibDiff;
    for (size_t i=0;i<sfBytes;++i)  if (sF[i]!=sR[i]) ++sfDiff;
    std::printf("PARITY: nibble byte-diffs=%zu/%zu  SF byte-diffs=%zu/%zu  %s\n",
                nibDiff,outBytes,sfDiff,sfBytes,
                (nibDiff==0 && sfDiff==0)?"PASS (bit-identical)":"FAIL");

    // ---- PERF ----
    auto ref = [&]{
        silu_mul<<<(unsigned)((elems+255)/256),256>>>(dG,dU,(int)elems);
        moe_act_quant_nvfp4_rows<<<qgrid,qblk>>>(dG,dOut,dSF,1.0f,dMap,nRows,K);
    };
    auto fused = [&]{
        moe_silu_mul_quant_nvfp4_rows<<<qgrid,qblk>>>(dG,dU,dOut,dSF,1.0f,dMap,nRows,K);
    };
    const float msRef=timeit(ref,iters), msF=timeit(fused,iters);
    std::printf("PERF: ref(silu+quant)=%.4f ms  fused=%.4f ms  %.2fx\n", msRef, msF, msRef/msF);
    return 0;
}

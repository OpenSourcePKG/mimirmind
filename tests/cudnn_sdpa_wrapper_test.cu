// Parity test for CudnnSdpaPrefill vs a CPU F32 causal-GQA reference, in the
// POSITION-major [pos, head, dim] layout the engine actually uses. Covers both
// a first chunk (T_kv==T_q, plain causal) and a continuation chunk
// (T_kv>T_q, bottom-right causal). Built only under MIMIRMIND_ENABLE_CUDNN.
#include "compute/cuda/CudnnSdpaPrefill.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using mimirmind::compute::cuda::CudnnSdpaPrefill;

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e));return 90;}}while(0)

// One (T_q, T_kv) case. Returns 0 on pass.
static int run_case(CudnnSdpaPrefill& sdpa, int T_q, int T_kv, int H, int Hkv, int D) {
    const float scale = 1.0f/std::sqrt((float)D);
    const int off = T_kv - T_q;   // absolute position of query row 0
    printf("--- case T_q=%d T_kv=%d (off=%d, %s) ---\n", T_q, T_kv, off,
           off==0 ? "plain causal" : "bottom-right causal");

    std::mt19937 rng(7 + T_kv); std::normal_distribution<float> nd(0,1);
    const size_t nQ=(size_t)T_q*H*D, nKV=(size_t)T_kv*Hkv*D;
    std::vector<float> hQ(nQ),hK(nKV),hV(nKV),hO(nQ);
    for(auto&x:hQ)x=nd(rng); for(auto&x:hK)x=nd(rng); for(auto&x:hV)x=nd(rng);

    float *dQ,*dK,*dV,*dO;
    CK(cudaMalloc(&dQ,nQ*4)); CK(cudaMalloc(&dK,nKV*4)); CK(cudaMalloc(&dV,nKV*4)); CK(cudaMalloc(&dO,nQ*4));
    CK(cudaMemcpy(dQ,hQ.data(),nQ*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,hK.data(),nKV*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,hV.data(),nKV*4,cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    bool ok = sdpa.runF32Causal(s, dQ,dK,dV,dO, T_q,T_kv,H,Hkv,D, scale);
    CK(cudaStreamSynchronize(s));
    if(!ok){ printf("RESULT: FAIL (runF32Causal returned false)\n"); return 1; }
    CK(cudaMemcpy(hO.data(),dO,nQ*4,cudaMemcpyDeviceToHost));

    // POSITION-major indexing: q[t][h][d]=(t*H+h)*D+d ; kv[t][h][d]=(t*Hkv+h)*D+d
    auto qi =[&](int t,int h,int k){return ((size_t)t*H+h)*D+k;};
    auto kvi=[&](int t,int h,int k){return ((size_t)t*Hkv+h)*D+k;};
    double maxAbs=0; int nan=0; std::vector<float> sc(T_kv);
    for(int h=0;h<H;++h){ int hk=h/(H/Hkv);
        for(int i=0;i<T_q;++i){ int kmax=off+i;   // inclusive last key (bottom-right causal)
            float m=-1e30f;
            for(int j=0;j<=kmax;++j){ float d=0; for(int k=0;k<D;++k) d+=hQ[qi(i,h,k)]*hK[kvi(j,hk,k)]; sc[j]=d*scale; if(sc[j]>m)m=sc[j]; }
            float sum=0; for(int j=0;j<=kmax;++j){ sc[j]=std::exp(sc[j]-m); sum+=sc[j]; }
            for(int k=0;k<D;++k){ float o=0; for(int j=0;j<=kmax;++j) o+=sc[j]/sum*hV[kvi(j,hk,k)];
                float a=hO[qi(i,h,k)]; if(std::isnan(a)||std::isinf(a))nan++; double ad=std::fabs(a-o); if(ad>maxAbs)maxAbs=ad; }
        }
    }
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaStreamDestroy(s);
    printf("NaN/Inf: %d   max abs diff: %.4g   O[0..3]: %.4f %.4f %.4f %.4f\n",
           nan, maxAbs, hO[0],hO[1],hO[2],hO[3]);
    bool pass=(nan==0)&&(maxAbs<0.03);
    printf("%s\n", pass?"  -> PASS":"  -> FAIL");
    return pass?0:2;
}

int main(int argc, char** argv) {
    const int H=16, Hkv=2, D=256;
    CudnnSdpaPrefill sdpa;
    int rc=0;
    rc |= run_case(sdpa, 256, 256,  H,Hkv,D);   // first chunk: plain causal
    rc |= run_case(sdpa, 512, 512,  H,Hkv,D);   // first chunk, larger
    rc |= run_case(sdpa, 512, 1024, H,Hkv,D);   // continuation chunk 2 (off=512)
    rc |= run_case(sdpa, 512, 2560, H,Hkv,D);   // continuation chunk 5 (off=2048)
    printf("\nOVERALL: %s\n", rc==0 ? "PASS (position-major + bottom-right causal correct)" : "FAIL");
    return rc==0?0:2;
}

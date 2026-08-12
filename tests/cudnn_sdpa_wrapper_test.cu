// Parity test for CudnnSdpaPrefill (F32-in/F32-out wrapper) vs a CPU F32
// causal-GQA softmax reference. Standalone: no engine deps beyond the wrapper.
// Built only under MIMIRMIND_ENABLE_CUDNN.
#include "compute/cuda/CudnnSdpaPrefill.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using mimirmind::compute::cuda::CudnnSdpaPrefill;

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e));return 90;}}while(0)

int main(int argc, char** argv) {
    const int T = (argc>1)?atoi(argv[1]):256, H=16, Hkv=2, D=256;
    const float scale = 1.0f/std::sqrt((float)D);
    printf("CudnnSdpaPrefill wrapper test: T=%d H=%d Hkv=%d D=%d (F32 in/out, causal GQA)\n",T,H,Hkv,D);

    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
    const size_t nQ=(size_t)T*H*D, nK=(size_t)T*Hkv*D;
    std::vector<float> hQ(nQ),hK(nK),hV(nK),hO(nQ);
    for(auto&x:hQ)x=nd(rng); for(auto&x:hK)x=nd(rng); for(auto&x:hV)x=nd(rng);

    float *dQ,*dK,*dV,*dO;
    CK(cudaMalloc(&dQ,nQ*4)); CK(cudaMalloc(&dK,nK*4)); CK(cudaMalloc(&dV,nK*4)); CK(cudaMalloc(&dO,nQ*4));
    CK(cudaMemcpy(dQ,hQ.data(),nQ*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,hK.data(),nK*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,hV.data(),nK*4,cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    CudnnSdpaPrefill sdpa;
    bool ok = sdpa.runF32Causal(s, dQ,dK,dV,dO, T,H,Hkv,D, scale);
    CK(cudaStreamSynchronize(s));
    if(!ok){ printf("RESULT: FAIL (runF32Causal returned false — cuDNN error/unsupported)\n"); return 1; }
    CK(cudaMemcpy(hO.data(),dO,nQ*4,cudaMemcpyDeviceToHost));

    // CPU F32 causal GQA reference
    auto qi=[&](int h,int i,int k){return ((size_t)h*T+i)*D+k;};
    auto kvi=[&](int h,int j,int k){return ((size_t)h*T+j)*D+k;};
    double maxAbs=0; int nan=0;
    std::vector<float> sc(T);
    for(int h=0;h<H;++h){ int hk=h/(H/Hkv);
        for(int i=0;i<T;++i){ float m=-1e30f;
            for(int j=0;j<=i;++j){ float d=0; for(int k=0;k<D;++k) d+=hQ[qi(h,i,k)]*hK[kvi(hk,j,k)]; sc[j]=d*scale; if(sc[j]>m)m=sc[j]; }
            float sum=0; for(int j=0;j<=i;++j){ sc[j]=std::exp(sc[j]-m); sum+=sc[j]; }
            for(int k=0;k<D;++k){ float o=0; for(int j=0;j<=i;++j) o+=sc[j]/sum*hV[kvi(hk,j,k)];
                float a=hO[qi(h,i,k)]; if(std::isnan(a)||std::isinf(a))nan++; double ad=std::fabs(a-o); if(ad>maxAbs)maxAbs=ad; }
        }
    }
    printf("NaN/Inf: %d   max abs diff: %.4g\n", nan, maxAbs);
    printf("O[0..3]: %.4f %.4f %.4f %.4f\n", hO[0],hO[1],hO[2],hO[3]);
    bool pass = (nan==0) && (maxAbs<0.03);
    printf("RESULT: %s\n", pass?"PASS (wrapper F32->bf16->SDPA->F32 correct)":"FAIL");
    return pass?0:2;
}

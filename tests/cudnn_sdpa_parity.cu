// cuDNN 9 SDPA D256 GQA causal smoke test for GB10 / sm_121a.
// Answers: (a) does the graph finalize an engine for consumer Blackwell,
// (b) does execute() run on sm_121 without wrong-arch garbage,
// (c) is it numerically correct vs a CPU reference (bf16 tolerance).
#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <unordered_map>

namespace fe = cudnn_frontend;

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e));return 90;} } while(0)
#define ST(stage, expr) do { auto s=(expr); if(!s.is_good()){printf("[FAIL @ %s] %s\n", stage, s.get_message().c_str()); return 1;} printf("[ ok ] %s\n", stage);} while(0)

int main(int argc, char** argv) {
    const int64_t b=1, h_q=16, h_kv=2, d=256;
    const int64_t s = (argc>1) ? atol(argv[1]) : 512;
    const bool doRef = (s <= 1024);   // CPU reference is O(s^2 d h) — only for small s
    const float scale = 1.0f/std::sqrt((float)d);
    printf("cuDNN SDPA smoke: b=%ld h_q=%ld h_kv=%ld s=%ld d=%ld (GQA %ld:1, causal, bf16)\n",
           b,h_q,h_kv,s,d,h_q/h_kv);
    int devId=0; cudaDeviceProp prop; CK(cudaGetDevice(&devId)); CK(cudaGetDeviceProperties(&prop,devId));
    printf("device: %s  sm_%d%d\n", prop.name, prop.major, prop.minor);
    printf("cudnn runtime version: %zu\n", (size_t)cudnnGetVersion());

    // ---- build graph ----
    fe::graph::Graph g;
    g.set_io_data_type(fe::DataType_t::BFLOAT16)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);
    auto Q = g.tensor(fe::graph::Tensor_attributes().set_name("Q")
                .set_dim({b,h_q,s,d}).set_stride({h_q*s*d, s*d, d, 1}));
    auto K = g.tensor(fe::graph::Tensor_attributes().set_name("K")
                .set_dim({b,h_kv,s,d}).set_stride({h_kv*s*d, s*d, d, 1}));
    auto V = g.tensor(fe::graph::Tensor_attributes().set_name("V")
                .set_dim({b,h_kv,s,d}).set_stride({h_kv*s*d, s*d, d, 1}));
    auto attrs = fe::graph::SDPA_attributes().set_name("sdpa")
                    .set_is_inference(true).set_causal_mask(true).set_attn_scale(scale);
    auto out = g.sdpa(Q, K, V, attrs);
    auto O = out[0];
    O->set_output(true).set_dim({b,h_q,s,d}).set_stride({h_q*s*d, s*d, d, 1});

    cudnnHandle_t handle; if(cudnnCreate(&handle)){printf("cudnnCreate failed\n");return 91;}
    ST("validate",              g.validate());
    ST("build_operation_graph", g.build_operation_graph(handle));
    ST("create_execution_plans",g.create_execution_plans({fe::HeurMode_t::A}));
    ST("check_support",         g.check_support(handle));   // <-- sm_121 D256 support gate
    ST("build_plans",           g.build_plans(handle));

    int64_t wsSize=0; ST("get_workspace_size", g.get_workspace_size(wsSize));
    printf("workspace bytes: %ld\n", wsSize);

    // ---- host data (float), bf16-rounded ----
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f);
    auto nQ=b*h_q*s*d, nK=b*h_kv*s*d, nV=nK, nO=nQ;
    std::vector<float> hQf(nQ), hKf(nK), hVf(nV);
    std::vector<__nv_bfloat16> hQ(nQ), hK(nK), hV(nV);
    auto fill=[&](std::vector<float>&f, std::vector<__nv_bfloat16>&h){
        for(size_t i=0;i<f.size();++i){ __nv_bfloat16 b=__float2bfloat16(nd(rng)); h[i]=b; f[i]=__bfloat162float(b);} };
    fill(hQf,hQ); fill(hKf,hK); fill(hVf,hV);

    void *dQ,*dK,*dV,*dO,*dWs=nullptr;
    CK(cudaMalloc(&dQ,nQ*2)); CK(cudaMalloc(&dK,nK*2)); CK(cudaMalloc(&dV,nV*2)); CK(cudaMalloc(&dO,nO*2));
    if(wsSize>0) CK(cudaMalloc(&dWs,wsSize));
    CK(cudaMemcpy(dQ,hQ.data(),nQ*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,hK.data(),nK*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,hV.data(),nV*2,cudaMemcpyHostToDevice));

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack =
        {{Q,dQ},{K,dK},{V,dV},{O,dO}};
    ST("execute", g.execute(handle, pack, dWs));
    CK(cudaDeviceSynchronize());

    // ---- rough latency: warmup + timed loop (one attn layer, one seq) ----
    for(int i=0;i<10;++i){ g.execute(handle, pack, dWs); } CK(cudaDeviceSynchronize());
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int iters=100; CK(cudaEventRecord(t0));
    for(int i=0;i<iters;++i){ g.execute(handle, pack, dWs); }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    const double perCall = ms/iters;
    printf("latency: %.4f ms / SDPA call (s=%ld, d=%ld, %ld q-heads; 1 layer)\n", perCall, s, d, h_q);
    printf("PROJECTION: x10 full-attn layers = %.3f ms total prefill attn  (vs P3.a scalar 237 ms)\n",
           perCall*10.0);

    std::vector<__nv_bfloat16> hOb(nO); CK(cudaMemcpy(hOb.data(),dO,nO*2,cudaMemcpyDeviceToHost));
    std::vector<float> hO(nO); for(size_t i=0;i<hOb.size();++i) hO[i]=__bfloat162float(hOb[i]);

    if(!doRef){
        int nan=0; for(float x: hO) if(std::isnan(x)||std::isinf(x)) nan++;
        printf("NaN/Inf outputs: %d (numeric parity validated separately at s<=1024)\n", nan);
        printf("\nRESULT: %s\n", nan==0 ? "OK (ran clean at prompt scale; latency above)"
                                        : "FAIL (NaN/Inf at prompt scale)");
        return nan==0 ? 0 : 2;
    }

    // ---- CPU reference: causal GQA softmax over bf16-rounded inputs ----
    auto qIdx=[&](int64_t h,int64_t i,int64_t k){ return ((h*s)+i)*d + k; };
    auto kvIdx=[&](int64_t h,int64_t j,int64_t k){ return ((h*s)+j)*d + k; };
    std::vector<float> ref(nO);
    double maxAbs=0, maxRel=0; int nanCount=0;
    for(int64_t h=0; h<h_q; ++h){ int64_t hk=h/(h_q/h_kv);
        for(int64_t i=0;i<s;++i){
            float m=-1e30f; std::vector<float> sc(i+1);
            for(int64_t j=0;j<=i;++j){ float dot=0; for(int64_t k=0;k<d;++k) dot+=hQf[qIdx(h,i,k)]*hKf[kvIdx(hk,j,k)];
                sc[j]=dot*scale; if(sc[j]>m)m=sc[j]; }
            float sum=0; for(int64_t j=0;j<=i;++j){ sc[j]=std::exp(sc[j]-m); sum+=sc[j]; }
            for(int64_t k=0;k<d;++k){ float o=0; for(int64_t j=0;j<=i;++j) o+=sc[j]/sum*hVf[kvIdx(hk,j,k)];
                ref[qIdx(h,i,k)]=o; }
        }
    }
    for(size_t idx=0; idx<ref.size(); ++idx){ float a=hO[idx], r=ref[idx];
        if(std::isnan(a)||std::isinf(a)) nanCount++;
        double ad=std::fabs(a-r);
        if(ad>maxAbs)maxAbs=ad;
        if(std::fabs(r)>0.05){ double rd=ad/std::fabs(r); if(rd>maxRel)maxRel=rd; } }
    printf("\n=== numeric check vs CPU ref (bf16) ===\n");
    printf("NaN/Inf outputs: %d\n", nanCount);
    // rel diff only where the reference is not near-zero (avoids /~0 false-fail)
    printf("max abs diff: %.4g   max rel diff (|ref|>0.05 only): %.4g\n", maxAbs, maxRel);
    printf("sample O[0,0,0..3]: %.4f %.4f %.4f %.4f  ref: %.4f %.4f %.4f %.4f\n",
           hO[0],hO[1],hO[2],hO[3], ref[0],ref[1],ref[2],ref[3]);
    // bf16 attention outputs are O(1): absolute diff is the right metric.
    bool pass = (nanCount==0) && (maxAbs < 0.03);
    printf("\nRESULT: %s\n", pass ? "PASS (cuDNN D256 SDPA runs + correct on this GPU)"
                                  : "FAIL (garbage / wrong-arch / unsupported)");
    return pass ? 0 : 2;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// cuDNN 9 SDPA prefill-attention wrapper — see CudnnSdpaPrefill.hpp.

#include "compute/cuda/CudnnSdpaPrefill.hpp"

#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace mimirmind::compute::cuda {

namespace {

__global__ void castF32ToBf16(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                              std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ void castBf16ToF32(const __nv_bfloat16* __restrict__ in, float* __restrict__ out,
                              std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

struct CachedGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O;
    int64_t workspaceBytes{0};
};

}  // namespace

struct CudnnSdpaPrefill::Impl {
    cudnnHandle_t handle{nullptr};
    bool handleOk{false};
    std::unordered_map<uint64_t, CachedGraph> cache;

    // bf16 + workspace device scratch (grown on demand).
    __nv_bfloat16 *dQ{nullptr}, *dK{nullptr}, *dV{nullptr}, *dO{nullptr};
    std::size_t qCap{0}, kCap{0}, vCap{0}, oCap{0};   // K and V need SEPARATE caps
    void*       dWs{nullptr};
    std::size_t wsCap{0};

    Impl() {
        if (cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS) handleOk = true;
    }
    ~Impl() {
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if (dV) cudaFree(dV);
        if (dO) cudaFree(dO);
        if (dWs) cudaFree(dWs);
        if (handle) cudnnDestroy(handle);
    }

    static uint64_t key(int T_q, int T_kv, int nHeads, int nKvHeads, int headDim) {
        // T_q,T_kv <= 8192 (13b); nHeads,nKvHeads <= 127 (7b); headDim <= 1023 (10b)
        return ((uint64_t)(uint32_t)T_q     << 37) |
               ((uint64_t)(uint32_t)T_kv    << 24) |
               ((uint64_t)(uint32_t)nHeads  << 17) |
               ((uint64_t)(uint32_t)nKvHeads << 10) |
                (uint64_t)(uint32_t)headDim;
    }

    template <typename T>
    static bool grow(T*& ptr, std::size_t& cap, std::size_t need) {
        if (need <= cap) return true;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        if (cudaMalloc(&ptr, need * sizeof(T)) != cudaSuccess) { cap = 0; return false; }
        cap = need;
        return true;
    }

    CachedGraph* getOrBuild(int T_q, int T_kv, int nHeads, int nKvHeads, int headDim,
                            float scale) {
        uint64_t k = key(T_q, T_kv, nHeads, nKvHeads, headDim);
        auto it = cache.find(k);
        if (it != cache.end()) return &it->second;

        CachedGraph cg;
        cg.graph = std::make_shared<fe::graph::Graph>();
        auto& g  = *cg.graph;
        g.set_io_data_type(fe::DataType_t::BFLOAT16)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
        const int64_t b = 1;
        const int64_t D = headDim;
        // POSITION-major physical layout: Q is [T_q, nHeads, D], K/V are
        // [T_kv, nKvHeads, D]. Logical cuDNN dims are [b, h, s, d]; the strides
        // describe the physical layout: head stride = D, seq stride = h*D.
        cg.Q = g.tensor(fe::graph::Tensor_attributes().set_name("Q")
                    .set_dim({b, nHeads, T_q, D})
                    .set_stride({(int64_t)nHeads*T_q*D, D, (int64_t)nHeads*D, 1}));
        cg.K = g.tensor(fe::graph::Tensor_attributes().set_name("K")
                    .set_dim({b, nKvHeads, T_kv, D})
                    .set_stride({(int64_t)nKvHeads*T_kv*D, D, (int64_t)nKvHeads*D, 1}));
        cg.V = g.tensor(fe::graph::Tensor_attributes().set_name("V")
                    .set_dim({b, nKvHeads, T_kv, D})
                    .set_stride({(int64_t)nKvHeads*T_kv*D, D, (int64_t)nKvHeads*D, 1}));
        // Bottom-right causal: the T_q queries are the last rows of the T_kv
        // range (query i at absolute pos T_kv-T_q+i attends keys [0, that pos]).
        // Reduces to plain causal when T_kv == T_q.
        auto attrs = fe::graph::SDPA_attributes().set_name("sdpa")
                        .set_is_inference(true).set_causal_mask_bottom_right(true)
                        .set_attn_scale(scale);
        auto outs = g.sdpa(cg.Q, cg.K, cg.V, attrs);
        cg.O = outs[0];
        cg.O->set_output(true).set_dim({b, nHeads, T_q, D})
             .set_stride({(int64_t)nHeads*T_q*D, D, (int64_t)nHeads*D, 1});

        if (g.validate().is_bad())               return nullptr;
        if (g.build_operation_graph(handle).is_bad()) return nullptr;
        if (g.create_execution_plans({fe::HeurMode_t::A}).is_bad()) return nullptr;
        if (g.check_support(handle).is_bad())     return nullptr;
        if (g.build_plans(handle).is_bad())       return nullptr;
        if (g.get_workspace_size(cg.workspaceBytes).is_bad()) return nullptr;

        auto res = cache.emplace(k, std::move(cg));
        return &res.first->second;
    }
};

CudnnSdpaPrefill::CudnnSdpaPrefill()  : _impl(new Impl()) {}
CudnnSdpaPrefill::~CudnnSdpaPrefill() { delete _impl; }

bool CudnnSdpaPrefill::runF32Causal(void* stream,
                                    const float* q, const float* k, const float* v, float* out,
                                    int T_q, int T_kv, int nHeads, int nKvHeads, int headDim,
                                    float scale) {
    Impl& I = *_impl;
    if (!I.handleOk) return false;
    if (T_kv < T_q) return false;   // K/V must cover the query range
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    const std::size_t nQ  = (std::size_t)T_q  * nHeads   * headDim;   // Q / O
    const std::size_t nKV = (std::size_t)T_kv * nKvHeads * headDim;   // K / V (full range)
    if (!Impl::grow(I.dQ, I.qCap, nQ) || !Impl::grow(I.dK, I.kCap, nKV) ||
        !Impl::grow(I.dV, I.vCap, nKV) || !Impl::grow(I.dO, I.oCap, nQ))
        return false;

    auto launchCastTo = [&](const float* in, __nv_bfloat16* o, std::size_t n) {
        const int tpb = 256; const std::size_t blocks = (n + tpb - 1) / tpb;
        castF32ToBf16<<<(unsigned)blocks, tpb, 0, s>>>(in, o, n);
    };
    launchCastTo(q, I.dQ, nQ);
    launchCastTo(k, I.dK, nKV);
    launchCastTo(v, I.dV, nKV);

    CachedGraph* cg = I.getOrBuild(T_q, T_kv, nHeads, nKvHeads, headDim, scale);
    if (!cg) return false;
    if (cg->workspaceBytes > 0 &&
        !Impl::grow(reinterpret_cast<char*&>(I.dWs), I.wsCap, (std::size_t)cg->workspaceBytes))
        return false;

    if (cudnnSetStream(I.handle, s) != CUDNN_STATUS_SUCCESS) return false;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
        {cg->Q, I.dQ}, {cg->K, I.dK}, {cg->V, I.dV}, {cg->O, I.dO}};
    if (cg->graph->execute(I.handle, pack, I.dWs).is_bad()) return false;

    const int tpb = 256; const std::size_t blocks = (nQ + tpb - 1) / tpb;
    castBf16ToF32<<<(unsigned)blocks, tpb, 0, s>>>(I.dO, out, nQ);
    return true;
}

}  // namespace mimirmind::compute::cuda

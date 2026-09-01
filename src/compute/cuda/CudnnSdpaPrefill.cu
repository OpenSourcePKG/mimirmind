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
// Write the two actual seqlens on-device (values marshaled as launch args, so no
// host buffer + no per-call cudaStreamSynchronize — the old sync serialized every
// call, which dominated the per-seq serving prefill loop).
__global__ void setSeqLens(int* __restrict__ seqQ, int* __restrict__ seqKV,
                           int tq, int tkv) {
    *seqQ  = tq;
    *seqKV = tkv;
}
__global__ void castBf16ToF32(const __nv_bfloat16* __restrict__ in, float* __restrict__ out,
                              std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

struct CachedGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O, seqQ, seqKV;
    int64_t workspaceBytes{0};
};

// Fixed max sequence length the single ragged graph is built for. Covers
// maxContextTokens=8192; requests beyond fall back to the hand kernel.
static constexpr int kSmax = 8192;

}  // namespace

struct CudnnSdpaPrefill::Impl {
    cudnnHandle_t handle{nullptr};
    bool handleOk{false};
    std::unordered_map<uint64_t, CachedGraph> cache;

    // bf16 + workspace device scratch (grown on demand).
    __nv_bfloat16 *dQ{nullptr}, *dK{nullptr}, *dV{nullptr}, *dO{nullptr};
    std::size_t qCap{0}, kCap{0}, vCap{0}, oCap{0};   // K and V need SEPARATE caps
    int32_t     *dSeqQ{nullptr}, *dSeqKV{nullptr};    // per-call actual seqlens (device)
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
        if (dSeqQ) cudaFree(dSeqQ);
        if (dSeqKV) cudaFree(dSeqKV);
        if (dWs) cudaFree(dWs);
        if (handle) cudnnDestroy(handle);
    }

    // One ragged graph per attention shape (nHeads,nKvHeads,headDim); the
    // variable T_q/T_kv are passed at execute via seq_len tensors, so no
    // per-length rebuild.
    static uint64_t key(int nHeads, int nKvHeads, int headDim, int sBucket) {
        return ((uint64_t)(uint32_t)sBucket << 32) |
               ((uint64_t)(uint32_t)nHeads  << 20) |
               ((uint64_t)(uint32_t)nKvHeads << 12) |
                (uint64_t)(uint32_t)headDim;
    }

    // Round T_kv up to a power-of-2 bucket in [1024, kSmax]. A graph built for a
    // tight bucket picks a far better plan than the fixed-8192 one (~5-15x on the
    // attention term, measured), while keeping the cache to a handful of graphs.
    static int bucketS(int Tkv) {
        int s = 1024;
        while (s < Tkv && s < kSmax) s <<= 1;
        return s;
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

    CachedGraph* getOrBuild(int nHeads, int nKvHeads, int headDim, float scale,
                            int sBucket) {
        uint64_t k = key(nHeads, nKvHeads, headDim, sBucket);
        auto it = cache.find(k);
        if (it != cache.end()) return &it->second;

        CachedGraph cg;
        cg.graph = std::make_shared<fe::graph::Graph>();
        auto& g  = *cg.graph;
        g.set_io_data_type(fe::DataType_t::BFLOAT16)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
        const int64_t b = 1, S = sBucket;
        const int64_t D = headDim;
        // ONE ragged graph built for the max seqlen S. Actual per-call T_q/T_kv
        // come via seq_len tensors + padding mask, so there is no per-length
        // rebuild. POSITION-major physical layout ([pos,head,dim]): head
        // stride = D, seq stride = heads*D. Only rows [0,T_q)/[0,T_kv) hold
        // valid data; the padding mask + bottom-right causal exclude the rest.
        cg.Q = g.tensor(fe::graph::Tensor_attributes().set_name("Q")
                    .set_dim({b, nHeads, S, D})
                    .set_stride({(int64_t)nHeads*S*D, D, (int64_t)nHeads*D, 1}));
        cg.K = g.tensor(fe::graph::Tensor_attributes().set_name("K")
                    .set_dim({b, nKvHeads, S, D})
                    .set_stride({(int64_t)nKvHeads*S*D, D, (int64_t)nKvHeads*D, 1}));
        cg.V = g.tensor(fe::graph::Tensor_attributes().set_name("V")
                    .set_dim({b, nKvHeads, S, D})
                    .set_stride({(int64_t)nKvHeads*S*D, D, (int64_t)nKvHeads*D, 1}));
        cg.seqQ = g.tensor(fe::graph::Tensor_attributes().set_name("seq_q")
                    .set_dim({b,1,1,1}).set_stride({1,1,1,1})
                    .set_data_type(fe::DataType_t::INT32));
        cg.seqKV = g.tensor(fe::graph::Tensor_attributes().set_name("seq_kv")
                    .set_dim({b,1,1,1}).set_stride({1,1,1,1})
                    .set_data_type(fe::DataType_t::INT32));
        // Bottom-right causal + padding: query i (i<T_q) attends keys
        // [0, T_kv-T_q+i]; rows >= seq_len are padding. Covers a first chunk
        // (T_kv==T_q, plain causal) and continuation chunks (T_kv>T_q).
        auto attrs = fe::graph::SDPA_attributes().set_name("sdpa")
                        .set_is_inference(true).set_causal_mask_bottom_right(true)
                        .set_padding_mask(true)
                        .set_seq_len_q(cg.seqQ).set_seq_len_kv(cg.seqKV)
                        .set_attn_scale(scale);
        auto outs = g.sdpa(cg.Q, cg.K, cg.V, attrs);
        cg.O = outs[0];
        cg.O->set_output(true).set_dim({b, nHeads, S, D})
             .set_stride({(int64_t)nHeads*S*D, D, (int64_t)nHeads*D, 1});

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
    if (T_kv < T_q || T_kv > kSmax || T_q > kSmax) return false;  // K/V covers Q, within Smax
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    const std::size_t nQ  = (std::size_t)T_q  * nHeads   * headDim;   // valid Q / O rows
    const std::size_t nKV = (std::size_t)T_kv * nKvHeads * headDim;   // valid K / V rows
    // The ragged graph reads Smax rows -> scratch is Smax-sized (fixed, grown once).
    const std::size_t capQ  = (std::size_t)kSmax * nHeads   * headDim;
    const std::size_t capKV = (std::size_t)kSmax * nKvHeads * headDim;
    if (!Impl::grow(I.dQ, I.qCap, capQ) || !Impl::grow(I.dK, I.kCap, capKV) ||
        !Impl::grow(I.dV, I.vCap, capKV) || !Impl::grow(I.dO, I.oCap, capQ))
        return false;
    if (!I.dSeqQ  && cudaMalloc(&I.dSeqQ,  sizeof(int32_t)) != cudaSuccess) return false;
    if (!I.dSeqKV && cudaMalloc(&I.dSeqKV, sizeof(int32_t)) != cudaSuccess) return false;

    auto launchCastTo = [&](const float* in, __nv_bfloat16* o, std::size_t n) {
        const int tpb = 256; const std::size_t blocks = (n + tpb - 1) / tpb;
        castF32ToBf16<<<(unsigned)blocks, tpb, 0, s>>>(in, o, n);
    };
    launchCastTo(q, I.dQ, nQ);      // only the valid rows; rest is stale-but-finite (masked)
    launchCastTo(k, I.dK, nKV);
    launchCastTo(v, I.dV, nKV);

    // Actual seqlens (bottom-right causal + padding uses these). Set on-device via
    // a 1-thread kernel — no host buffer, no per-call stream sync.
    setSeqLens<<<1, 1, 0, s>>>(I.dSeqQ, I.dSeqKV, T_q, T_kv);

    CachedGraph* cg = I.getOrBuild(nHeads, nKvHeads, headDim, scale,
                                   Impl::bucketS(T_kv));
    if (!cg) return false;
    if (cg->workspaceBytes > 0 &&
        !Impl::grow(reinterpret_cast<char*&>(I.dWs), I.wsCap, (std::size_t)cg->workspaceBytes))
        return false;

    if (cudnnSetStream(I.handle, s) != CUDNN_STATUS_SUCCESS) return false;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
        {cg->Q, I.dQ}, {cg->K, I.dK}, {cg->V, I.dV}, {cg->O, I.dO},
        {cg->seqQ, I.dSeqQ}, {cg->seqKV, I.dSeqKV}};
    if (cg->graph->execute(I.handle, pack, I.dWs).is_bad()) return false;

    const int tpb = 256; const std::size_t blocks = (nQ + tpb - 1) / tpb;
    castBf16ToF32<<<(unsigned)blocks, tpb, 0, s>>>(I.dO, out, nQ);   // valid O rows only
    return true;
}

}  // namespace mimirmind::compute::cuda

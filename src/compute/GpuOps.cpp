#include "compute/GpuOps.hpp"

#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mimirmind::compute {

namespace {

std::int32_t toInt32(std::size_t v, const char* tag) {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            std::string{"GpuOps: "} + tag +
            " overflows int32 ("  + std::to_string(v) + ")");
    }
    return static_cast<std::int32_t>(v);
}

std::uint32_t groupsForN(std::size_t n, std::uint32_t local) {
    const std::size_t g = (n + local - 1) / local;
    if (g > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("GpuOps: workgroup count overflows uint32");
    }
    return static_cast<std::uint32_t>(g);
}

} // namespace

GpuOps::GpuOps(runtime::L0Context& ctx, runtime::CommandQueue& queue)
    : _ctx{ctx},
      _queue{queue},
      _rmsnormModule    {ctx, "rmsnorm"},
      _rmsnormKernel    {_rmsnormModule.kernel("rmsnorm")},
      _addBiasModule    {ctx, "add_bias"},
      _addBiasKernel    {_addBiasModule.kernel("add_bias")},
      _addResidualModule{ctx, "add_residual"},
      _addResidualKernel{_addResidualModule.kernel("add_residual")},
      _siluMulModule    {ctx, "silu_mul"},
      _siluMulKernel    {_siluMulModule.kernel("silu_mul")},
      _ropeModule       {ctx, "rope_inplace"},
      _ropeKernel       {_ropeModule.kernel("rope_inplace")},
      _mulScalarModule  {ctx, "mul_scalar"},
      _mulScalarKernel  {_mulScalarModule.kernel("mul_scalar")},
      _geluMulModule    {ctx, "gelu_mul"},
      _geluMulKernel    {_geluMulModule.kernel("gelu_mul")},
      _rmsnormGemmaModule{ctx, "rmsnorm_gemma"},
      _rmsnormGemmaKernel{_rmsnormGemmaModule.kernel("rmsnorm_gemma")},
      _rmsnormNoWeightModule{ctx, "rmsnorm_no_weight"},
      _rmsnormNoWeightKernel{_rmsnormNoWeightModule.kernel("rmsnorm_no_weight")},
      _ropeFfModule    {ctx, "rope_inplace_ff"},
      _ropeFfKernel    {_ropeFfModule.kernel("rope_inplace_ff")},
      _attentionModule {ctx, "attention"},
      _attentionKernel {_attentionModule.kernel("attention")}
{
    MM_LOG_INFO("gpuops",
                "GpuOps ready — rmsnorm/rmsnorm_gemma/rmsnorm_no_weight/"
                "add_bias/add_residual/silu_mul/rope/rope_ff/mul_scalar/"
                "gelu_mul/attention loaded (rms local={}, elementwise "
                "local={}, rope local={}, attention local={}, attention "
                "max T_k={})",
                kRmsnormLocalSize, kElementwiseLocalSize, kRopeLocalSize,
                kAttentionLocalSize, kAttentionMaxTk);
}

void GpuOps::rmsNormAsync(const float* x,
                          std::size_t  M,
                          std::size_t  K,
                          const float* weight,
                          float        eps,
                          float*       y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNorm K");
    _rmsnormKernel.setPtr(0, x);
    _rmsnormKernel.setPtr(1, weight);
    _rmsnormKernel.setPtr(2, y);
    _rmsnormKernel.setValue<float>(3, eps);
    _rmsnormKernel.setValue<std::int32_t>(4, Ki);
    _rmsnormKernel.setGroupSize(kRmsnormLocalSize, 1, 1);
    // One workgroup per row.
    _queue.appendLaunch(_rmsnormKernel,
                        static_cast<std::uint32_t>(M), 1, 1);
}

void GpuOps::rmsNormGemmaAsync(const float* x,
                               std::size_t  M,
                               std::size_t  K,
                               const float* weight,
                               float        eps,
                               float*       y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNormGemma K");
    _rmsnormGemmaKernel.setPtr(0, x);
    _rmsnormGemmaKernel.setPtr(1, weight);
    _rmsnormGemmaKernel.setPtr(2, y);
    _rmsnormGemmaKernel.setValue<float>(3, eps);
    _rmsnormGemmaKernel.setValue<std::int32_t>(4, Ki);
    _rmsnormGemmaKernel.setGroupSize(kRmsnormLocalSize, 1, 1);
    _queue.appendLaunch(_rmsnormGemmaKernel,
                        static_cast<std::uint32_t>(M), 1, 1);
}

void GpuOps::rmsNormNoWeightAsync(const float* x,
                                  std::size_t  M,
                                  std::size_t  K,
                                  float        eps,
                                  float*       y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNormNoWeight K");
    _rmsnormNoWeightKernel.setPtr(0, x);
    _rmsnormNoWeightKernel.setPtr(1, y);
    _rmsnormNoWeightKernel.setValue<float>(2, eps);
    _rmsnormNoWeightKernel.setValue<std::int32_t>(3, Ki);
    _rmsnormNoWeightKernel.setGroupSize(kRmsnormLocalSize, 1, 1);
    _queue.appendLaunch(_rmsnormNoWeightKernel,
                        static_cast<std::uint32_t>(M), 1, 1);
}

void GpuOps::addBiasAsync(float*       y,
                          std::size_t  M,
                          std::size_t  K,
                          const float* bias) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Mi = toInt32(M, "addBias M");
    const std::int32_t Ki = toInt32(K, "addBias K");
    _addBiasKernel.setPtr(0, y);
    _addBiasKernel.setPtr(1, bias);
    _addBiasKernel.setValue<std::int32_t>(2, Mi);
    _addBiasKernel.setValue<std::int32_t>(3, Ki);
    _addBiasKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_addBiasKernel,
                        groupsForN(M * K, kElementwiseLocalSize), 1, 1);
}

void GpuOps::addResidualAsync(float*       y,
                              const float* x,
                              std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "addResidual n");
    _addResidualKernel.setPtr(0, y);
    _addResidualKernel.setPtr(1, x);
    _addResidualKernel.setValue<std::int32_t>(2, ni);
    _addResidualKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_addResidualKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::siluMulAsync(float*       gate,
                          const float* up,
                          std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "siluMul n");
    _siluMulKernel.setPtr(0, gate);
    _siluMulKernel.setPtr(1, up);
    _siluMulKernel.setValue<std::int32_t>(2, ni);
    _siluMulKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_siluMulKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::mulScalarAsync(float*       y,
                            float        s,
                            std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "mulScalar n");
    _mulScalarKernel.setPtr(0, y);
    _mulScalarKernel.setValue<float>(1, s);
    _mulScalarKernel.setValue<std::int32_t>(2, ni);
    _mulScalarKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_mulScalarKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::geluMulAsync(float*       gate,
                          const float* up,
                          std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "geluMul n");
    _geluMulKernel.setPtr(0, gate);
    _geluMulKernel.setPtr(1, up);
    _geluMulKernel.setValue<std::int32_t>(2, ni);
    _geluMulKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_geluMulKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::ropeInPlaceAsync(float*       x,
                              std::size_t  seqLen,
                              std::size_t  numHeads,
                              std::size_t  headDim,
                              std::size_t  startPos,
                              float        base) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "GpuOps::ropeInPlace: headDim must be even");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    _ropeKernel.setPtr(0, x);
    _ropeKernel.setValue<std::int32_t>(1, toInt32(seqLen,   "rope seqLen"));
    _ropeKernel.setValue<std::int32_t>(2, toInt32(numHeads, "rope numHeads"));
    _ropeKernel.setValue<std::int32_t>(3, toInt32(headDim,  "rope headDim"));
    _ropeKernel.setValue<std::int32_t>(4, toInt32(startPos, "rope startPos"));
    _ropeKernel.setValue<float>(5, base);
    _ropeKernel.setGroupSize(kRopeLocalSize, 1, 1);
    _queue.appendLaunch(_ropeKernel,
                        groupsForN(total, kRopeLocalSize), 1, 1);
}

void GpuOps::ropeInPlaceWithFactorsAsync(float*       x,
                                         const float* freqFactors,
                                         std::size_t  seqLen,
                                         std::size_t  numHeads,
                                         std::size_t  headDim,
                                         std::size_t  startPos,
                                         float        base) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "GpuOps::ropeInPlaceWithFactors: headDim must be even");
    }
    if (freqFactors == nullptr) {
        throw std::runtime_error(
            "GpuOps::ropeInPlaceWithFactors: freqFactors is null");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    _ropeFfKernel.setPtr(0, x);
    _ropeFfKernel.setPtr(1, freqFactors);
    _ropeFfKernel.setValue<std::int32_t>(2, toInt32(seqLen,   "rope_ff seqLen"));
    _ropeFfKernel.setValue<std::int32_t>(3, toInt32(numHeads, "rope_ff numHeads"));
    _ropeFfKernel.setValue<std::int32_t>(4, toInt32(headDim,  "rope_ff headDim"));
    _ropeFfKernel.setValue<std::int32_t>(5, toInt32(startPos, "rope_ff startPos"));
    _ropeFfKernel.setValue<float>(6, base);
    _ropeFfKernel.setGroupSize(kRopeLocalSize, 1, 1);
    _queue.appendLaunch(_ropeFfKernel,
                        groupsForN(total, kRopeLocalSize), 1, 1);
}

void GpuOps::attentionAsync(const float* q,
                            const float* k,
                            const float* v,
                            std::size_t  T_q,
                            std::size_t  T_k,
                            std::size_t  nHeads,
                            std::size_t  nKvHeads,
                            std::size_t  headDim,
                            std::size_t  positionOffset,
                            float        scale,
                            float*       out) {
    if (T_q == 0 || T_k == 0 || nHeads == 0 || headDim == 0) {
        return;
    }
    if (nKvHeads == 0 || nHeads % nKvHeads != 0) {
        throw std::runtime_error(
            "GpuOps::attentionAsync: nHeads (" + std::to_string(nHeads) +
            ") must be a positive multiple of nKvHeads (" +
            std::to_string(nKvHeads) + ")");
    }
    if (T_k > kAttentionMaxTk) {
        throw std::runtime_error(
            "GpuOps::attentionAsync: T_k=" + std::to_string(T_k) +
            " exceeds compile-time bound ATTN_MAX_TK=" +
            std::to_string(kAttentionMaxTk) +
            " — bump ATTN_MAX_TK in kernels/attention.cl + kAttentionMaxTk "
            "in GpuOps.hpp together");
    }
    _attentionKernel.setPtr(0, q);
    _attentionKernel.setPtr(1, k);
    _attentionKernel.setPtr(2, v);
    _attentionKernel.setPtr(3, out);
    _attentionKernel.setValue<std::int32_t>(4, toInt32(T_q,            "attention T_q"));
    _attentionKernel.setValue<std::int32_t>(5, toInt32(T_k,            "attention T_k"));
    _attentionKernel.setValue<std::int32_t>(6, toInt32(nHeads,         "attention nHeads"));
    _attentionKernel.setValue<std::int32_t>(7, toInt32(nKvHeads,       "attention nKvHeads"));
    _attentionKernel.setValue<std::int32_t>(8, toInt32(headDim,        "attention headDim"));
    _attentionKernel.setValue<std::int32_t>(9, toInt32(positionOffset, "attention positionOffset"));
    _attentionKernel.setValue<float>(10, scale);
    _attentionKernel.setGroupSize(kAttentionLocalSize, 1, 1);
    // One workgroup per (head, query-position).
    _queue.appendLaunch(_attentionKernel,
                        static_cast<std::uint32_t>(nHeads),
                        static_cast<std::uint32_t>(T_q),
                        1);
}

} // namespace mimirmind::compute
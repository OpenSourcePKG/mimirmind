#include "compute/GpuOps.hpp"

#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

GpuOps::GpuOps(runtime::L0Context&    ctx,
               runtime::UsmAllocator& alloc,
               runtime::CommandQueue& queue)
    : _ctx{ctx},
      _queue{queue},
      _alloc{alloc},
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
      _attentionKernel {_attentionModule.kernel("attention")},
      _attentionFlashPartialModule{ctx, "attention_flash_partial"},
      _attentionFlashPartialKernel{
          _attentionFlashPartialModule.kernel("attention_flash_partial")},
      _attentionFlashMergeModule  {ctx, "attention_flash_merge"},
      _attentionFlashMergeKernel  {
          _attentionFlashMergeModule.kernel("attention_flash_merge")},
      _scaledAddResidualModule    {ctx, "scaled_add_residual"},
      _scaledAddResidualKernel    {
          _scaledAddResidualModule.kernel("scaled_add_residual")},
      _qkvSplitModule             {ctx, "qkv_split"},
      _qkvSplitKernel             {_qkvSplitModule.kernel("qkv_split")},
      _xQuantI8Module             {ctx, "x_quant_i8"},
      _xQuantI8Kernel             {_xQuantI8Module.kernel("x_quant_i8")}
{
    // Persistent FlashAttention partial-tile scratch. Sized for the
    // worst case across our target models; reused across every decode
    // call. Layout: [nHeads, K_TILES, (2 + headDim)] f32.
    _flashPartialBytes =
        kFlashMaxHeads * kFlashMaxKTiles *
        (2 + kFlashMaxHeadDim) * sizeof(float);
    _flashPartialUsm = _alloc.allocate(_flashPartialBytes);

    MM_LOG_INFO("gpuops",
                "GpuOps ready — rmsnorm/rmsnorm_gemma/rmsnorm_no_weight/"
                "add_bias/add_residual/silu_mul/rope/rope_ff/mul_scalar/"
                "gelu_mul/attention/attention_flash/scaled_add_residual/"
                "qkv_split/x_quant_i8 loaded (rms local={}, "
                "elementwise local={}, rope local={}, attention local={}, "
                "attention max T_k={}, flash kTile={}, flash maxHeads={}, "
                "flash maxHeadDim={}, flash partial scratch={} bytes)",
                kRmsnormLocalSize, kElementwiseLocalSize, kRopeLocalSize,
                kAttentionLocalSize, kAttentionMaxTk,
                kFlashKTileSize, kFlashMaxHeads, kFlashMaxHeadDim,
                _flashPartialBytes);
}

GpuOps::~GpuOps() {
    if (_flashPartialUsm != nullptr) {
        _alloc.deallocate(_flashPartialUsm, _flashPartialBytes);
        _flashPartialUsm = nullptr;
    }
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

void GpuOps::scaledAddResidualAsync(float*       dst,
                                    const float* src,
                                    float        scale,
                                    std::size_t  n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "scaledAddResidual n");
    _scaledAddResidualKernel.setPtr(0, dst);
    _scaledAddResidualKernel.setPtr(1, src);
    _scaledAddResidualKernel.setValue<float>(2, scale);
    _scaledAddResidualKernel.setValue<std::int32_t>(3, ni);
    _scaledAddResidualKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_scaledAddResidualKernel,
                        groupsForN(n, kElementwiseLocalSize), 1, 1);
}

void GpuOps::selfTest(runtime::UsmAllocator& allocator) {
    // x_quant_i8: per-row symmetric int8 quantisation. Feeds the DP4A
    // Q8_0 matmul (M8.H) — a broken quant kernel silently corrupts
    // every DP4A matmul on the target iGPU, so this runs first.
    {
        constexpr std::size_t M    = 2;
        constexpr std::size_t K    = 64;
        constexpr std::size_t xBytes = M * K * sizeof(float);
        constexpr std::size_t yBytes = M * K * sizeof(std::int8_t);
        constexpr std::size_t sBytes = M     * sizeof(float);

        void* xUsm = allocator.allocate(xBytes);
        void* yUsm = allocator.allocate(yBytes);
        void* sUsm = allocator.allocate(sBytes);

        // Row 0: mixed positive/negative, max at k=13. Row 1: all zeros
        // to exercise the amax=0 branch.
        std::vector<float> xh(M * K, 0.0F);
        for (std::size_t k = 0; k < K; ++k) {
            xh[k] = static_cast<float>(k) * 0.125F - 3.0F;
        }
        xh[13] = 5.5F; // guaranteed row-0 amax
        std::memcpy(xUsm, xh.data(), xBytes);
        std::memset(yUsm, 0x7F, yBytes);
        const float sPoison = -1.0e6F;
        for (std::size_t m = 0; m < M; ++m) {
            std::memcpy(static_cast<char*>(sUsm) + m * sizeof(float),
                        &sPoison, sizeof(float));
        }

        xQuantI8Async(static_cast<const float*>(xUsm),
                      static_cast<std::int8_t*>(yUsm),
                      static_cast<float*>(sUsm),
                      M, K);
        _queue.flush();

        std::vector<std::int8_t> qGot(M * K);
        std::vector<float>       sGot(M);
        std::memcpy(qGot.data(), yUsm, yBytes);
        std::memcpy(sGot.data(), sUsm, sBytes);

        // Row 0: scale = 5.5/127, dequant round-trip within 0.5*scale.
        const float amax0 = 5.5F;
        const float sRef0 = amax0 / 127.0F;
        if (!(std::fabs(sGot[0] - sRef0) <= 1e-6F)) {
            std::ostringstream os;
            os << "GpuOps::selfTest[x_quant_i8]: scale[0] mismatch — got="
               << sGot[0] << " ref=" << sRef0;
            throw std::runtime_error(os.str());
        }
        const float tol0 = 0.5F * sRef0 + 1e-6F;
        for (std::size_t k = 0; k < K; ++k) {
            const float deq = static_cast<float>(qGot[k]) * sGot[0];
            const float d   = std::fabs(deq - xh[k]);
            if (!(d <= tol0)) {
                std::ostringstream os;
                os << "GpuOps::selfTest[x_quant_i8]: row 0 dequant "
                   << "mismatch at k=" << k << " x=" << xh[k]
                   << " q=" << static_cast<int>(qGot[k])
                   << " deq=" << deq << " diff=" << d
                   << " tol=" << tol0;
                throw std::runtime_error(os.str());
            }
        }

        // Row 1: all-zero input → scale=0, all quants=0.
        if (!(sGot[1] == 0.0F)) {
            std::ostringstream os;
            os << "GpuOps::selfTest[x_quant_i8]: zero-row scale[1] "
               << "expected 0 got=" << sGot[1];
            throw std::runtime_error(os.str());
        }
        for (std::size_t k = 0; k < K; ++k) {
            if (qGot[K + k] != 0) {
                std::ostringstream os;
                os << "GpuOps::selfTest[x_quant_i8]: zero-row quant "
                   << "at k=" << k << " expected 0 got="
                   << static_cast<int>(qGot[K + k]);
                throw std::runtime_error(os.str());
            }
        }

        allocator.deallocate(xUsm, xBytes);
        allocator.deallocate(yUsm, yBytes);
        allocator.deallocate(sUsm, sBytes);
    }

    // qkv_split: full QKV path (hasV=true) plus alt-attention path
    // (hasV=false) on a tiny fixed shape.
    constexpr std::size_t M   = 3;
    constexpr std::size_t Nq  = 8;
    constexpr std::size_t Nkv = 4;

    auto runCase = [&](bool hasV, const char* label) {
        const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);

        void* fUsm = allocator.allocate(M * Nfused * sizeof(float));
        void* qUsm = allocator.allocate(M * Nq     * sizeof(float));
        void* kUsm = allocator.allocate(M * Nkv    * sizeof(float));
        void* vUsm = allocator.allocate(M * Nkv    * sizeof(float));

        std::vector<float> fused(M * Nfused);
        for (std::size_t i = 0; i < fused.size(); ++i) {
            fused[i] = static_cast<float>(i) * 0.125F;
        }
        std::memcpy(fUsm, fused.data(), fused.size() * sizeof(float));
        // Poison the outputs so under-fills show up.
        std::vector<float> poison(M * Nkv, -1.0e6F);
        std::memcpy(qUsm, poison.data(), M * Nq  * sizeof(float));
        std::memcpy(kUsm, poison.data(), M * Nkv * sizeof(float));
        std::memcpy(vUsm, poison.data(), M * Nkv * sizeof(float));

        qkvSplitAsync(static_cast<const float*>(fUsm),
                      static_cast<float*>(qUsm),
                      static_cast<float*>(kUsm),
                      static_cast<float*>(vUsm),
                      M, Nq, Nkv, hasV);
        _queue.flush();

        auto verify = [&](const void* usm, const float* ref,
                          std::size_t n, const char* which) {
            std::vector<float> got(n);
            std::memcpy(got.data(), usm, n * sizeof(float));
            float maxDiff = 0.0F;
            std::size_t badIdx = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const float d = std::fabs(got[i] - ref[i]);
                if (d > maxDiff) { maxDiff = d; badIdx = i; }
            }
            if (!(maxDiff <= 1e-6F)) {
                std::ostringstream os;
                os << "GpuOps::selfTest[" << label << "/" << which
                   << "]: qkv_split output mismatch — maxDiff=" << maxDiff
                   << " at i=" << badIdx
                   << " got=" << got[badIdx] << " ref=" << ref[badIdx];
                throw std::runtime_error(os.str());
            }
        };

        std::vector<float> refQ(M * Nq);
        std::vector<float> refK(M * Nkv);
        std::vector<float> refV(M * Nkv, -1.0e6F);
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t i = 0; i < Nq; ++i) {
                refQ[m * Nq + i] = fused[m * Nfused + i];
            }
            for (std::size_t j = 0; j < Nkv; ++j) {
                refK[m * Nkv + j] = fused[m * Nfused + Nq + j];
            }
            if (hasV) {
                for (std::size_t j = 0; j < Nkv; ++j) {
                    refV[m * Nkv + j] =
                        fused[m * Nfused + Nq + Nkv + j];
                }
            }
        }
        verify(qUsm, refQ.data(), M * Nq,  "Q");
        verify(kUsm, refK.data(), M * Nkv, "K");
        if (hasV) {
            verify(vUsm, refV.data(), M * Nkv, "V");
        }

        allocator.deallocate(fUsm, M * Nfused * sizeof(float));
        allocator.deallocate(qUsm, M * Nq     * sizeof(float));
        allocator.deallocate(kUsm, M * Nkv    * sizeof(float));
        allocator.deallocate(vUsm, M * Nkv    * sizeof(float));
    };

    runCase(/*hasV=*/true,  "full");
    runCase(/*hasV=*/false, "qk-only");

    _selfTestStatus = "ok";
    MM_LOG_INFO("gpuops",
                "selfTest OK — x_quant_i8 (M=2,K=64) + qkv_split full + "
                "qk-only paths verified");
}

void GpuOps::qkvSplitAsync(const float* fused,
                           float*       Yq,
                           float*       Yk,
                           float*       Yv,
                           std::size_t  M,
                           std::size_t  Nq,
                           std::size_t  Nkv,
                           bool         hasV) {
    if (M == 0 || Nq == 0 || Nkv == 0) {
        return;
    }
    const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);
    const std::size_t total  = M * Nfused;

    // Yv may legitimately be nullptr when hasV is false — the kernel
    // guards against dereferencing it, but Level Zero still expects a
    // valid pointer for the argument. Route to `fused` as a safe stub.
    const float* YvPtr = hasV ? Yv : fused;

    _qkvSplitKernel.setPtr(0, fused);
    _qkvSplitKernel.setPtr(1, Yq);
    _qkvSplitKernel.setPtr(2, Yk);
    _qkvSplitKernel.setPtr(3, YvPtr);
    _qkvSplitKernel.setValue<std::int32_t>(4, toInt32(M,      "qkvSplit M"));
    _qkvSplitKernel.setValue<std::int32_t>(5, toInt32(Nq,     "qkvSplit Nq"));
    _qkvSplitKernel.setValue<std::int32_t>(6, toInt32(Nkv,    "qkvSplit Nkv"));
    _qkvSplitKernel.setValue<std::int32_t>(7, hasV ? 1 : 0);
    _qkvSplitKernel.setValue<std::int32_t>(8, toInt32(Nfused, "qkvSplit Nfused"));
    _qkvSplitKernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(_qkvSplitKernel,
                        groupsForN(total, kElementwiseLocalSize), 1, 1);
}

void GpuOps::xQuantI8Async(const float* x,
                           std::int8_t* y,
                           float*       scale,
                           std::size_t  M,
                           std::size_t  K) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "xQuantI8 K");
    _xQuantI8Kernel.setPtr(0, x);
    _xQuantI8Kernel.setPtr(1, y);
    _xQuantI8Kernel.setPtr(2, scale);
    _xQuantI8Kernel.setValue<std::int32_t>(3, Ki);
    _xQuantI8Kernel.setGroupSize(kXQuantI8LocalSize, 1, 1);
    // One workgroup per row.
    _queue.appendLaunch(_xQuantI8Kernel,
                        static_cast<std::uint32_t>(M), 1, 1);
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

    // M5f.3.2: dispatch by query length. Decode (T_q == 1) goes through
    // the FlashAttention K-tiled path which launches nHeads × K_TILES
    // workgroups, getting the iGPU off the under-saturated nHeads-only
    // geometry of variant (a). Prefill keeps variant (a) because T_q ×
    // nHeads is already plenty of workgroups, and the scratch buffer
    // for partials would scale quadratically with context length.
    if (T_q == 1) {
        if (nHeads > kFlashMaxHeads || headDim > kFlashMaxHeadDim) {
            throw std::runtime_error(
                "GpuOps::attentionAsync: flash path needs nHeads<=" +
                std::to_string(kFlashMaxHeads) + " and headDim<=" +
                std::to_string(kFlashMaxHeadDim) + " (got " +
                std::to_string(nHeads) + " / " + std::to_string(headDim) +
                "); bump kFlashMax* and the kernel constants together");
        }
        attentionDecodeFlashAsync(q, k, v, T_k, nHeads, nKvHeads, headDim,
                                  positionOffset, scale, out);
    } else {
        attentionPlainAsync(q, k, v, T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale, out);
    }
}

void GpuOps::attentionPlainAsync(const float* q,
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

void GpuOps::attentionDecodeFlashAsync(const float* q,
                                       const float* k,
                                       const float* v,
                                       std::size_t  T_k,
                                       std::size_t  nHeads,
                                       std::size_t  nKvHeads,
                                       std::size_t  headDim,
                                       std::size_t  positionOffset,
                                       float        scale,
                                       float*       out) {
    // kMax = positionOffset + 1 in decode mode. Cap to T_k just in
    // case a caller passes a positionOffset >= T_k - 1.
    const std::size_t kMax =
        (positionOffset + 1 < T_k) ? (positionOffset + 1) : T_k;
    const std::size_t nKTiles =
        (kMax + kFlashKTileSize - 1) / kFlashKTileSize;
    if (nKTiles == 0 || nKTiles > kFlashMaxKTiles) {
        throw std::runtime_error(
            "GpuOps::attentionDecodeFlashAsync: nKTiles=" +
            std::to_string(nKTiles) + " out of [1, " +
            std::to_string(kFlashMaxKTiles) + "]");
    }

    // Pass 1 — per-tile partial (m, l, o_unnorm) into the persistent
    // USM scratch.
    _attentionFlashPartialKernel.setPtr(0, q);
    _attentionFlashPartialKernel.setPtr(1, k);
    _attentionFlashPartialKernel.setPtr(2, v);
    _attentionFlashPartialKernel.setPtr(3, _flashPartialUsm);
    _attentionFlashPartialKernel.setValue<std::int32_t>(4, toInt32(T_k,            "flash T_k"));
    _attentionFlashPartialKernel.setValue<std::int32_t>(5, toInt32(nHeads,         "flash nHeads"));
    _attentionFlashPartialKernel.setValue<std::int32_t>(6, toInt32(nKvHeads,       "flash nKvHeads"));
    _attentionFlashPartialKernel.setValue<std::int32_t>(7, toInt32(headDim,        "flash headDim"));
    _attentionFlashPartialKernel.setValue<std::int32_t>(8, toInt32(positionOffset, "flash positionOffset"));
    _attentionFlashPartialKernel.setValue<std::int32_t>(9, toInt32(nKTiles,        "flash nKTiles"));
    _attentionFlashPartialKernel.setValue<float>(10, scale);
    _attentionFlashPartialKernel.setGroupSize(kAttentionLocalSize, 1, 1);
    _queue.appendLaunch(_attentionFlashPartialKernel,
                        static_cast<std::uint32_t>(nHeads),
                        static_cast<std::uint32_t>(nKTiles),
                        1);

    // Pass 2 — merge the per-tile partials into the final output. The
    // auto-barrier between launches makes partials visible to the merge.
    _attentionFlashMergeKernel.setPtr(0, _flashPartialUsm);
    _attentionFlashMergeKernel.setPtr(1, out);
    _attentionFlashMergeKernel.setValue<std::int32_t>(2, toInt32(nHeads,  "flash_merge nHeads"));
    _attentionFlashMergeKernel.setValue<std::int32_t>(3, toInt32(headDim, "flash_merge headDim"));
    _attentionFlashMergeKernel.setValue<std::int32_t>(4, toInt32(nKTiles, "flash_merge nKTiles"));
    _attentionFlashMergeKernel.setGroupSize(kAttentionLocalSize, 1, 1);
    _queue.appendLaunch(_attentionFlashMergeKernel,
                        static_cast<std::uint32_t>(nHeads),
                        1,
                        1);
}

} // namespace mimirmind::compute
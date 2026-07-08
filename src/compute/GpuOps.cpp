#include "compute/GpuOps.hpp"

#include "compute/Attention.hpp"
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

bool envSet(const char* name) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string_view s{v};
    return !s.empty() && s != "0" && s != "false" && s != "off";
}

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
      _rmsnormQkvModule{ctx, "rmsnorm_qkv"},
      _rmsnormQkvKernel{_rmsnormQkvModule.kernel("rmsnorm_qkv")},
      _addRmsNormModule{ctx, "add_rmsnorm"},
      _addRmsNormKernel{_addRmsNormModule.kernel("add_rmsnorm")},
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
      _attentionPrefillFlashModule{ctx, "attention_prefill_flash"},
      _attentionPrefillFlashKernel{
          _attentionPrefillFlashModule.kernel("attention_prefill_flash")},
      _attentionFp16Module        {ctx, "attention_fp16"},
      _attentionFp16Kernel        {
          _attentionFp16Module.kernel("attention_fp16")},
      _attentionFlashPartialFp16Module{ctx, "attention_flash_partial_fp16"},
      _attentionFlashPartialFp16Kernel{
          _attentionFlashPartialFp16Module.kernel("attention_flash_partial_fp16")},
      _attentionPrefillFlashFp16Module{ctx, "attention_prefill_flash_fp16"},
      _attentionPrefillFlashFp16Kernel{
          _attentionPrefillFlashFp16Module.kernel("attention_prefill_flash_fp16")},
      _rmsnormQkvFp16Module    {ctx, "rmsnorm_qkv_fp16"},
      _rmsnormQkvFp16Kernel    {
          _rmsnormQkvFp16Module.kernel("rmsnorm_qkv_fp16")},
      _qkvSplitFp16Module      {ctx, "qkv_split_fp16"},
      _qkvSplitFp16Kernel      {
          _qkvSplitFp16Module.kernel("qkv_split_fp16")},
      _ropeFp16Module          {ctx, "rope_inplace_fp16"},
      _ropeFp16Kernel          {
          _ropeFp16Module.kernel("rope_inplace_fp16")},
      _ropeFfFp16Module        {ctx, "rope_inplace_ff_fp16"},
      _ropeFfFp16Kernel        {
          _ropeFfFp16Module.kernel("rope_inplace_ff_fp16")},
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

    // M-CLR.2: single-int USM slot for the current decode position.
    // Kernels bind it via setPtr; host writes the value before each
    // dispatch in immediate mode and between replays in replay mode.
    _curLenSlotUsm = static_cast<std::int32_t*>(
        _alloc.allocate(sizeof(std::int32_t)));
    *_curLenSlotUsm = 0;

    // M5i.J: cache the prefill-flash rollback flag once at startup so the
    // dispatcher hot path stays branch-cheap.
    _prefillFlashDisabled = envSet("MIMIRMIND_DISABLE_FLASH_PREFILL");

    MM_LOG_INFO("gpuops",
                "GpuOps ready — rmsnorm/rmsnorm_gemma/rmsnorm_no_weight/"
                "rmsnorm_qkv/add_rmsnorm/"
                "add_bias/add_residual/silu_mul/rope/rope_ff/mul_scalar/"
                "gelu_mul/attention/attention_flash/attention_prefill_flash/"
                "attention_fp16/attention_flash_partial_fp16/"
                "attention_prefill_flash_fp16/"
                "rmsnorm_qkv_fp16/qkv_split_fp16/"
                "rope_inplace_fp16/rope_inplace_ff_fp16/"
                "scaled_add_residual/qkv_split/x_quant_i8 loaded "
                "(rms local={}, elementwise local={}, rope local={}, "
                "attention local={}, attention max T_k={}, flash kTile={}, "
                "flash maxHeads={}, flash maxHeadDim={}, "
                "flash partial scratch={} bytes, prefill_flash={})",
                kRmsnormLocalSize, kElementwiseLocalSize, kRopeLocalSize,
                kAttentionLocalSize, kAttentionMaxTk,
                kFlashKTileSize, kFlashMaxHeads, kFlashMaxHeadDim,
                _flashPartialBytes,
                _prefillFlashDisabled ? "disabled (env)" : "enabled");
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

void GpuOps::rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                             float*           kBase,  const float* kWeight,
                             float*           vBase,
                             std::size_t      qRows,
                             std::size_t      kvRows,
                             std::size_t      headDim,
                             float            eps,
                             std::size_t      writeOffset,
                             std::size_t      kvDim,
                             runtime::KvDtype kvDtype) {
    if ((qRows == 0 && kvRows == 0) || headDim == 0) {
        return;
    }
    const std::int32_t Ki      = toInt32(headDim, "rmsNormQkv headDim");
    const std::int32_t qRowsI  = toInt32(qRows,  "rmsNormQkv qRows");
    const std::int32_t kvRowsI = toInt32(kvRows, "rmsNormQkv kvRows");
    const std::int32_t kvDimI  = toInt32(kvDim,  "rmsNormQkv kvDim");

    // M-CLR.2: writeOffset (= current KV-cache length) via shared USM
    // slot. Kernel adds `curLen * kvDim` to K/V rows so the kBase/vBase
    // pointers stay stable across replays.
    // M10.2 Phase 0 Commit 4 — pick the fp16-KV write variant when the
    // caller signals FP16 storage. Kernel signature is identical (setPtr
    // is untyped); only the K/V write paths differ (vstore_half vs
    // scalar store) inside the kernel body.
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _rmsnormQkvFp16Kernel
                                            : _rmsnormQkvKernel;
    *_curLenSlotUsm = toInt32(writeOffset, "rmsNormQkv writeOffset");
    kernel.setPtr(0, qBuf);
    kernel.setPtr(1, qWeight);
    kernel.setPtr(2, qBuf);           // in-place
    kernel.setPtr(3, kBase);
    kernel.setPtr(4, kWeight);
    kernel.setPtr(5, kBase);          // in-place (same base)
    kernel.setPtr(6, vBase);
    kernel.setPtr(7, vBase);          // in-place (same base)
    kernel.setValue<std::int32_t>(8,  qRowsI);
    kernel.setValue<std::int32_t>(9,  kvRowsI);
    kernel.setValue<std::int32_t>(10, Ki);
    kernel.setValue<float>(11, eps);
    kernel.setPtr(12, _curLenSlotUsm);
    kernel.setValue<std::int32_t>(13, kvDimI);
    kernel.setGroupSize(kRmsnormLocalSize, 1, 1);

    // Total workgroups = qRows + 2 * kvRows. Q rows first, then K, then V.
    const std::uint32_t totalRows =
        static_cast<std::uint32_t>(qRows + 2 * kvRows);
    _queue.appendLaunch(kernel, totalRows, 1, 1);
}

void GpuOps::addRmsNormAsync(float*       x,
                             const float* delta,
                             std::size_t  M,
                             std::size_t  K,
                             const float* weight,
                             float        eps,
                             float*       y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "addRmsNorm K");
    _addRmsNormKernel.setPtr(0, x);
    _addRmsNormKernel.setPtr(1, delta);
    _addRmsNormKernel.setPtr(2, weight);
    _addRmsNormKernel.setPtr(3, y);
    _addRmsNormKernel.setValue<float>(4, eps);
    _addRmsNormKernel.setValue<std::int32_t>(5, Ki);
    _addRmsNormKernel.setGroupSize(kRmsnormLocalSize, 1, 1);
    _queue.appendLaunch(_addRmsNormKernel,
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

void GpuOps::ropeInPlaceAsync(float*           xBase,
                              std::size_t      seqLen,
                              std::size_t      numHeads,
                              std::size_t      headDim,
                              std::size_t      startPos,
                              float            base,
                              std::size_t      writeOffsetStride,
                              runtime::KvDtype kvDtype) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "GpuOps::ropeInPlace: headDim must be even");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    // M-CLR.2: startPos goes through the shared USM slot. The write
    // happens before appendLaunch so submit-and-wait mode sees the
    // fresh value; recorded/replayed lists pick up whatever the host
    // sets between replays.
    // M-CLR.2 Wave 3b: `writeOffsetStride` == 0 means the kernel takes
    // `xBase` as-is (Q-rope, workspace); != 0 shifts it by
    // `startPos * stride` inside the kernel (K-rope, cache slot).
    // M10.2 Phase 0 Commit 4 — pick the fp16-KV in-place variant when
    // caller signals FP16 storage. Rotation is still done in fp32 in
    // registers (vload_half → rotate → vstore_half); only the read/write
    // sites change dtype.
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _ropeFp16Kernel
                                            : _ropeKernel;
    *_curLenSlotUsm = toInt32(startPos, "rope startPos");
    kernel.setPtr(0, xBase);
    kernel.setValue<std::int32_t>(1, toInt32(seqLen,   "rope seqLen"));
    kernel.setValue<std::int32_t>(2, toInt32(numHeads, "rope numHeads"));
    kernel.setValue<std::int32_t>(3, toInt32(headDim,  "rope headDim"));
    kernel.setPtr(4, _curLenSlotUsm);
    kernel.setValue<float>(5, base);
    kernel.setValue<std::int32_t>(
        6, toInt32(writeOffsetStride, "rope writeOffsetStride"));
    kernel.setGroupSize(kRopeLocalSize, 1, 1);
    _queue.appendLaunch(kernel,
                        groupsForN(total, kRopeLocalSize), 1, 1);
}

void GpuOps::ropeInPlaceWithFactorsAsync(float*           xBase,
                                         const float*     freqFactors,
                                         std::size_t      seqLen,
                                         std::size_t      numHeads,
                                         std::size_t      headDim,
                                         std::size_t      startPos,
                                         float            base,
                                         std::size_t      writeOffsetStride,
                                         runtime::KvDtype kvDtype) {
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

    // M-CLR.2: startPos via shared USM slot. See ropeInPlaceAsync().
    // M-CLR.2 Wave 3b: writeOffsetStride shifts xBase for K-rope.
    // M10.2 Phase 0 Commit 4 — fp16-KV variant dispatch, see
    // ropeInPlaceAsync above.
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _ropeFfFp16Kernel
                                            : _ropeFfKernel;
    *_curLenSlotUsm = toInt32(startPos, "rope_ff startPos");
    kernel.setPtr(0, xBase);
    kernel.setPtr(1, freqFactors);
    kernel.setValue<std::int32_t>(2, toInt32(seqLen,   "rope_ff seqLen"));
    kernel.setValue<std::int32_t>(3, toInt32(numHeads, "rope_ff numHeads"));
    kernel.setValue<std::int32_t>(4, toInt32(headDim,  "rope_ff headDim"));
    kernel.setPtr(5, _curLenSlotUsm);
    kernel.setValue<float>(6, base);
    kernel.setValue<std::int32_t>(
        7, toInt32(writeOffsetStride, "rope_ff writeOffsetStride"));
    kernel.setGroupSize(kRopeLocalSize, 1, 1);
    _queue.appendLaunch(kernel,
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

    // M5i.J: verify the single-WG streaming FlashAttention prefill kernel
    // against compute::multiHeadAttention (CPU reference) on a tiny
    // T_q=8 case. Same SPV-regression-guard role as the qkv_split block
    // above — catches driver/ocloc miscompilation before the first block
    // runs and gives a targeted error rather than a mysterious model
    // divergence later. Skipped when the rollback flag is on: there's
    // nothing to gate.
    if (!_prefillFlashDisabled) {
        constexpr std::size_t T_q      = 8;
        constexpr std::size_t T_k      = 8;
        constexpr std::size_t nHeads   = 4;
        constexpr std::size_t nKvHeads = 2;
        constexpr std::size_t headDim  = 32;
        const float scale = 1.0F / std::sqrt(
            static_cast<float>(headDim));

        const std::size_t qN  = T_q * nHeads   * headDim;
        const std::size_t kvN = T_k * nKvHeads * headDim;
        const std::size_t oN  = qN;

        void* qUsm = allocator.allocate(qN  * sizeof(float));
        void* kUsm = allocator.allocate(kvN * sizeof(float));
        void* vUsm = allocator.allocate(kvN * sizeof(float));
        void* oUsm = allocator.allocate(oN  * sizeof(float));

        // Deterministic ramp inputs — same seed math wouldn't help since
        // we have no RNG in the runtime, and this keeps the test
        // reproducible without pulling <random> into a load-time gate.
        std::vector<float> qHost(qN), kHost(kvN), vHost(kvN);
        for (std::size_t i = 0; i < qN;  ++i)
            qHost[i] = static_cast<float>((i * 7  + 1) % 17) * 0.125F - 1.0F;
        for (std::size_t i = 0; i < kvN; ++i)
            kHost[i] = static_cast<float>((i * 11 + 3) % 19) * 0.125F - 1.25F;
        for (std::size_t i = 0; i < kvN; ++i)
            vHost[i] = static_cast<float>((i * 13 + 5) % 23) * 0.0625F - 0.75F;
        std::memcpy(qUsm, qHost.data(), qN  * sizeof(float));
        std::memcpy(kUsm, kHost.data(), kvN * sizeof(float));
        std::memcpy(vUsm, vHost.data(), kvN * sizeof(float));
        std::memset(oUsm, 0,            oN  * sizeof(float));

        attentionPrefillFlashAsync(
            static_cast<const float*>(qUsm),
            static_cast<const float*>(kUsm),
            static_cast<const float*>(vUsm),
            T_q, nHeads, nKvHeads, headDim,
            /*positionOffset=*/0, scale,
            static_cast<float*>(oUsm),
            /*slidingWindow=*/0,
            runtime::KvDtype::F32);
        _queue.flush();

        std::vector<float> outCpu(oN, 0.0F);
        std::vector<float> scratch(T_k);
        // compute::multiHeadAttention bakes 1/sqrt(headDim) into its Q·K
        // scale — matches what we pass here, no pre-scale needed.
        multiHeadAttention(qHost.data(), kHost.data(), vHost.data(),
                           T_q, T_k, nHeads, nKvHeads, headDim,
                           /*positionOffset=*/0,
                           scratch.data(), outCpu.data());

        std::vector<float> outGpu(oN);
        std::memcpy(outGpu.data(), oUsm, oN * sizeof(float));

        constexpr float kTol = 5e-4F;
        float       maxDiff = 0.0F;
        std::size_t badIdx  = 0;
        for (std::size_t i = 0; i < oN; ++i) {
            const float d = std::fabs(outGpu[i] - outCpu[i]);
            if (d > maxDiff) { maxDiff = d; badIdx = i; }
        }
        if (!(maxDiff <= kTol)) {
            std::ostringstream os;
            os << "GpuOps::selfTest[attention_prefill_flash]: "
               << "output mismatch — maxDiff=" << maxDiff
               << " at i=" << badIdx
               << " gpu=" << outGpu[badIdx]
               << " cpu=" << outCpu[badIdx]
               << " tol="  << kTol;
            throw std::runtime_error(os.str());
        }

        allocator.deallocate(qUsm, qN  * sizeof(float));
        allocator.deallocate(kUsm, kvN * sizeof(float));
        allocator.deallocate(vUsm, kvN * sizeof(float));
        allocator.deallocate(oUsm, oN  * sizeof(float));
    }

    _selfTestStatus = "ok";
    MM_LOG_INFO("gpuops",
                "selfTest OK — x_quant_i8 (M=2,K=64) + qkv_split full + "
                "qk-only paths{}",
                _prefillFlashDisabled
                    ? " (attention_prefill_flash skipped: "
                      "MIMIRMIND_DISABLE_FLASH_PREFILL=1)"
                    : " + attention_prefill_flash (T_q=8) verified");
}

void GpuOps::qkvSplitAsync(const float*     fused,
                           float*           Yq,
                           float*           YkBase,
                           float*           YvBase,
                           std::size_t      M,
                           std::size_t      Nq,
                           std::size_t      Nkv,
                           bool             hasV,
                           std::size_t      writeOffset,
                           runtime::KvDtype kvDtype) {
    if (M == 0 || Nq == 0 || Nkv == 0) {
        return;
    }
    const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);
    const std::size_t total  = M * Nfused;

    // Yv may legitimately be nullptr when hasV is false — the kernel
    // guards against dereferencing it, but Level Zero still expects a
    // valid pointer for the argument. Route to `fused` as a safe stub.
    const float* YvPtr = hasV ? YvBase : fused;

    // M-CLR.2: curLen via shared USM slot. Kernel adds
    // `curLen * Nkv` to reach the row inside the K/V cache.
    // M10.2 Phase 0 Commit 4 — pick fp16-KV variant when caller signals
    // FP16. Yq stays fp32 in both paths; Yk/Yv are stored fp16 in the
    // FP16 dispatch and the fp16 kernel uses vstore_half on the scatter.
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _qkvSplitFp16Kernel
                                            : _qkvSplitKernel;
    *_curLenSlotUsm = toInt32(writeOffset, "qkvSplit writeOffset");
    kernel.setPtr(0, fused);
    kernel.setPtr(1, Yq);
    kernel.setPtr(2, YkBase);
    kernel.setPtr(3, YvPtr);
    kernel.setValue<std::int32_t>(4, toInt32(M,      "qkvSplit M"));
    kernel.setValue<std::int32_t>(5, toInt32(Nq,     "qkvSplit Nq"));
    kernel.setValue<std::int32_t>(6, toInt32(Nkv,    "qkvSplit Nkv"));
    kernel.setValue<std::int32_t>(7, hasV ? 1 : 0);
    kernel.setValue<std::int32_t>(8, toInt32(Nfused, "qkvSplit Nfused"));
    kernel.setPtr(9, _curLenSlotUsm);
    kernel.setGroupSize(kElementwiseLocalSize, 1, 1);
    _queue.appendLaunch(kernel,
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

void GpuOps::attentionAsync(const float*      q,
                            const float*      k,
                            const float*      v,
                            std::size_t       T_q,
                            std::size_t       T_k,
                            std::size_t       nHeads,
                            std::size_t       nKvHeads,
                            std::size_t       headDim,
                            std::size_t       positionOffset,
                            float             scale,
                            float*            out,
                            std::size_t       slidingWindow,
                            runtime::KvDtype  kvDtype) {
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

    // Dispatch by query length.
    //  - T_q == 1 (M5f.3.2): FlashAttention two-kernel decode. Launches
    //    nHeads × K_TILES workgroups, saturating the iGPU past the
    //    variant-(a) nHeads-only geometry.
    //  - T_q >  1 (M5i.J):   single-WG streaming FlashAttention. Same
    //    (nHeads, T_q) launch geometry as variant (a) but the K-loop
    //    runs tiled online-softmax intra-WG, shrinking SLM from 64 KiB
    //    (variant (a)) to ~2.5 KiB per WG. Rollback via
    //    MIMIRMIND_DISABLE_FLASH_PREFILL=1.
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
                                  positionOffset, scale, out, slidingWindow,
                                  kvDtype);
    } else if (!_prefillFlashDisabled && headDim <= kFlashMaxHeadDim) {
        attentionPrefillFlashAsync(q, k, v, T_q, nHeads, nKvHeads, headDim,
                                   positionOffset, scale, out, slidingWindow,
                                   kvDtype);
    } else {
        attentionPlainAsync(q, k, v, T_q, T_k, nHeads, nKvHeads, headDim,
                            positionOffset, scale, out, slidingWindow,
                            kvDtype);
    }
}

void GpuOps::attentionPlainAsync(const float*     q,
                                 const float*     k,
                                 const float*     v,
                                 std::size_t      T_q,
                                 std::size_t      T_k,
                                 std::size_t      nHeads,
                                 std::size_t      nKvHeads,
                                 std::size_t      headDim,
                                 std::size_t      positionOffset,
                                 float            scale,
                                 float*           out,
                                 std::size_t      slidingWindow,
                                 runtime::KvDtype kvDtype) {
    // M-CLR.2: positionOffset via shared USM slot. T_k was previously
    // an arg used only to cap kMax; every caller satisfies
    // T_k = positionOffset + T_q so the cap is a no-op, and the arg
    // drops out. See kernels/attention.cl for the signature.
    // The (void)T_k below documents that the wrapper still accepts it
    // but no longer forwards it — callers pass totalLen for historical
    // parity with the CPU reference.
    (void)T_k;
    // M10.2 Phase 0 Commit 3 — pick the fp16-KV read variant when the
    // caller signals FP16 storage. Same signature/argument layout, only
    // the K/V loads differ (vload_half + fp32 promote inside the kernel).
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _attentionFp16Kernel
                                            : _attentionKernel;
    *_curLenSlotUsm = toInt32(positionOffset, "attention positionOffset");
    kernel.setPtr(0, q);
    kernel.setPtr(1, k);
    kernel.setPtr(2, v);
    kernel.setPtr(3, out);
    kernel.setValue<std::int32_t>(4, toInt32(T_q,      "attention T_q"));
    kernel.setValue<std::int32_t>(5, toInt32(nHeads,   "attention nHeads"));
    kernel.setValue<std::int32_t>(6, toInt32(nKvHeads, "attention nKvHeads"));
    kernel.setValue<std::int32_t>(7, toInt32(headDim,  "attention headDim"));
    kernel.setPtr(8, _curLenSlotUsm);
    kernel.setValue<float>(9, scale);
    kernel.setValue<std::int32_t>(
        10, toInt32(slidingWindow, "attention slidingWindow"));
    kernel.setGroupSize(kAttentionLocalSize, 1, 1);
    // One workgroup per (head, query-position).
    _queue.appendLaunch(kernel,
                        static_cast<std::uint32_t>(nHeads),
                        static_cast<std::uint32_t>(T_q),
                        1);
}

void GpuOps::attentionPrefillFlashAsync(const float*     q,
                                        const float*     k,
                                        const float*     v,
                                        std::size_t      T_q,
                                        std::size_t      nHeads,
                                        std::size_t      nKvHeads,
                                        std::size_t      headDim,
                                        std::size_t      positionOffset,
                                        float            scale,
                                        float*           out,
                                        std::size_t      slidingWindow,
                                        runtime::KvDtype kvDtype) {
    // M-CLR.2: positionOffset comes via the shared USM slot. The kernel
    // dereferences curLenPtr[0] to compute kMax = positionOffset + pq + 1
    // per query position.
    // M10.2 Phase 0 Commit 3 — pick the fp16-KV read variant when the
    // caller signals FP16 storage.
    runtime::GpuKernel& kernel =
        (kvDtype == runtime::KvDtype::FP16) ? _attentionPrefillFlashFp16Kernel
                                            : _attentionPrefillFlashKernel;
    *_curLenSlotUsm =
        toInt32(positionOffset, "prefill_flash positionOffset");
    kernel.setPtr(0, q);
    kernel.setPtr(1, k);
    kernel.setPtr(2, v);
    kernel.setPtr(3, out);
    kernel.setValue<std::int32_t>(
        4, toInt32(T_q,      "prefill_flash T_q"));
    kernel.setValue<std::int32_t>(
        5, toInt32(nHeads,   "prefill_flash nHeads"));
    kernel.setValue<std::int32_t>(
        6, toInt32(nKvHeads, "prefill_flash nKvHeads"));
    kernel.setValue<std::int32_t>(
        7, toInt32(headDim,  "prefill_flash headDim"));
    kernel.setPtr(8, _curLenSlotUsm);
    kernel.setValue<float>(9, scale);
    kernel.setValue<std::int32_t>(
        10, toInt32(slidingWindow, "prefill_flash slidingWindow"));
    kernel.setGroupSize(kAttentionLocalSize, 1, 1);
    // Same (nHeads, T_q) grid as variant (a). Streaming K-tile loop
    // stays inside each workgroup.
    _queue.appendLaunch(kernel,
                        static_cast<std::uint32_t>(nHeads),
                        static_cast<std::uint32_t>(T_q),
                        1);
}

void GpuOps::attentionDecodeFlashAsync(const float*     q,
                                       const float*     k,
                                       const float*     v,
                                       std::size_t      T_k,
                                       std::size_t      nHeads,
                                       std::size_t      nKvHeads,
                                       std::size_t      headDim,
                                       std::size_t      positionOffset,
                                       float            scale,
                                       float*           out,
                                       std::size_t      slidingWindow,
                                       runtime::KvDtype kvDtype) {
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

    // M-CLR.2: T_k, positionOffset and nKTiles are all derivable from
    // positionOffset (= curLen). The kernels dereference curLenPtr and
    // compute the rest. Launch geometry still uses the host-computed
    // nKTiles — M-CLR.3 will move to a max-tile launch with kernel-side
    // early-exit so the replayed geometry doesn't need to update.
    (void)T_k;
    *_curLenSlotUsm = toInt32(positionOffset, "flash positionOffset");

    // Pass 1 — per-tile partial (m, l, o_unnorm) into the persistent
    // USM scratch. Kernel picked by KV dtype; partial layout stays fp32
    // regardless so the merge kernel is dtype-agnostic.
    runtime::GpuKernel& partialKernel =
        (kvDtype == runtime::KvDtype::FP16) ? _attentionFlashPartialFp16Kernel
                                            : _attentionFlashPartialKernel;
    partialKernel.setPtr(0, q);
    partialKernel.setPtr(1, k);
    partialKernel.setPtr(2, v);
    partialKernel.setPtr(3, _flashPartialUsm);
    partialKernel.setValue<std::int32_t>(4, toInt32(nHeads,   "flash nHeads"));
    partialKernel.setValue<std::int32_t>(5, toInt32(nKvHeads, "flash nKvHeads"));
    partialKernel.setValue<std::int32_t>(6, toInt32(headDim,  "flash headDim"));
    partialKernel.setPtr(7, _curLenSlotUsm);
    partialKernel.setValue<float>(8, scale);
    partialKernel.setValue<std::int32_t>(
        9, toInt32(slidingWindow, "flash slidingWindow"));
    partialKernel.setGroupSize(kAttentionLocalSize, 1, 1);
    // M-CLR.4 follow-up: launch geometry is
    //  - `_replayMaxKTiles` during recording (right-sized upper bound
    //     the InferenceEngine computed from prompt+max_new_tokens),
    //  - actual `nKTiles` in immediate mode (optimal, no waste).
    // The kernel's early-exit at `kStart >= kMax` still covers unused
    // tiles inside the recorded launch when curLen hasn't caught up
    // yet during replay.
    const std::uint32_t launchKTiles =
        (_queue.isRecording() && _replayMaxKTiles > 0)
            ? static_cast<std::uint32_t>(_replayMaxKTiles)
            : static_cast<std::uint32_t>(nKTiles);
    _queue.appendLaunch(partialKernel,
                        static_cast<std::uint32_t>(nHeads),
                        launchKTiles,
                        1);

    // Pass 2 — merge the per-tile partials into the final output. The
    // auto-barrier between launches makes partials visible to the merge.
    _attentionFlashMergeKernel.setPtr(0, _flashPartialUsm);
    _attentionFlashMergeKernel.setPtr(1, out);
    _attentionFlashMergeKernel.setValue<std::int32_t>(2, toInt32(nHeads,  "flash_merge nHeads"));
    _attentionFlashMergeKernel.setValue<std::int32_t>(3, toInt32(headDim, "flash_merge headDim"));
    _attentionFlashMergeKernel.setPtr(4, _curLenSlotUsm);
    _attentionFlashMergeKernel.setGroupSize(kAttentionLocalSize, 1, 1);
    _queue.appendLaunch(_attentionFlashMergeKernel,
                        static_cast<std::uint32_t>(nHeads),
                        1,
                        1);
}

} // namespace mimirmind::compute
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/hip/GpuOps.hpp"

#include "core/gpu/hip/HipComputeContext.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::hip {

namespace {

constexpr const char* kDefaultHsacoDir = "/usr/local/share/mimirmind/hsaco";

// Range-check + narrow to int32 for kernel scalar args. Kernels bind
// their shape arguments as `const int` so an oversized `size_t` would
// silently truncate — this helper throws instead, matching the L0
// side's `toInt32` in GpuOps.cpp.
std::int32_t toInt32(std::size_t v, const char* tag) {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            std::string{"hip::GpuOps: "} + tag +
            " overflows int32 ("  + std::to_string(v) + ")");
    }
    return static_cast<std::int32_t>(v);
}

// Ceiling division for launch grid geometry. Same shape as the L0
// `groupsForN`. Throws if the resulting group count would overflow a
// `uint32_t` (kernel launch API takes 32-bit dims).
std::uint32_t groupsForN(std::size_t n, std::uint32_t local) {
    const std::size_t g = (n + local - 1) / local;
    if (g > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("hip::GpuOps: workgroup count overflows uint32");
    }
    return static_cast<std::uint32_t>(g);
}

// Resolve `<name>.hsaco` in one of:
//   1. $MIMIRMIND_HSACO_DIR (env var — HIP analog of runtime.spvDir)
//   2. /usr/local/share/mimirmind/hsaco (production install)
//   3. build-tree fallbacks (build/hsaco, ../build/hsaco, hsaco)
// Mirrors `resolveSpvPath` in `GpuModule.cpp` — same three-tier lookup
// so the deployment stories stay parallel.
std::filesystem::path resolveHsacoPath(std::string_view name) {
    const std::string filename = std::string{name} + ".hsaco";

    if (const char* env = std::getenv("MIMIRMIND_HSACO_DIR")) {
        if (env[0] != '\0') {
            const std::filesystem::path p =
                std::filesystem::path{env} / filename;
            if (std::filesystem::exists(p)) {
                return p;
            }
        }
    }

    {
        const std::filesystem::path p =
            std::filesystem::path{kDefaultHsacoDir} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    for (auto rel : std::array<const char*, 3>{
             "build/hsaco", "../build/hsaco", "hsaco"}) {
        const std::filesystem::path p =
            std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    throw std::runtime_error(
        "hip::GpuOps: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultHsacoDir);
}

// Convenience: load a HipModule by kernel name, resolving the .hsaco
// path through `resolveHsacoPath`. Symbol name inside the module is
// assumed to be identical to the file basename (mirrors the L0 side
// where `.spv` filename == kernel `__kernel` symbol).
core::hip::HipModule loadHipModule(core::hip::HipContext& ctx,
                                   std::string_view       name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("hipgpuops", "loading module '{}' from {}",
                std::string{name}, path.string());
    return core::hip::HipModule::fromFile(ctx, path.string());
}

// Named throw helper — keeps every stub one line and gives an
// immediately-actionable message when a caller trips on a not-yet-
// implemented dispatch.
[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::hip::GpuOps::"} + method +
        ": not yet implemented (Schritt 3b skeleton — kernel-launch "
        "impl lands in follow-up commits)");
}

} // namespace

// Pimpl body — one HipModule + HipKernel pair per compiled `.hip`
// source under `kernels_hip/` that corresponds to a `ComputeOps`
// entry point. HIP-only kernels used by `HipGpuMatmul` (matmul
// variants + moe_down) live on that class, not here. Kernels that
// exist in the L0 `GpuOps::Impl` but haven't been ported to HIP yet
// (rope_inplace_ff{,_fp16}, rope_inplace_fp16,
// attention_prefill_flash_q8_0_gqa KTILE=64 variant) are deliberately
// absent — the corresponding ComputeOps overrides throw
// `not yet implemented` at dispatch time.
struct GpuOps::Impl {
    core::hip::HipModule _rmsnormModule;
    core::hip::HipKernel _rmsnormKernel;
    core::hip::HipModule _rmsnormGemmaModule;
    core::hip::HipKernel _rmsnormGemmaKernel;
    core::hip::HipModule _rmsnormNoWeightModule;
    core::hip::HipKernel _rmsnormNoWeightKernel;
    core::hip::HipModule _rmsnormQkvModule;
    core::hip::HipKernel _rmsnormQkvKernel;
    core::hip::HipModule _rmsnormQkvFp16Module;
    core::hip::HipKernel _rmsnormQkvFp16Kernel;
    core::hip::HipModule _addRmsNormModule;
    core::hip::HipKernel _addRmsNormKernel;

    core::hip::HipModule _addBiasModule;
    core::hip::HipKernel _addBiasKernel;
    core::hip::HipModule _addResidualModule;
    core::hip::HipKernel _addResidualKernel;
    core::hip::HipModule _siluMulModule;
    core::hip::HipKernel _siluMulKernel;
    core::hip::HipModule _geluMulModule;
    core::hip::HipKernel _geluMulKernel;
    core::hip::HipModule _mulScalarModule;
    core::hip::HipKernel _mulScalarKernel;
    core::hip::HipModule _scaledAddResidualModule;
    core::hip::HipKernel _scaledAddResidualKernel;
    core::hip::HipModule _xQuantI8Module;
    core::hip::HipKernel _xQuantI8Kernel;
    core::hip::HipModule _ropeModule;
    core::hip::HipKernel _ropeKernel;

    core::hip::HipModule _attentionModule;
    core::hip::HipKernel _attentionKernel;
    core::hip::HipModule _attentionFp16Module;
    core::hip::HipKernel _attentionFp16Kernel;
    core::hip::HipModule _attentionQ8Module;
    core::hip::HipKernel _attentionQ8Kernel;
    core::hip::HipModule _attentionFlashPartialModule;
    core::hip::HipKernel _attentionFlashPartialKernel;
    core::hip::HipModule _attentionFlashPartialFp16Module;
    core::hip::HipKernel _attentionFlashPartialFp16Kernel;
    core::hip::HipModule _attentionFlashPartialQ8Module;
    core::hip::HipKernel _attentionFlashPartialQ8Kernel;
    core::hip::HipModule _attentionFlashMergeModule;
    core::hip::HipKernel _attentionFlashMergeKernel;

    core::hip::HipModule _attentionPrefillFlashModule;
    core::hip::HipKernel _attentionPrefillFlashKernel;
    core::hip::HipModule _attentionPrefillFlashFp16Module;
    core::hip::HipKernel _attentionPrefillFlashFp16Kernel;
    core::hip::HipModule _attentionPrefillFlashQ8Module;
    core::hip::HipKernel _attentionPrefillFlashQ8Kernel;
    core::hip::HipModule _attentionPrefillFlashQ8GqaModule;
    core::hip::HipKernel _attentionPrefillFlashQ8GqaKernel;

    core::hip::HipModule _qkvSplitModule;
    core::hip::HipKernel _qkvSplitKernel;
    core::hip::HipModule _qkvSplitFp16Module;
    core::hip::HipKernel _qkvSplitFp16Kernel;

    core::hip::HipModule _kvQuantCommitQ8Module;
    core::hip::HipKernel _kvQuantCommitQ8Kernel;

    core::hip::HipModule _matmulQ8_0VecReorderModule;
    core::hip::HipKernel _matmulQ8_0VecReorderKernel;

    explicit Impl(core::hip::HipContext& ctx)
        : _rmsnormModule           {loadHipModule(ctx, "rmsnorm")},
          _rmsnormKernel           {_rmsnormModule.getKernel("rmsnorm")},
          _rmsnormGemmaModule      {loadHipModule(ctx, "rmsnorm_gemma")},
          _rmsnormGemmaKernel      {_rmsnormGemmaModule.getKernel("rmsnorm_gemma")},
          _rmsnormNoWeightModule   {loadHipModule(ctx, "rmsnorm_no_weight")},
          _rmsnormNoWeightKernel   {_rmsnormNoWeightModule.getKernel("rmsnorm_no_weight")},
          _rmsnormQkvModule        {loadHipModule(ctx, "rmsnorm_qkv")},
          _rmsnormQkvKernel        {_rmsnormQkvModule.getKernel("rmsnorm_qkv")},
          _rmsnormQkvFp16Module    {loadHipModule(ctx, "rmsnorm_qkv_fp16")},
          _rmsnormQkvFp16Kernel    {_rmsnormQkvFp16Module.getKernel("rmsnorm_qkv_fp16")},
          _addRmsNormModule        {loadHipModule(ctx, "add_rmsnorm")},
          _addRmsNormKernel        {_addRmsNormModule.getKernel("add_rmsnorm")},

          _addBiasModule           {loadHipModule(ctx, "add_bias")},
          _addBiasKernel           {_addBiasModule.getKernel("add_bias")},
          _addResidualModule       {loadHipModule(ctx, "add_residual")},
          _addResidualKernel       {_addResidualModule.getKernel("add_residual")},
          _siluMulModule           {loadHipModule(ctx, "silu_mul")},
          _siluMulKernel           {_siluMulModule.getKernel("silu_mul")},
          _geluMulModule           {loadHipModule(ctx, "gelu_mul")},
          _geluMulKernel           {_geluMulModule.getKernel("gelu_mul")},
          _mulScalarModule         {loadHipModule(ctx, "mul_scalar")},
          _mulScalarKernel         {_mulScalarModule.getKernel("mul_scalar")},
          _scaledAddResidualModule {loadHipModule(ctx, "scaled_add_residual")},
          _scaledAddResidualKernel {
              _scaledAddResidualModule.getKernel("scaled_add_residual")},
          _xQuantI8Module          {loadHipModule(ctx, "x_quant_i8")},
          _xQuantI8Kernel          {_xQuantI8Module.getKernel("x_quant_i8")},
          _ropeModule              {loadHipModule(ctx, "rope_inplace")},
          _ropeKernel              {_ropeModule.getKernel("rope_inplace")},

          _attentionModule         {loadHipModule(ctx, "attention")},
          _attentionKernel         {_attentionModule.getKernel("attention")},
          _attentionFp16Module     {loadHipModule(ctx, "attention_fp16")},
          _attentionFp16Kernel     {_attentionFp16Module.getKernel("attention_fp16")},
          _attentionQ8Module       {loadHipModule(ctx, "attention_q8_0")},
          _attentionQ8Kernel       {_attentionQ8Module.getKernel("attention_q8_0")},
          _attentionFlashPartialModule{loadHipModule(ctx, "attention_flash_partial")},
          _attentionFlashPartialKernel{
              _attentionFlashPartialModule.getKernel("attention_flash_partial")},
          _attentionFlashPartialFp16Module{
              loadHipModule(ctx, "attention_flash_partial_fp16")},
          _attentionFlashPartialFp16Kernel{
              _attentionFlashPartialFp16Module.getKernel("attention_flash_partial_fp16")},
          _attentionFlashPartialQ8Module{
              loadHipModule(ctx, "attention_flash_partial_q8_0")},
          _attentionFlashPartialQ8Kernel{
              _attentionFlashPartialQ8Module.getKernel("attention_flash_partial_q8_0")},
          _attentionFlashMergeModule{loadHipModule(ctx, "attention_flash_merge")},
          _attentionFlashMergeKernel{
              _attentionFlashMergeModule.getKernel("attention_flash_merge")},

          _attentionPrefillFlashModule{loadHipModule(ctx, "attention_prefill_flash")},
          _attentionPrefillFlashKernel{
              _attentionPrefillFlashModule.getKernel("attention_prefill_flash")},
          _attentionPrefillFlashFp16Module{
              loadHipModule(ctx, "attention_prefill_flash_fp16")},
          _attentionPrefillFlashFp16Kernel{
              _attentionPrefillFlashFp16Module.getKernel("attention_prefill_flash_fp16")},
          _attentionPrefillFlashQ8Module{
              loadHipModule(ctx, "attention_prefill_flash_q8_0")},
          _attentionPrefillFlashQ8Kernel{
              _attentionPrefillFlashQ8Module.getKernel("attention_prefill_flash_q8_0")},
          _attentionPrefillFlashQ8GqaModule{
              loadHipModule(ctx, "attention_prefill_flash_q8_0_gqa")},
          _attentionPrefillFlashQ8GqaKernel{
              _attentionPrefillFlashQ8GqaModule.getKernel(
                  "attention_prefill_flash_q8_0_gqa")},

          _qkvSplitModule          {loadHipModule(ctx, "qkv_split")},
          _qkvSplitKernel          {_qkvSplitModule.getKernel("qkv_split")},
          _qkvSplitFp16Module      {loadHipModule(ctx, "qkv_split_fp16")},
          _qkvSplitFp16Kernel      {_qkvSplitFp16Module.getKernel("qkv_split_fp16")},

          _kvQuantCommitQ8Module   {loadHipModule(ctx, "kv_quant_commit_q8_0")},
          _kvQuantCommitQ8Kernel   {
              _kvQuantCommitQ8Module.getKernel("kv_quant_commit_q8_0")},

          _matmulQ8_0VecReorderModule{loadHipModule(ctx, "matmul_q8_0_vec_reorder")},
          _matmulQ8_0VecReorderKernel{
              _matmulQ8_0VecReorderModule.getKernel("matmul_q8_0_vec_reorder")}
    {}
};

GpuOps::GpuOps(core::hip::HipComputeContext& ctx,
                     bool                          flashPrefillEnabled,
                     bool                          flashPrefillGqaQ8Enabled,
                     std::size_t                   flashPrefillKTileQ8,
                     core::config::TriState        q8_0ReorderMode)
    : _ctx{ctx},
      _pimpl{std::make_unique<Impl>(ctx.hipContext())}
{
    auto& alloc = ctx.allocator();

    // Persistent FlashAttention partial-tile scratch — same layout +
    // worst-case sizing as the L0 side. Reused across every decode.
    _flashPartialBytes =
        kFlashMaxHeads * kFlashMaxKTiles *
        (2 + kFlashMaxHeadDim) * sizeof(float);
    _flashPartialUsm = alloc.allocate(_flashPartialBytes);

    // Shared curLen slot for kernels that dereference the KV-cache
    // length at launch. On HIP without USM the host must
    // `hipMemcpy(H2D)` this before each dispatch — no zero-copy path
    // on gfx1101. Persistent single-int allocation, initialised to 0.
    _curLenSlotUsm = static_cast<std::int32_t*>(
        alloc.allocate(sizeof(std::int32_t)));
    {
        const std::int32_t zero = 0;
        alloc.copyH2D(_curLenSlotUsm, &zero, sizeof(std::int32_t));
    }

    // Second slot: always-0 sentinel for the Q8_0 fp32-staging pipeline.
    // Same design point as the L0 side — one slot advances with curLen,
    // the other stays pinned at zero, and the two never race.
    _stagingOffsetSlotUsm = static_cast<std::int32_t*>(
        alloc.allocate(sizeof(std::int32_t)));
    {
        const std::int32_t zero = 0;
        alloc.copyH2D(_stagingOffsetSlotUsm, &zero, sizeof(std::int32_t));
    }

    _prefillFlashDisabled      = !flashPrefillEnabled;
    _prefillFlashGqaQ8Disabled = !flashPrefillGqaQ8Enabled;
    _q8_0ReorderMode           = q8_0ReorderMode;

    _prefillFlashKTileQ8Configured = flashPrefillKTileQ8;
    if (flashPrefillKTileQ8 == 0) {
        _prefillFlashKTileQ8       = 128;
        _prefillFlashKTileQ8Source = "pending (autotune)";
    } else if (flashPrefillKTileQ8 == 64 || flashPrefillKTileQ8 == 128) {
        _prefillFlashKTileQ8       = flashPrefillKTileQ8;
        _prefillFlashKTileQ8Source = "pinned (config)";
    } else {
        throw std::runtime_error(
            "hip::GpuOps: features.flashPrefillKTileQ8=" +
            std::to_string(flashPrefillKTileQ8) +
            " unexpected — Config.cpp parser should have rejected this");
    }

    MM_LOG_INFO("hipgpuops",
                "hip::GpuOps ready — 29 modules loaded (rmsnorm variants, "
                "elementwise, rope, attention decode/prefill × f32/fp16/Q8_0, "
                "qkv_split × f32/fp16, kv_quant_commit_q8_0, "
                "matmul_q8_0_vec_reorder). "
                "flash_partial_scratch={} bytes, prefill_flash={}, "
                "prefill_flash_gqa_q8={}, prefill_flash_ktile_q8={}, "
                "q8_0_reorder={}",
                _flashPartialBytes,
                _prefillFlashDisabled      ? "disabled (config)" : "enabled",
                _prefillFlashGqaQ8Disabled ? "disabled (config)" : "enabled",
                _prefillFlashKTileQ8,
                q8_0ReorderModeName());
}

GpuOps::~GpuOps() {
    auto& alloc = _ctx.allocator();
    if (_stagingOffsetSlotUsm) {
        alloc.deallocate(_stagingOffsetSlotUsm, sizeof(std::int32_t),
                         core::hip::HipAllocKind::Device);
    }
    if (_curLenSlotUsm) {
        alloc.deallocate(_curLenSlotUsm, sizeof(std::int32_t),
                         core::hip::HipAllocKind::Device);
    }
    if (_flashPartialUsm) {
        alloc.deallocate(_flashPartialUsm, _flashPartialBytes,
                         core::hip::HipAllocKind::Device);
    }
}

// ---- Real (non-stub) implementations --------------------------------

core::hip::HipStream& GpuOps::stream() noexcept {
    return _ctx.stream();
}

core::hip::HipMemoryAllocator& GpuOps::allocator() noexcept {
    return _ctx.allocator();
}

std::string_view GpuOps::q8_0ReorderModeName() const noexcept {
    switch (_q8_0ReorderMode) {
        case core::config::TriState::Auto:    return "auto";
        case core::config::TriState::Force:   return "force";
        case core::config::TriState::Disable: return "disable";
    }
    return "unknown";
}

void GpuOps::noteQ8_0ReorderApplied(std::size_t bytes,
                                       std::string_view label) noexcept {
    _q8_0ReorderTensorCount += 1;
    _q8_0ReorderTotalBytes  += bytes;
    MM_LOG_INFO("hipgpuops",
                "q8_0 reorder applied to '{}' ({} bytes) — running total "
                "tensors={} bytes={}",
                std::string{label}, bytes,
                _q8_0ReorderTensorCount, _q8_0ReorderTotalBytes);
}

// ---- Stubbed kernel-launch overrides --------------------------------
//
// Every method below throws `std::runtime_error` with a clear
// diagnostic. Follow-up commits (Schritt 3b sub-B..sub-E) fill them
// group-by-group. Order matches the header layout.

void GpuOps::rmsNormAsync(const float* x, std::size_t M, std::size_t K,
                             const float* weight, float eps, float* y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNorm K");
    auto& k = _pimpl->_rmsnormKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, weight);
    k.setPtr  (2, y);
    k.setValue(3, eps);
    k.setValue(4, Ki);
    // One workgroup per row — mirrors L0 GpuOps.
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kRmsnormLocalSize, 1, 1);
}

void GpuOps::rmsNormGemmaAsync(const float* x, std::size_t M, std::size_t K,
                                  const float* weight, float eps, float* y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNormGemma K");
    auto& k = _pimpl->_rmsnormGemmaKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, weight);
    k.setPtr  (2, y);
    k.setValue(3, eps);
    k.setValue(4, Ki);
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kRmsnormLocalSize, 1, 1);
}

void GpuOps::rmsNormNoWeightAsync(const float* x, std::size_t M, std::size_t K,
                                     float eps, float* y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "rmsNormNoWeight K");
    auto& k = _pimpl->_rmsnormNoWeightKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, y);
    k.setValue(2, eps);
    k.setValue(3, Ki);
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kRmsnormLocalSize, 1, 1);
}

void GpuOps::rmsNormQkvAsync(float* qBuf, const float* qWeight,
                                void* kBase, const float* kWeight,
                                void* vBase,
                                std::size_t qRows, std::size_t kvRows,
                                std::size_t headDim, float eps,
                                std::size_t writeOffset, std::size_t kvDim,
                                runtime::KvDtype kvDtype, bool useStagingSlot) {
    if ((qRows == 0 && kvRows == 0) || headDim == 0) {
        return;
    }
    const std::int32_t Ki      = toInt32(headDim, "rmsNormQkv headDim");
    const std::int32_t qRowsI  = toInt32(qRows,   "rmsNormQkv qRows");
    const std::int32_t kvRowsI = toInt32(kvRows,  "rmsNormQkv kvRows");
    const std::int32_t kvDimI  = toInt32(kvDim,   "rmsNormQkv kvDim");

    // Pick f32 vs fp16 KV variant — arg layout is identical, only the
    // K/V store lowering differs inside the kernel body.
    auto& k = (kvDtype == runtime::KvDtype::FP16)
                  ? _pimpl->_rmsnormQkvFp16Kernel
                  : _pimpl->_rmsnormQkvKernel;

    // Bind the offset slot. Staging path always reads 0; the non-
    // staging path writes `writeOffset` into the shared curLen slot
    // via a synchronous H2D copy before launch. Mirror of the L0
    // dispatcher's slot-swap logic — see GpuOps::rmsNormQkvAsync.
    std::int32_t* offsetSlot;
    if (useStagingSlot) {
        offsetSlot = _stagingOffsetSlotUsm;
    } else {
        const std::int32_t v = toInt32(writeOffset, "rmsNormQkv writeOffset");
        _ctx.allocator().copyH2D(_curLenSlotUsm, &v, sizeof(std::int32_t));
        offsetSlot = _curLenSlotUsm;
    }

    k.setPtr  (0,  qBuf);
    k.setPtr  (1,  qWeight);
    k.setPtr  (2,  qBuf);            // in-place
    k.setPtr  (3,  kBase);
    k.setPtr  (4,  kWeight);
    k.setPtr  (5,  kBase);           // in-place
    k.setPtr  (6,  vBase);
    k.setPtr  (7,  vBase);           // in-place
    k.setValue(8,  qRowsI);
    k.setValue(9,  kvRowsI);
    k.setValue(10, Ki);
    k.setValue(11, eps);
    k.setPtr  (12, offsetSlot);
    k.setValue(13, kvDimI);

    // Total workgroups = qRows + 2*kvRows. Q rows first, then K, then V.
    const std::uint32_t totalRows =
        static_cast<std::uint32_t>(qRows + 2 * kvRows);
    k.launch(_ctx.stream(),
             totalRows, 1, 1,
             kRmsnormLocalSize, 1, 1);
}

void GpuOps::addRmsNormAsync(float* x, const float* delta,
                             std::size_t M, std::size_t K,
                             const float* weight, float eps, float* y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "addRmsNorm K");
    auto& k = _pimpl->_addRmsNormKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, delta);
    k.setPtr  (2, weight);
    k.setPtr  (3, y);
    k.setValue(4, eps);
    k.setValue(5, Ki);
    // One workgroup per row — same as rmsnorm.
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kRmsnormLocalSize, 1, 1);
}

void GpuOps::addBiasAsync(float* y, std::size_t M, std::size_t K,
                             const float* bias) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Mi = toInt32(M, "addBias M");
    const std::int32_t Ki = toInt32(K, "addBias K");
    auto& k = _pimpl->_addBiasKernel;
    k.setPtr  (0, y);
    k.setPtr  (1, bias);
    k.setValue(2, Mi);
    k.setValue(3, Ki);
    k.launch(_ctx.stream(),
             groupsForN(M * K, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::addResidualAsync(float* y, const float* x, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "addResidual n");
    auto& k = _pimpl->_addResidualKernel;
    k.setPtr  (0, y);
    k.setPtr  (1, x);
    k.setValue(2, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::siluMulAsync(float* gate, const float* up, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "siluMul n");
    auto& k = _pimpl->_siluMulKernel;
    k.setPtr  (0, gate);
    k.setPtr  (1, up);
    k.setValue(2, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::geluMulAsync(float* gate, const float* up, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "geluMul n");
    auto& k = _pimpl->_geluMulKernel;
    k.setPtr  (0, gate);
    k.setPtr  (1, up);
    k.setValue(2, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::mulScalarAsync(float* y, float s, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "mulScalar n");
    auto& k = _pimpl->_mulScalarKernel;
    k.setPtr  (0, y);
    k.setValue(1, s);
    k.setValue(2, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::scaledAddResidualAsync(float* dst, const float* src,
                                    float scale, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "scaledAddResidual n");
    auto& k = _pimpl->_scaledAddResidualKernel;
    k.setPtr  (0, dst);
    k.setPtr  (1, src);
    k.setValue(2, scale);
    k.setValue(3, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::ropeInPlaceAsync(void* xBase, std::size_t seqLen,
                                 std::size_t numHeads, std::size_t headDim,
                                 std::size_t startPos, float base,
                                 std::size_t writeOffsetStride,
                                 runtime::KvDtype kvDtype) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "GpuOps::ropeInPlace: headDim must be even");
    }
    // fp16 in-place variant needs `rope_inplace_fp16.hip` — not
    // ported yet. Refuse loudly rather than silently corrupt an
    // fp16 KV cache with the fp32 kernel.
    if (kvDtype == runtime::KvDtype::FP16) {
        throw std::runtime_error(
            "GpuOps::ropeInPlaceAsync: FP16 KV path requires "
            "rope_inplace_fp16.hip — not yet ported on the HIP side");
    }

    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    // startPos flows through the shared curLen slot — kernel binds it
    // as a device pointer and dereferences at launch. Sync H2D write
    // before launch matches L0's `*_curLenSlotUsm = startPos` store
    // (works there via USM). See rmsNormQkvAsync for the same trick.
    const std::int32_t startI = toInt32(startPos, "rope startPos");
    _ctx.allocator().copyH2D(_curLenSlotUsm, &startI, sizeof(std::int32_t));

    auto& k = _pimpl->_ropeKernel;
    k.setPtr  (0, xBase);
    k.setValue(1, toInt32(seqLen,   "rope seqLen"));
    k.setValue(2, toInt32(numHeads, "rope numHeads"));
    k.setValue(3, toInt32(headDim,  "rope headDim"));
    k.setPtr  (4, _curLenSlotUsm);
    k.setValue(5, base);
    k.setValue(6, toInt32(writeOffsetStride, "rope writeOffsetStride"));
    k.launch(_ctx.stream(),
             groupsForN(total, kRopeLocalSize), 1, 1,
             kRopeLocalSize, 1, 1);
}

void GpuOps::ropeInPlaceWithFactorsAsync(void*, const float*,
                                            std::size_t, std::size_t,
                                            std::size_t, std::size_t, float,
                                            std::size_t, runtime::KvDtype) {
    // rope_inplace_ff.hip not yet ported (nor its _fp16 variant).
    throwNotImplemented("ropeInPlaceWithFactorsAsync");
}

void GpuOps::xQuantI8Async(const float* x, std::int8_t* y, float* scale,
                           std::size_t M, std::size_t K) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "xQuantI8 K");
    auto& k = _pimpl->_xQuantI8Kernel;
    k.setPtr  (0, x);
    k.setPtr  (1, y);
    k.setPtr  (2, scale);
    k.setValue(3, Ki);
    // One workgroup per row — kernel LOCAL=128, matches L0.
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kXQuantI8LocalSize, 1, 1);
}

void GpuOps::kvQuantCommitQ8Async(const float* xSrc, void* kvDst,
                                     std::size_t T, std::size_t kvDim,
                                     std::size_t writeOffset) {
    if (T == 0 || kvDim == 0) {
        return;
    }
    // Q8_0 is inherently 32-element block based. A partial block
    // would leave stale bytes in the fp16 scale slot and mis-index
    // every following row. Same guard as GpuOps + KvCache ctor.
    constexpr std::size_t kBlockElems = 32;
    if (kvDim % kBlockElems != 0) {
        throw std::runtime_error(
            "GpuOps::kvQuantCommitQ8Async: kvDim=" +
            std::to_string(kvDim) +
            " must be a multiple of " + std::to_string(kBlockElems));
    }
    const std::size_t nBlocksPerRow = kvDim / kBlockElems;

    // writeOffset goes through the shared curLen slot — kernel adds
    // `curLen * nBlocksPerRow * 34` to reach the row-aligned start,
    // so `kvDst` stays a stable layer-base pointer across replays.
    const std::int32_t offI =
        toInt32(writeOffset, "kvQuantCommitQ8 writeOffset");
    _ctx.allocator().copyH2D(_curLenSlotUsm, &offI, sizeof(std::int32_t));

    auto& k = _pimpl->_kvQuantCommitQ8Kernel;
    k.setPtr  (0, xSrc);
    k.setPtr  (1, kvDst);
    k.setValue(2, toInt32(kvDim, "kvQuantCommitQ8 kvDim"));
    k.setPtr  (3, _curLenSlotUsm);
    // One workgroup per (t, block). Kernel LOCAL=32 == Q8_0 block
    // size so each thread owns one element of one block.
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(T),
             static_cast<std::uint32_t>(nBlocksPerRow),
             1,
             kKvQuantCommitLocalSize, 1, 1);
}

void GpuOps::qkvSplitAsync(const float* fused, float* Yq,
                              void* YkBase, void* YvBase,
                              std::size_t M, std::size_t Nq, std::size_t Nkv,
                              bool hasV,
                              std::size_t writeOffset,
                              runtime::KvDtype kvDtype,
                              bool useStagingSlot) {
    if (M == 0 || Nq == 0 || Nkv == 0) {
        return;
    }
    const std::size_t Nfused = Nq + Nkv * (hasV ? 2 : 1);
    const std::size_t total  = M * Nfused;

    // Yv may legitimately be nullptr when hasV is false. The kernel
    // guards against dereferencing it, but the HIP launch API still
    // expects a valid pointer for slot 3 — route to `fused` as a
    // safe non-null stub. Same trick as L0.
    const void* YvPtr = hasV ? YvBase : static_cast<const void*>(fused);

    // fp16 KV path routes to the fp16 variant. Yq stays fp32 in both;
    // only the K/V scatter store differs (vstore_half vs scalar).
    auto& k = (kvDtype == runtime::KvDtype::FP16)
                  ? _pimpl->_qkvSplitFp16Kernel
                  : _pimpl->_qkvSplitKernel;

    // Slot-swap discipline mirror of rmsNormQkvAsync — see that
    // method for the full rationale (avoids CLR replay races).
    std::int32_t* offsetSlot;
    if (useStagingSlot) {
        offsetSlot = _stagingOffsetSlotUsm;
    } else {
        const std::int32_t v = toInt32(writeOffset, "qkvSplit writeOffset");
        _ctx.allocator().copyH2D(_curLenSlotUsm, &v, sizeof(std::int32_t));
        offsetSlot = _curLenSlotUsm;
    }

    k.setPtr  (0, fused);
    k.setPtr  (1, Yq);
    k.setPtr  (2, YkBase);
    k.setPtr  (3, YvPtr);
    k.setValue(4, toInt32(M,      "qkvSplit M"));
    k.setValue(5, toInt32(Nq,     "qkvSplit Nq"));
    k.setValue(6, toInt32(Nkv,    "qkvSplit Nkv"));
    k.setValue(7, hasV ? std::int32_t{1} : std::int32_t{0});
    k.setValue(8, toInt32(Nfused, "qkvSplit Nfused"));
    k.setPtr  (9, offsetSlot);

    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::attentionAsync(const float* q, const void* k, const void* v,
                            std::size_t T_q, std::size_t T_k,
                            std::size_t nHeads, std::size_t nKvHeads,
                            std::size_t headDim,
                            std::size_t positionOffset,
                            float scale, float* out,
                            std::size_t slidingWindow,
                            runtime::KvDtype kvDtype) {
    if (T_q == 0 || T_k == 0 || nHeads == 0 || headDim == 0) {
        return;
    }
    if (nKvHeads == 0 || nHeads % nKvHeads != 0) {
        throw std::runtime_error(
            "compute::hip::GpuOps::attentionAsync: nHeads (" +
            std::to_string(nHeads) + ") must be a positive multiple of "
            "nKvHeads (" + std::to_string(nKvHeads) + ")");
    }

    // Dispatch mirrors compute::l0::GpuOps::attentionAsync:
    //   T_q == 1 → decode-flash (two-pass partial + merge)
    //   T_q >  1 → prefill-flash (single-WG streaming, if enabled)
    //   fallback → plain attention (T_k <= kAttentionMaxTk)
    if (T_q == 1) {
        if (nHeads > kFlashMaxHeads || headDim > kFlashMaxHeadDim) {
            throw std::runtime_error(
                "compute::hip::GpuOps::attentionAsync: flash path needs "
                "nHeads<=" + std::to_string(kFlashMaxHeads) +
                " and headDim<=" + std::to_string(kFlashMaxHeadDim) +
                " (got " + std::to_string(nHeads) + " / " +
                std::to_string(headDim) + ")");
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

void GpuOps::attentionPlainAsync(const float* q, const void* k, const void* v,
                                 std::size_t T_q, std::size_t T_k,
                                 std::size_t nHeads, std::size_t nKvHeads,
                                 std::size_t headDim,
                                 std::size_t positionOffset,
                                 float scale, float* out,
                                 std::size_t slidingWindow,
                                 runtime::KvDtype kvDtype) {
    if (T_k > kAttentionMaxTk) {
        throw std::runtime_error(
            "compute::hip::GpuOps::attentionPlainAsync: T_k=" +
            std::to_string(T_k) + " exceeds compile-time bound "
            "ATTN_MAX_TK=" + std::to_string(kAttentionMaxTk) +
            " — the plain-attention kernel holds scores[ATTN_MAX_TK] in "
            "LDS. Re-enable the flash path (features.prefillFlash: true) "
            "or reduce runtime.maxContextTokens below " +
            std::to_string(kAttentionMaxTk));
    }
    (void)T_k;

    core::hip::HipKernel* kernelPtr = &_pimpl->_attentionKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        kernelPtr = &_pimpl->_attentionFp16Kernel;
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        kernelPtr = &_pimpl->_attentionQ8Kernel;
    }
    auto& kernel = *kernelPtr;

    const std::int32_t posI =
        toInt32(positionOffset, "attention positionOffset");
    _ctx.allocator().copyH2D(_curLenSlotUsm, &posI, sizeof(std::int32_t));

    kernel.setPtr  (0, q);
    kernel.setPtr  (1, k);
    kernel.setPtr  (2, v);
    kernel.setPtr  (3, out);
    kernel.setValue(4, toInt32(T_q,      "attention T_q"));
    kernel.setValue(5, toInt32(nHeads,   "attention nHeads"));
    kernel.setValue(6, toInt32(nKvHeads, "attention nKvHeads"));
    kernel.setValue(7, toInt32(headDim,  "attention headDim"));
    kernel.setPtr  (8, _curLenSlotUsm);
    kernel.setValue(9, scale);
    kernel.setValue(10, toInt32(slidingWindow, "attention slidingWindow"));

    // One workgroup per (head, query-position).
    kernel.launch(_ctx.stream(),
                  static_cast<std::uint32_t>(nHeads),
                  static_cast<std::uint32_t>(T_q),
                  1,
                  kAttentionLocalSize, 1, 1);
}

void GpuOps::attentionPrefillFlashAsync(const float* q, const void* k,
                                        const void* v,
                                        std::size_t T_q,
                                        std::size_t nHeads,
                                        std::size_t nKvHeads,
                                        std::size_t headDim,
                                        std::size_t positionOffset,
                                        float scale, float* out,
                                        std::size_t slidingWindow,
                                        runtime::KvDtype kvDtype) {
    // Under Q8_0 with GQA shape, route to the head-packed kernel when
    // the config allows and nQPerKv is within the packed kernel's cap.
    // The HIP port currently ships only the KTILE=128 variant of the
    // packed kernel (attention_prefill_flash_q8_0_gqa.hip). L0 has a
    // second SPV compiled from the same source with -DATTN_KTILE=64;
    // on HIP that's a follow-up kernel-port. Until then, honour
    // `_prefillFlashKTileQ8` only as configuration bookkeeping — the
    // dispatch always uses the 128-tile kernel.
    const std::size_t nQPerKv = nHeads / nKvHeads;
    const bool useQ8Gqa =
        (kvDtype == runtime::KvDtype::Q8_0) &&
        !_prefillFlashGqaQ8Disabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv);

    core::hip::HipKernel* kernelPtr = &_pimpl->_attentionPrefillFlashKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        kernelPtr = &_pimpl->_attentionPrefillFlashFp16Kernel;
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        kernelPtr = useQ8Gqa
            ? &_pimpl->_attentionPrefillFlashQ8GqaKernel
            : &_pimpl->_attentionPrefillFlashQ8Kernel;
    }
    auto& kernel = *kernelPtr;

    const std::int32_t posI =
        toInt32(positionOffset, "prefill_flash positionOffset");
    _ctx.allocator().copyH2D(_curLenSlotUsm, &posI, sizeof(std::int32_t));

    kernel.setPtr  (0, q);
    kernel.setPtr  (1, k);
    kernel.setPtr  (2, v);
    kernel.setPtr  (3, out);
    kernel.setValue(4, toInt32(T_q,      "prefill_flash T_q"));
    kernel.setValue(5, toInt32(nHeads,   "prefill_flash nHeads"));
    kernel.setValue(6, toInt32(nKvHeads, "prefill_flash nKvHeads"));
    kernel.setValue(7, toInt32(headDim,  "prefill_flash headDim"));
    kernel.setPtr  (8, _curLenSlotUsm);
    kernel.setValue(9, scale);
    kernel.setValue(10, toInt32(slidingWindow, "prefill_flash slidingWindow"));

    // Plain kernels: one WG per (query-head, query-position).
    // GQA-packed kernel: one WG per (kv-head, query-position).
    const std::uint32_t dim0 = useQ8Gqa
        ? static_cast<std::uint32_t>(nKvHeads)
        : static_cast<std::uint32_t>(nHeads);
    kernel.launch(_ctx.stream(),
                  dim0,
                  static_cast<std::uint32_t>(T_q),
                  1,
                  kAttentionLocalSize, 1, 1);
}

void GpuOps::attentionDecodeFlashAsync(const float* q, const void* k,
                                       const void* v,
                                       std::size_t T_k,
                                       std::size_t nHeads,
                                       std::size_t nKvHeads,
                                       std::size_t headDim,
                                       std::size_t positionOffset,
                                       float scale, float* out,
                                       std::size_t slidingWindow,
                                       runtime::KvDtype kvDtype) {
    const std::size_t kMax =
        (positionOffset + 1 < T_k) ? (positionOffset + 1) : T_k;
    const std::size_t nKTiles =
        (kMax + kFlashKTileSize - 1) / kFlashKTileSize;
    if (nKTiles == 0 || nKTiles > kFlashMaxKTiles) {
        throw std::runtime_error(
            "compute::hip::GpuOps::attentionDecodeFlashAsync: nKTiles=" +
            std::to_string(nKTiles) + " out of [1, " +
            std::to_string(kFlashMaxKTiles) + "]");
    }
    (void)T_k;

    const std::int32_t posI =
        toInt32(positionOffset, "flash positionOffset");
    _ctx.allocator().copyH2D(_curLenSlotUsm, &posI, sizeof(std::int32_t));

    // Pass 1 — per-tile partial (m, l, o_unnorm) into persistent scratch.
    // Kernel by KV dtype; partial layout stays fp32 regardless so the
    // merge kernel is dtype-agnostic.
    core::hip::HipKernel* partialKernelPtr =
        &_pimpl->_attentionFlashPartialKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        partialKernelPtr = &_pimpl->_attentionFlashPartialFp16Kernel;
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        partialKernelPtr = &_pimpl->_attentionFlashPartialQ8Kernel;
    }
    auto& partialKernel = *partialKernelPtr;

    partialKernel.setPtr  (0, q);
    partialKernel.setPtr  (1, k);
    partialKernel.setPtr  (2, v);
    partialKernel.setPtr  (3, _flashPartialUsm);
    partialKernel.setValue(4, toInt32(nHeads,   "flash nHeads"));
    partialKernel.setValue(5, toInt32(nKvHeads, "flash nKvHeads"));
    partialKernel.setValue(6, toInt32(headDim,  "flash headDim"));
    partialKernel.setPtr  (7, _curLenSlotUsm);
    partialKernel.setValue(8, scale);
    partialKernel.setValue(9, toInt32(slidingWindow, "flash slidingWindow"));

    // HIP has no hipGraph recording yet, so launch geometry always uses
    // the actual nKTiles (immediate-mode optimal). `_replayMaxKTiles` is
    // kept as bookkeeping for the future hipGraph landing (Schritt 5).
    partialKernel.launch(_ctx.stream(),
                         static_cast<std::uint32_t>(nHeads),
                         static_cast<std::uint32_t>(nKTiles),
                         1,
                         kAttentionLocalSize, 1, 1);

    // Pass 2 — merge per-tile partials into the final output. HIP kernel
    // launches on the same stream serialise implicitly, so the merge
    // sees committed partials without an explicit barrier.
    auto& mergeKernel = _pimpl->_attentionFlashMergeKernel;
    mergeKernel.setPtr  (0, _flashPartialUsm);
    mergeKernel.setPtr  (1, out);
    mergeKernel.setValue(2, toInt32(nHeads,  "flash_merge nHeads"));
    mergeKernel.setValue(3, toInt32(headDim, "flash_merge headDim"));
    mergeKernel.setPtr  (4, _curLenSlotUsm);
    mergeKernel.launch(_ctx.stream(),
                       static_cast<std::uint32_t>(nHeads),
                       1, 1,
                       kAttentionLocalSize, 1, 1);
}

void GpuOps::matmulQ8_0VecReorderAsync(const void* wReordered,
                                          std::size_t N, std::size_t K,
                                          const float* x, float* y) {
    if (N == 0 || K == 0) {
        return;
    }
    // Launch geometry matches matmul_q8_0_vec: local=64, subgroup=16,
    // 4 outputs per workgroup, one workgroup per group of 4 output rows.
    // Kept in sync with GpuMatmul::kLocalSize / kOutputsPerGroup by the
    // kernel macros MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG.
    constexpr std::uint32_t kLocalSize       = 64;
    constexpr std::uint32_t kOutputsPerGroup = 4;

    const std::int32_t Ki = toInt32(K, "matmulQ8_0VecReorder K");
    const std::int32_t Ni = toInt32(N, "matmulQ8_0VecReorder N");

    auto& k = _pimpl->_matmulQ8_0VecReorderKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, wReordered);
    k.setPtr  (2, y);
    k.setValue(3, Ki);
    k.setValue(4, Ni);

    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    k.launch(_ctx.stream(),
             nGroups, 1, 1,
             kLocalSize, 1, 1);
}

} // namespace mimirmind::compute::hip
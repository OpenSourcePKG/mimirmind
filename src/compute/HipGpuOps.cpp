// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/HipGpuOps.hpp"

#include "core/gpu/hip/HipComputeContext.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mimirmind::compute {

namespace {

constexpr const char* kDefaultHsacoDir = "/usr/local/share/mimirmind/hsaco";

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
        "HipGpuOps: cannot find " + filename +
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
        std::string{"HipGpuOps::"} + method +
        ": not yet implemented (Schritt 3b skeleton — kernel-launch "
        "impl lands in follow-up commits)");
}

} // namespace

// Pimpl body — one HipModule + HipKernel pair per compiled `.hip`
// source under `kernels_hip/` that corresponds to a `ComputeOps`
// entry point. HIP-only kernels used by `HipGpuMatmul` (matmul
// variants + moe_down) live on that class, not here. Kernels that
// exist in the L0 `GpuOps::Impl` but haven't been ported to HIP yet
// (gelu_mul, add_rmsnorm, rope_inplace_ff{,_fp16}, rope_inplace_fp16,
// scaled_add_residual, x_quant_i8, attention_prefill_flash_q8_0_gqa
// KTILE=64 variant) are deliberately absent — the corresponding
// ComputeOps overrides throw `not yet implemented` at dispatch time.
struct HipGpuOps::Impl {
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

    core::hip::HipModule _addBiasModule;
    core::hip::HipKernel _addBiasKernel;
    core::hip::HipModule _addResidualModule;
    core::hip::HipKernel _addResidualKernel;
    core::hip::HipModule _siluMulModule;
    core::hip::HipKernel _siluMulKernel;
    core::hip::HipModule _mulScalarModule;
    core::hip::HipKernel _mulScalarKernel;
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

          _addBiasModule           {loadHipModule(ctx, "add_bias")},
          _addBiasKernel           {_addBiasModule.getKernel("add_bias")},
          _addResidualModule       {loadHipModule(ctx, "add_residual")},
          _addResidualKernel       {_addResidualModule.getKernel("add_residual")},
          _siluMulModule           {loadHipModule(ctx, "silu_mul")},
          _siluMulKernel           {_siluMulModule.getKernel("silu_mul")},
          _mulScalarModule         {loadHipModule(ctx, "mul_scalar")},
          _mulScalarKernel         {_mulScalarModule.getKernel("mul_scalar")},
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

HipGpuOps::HipGpuOps(core::hip::HipComputeContext& ctx,
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
            "HipGpuOps: features.flashPrefillKTileQ8=" +
            std::to_string(flashPrefillKTileQ8) +
            " unexpected — Config.cpp parser should have rejected this");
    }

    MM_LOG_INFO("hipgpuops",
                "HipGpuOps ready — 25 modules loaded (rmsnorm variants, "
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

HipGpuOps::~HipGpuOps() {
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

core::hip::HipStream& HipGpuOps::stream() noexcept {
    return _ctx.stream();
}

core::hip::HipMemoryAllocator& HipGpuOps::allocator() noexcept {
    return _ctx.allocator();
}

std::string_view HipGpuOps::q8_0ReorderModeName() const noexcept {
    switch (_q8_0ReorderMode) {
        case core::config::TriState::Auto:    return "auto";
        case core::config::TriState::Force:   return "force";
        case core::config::TriState::Disable: return "disable";
    }
    return "unknown";
}

void HipGpuOps::noteQ8_0ReorderApplied(std::size_t bytes,
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

void HipGpuOps::rmsNormAsync(const float*, std::size_t, std::size_t,
                             const float*, float, float*) {
    throwNotImplemented("rmsNormAsync");
}

void HipGpuOps::rmsNormGemmaAsync(const float*, std::size_t, std::size_t,
                                  const float*, float, float*) {
    throwNotImplemented("rmsNormGemmaAsync");
}

void HipGpuOps::rmsNormNoWeightAsync(const float*, std::size_t, std::size_t,
                                     float, float*) {
    throwNotImplemented("rmsNormNoWeightAsync");
}

void HipGpuOps::rmsNormQkvAsync(float*, const float*,
                                void*, const float*,
                                void*,
                                std::size_t, std::size_t, std::size_t,
                                float,
                                std::size_t, std::size_t,
                                runtime::KvDtype, bool) {
    throwNotImplemented("rmsNormQkvAsync");
}

void HipGpuOps::addRmsNormAsync(float*, const float*,
                                std::size_t, std::size_t,
                                const float*, float, float*) {
    // add_rmsnorm kernel not yet ported to HIP — the L0 side has
    // a fused kernel; HIP would either need the port or a two-launch
    // fallback (addResidualAsync + rmsNormAsync) once those exist.
    throwNotImplemented("addRmsNormAsync");
}

void HipGpuOps::addBiasAsync(float*, std::size_t, std::size_t,
                             const float*) {
    throwNotImplemented("addBiasAsync");
}

void HipGpuOps::addResidualAsync(float*, const float*, std::size_t) {
    throwNotImplemented("addResidualAsync");
}

void HipGpuOps::siluMulAsync(float*, const float*, std::size_t) {
    throwNotImplemented("siluMulAsync");
}

void HipGpuOps::geluMulAsync(float*, const float*, std::size_t) {
    // gelu_mul.hip not yet ported.
    throwNotImplemented("geluMulAsync");
}

void HipGpuOps::mulScalarAsync(float*, float, std::size_t) {
    throwNotImplemented("mulScalarAsync");
}

void HipGpuOps::scaledAddResidualAsync(float*, const float*, float, std::size_t) {
    // scaled_add_residual.hip not yet ported.
    throwNotImplemented("scaledAddResidualAsync");
}

void HipGpuOps::ropeInPlaceAsync(void*, std::size_t, std::size_t,
                                 std::size_t, std::size_t, float,
                                 std::size_t, runtime::KvDtype) {
    // f32 variant loaded (`rope_inplace.hip`); fp16 variant not ported.
    // Once the launch code lands, `kvDtype == FP16` will still throw
    // until `rope_inplace_fp16.hip` gets a port.
    throwNotImplemented("ropeInPlaceAsync");
}

void HipGpuOps::ropeInPlaceWithFactorsAsync(void*, const float*,
                                            std::size_t, std::size_t,
                                            std::size_t, std::size_t, float,
                                            std::size_t, runtime::KvDtype) {
    // rope_inplace_ff.hip not yet ported (nor its _fp16 variant).
    throwNotImplemented("ropeInPlaceWithFactorsAsync");
}

void HipGpuOps::xQuantI8Async(const float*, std::int8_t*, float*,
                              std::size_t, std::size_t) {
    // x_quant_i8.hip not yet ported. Blocks the DP4A matvec path in
    // `HipGpuMatmul` — plain matvec will still work once that lands.
    throwNotImplemented("xQuantI8Async");
}

void HipGpuOps::kvQuantCommitQ8Async(const float*, void*,
                                     std::size_t, std::size_t, std::size_t) {
    throwNotImplemented("kvQuantCommitQ8Async");
}

void HipGpuOps::qkvSplitAsync(const float*, float*, void*, void*,
                              std::size_t, std::size_t, std::size_t,
                              bool,
                              std::size_t, runtime::KvDtype, bool) {
    throwNotImplemented("qkvSplitAsync");
}

void HipGpuOps::attentionAsync(const float*, const void*, const void*,
                               std::size_t, std::size_t,
                               std::size_t, std::size_t, std::size_t,
                               std::size_t,
                               float, float*,
                               std::size_t, runtime::KvDtype) {
    // Dispatch fan-out (plain / prefill-flash / decode-flash by T_q,
    // T_k, positionOffset) lands in the follow-up. Both KTILE variants
    // will need the second .hip compile that L0 gets from ocloc; on
    // HIP we either compile with -DATTN_KTILE=<n> a second time or
    // fold both KTILE=64 and KTILE=128 into the same kernel with a
    // runtime arg.
    throwNotImplemented("attentionAsync");
}

void HipGpuOps::matmulQ8_0VecReorderAsync(const void*, std::size_t, std::size_t,
                                          const float*, float*) {
    throwNotImplemented("matmulQ8_0VecReorderAsync");
}

} // namespace mimirmind::compute
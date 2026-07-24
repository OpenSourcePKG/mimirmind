// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/GpuOps.hpp"

#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaMemoryAllocator.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/log/Log.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::cuda {

namespace {

constexpr const char* kDefaultPtxDir = "/usr/local/share/mimirmind/ptx";

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

// Resolve `<name>.ptx` in one of:
//   1. $MIMIRMIND_HSACO_DIR (env var — HIP analog of runtime.spvDir)
//   2. /usr/local/share/mimirmind/ptx (production install)
//   3. build-tree fallbacks (build*/ptx, ../build*/ptx, ptx)
// Mirrors `resolveSpvPath` in `GpuModule.cpp` — same three-tier lookup
// so the deployment stories stay parallel.
std::filesystem::path resolveHsacoPath(std::string_view name) {
    const std::string filename = std::string{name} + ".ptx";

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
            std::filesystem::path{kDefaultPtxDir} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    for (auto rel : std::array<const char*, 5>{
             "build/ptx", "build-both/ptx",
             "../build/ptx", "../build-both/ptx",
             "ptx"}) {
        const std::filesystem::path p =
            std::filesystem::path{rel} / filename;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    throw std::runtime_error(
        "hip::GpuOps: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultPtxDir);
}

// Convenience: load a CudaModule by kernel name, resolving the .ptx
// path through `resolveHsacoPath`. Symbol name inside the module is
// assumed to be identical to the file basename (mirrors the L0 side
// where `.spv` filename == kernel `__kernel` symbol).
core::cuda::CudaModule loadCudaModule(core::cuda::CudaContext& ctx,
                                   std::string_view       name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("hipgpuops", "loading module '{}' from {}",
                std::string{name}, path.string());
    return core::cuda::CudaModule::fromFile(ctx, path.string());
}

// Named throw helper — keeps every stub one line and gives an
// immediately-actionable message when a caller trips on a not-yet-
// implemented dispatch.
[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::cuda::GpuOps::"} + method +
        ": not yet implemented (Schritt 3b skeleton — kernel-launch "
        "impl lands in follow-up commits)");
}

} // namespace

// Pimpl body — one CudaModule + CudaKernel pair per compiled `.hip`
// source under `kernels_hip/` that corresponds to a `ComputeOps`
// entry point. HIP-only kernels used by `HipGpuMatmul` (matmul
// variants + moe_down) live on that class, not here. Kernels that
// exist in the L0 `GpuOps::Impl` but haven't been ported to HIP yet
// (rope_inplace_ff_fp16) are deliberately absent — the corresponding
// ComputeOps overrides throw a clear diagnostic at dispatch time.
struct GpuOps::Impl {
    core::cuda::CudaModule _rmsnormModule;
    core::cuda::CudaKernel _rmsnormKernel;
    core::cuda::CudaModule _rmsnormGemmaModule;
    core::cuda::CudaKernel _rmsnormGemmaKernel;
    core::cuda::CudaModule _rmsnormNoWeightModule;
    core::cuda::CudaKernel _rmsnormNoWeightKernel;
    core::cuda::CudaModule _rmsnormQkvModule;
    core::cuda::CudaKernel _rmsnormQkvKernel;
    core::cuda::CudaModule _rmsnormQkvFp16Module;
    core::cuda::CudaKernel _rmsnormQkvFp16Kernel;
    core::cuda::CudaModule _addRmsNormModule;
    core::cuda::CudaKernel _addRmsNormKernel;

    core::cuda::CudaModule _addBiasModule;
    core::cuda::CudaKernel _addBiasKernel;
    core::cuda::CudaModule _addResidualModule;
    core::cuda::CudaKernel _addResidualKernel;
    core::cuda::CudaModule _siluMulModule;
    core::cuda::CudaKernel _siluMulKernel;
    core::cuda::CudaModule _geluMulModule;
    core::cuda::CudaKernel _geluMulKernel;
    core::cuda::CudaModule _mulScalarModule;
    core::cuda::CudaKernel _mulScalarKernel;
    core::cuda::CudaModule _scaledAddResidualModule;
    core::cuda::CudaKernel _scaledAddResidualKernel;
    core::cuda::CudaModule _xQuantI8Module;
    core::cuda::CudaKernel _xQuantI8Kernel;
    core::cuda::CudaModule _ropeModule;
    core::cuda::CudaKernel _ropeKernel;
    core::cuda::CudaModule _ropeFp16Module;
    core::cuda::CudaKernel _ropeFp16Kernel;
    core::cuda::CudaModule _ropeFfModule;
    core::cuda::CudaKernel _ropeFfKernel;

    core::cuda::CudaModule _attentionModule;
    core::cuda::CudaKernel _attentionKernel;
    core::cuda::CudaModule _attentionFp16Module;
    core::cuda::CudaKernel _attentionFp16Kernel;
    core::cuda::CudaModule _attentionQ8Module;
    core::cuda::CudaKernel _attentionQ8Kernel;
    core::cuda::CudaModule _attentionFlashPartialModule;
    core::cuda::CudaKernel _attentionFlashPartialKernel;
    core::cuda::CudaModule _attentionFlashPartialFp16Module;
    core::cuda::CudaKernel _attentionFlashPartialFp16Kernel;
    core::cuda::CudaModule _attentionFlashPartialQ8Module;
    core::cuda::CudaKernel _attentionFlashPartialQ8Kernel;
    core::cuda::CudaModule _attentionFlashMergeModule;
    core::cuda::CudaKernel _attentionFlashMergeKernel;
    core::cuda::CudaModule _attentionFlashPartialBatchedModule;
    core::cuda::CudaKernel _attentionFlashPartialBatchedKernel;
    core::cuda::CudaModule _attentionFlashMergeBatchedModule;
    core::cuda::CudaKernel _attentionFlashMergeBatchedKernel;

    core::cuda::CudaModule _attentionPrefillFlashModule;
    core::cuda::CudaKernel _attentionPrefillFlashKernel;
    core::cuda::CudaModule _attentionPrefillFlashFp16Module;
    core::cuda::CudaKernel _attentionPrefillFlashFp16Kernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8Module;
    core::cuda::CudaKernel _attentionPrefillFlashQ8Kernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8GqaModule;
    core::cuda::CudaKernel _attentionPrefillFlashQ8GqaKernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8GqaKtile64Module;
    core::cuda::CudaKernel _attentionPrefillFlashQ8GqaKtile64Kernel;

    core::cuda::CudaModule _qkvSplitModule;
    core::cuda::CudaKernel _qkvSplitKernel;
    core::cuda::CudaModule _qkvSplitFp16Module;
    core::cuda::CudaKernel _qkvSplitFp16Kernel;

    core::cuda::CudaModule _kvQuantCommitQ8Module;
    core::cuda::CudaKernel _kvQuantCommitQ8Kernel;

    core::cuda::CudaModule _matmulQ8_0VecReorderModule;
    core::cuda::CudaKernel _matmulQ8_0VecReorderKernel;

    core::cuda::CudaModule _mropeModule;
    core::cuda::CudaKernel _mropeKernel;
    core::cuda::CudaModule _mropeBatchedModule;
    core::cuda::CudaKernel _mropeBatchedKernel;
    core::cuda::CudaModule _splitHeadPairModule;
    core::cuda::CudaKernel _splitHeadPairKernel;
    core::cuda::CudaModule _sigmoidGateMulModule;
    core::cuda::CudaKernel _sigmoidGateMulKernel;
    core::cuda::CudaModule _l2NormModule;
    core::cuda::CudaKernel _l2NormKernel;
    core::cuda::CudaModule _ssmConv1dModule;
    core::cuda::CudaKernel _ssmConv1dKernel;
    core::cuda::CudaModule _ssmConv1dBatchedModule;
    core::cuda::CudaKernel _ssmConv1dBatchedKernel;
    core::cuda::CudaModule _gatedDeltaNetArModule;
    core::cuda::CudaKernel _gatedDeltaNetArKernel;
    core::cuda::CudaModule _gatedDeltaNetArBatchedModule;
    core::cuda::CudaKernel _gatedDeltaNetArBatchedKernel;
    core::cuda::CudaModule _deltanetGateModule;
    core::cuda::CudaKernel _deltanetGateKernel;
    core::cuda::CudaModule _deltanetChunkCumGateModule;
    core::cuda::CudaKernel _deltanetChunkCumGateKernel;
    core::cuda::CudaModule _deltanetChunkForwardModule;
    core::cuda::CudaKernel _deltanetChunkForwardKernel;
    core::cuda::CudaModule _deltanetChunkCumGateBatchedModule;
    core::cuda::CudaKernel _deltanetChunkCumGateBatchedKernel;
    core::cuda::CudaModule _deltanetChunkForwardBatchedModule;
    core::cuda::CudaKernel _deltanetChunkForwardBatchedKernel;
    core::cuda::CudaModule _deltanetKktSolveModule;
    core::cuda::CudaKernel _deltanetKktSolveKernel;
    core::cuda::CudaModule _sigmoidInplaceModule;
    core::cuda::CudaKernel _sigmoidInplaceKernel;
    core::cuda::CudaModule _gatherHeadsModule;
    core::cuda::CudaKernel _gatherHeadsKernel;

    explicit Impl(core::cuda::CudaContext& ctx)
        : _rmsnormModule           {loadCudaModule(ctx, "rmsnorm")},
          _rmsnormKernel           {_rmsnormModule.getFunction("rmsnorm")},
          _rmsnormGemmaModule      {loadCudaModule(ctx, "rmsnorm_gemma")},
          _rmsnormGemmaKernel      {_rmsnormGemmaModule.getFunction("rmsnorm_gemma")},
          _rmsnormNoWeightModule   {loadCudaModule(ctx, "rmsnorm_no_weight")},
          _rmsnormNoWeightKernel   {_rmsnormNoWeightModule.getFunction("rmsnorm_no_weight")},
          _rmsnormQkvModule        {loadCudaModule(ctx, "rmsnorm_qkv")},
          _rmsnormQkvKernel        {_rmsnormQkvModule.getFunction("rmsnorm_qkv")},
          _rmsnormQkvFp16Module    {loadCudaModule(ctx, "rmsnorm_qkv_fp16")},
          _rmsnormQkvFp16Kernel    {_rmsnormQkvFp16Module.getFunction("rmsnorm_qkv_fp16")},
          _addRmsNormModule        {loadCudaModule(ctx, "add_rmsnorm")},
          _addRmsNormKernel        {_addRmsNormModule.getFunction("add_rmsnorm")},

          _addBiasModule           {loadCudaModule(ctx, "add_bias")},
          _addBiasKernel           {_addBiasModule.getFunction("add_bias")},
          _addResidualModule       {loadCudaModule(ctx, "add_residual")},
          _addResidualKernel       {_addResidualModule.getFunction("add_residual")},
          _siluMulModule           {loadCudaModule(ctx, "silu_mul")},
          _siluMulKernel           {_siluMulModule.getFunction("silu_mul")},
          _geluMulModule           {loadCudaModule(ctx, "gelu_mul")},
          _geluMulKernel           {_geluMulModule.getFunction("gelu_mul")},
          _mulScalarModule         {loadCudaModule(ctx, "mul_scalar")},
          _mulScalarKernel         {_mulScalarModule.getFunction("mul_scalar")},
          _scaledAddResidualModule {loadCudaModule(ctx, "scaled_add_residual")},
          _scaledAddResidualKernel {
              _scaledAddResidualModule.getFunction("scaled_add_residual")},
          _xQuantI8Module          {loadCudaModule(ctx, "x_quant_i8")},
          _xQuantI8Kernel          {_xQuantI8Module.getFunction("x_quant_i8")},
          _ropeModule              {loadCudaModule(ctx, "rope_inplace")},
          _ropeKernel              {_ropeModule.getFunction("rope_inplace")},
          _ropeFp16Module          {loadCudaModule(ctx, "rope_inplace_fp16")},
          _ropeFp16Kernel          {_ropeFp16Module.getFunction("rope_inplace_fp16")},
          _ropeFfModule            {loadCudaModule(ctx, "rope_inplace_ff")},
          _ropeFfKernel            {_ropeFfModule.getFunction("rope_inplace_ff")},

          _attentionModule         {loadCudaModule(ctx, "attention")},
          _attentionKernel         {_attentionModule.getFunction("attention")},
          _attentionFp16Module     {loadCudaModule(ctx, "attention_fp16")},
          _attentionFp16Kernel     {_attentionFp16Module.getFunction("attention_fp16")},
          _attentionQ8Module       {loadCudaModule(ctx, "attention_q8_0")},
          _attentionQ8Kernel       {_attentionQ8Module.getFunction("attention_q8_0")},
          _attentionFlashPartialModule{loadCudaModule(ctx, "attention_flash_partial")},
          _attentionFlashPartialKernel{
              _attentionFlashPartialModule.getFunction("attention_flash_partial")},
          _attentionFlashPartialFp16Module{
              loadCudaModule(ctx, "attention_flash_partial_fp16")},
          _attentionFlashPartialFp16Kernel{
              _attentionFlashPartialFp16Module.getFunction("attention_flash_partial_fp16")},
          _attentionFlashPartialQ8Module{
              loadCudaModule(ctx, "attention_flash_partial_q8_0")},
          _attentionFlashPartialQ8Kernel{
              _attentionFlashPartialQ8Module.getFunction("attention_flash_partial_q8_0")},
          _attentionFlashMergeModule{loadCudaModule(ctx, "attention_flash_merge")},
          _attentionFlashMergeKernel{
              _attentionFlashMergeModule.getFunction("attention_flash_merge")},
          _attentionFlashPartialBatchedModule{loadCudaModule(ctx, "attention_flash_partial_batched")},
          _attentionFlashPartialBatchedKernel{
              _attentionFlashPartialBatchedModule.getFunction("attention_flash_partial_batched")},
          _attentionFlashMergeBatchedModule{loadCudaModule(ctx, "attention_flash_merge_batched")},
          _attentionFlashMergeBatchedKernel{
              _attentionFlashMergeBatchedModule.getFunction("attention_flash_merge_batched")},

          _attentionPrefillFlashModule{loadCudaModule(ctx, "attention_prefill_flash")},
          _attentionPrefillFlashKernel{
              _attentionPrefillFlashModule.getFunction("attention_prefill_flash")},
          _attentionPrefillFlashFp16Module{
              loadCudaModule(ctx, "attention_prefill_flash_fp16")},
          _attentionPrefillFlashFp16Kernel{
              _attentionPrefillFlashFp16Module.getFunction("attention_prefill_flash_fp16")},
          _attentionPrefillFlashQ8Module{
              loadCudaModule(ctx, "attention_prefill_flash_q8_0")},
          _attentionPrefillFlashQ8Kernel{
              _attentionPrefillFlashQ8Module.getFunction("attention_prefill_flash_q8_0")},
          _attentionPrefillFlashQ8GqaModule{
              loadCudaModule(ctx, "attention_prefill_flash_q8_0_gqa")},
          _attentionPrefillFlashQ8GqaKernel{
              _attentionPrefillFlashQ8GqaModule.getFunction(
                  "attention_prefill_flash_q8_0_gqa")},
          _attentionPrefillFlashQ8GqaKtile64Module{
              loadCudaModule(ctx, "attention_prefill_flash_q8_0_gqa_ktile64")},
          _attentionPrefillFlashQ8GqaKtile64Kernel{
              _attentionPrefillFlashQ8GqaKtile64Module.getFunction(
                  "attention_prefill_flash_q8_0_gqa")},

          _qkvSplitModule          {loadCudaModule(ctx, "qkv_split")},
          _qkvSplitKernel          {_qkvSplitModule.getFunction("qkv_split")},
          _qkvSplitFp16Module      {loadCudaModule(ctx, "qkv_split_fp16")},
          _qkvSplitFp16Kernel      {_qkvSplitFp16Module.getFunction("qkv_split_fp16")},

          _kvQuantCommitQ8Module   {loadCudaModule(ctx, "kv_quant_commit_q8_0")},
          _kvQuantCommitQ8Kernel   {
              _kvQuantCommitQ8Module.getFunction("kv_quant_commit_q8_0")},

          _matmulQ8_0VecReorderModule{loadCudaModule(ctx, "matmul_q8_0_vec_reorder")},
          _matmulQ8_0VecReorderKernel{
              _matmulQ8_0VecReorderModule.getFunction("matmul_q8_0_vec_reorder")},
          _mropeModule             {loadCudaModule(ctx, "rope_mrope")},
          _mropeKernel             {_mropeModule.getFunction("rope_mrope")},
          _mropeBatchedModule      {loadCudaModule(ctx, "rope_mrope_batched")},
          _mropeBatchedKernel      {_mropeBatchedModule.getFunction("rope_mrope_batched")},
          _splitHeadPairModule     {loadCudaModule(ctx, "split_head_pair")},
          _splitHeadPairKernel     {_splitHeadPairModule.getFunction("split_head_pair")},
          _sigmoidGateMulModule    {loadCudaModule(ctx, "sigmoid_gate_mul")},
          _sigmoidGateMulKernel    {
              _sigmoidGateMulModule.getFunction("sigmoid_gate_mul")},
          _l2NormModule            {loadCudaModule(ctx, "l2_norm")},
          _l2NormKernel            {_l2NormModule.getFunction("l2_norm")},
          _ssmConv1dModule         {loadCudaModule(ctx, "ssm_conv1d")},
          _ssmConv1dKernel         {_ssmConv1dModule.getFunction("ssm_conv1d")},
          _ssmConv1dBatchedModule  {loadCudaModule(ctx, "ssm_conv1d_batched")},
          _ssmConv1dBatchedKernel  {_ssmConv1dBatchedModule.getFunction("ssm_conv1d_batched")},
          _gatedDeltaNetArModule   {loadCudaModule(ctx, "gated_deltanet_ar")},
          _gatedDeltaNetArKernel   {
              _gatedDeltaNetArModule.getFunction("gated_deltanet_ar")},
          _gatedDeltaNetArBatchedModule{loadCudaModule(ctx, "gated_deltanet_ar_batched")},
          _gatedDeltaNetArBatchedKernel{
              _gatedDeltaNetArBatchedModule.getFunction("gated_deltanet_ar_batched")},
          _deltanetGateModule      {loadCudaModule(ctx, "deltanet_gate")},
          _deltanetGateKernel      {_deltanetGateModule.getFunction("deltanet_gate")},
          _deltanetChunkCumGateModule{loadCudaModule(ctx, "deltanet_chunk_cumgate")},
          _deltanetChunkCumGateKernel{
              _deltanetChunkCumGateModule.getFunction("deltanet_chunk_cumgate")},
          _deltanetChunkForwardModule{loadCudaModule(ctx, "deltanet_chunk_forward")},
          _deltanetChunkForwardKernel{
              _deltanetChunkForwardModule.getFunction("deltanet_chunk_forward")},
          _deltanetChunkCumGateBatchedModule{loadCudaModule(ctx, "deltanet_chunk_cumgate_batched")},
          _deltanetChunkCumGateBatchedKernel{
              _deltanetChunkCumGateBatchedModule.getFunction("deltanet_chunk_cumgate_batched")},
          _deltanetChunkForwardBatchedModule{loadCudaModule(ctx, "deltanet_chunk_forward_batched")},
          _deltanetChunkForwardBatchedKernel{
              _deltanetChunkForwardBatchedModule.getFunction("deltanet_chunk_forward_batched")},
          _deltanetKktSolveModule{loadCudaModule(ctx, "deltanet_kkt_solve")},
          _deltanetKktSolveKernel{
              _deltanetKktSolveModule.getFunction("deltanet_kkt_solve")},
          _sigmoidInplaceModule    {loadCudaModule(ctx, "sigmoid_inplace")},
          _sigmoidInplaceKernel    {
              _sigmoidInplaceModule.getFunction("sigmoid_inplace")},
          _gatherHeadsModule       {loadCudaModule(ctx, "gather_heads_from_channels")},
          _gatherHeadsKernel       {
              _gatherHeadsModule.getFunction("gather_heads_from_channels")}
    {}
};

GpuOps::GpuOps(core::cuda::CudaComputeContext& ctx,
                     bool                          flashPrefillEnabled,
                     bool                          flashPrefillGqaQ8Enabled,
                     std::size_t                   flashPrefillKTileQ8,
                     core::config::TriState        q8_0ReorderMode)
    : _ctx{ctx},
      _pimpl{std::make_unique<Impl>(ctx.cudaContext())},
      _moeTopKRoute{ctx}
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
    // `cudaMemcpy(H2D)` this before each dispatch — no zero-copy path
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

    // Pinned host ring for scalar H2D staging — the hot decode path
    // updates `_curLenSlotUsm` once per attention/rope call (~3 per
    // block × N blocks per token). Sync `hipMemcpy` from stack forced
    // the host to wait for the compute stream to drain each time —
    // profiling showed the GPU was 96% idle in decode as a result.
    // Pinned source lets `cudaMemcpyAsync` truly enqueue without
    // stalling. Ring cycles cleanly (256 slots > any in-flight batch).
    _scalarRing = static_cast<std::int32_t*>(
        alloc.allocate(kScalarRingSize * sizeof(std::int32_t),
                       core::cuda::CudaAllocKind::HostPinned));
    _scalarRingIdx = 0;

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
                "hip::GpuOps ready — 32 modules loaded (rmsnorm variants, "
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
    if (_scalarRing) {
        alloc.deallocate(_scalarRing,
                         kScalarRingSize * sizeof(std::int32_t),
                         core::cuda::CudaAllocKind::HostPinned);
    }
    if (_stagingOffsetSlotUsm) {
        alloc.deallocate(_stagingOffsetSlotUsm, sizeof(std::int32_t),
                         core::cuda::CudaAllocKind::Device);
    }
    if (_curLenSlotUsm) {
        alloc.deallocate(_curLenSlotUsm, sizeof(std::int32_t),
                         core::cuda::CudaAllocKind::Device);
    }
    if (_flashPartialUsm) {
        alloc.deallocate(_flashPartialUsm, _flashPartialBytes,
                         core::cuda::CudaAllocKind::Device);
    }
}

void GpuOps::stagedInt32ToDevice(std::int32_t* devicePtr,
                                 std::int32_t  value) {
    // During graph capture/replay the engine owns _curLenSlotUsm and updates
    // it once per token outside the graph; skip the per-kernel copy so no
    // record-time value is baked into the captured DAG.
    if (!_perKernelCurLenStaging && devicePtr == _curLenSlotUsm) {
        return;
    }
    std::int32_t* slot = &_scalarRing[_scalarRingIdx];
    *slot = value;
    _scalarRingIdx = (_scalarRingIdx + 1) & (kScalarRingSize - 1);
    appendMemoryCopy(devicePtr, slot, sizeof(std::int32_t));
}

void GpuOps::updateDecodeCurLen(std::int32_t v) {
    // Raw staging bypassing the gate — the single per-token curLen update the
    // engine issues outside the captured region.
    std::int32_t* slot = &_scalarRing[_scalarRingIdx];
    *slot = v;
    _scalarRingIdx = (_scalarRingIdx + 1) & (kScalarRingSize - 1);
    appendMemoryCopy(_curLenSlotUsm, slot, sizeof(std::int32_t));
}

// ---- Real (non-stub) implementations --------------------------------

core::cuda::CudaStream& GpuOps::stream() noexcept {
    return _ctx.stream();
}

core::cuda::CudaMemoryAllocator& GpuOps::allocator() noexcept {
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

// Schritt 3c.1 — neutral stream / recording ops. HIP has no direct
// analogue of L0's UnorderedScope: streams on RDNA schedule kernel
// launches concurrently by default (dependency-tracked by the driver
// via resource use, not by a "strict order" flag). So push/pop are
// no-ops — the scope reads like documentation of a concurrent
// section rather than actually toggling behaviour. `flush()` is the
// stream sync; `appendMemoryCopy` is a stream-ordered async memcpy.
void GpuOps::pushUnorderedScope() { /* HIP streams reorder freely — no-op */ }
void GpuOps::popUnorderedScope()  { /* no-op counterpart */ }

void GpuOps::appendMemoryCopy(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    const cudaError_t rc = cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyDefault, _ctx.stream().handle());
    if (rc != cudaSuccess) {
        throw std::runtime_error(
            std::string{"compute::cuda::GpuOps::appendMemoryCopy: "
                        "cudaMemcpyAsync failed: "} + cudaGetErrorString(rc));
    }
}

void GpuOps::flush() {
    _ctx.stream().synchronize();
}

void GpuOps::readbackToHost(void* hostDst, const void* deviceSrc,
                            std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    // Wait for any pending compute stream work (lm_head matmul in
    // particular) before pulling bytes back — otherwise we'd read
    // stale contents. `hipMemcpy` D2H is itself synchronous, so the
    // stream sync is defensive against unrelated pending work on
    // other paths.
    _ctx.stream().synchronize();
    _ctx.allocator().copyD2H(hostDst, deviceSrc, bytes);
}

// Schritt 3c.2 — neutral buffer factory. Zero-byte request skips the
// allocator to keep parity with the L0 side and with the empty
// ComputeBuffer default-ctor semantics. The deleter closure captures
// `CudaAllocKind::Device` implicitly — every buffer this method hands
// out goes back through the device-free path on destruction.
compute::ComputeBuffer GpuOps::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    auto& alloc = _ctx.allocator();
    // Integrated (unified, coherent) devices — GB10 / Jetson — need weight
    // and buffer memory host-reachable: InferenceEngine::generate()
    // dereferences token_embd.usmPtr on the CPU (embeddingLookup). A
    // device-only cudaMalloc segfaults that host read; Managed is the same
    // physical LPDDR5x on unified silicon, so it is free here. Discrete GPUs
    // keep Device. Device and Managed share the cudaFree deallocate path,
    // so the captureless deleter below stays correct for both.
    const auto kind = _ctx.cudaContext().cudaDeviceInfo().isIntegrated
                          ? core::cuda::CudaAllocKind::Managed
                          : core::cuda::CudaAllocKind::Device;
    void* ptr = alloc.allocate(bytes, kind);
    return compute::ComputeBuffer{
        ptr,
        bytes,
        [](void* p, std::size_t b, void* ctx) noexcept {
            static_cast<core::cuda::CudaMemoryAllocator*>(ctx)
                ->deallocate(p, b, core::cuda::CudaAllocKind::Device);
        },
        &alloc};
}

// Schicht 5.2 — sync host-to-device copy. Blocking hipMemcpy so the
// caller can assume the bytes have landed on device by return. The
// stream-async variant lives on `appendMemoryCopy`; loaders that copy
// hundreds of tensors in a loop prefer the blocking path for its
// simpler ordering (no per-tensor flush needed).
void GpuOps::uploadHostBytes(void*       deviceDst,
                             const void* hostSrc,
                             std::size_t bytes) {
    if (bytes == 0) return;
    const cudaError_t rc = cudaMemcpy(
        deviceDst, hostSrc, bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        throw std::runtime_error(
            std::string{"compute::cuda::GpuOps::uploadHostBytes: "
                        "cudaMemcpy(H2D) failed: "} + cudaGetErrorString(rc));
    }
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
        stagedInt32ToDevice(_curLenSlotUsm, v);
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
    // Q8_0 not supported for RoPE — Q8_0 is a KV-storage-only format;
    // K-rope always runs against the fp32 workspace before the Q8_0
    // KV commit. Refuse loudly rather than pick a wrong kernel.
    if (kvDtype == runtime::KvDtype::Q8_0) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::ropeInPlaceAsync: kvDtype=Q8_0 not "
            "supported — K-rope target buffer is fp32 workspace, not "
            "the Q8_0 KV cache");
    }

    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    // Pick f32 vs fp16 kernel by KV dtype. Rotation stays fp32 in
    // registers on both paths — the fp16 kernel just wraps loads /
    // stores in __half2float / __float2half so precision matches the
    // f32 kernel up to the fp16 store round-trip. Same arg layout,
    // only the `xBase` pointer type differs.
    auto& k = (kvDtype == runtime::KvDtype::FP16)
                  ? _pimpl->_ropeFp16Kernel
                  : _pimpl->_ropeKernel;

    // startPos flows through the shared curLen slot — kernel binds it
    // as a device pointer and dereferences at launch. Sync H2D write
    // before launch matches L0's `*_curLenSlotUsm = startPos` store
    // (works there via USM). See rmsNormQkvAsync for the same trick.
    const std::int32_t startI = toInt32(startPos, "rope startPos");
    stagedInt32ToDevice(_curLenSlotUsm, startI);

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

void GpuOps::mropeInPlaceAsync(void* xBase, std::size_t seqLen,
                              std::size_t numHeads, std::size_t headDim,
                              std::size_t startPos, float base,
                              const std::int32_t* sections,
                              std::size_t writeOffsetStride,
                              runtime::KvDtype kvDtype) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error("GpuOps::mropeInPlace: headDim must be even");
    }
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::mropeInPlaceAsync: only KvDtype::F32 "
            "is supported (M-Q3N.2 F32-only IMRoPE path)");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    const std::int32_t startI = toInt32(startPos, "mrope startPos");
    stagedInt32ToDevice(_curLenSlotUsm, startI);

    auto& k = _pimpl->_mropeKernel;
    k.setPtr  (0, xBase);
    k.setValue(1, toInt32(seqLen,   "mrope seqLen"));
    k.setValue(2, toInt32(numHeads, "mrope numHeads"));
    k.setValue(3, toInt32(headDim,  "mrope headDim"));
    k.setPtr  (4, _curLenSlotUsm);
    k.setValue(5, base);
    k.setValue(6, toInt32(writeOffsetStride, "mrope writeOffsetStride"));
    k.setValue(7, sections ? sections[0] : 0);
    k.setValue(8, sections ? sections[1] : 0);
    k.setValue(9, sections ? sections[2] : 0);
    k.setValue(10, sections ? sections[3] : 0);
    k.launch(_ctx.stream(),
             groupsForN(total, kRopeLocalSize), 1, 1,
             kRopeLocalSize, 1, 1);
}

void GpuOps::mropeInPlaceBatchedAsync(void* xBase, std::size_t nSeq,
                                      std::size_t xSeqStride,
                                      std::size_t seqLen, std::size_t numHeads,
                                      std::size_t headDim,
                                      const std::int32_t* startPosDev,
                                      float base, const std::int32_t* sections,
                                      std::size_t writeOffsetStride,
                                      runtime::KvDtype kvDtype) {
    if (nSeq == 0 || seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::mropeInPlaceBatchedAsync: headDim must be even");
    }
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::mropeInPlaceBatchedAsync: only KvDtype::F32");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;
    // Per-seq start positions come from a caller-owned device int32[nSeq]
    // (unlike the single path's staged single slot). Provisional x layout:
    // seq s at xBase + s*xSeqStride (M-Cuda.Batch Cat B, parity-gated).
    auto& k = _pimpl->_mropeBatchedKernel;
    k.setPtr  (0, xBase);
    k.setValue(1, toInt32(seqLen,   "mropeB seqLen"));
    k.setValue(2, toInt32(numHeads, "mropeB numHeads"));
    k.setValue(3, toInt32(headDim,  "mropeB headDim"));
    k.setPtr  (4, startPosDev);
    k.setValue(5, base);
    k.setValue(6, toInt32(writeOffsetStride, "mropeB writeOffsetStride"));
    k.setValue(7, toInt32(xSeqStride, "mropeB xSeqStride"));
    k.setValue(8,  sections ? sections[0] : 0);
    k.setValue(9,  sections ? sections[1] : 0);
    k.setValue(10, sections ? sections[2] : 0);
    k.setValue(11, sections ? sections[3] : 0);
    k.launch(_ctx.stream(),
             groupsForN(total, kRopeLocalSize),
             static_cast<std::uint32_t>(nSeq), 1,
             kRopeLocalSize, 1, 1);
}

void GpuOps::splitHeadPairAsync(const float* src, float* a, float* b,
                                std::size_t seqLen, std::size_t numHeads,
                                std::size_t headDim) {
    const std::size_t total = seqLen * numHeads * headDim;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_splitHeadPairKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, a);
    k.setPtr  (2, b);
    k.setValue(3, toInt32(seqLen,   "splitHeadPair seqLen"));
    k.setValue(4, toInt32(numHeads, "splitHeadPair numHeads"));
    k.setValue(5, toInt32(headDim,  "splitHeadPair headDim"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::sigmoidGateMulAsync(float* y, const float* g, std::size_t rows,
                                 std::size_t dim, std::size_t gateDim) {
    const std::size_t total = rows * dim;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_sigmoidGateMulKernel;
    k.setPtr  (0, y);
    k.setPtr  (1, g);
    k.setValue(2, toInt32(rows,    "sigmoidGateMul rows"));
    k.setValue(3, toInt32(dim,     "sigmoidGateMul dim"));
    k.setValue(4, toInt32(gateDim, "sigmoidGateMul gateDim"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::l2NormInPlaceAsync(float* x, std::size_t rows, std::size_t dim,
                                float eps) {
    if (rows == 0 || dim == 0) {
        return;
    }
    constexpr std::size_t kL2NormLocal = 64;   // matches l2_norm.cu launch_bounds
    auto& k = _pimpl->_l2NormKernel;
    k.setPtr  (0, x);
    k.setValue(1, toInt32(rows, "l2norm rows"));
    k.setValue(2, toInt32(dim,  "l2norm dim"));
    k.setValue(3, eps);
    k.launch(_ctx.stream(),
             groupsForN(rows, kL2NormLocal), 1, 1,
             kL2NormLocal, 1, 1);
}

void GpuOps::causalConv1dSiluAsync(const float* convInput, const float* kernel,
                                   float* out, std::size_t T,
                                   std::size_t channels, std::size_t kernelSize) {
    const std::size_t total = T * channels;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_ssmConv1dKernel;
    k.setPtr  (0, convInput);
    k.setPtr  (1, kernel);
    k.setPtr  (2, out);
    k.setValue(3, toInt32(T,          "conv1d T"));
    k.setValue(4, toInt32(channels,   "conv1d channels"));
    k.setValue(5, toInt32(kernelSize, "conv1d K"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::causalConv1dSiluBatchedAsync(const float* convInput,
                                          const float* kernel, float* out,
                                          std::size_t nSeq, std::size_t T,
                                          std::size_t channels,
                                          std::size_t kernelSize) {
    const std::size_t total = T * channels;
    if (nSeq == 0 || total == 0) {
        return;
    }
    // grid = (ceil(T*channels/LOCAL), nSeq); each seq owns its own conv
    // input (caller prepends its rolling conv-tail). Math byte-identical
    // to the single-sequence causalConv1dSiluAsync (M-Cuda.Batch Cat C-P0).
    auto& k = _pimpl->_ssmConv1dBatchedKernel;
    k.setPtr  (0, convInput);
    k.setPtr  (1, kernel);
    k.setPtr  (2, out);
    k.setValue(3, toInt32(T,          "conv1dB T"));
    k.setValue(4, toInt32(channels,   "conv1dB channels"));
    k.setValue(5, toInt32(kernelSize, "conv1dB K"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize),
             static_cast<std::uint32_t>(nSeq), 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::gatedDeltaNetRecurrentAsync(const float* q, const float* k_,
                                         const float* v, const float* gLog,
                                         const float* beta, float* state,
                                         float* out, std::size_t T,
                                         std::size_t H, std::size_t S) {
    if (T == 0 || H == 0 || S == 0) {
        return;
    }
    // grid = H blocks (one per head), block = S threads (one per state column).
    auto& k = _pimpl->_gatedDeltaNetArKernel;
    k.setPtr  (0, q);
    k.setPtr  (1, k_);
    k.setPtr  (2, v);
    k.setPtr  (3, gLog);
    k.setPtr  (4, beta);
    k.setPtr  (5, state);
    k.setPtr  (6, out);
    k.setValue(7, toInt32(T, "gdn T"));
    k.setValue(8, toInt32(H, "gdn H"));
    k.setValue(9, toInt32(S, "gdn S"));
    // grid = H blocks, block = S threads (S <= GATED_DELTANET_AR_MAX_S).
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H), 1, 1,
             static_cast<std::uint32_t>(S), 1, 1);
}

void GpuOps::gatedDeltaNetRecurrentBatchedAsync(
        const float* q, const float* k_, const float* v, const float* gLog,
        const float* beta, float* state, float* out, std::size_t nSeq,
        std::size_t T, std::size_t H, std::size_t S) {
    if (nSeq == 0 || T == 0 || H == 0 || S == 0) {
        return;
    }
    // grid = (H, nSeq) blocks, block = S threads. Each (head, seq) block
    // owns one sequence [S,S] state; math is byte-identical to the
    // single-sequence gatedDeltaNetRecurrentAsync (M-Cuda.Batch Cat C-P0).
    auto& k = _pimpl->_gatedDeltaNetArBatchedKernel;
    k.setPtr  (0, q);
    k.setPtr  (1, k_);
    k.setPtr  (2, v);
    k.setPtr  (3, gLog);
    k.setPtr  (4, beta);
    k.setPtr  (5, state);
    k.setPtr  (6, out);
    k.setValue(7, toInt32(T, "gdnB T"));
    k.setValue(8, toInt32(H, "gdnB H"));
    k.setValue(9, toInt32(S, "gdnB S"));
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H),
             static_cast<std::uint32_t>(nSeq), 1,
             static_cast<std::uint32_t>(S), 1, 1);
}

void GpuOps::deltanetGateAsync(const float* alpha, const float* ssmA,
                               const float* ssmDt, float* gLog,
                               std::size_t T, std::size_t H) {
    const std::size_t total = T * H;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_deltanetGateKernel;
    k.setPtr  (0, alpha);
    k.setPtr  (1, ssmA);
    k.setPtr  (2, ssmDt);
    k.setPtr  (3, gLog);
    k.setValue(4, toInt32(T, "deltanetGate T"));
    k.setValue(5, toInt32(H, "deltanetGate H"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::deltanetChunkCumGateAsync(const float* gLog, float* gCum,
                                       std::size_t T, std::size_t H,
                                       std::size_t chunkSize) {
    if (T == 0 || H == 0) {
        return;
    }
    const std::size_t C       = chunkSize ? chunkSize : 64;
    const std::size_t nChunks = (T + C - 1) / C;
    const std::size_t total   = H * nChunks;   // one thread per (head, chunk)
    auto& k = _pimpl->_deltanetChunkCumGateKernel;
    k.setPtr  (0, gLog);
    k.setPtr  (1, gCum);
    k.setValue(2, toInt32(T, "cumgate T"));
    k.setValue(3, toInt32(H, "cumgate H"));
    k.setValue(4, toInt32(C, "cumgate C"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::deltanetChunkCumGateBatchedAsync(const float* gLog, float* gCum,
                                              std::size_t nSeq, std::size_t T,
                                              std::size_t H,
                                              std::size_t chunkSize) {
    if (nSeq == 0 || T == 0 || H == 0) {
        return;
    }
    const std::size_t C       = chunkSize ? chunkSize : 64;
    const std::size_t nChunks = (T + C - 1) / C;
    const std::size_t total   = H * nChunks;   // one thread per (head, chunk)
    // grid.y = nSeq; each sequence prefix-sums its own gLog slab (M-Cuda.Batch
    // Cat C-P1). Byte-identical to nSeq single deltanetChunkCumGateAsync.
    auto& k = _pimpl->_deltanetChunkCumGateBatchedKernel;
    k.setPtr  (0, gLog);
    k.setPtr  (1, gCum);
    k.setValue(2, toInt32(T, "cumgateB T"));
    k.setValue(3, toInt32(H, "cumgateB H"));
    k.setValue(4, toInt32(C, "cumgateB C"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize),
             static_cast<std::uint32_t>(nSeq), 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::deltanetChunkForwardAsync(const float* q, const float* k_,
                                       const float* v, const float* gCum,
                                       const float* beta, const float* a0,
                                       float* state, float* out,
                                       std::size_t T, std::size_t H,
                                       std::size_t S, std::size_t chunkSize) {
    if (T == 0 || H == 0 || S == 0) {
        return;
    }
    const std::size_t C = chunkSize ? chunkSize : 64;
    const std::size_t f = sizeof(float);
    // Per-head global scratch: the [S,S] state snapshot plus the [C,S] chunk
    // working tensors exceed the shared budget at prod width. Freed after the
    // sync below (the kernel needs it live). Correctness-first; a persistent
    // scratch is a perf follow-up.
    // One combined scratch buffer (kMaxArgs=16 limit): s0[H,S,S] followed by
    // u|uq|qs|rp|d ([H,C,S] each). The kernel slices it by offset.
    const std::size_t scratchElems = H * S * S + 5 * H * C * S;
    auto scratch = allocate(scratchElems * f);

    auto& kern = _pimpl->_deltanetChunkForwardKernel;
    kern.setPtr  (0,  q);
    kern.setPtr  (1,  k_);
    kern.setPtr  (2,  v);
    kern.setPtr  (3,  gCum);
    kern.setPtr  (4,  beta);
    kern.setPtr  (5,  a0);
    kern.setPtr  (6,  state);
    kern.setPtr  (7,  out);
    kern.setPtr  (8,  scratch.get());
    kern.setValue(9,  toInt32(T, "chunkfwd T"));
    kern.setValue(10, toInt32(H, "chunkfwd H"));
    kern.setValue(11, toInt32(S, "chunkfwd S"));
    kern.setValue(12, toInt32(C, "chunkfwd C"));
    // grid = H blocks (one per head), block = S threads (one per state column).
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(H), 1, 1,
                static_cast<std::uint32_t>(S), 1, 1);
    // Sync so the scratch ComputeBuffers stay alive until the kernel is done
    // (prefill-only path; not the decode hot loop).
    _ctx.stream().synchronize();
}

void GpuOps::deltanetChunkForwardBatchedAsync(
        const float* q, const float* k_, const float* v, const float* gCum,
        const float* beta, const float* a0, float* state, float* out,
        std::size_t nSeq, std::size_t T, std::size_t H, std::size_t S,
        std::size_t chunkSize) {
    if (nSeq == 0 || T == 0 || H == 0 || S == 0) {
        return;
    }
    const std::size_t C = chunkSize ? chunkSize : 64;
    const std::size_t f = sizeof(float);
    // nSeq copies of the single-seq global scratch ([s0 [H,S,S] + 5 chunk
    // tensors [H,C,S]]); the kernel offsets by seq*scratchPerSeq. grid.y =
    // nSeq. Byte-identical to nSeq single deltanetChunkForwardAsync.
    const std::size_t scratchPerSeq = H * S * S + 5 * H * C * S;
    auto scratch = allocate(nSeq * scratchPerSeq * f);

    auto& kern = _pimpl->_deltanetChunkForwardBatchedKernel;
    kern.setPtr  (0,  q);
    kern.setPtr  (1,  k_);
    kern.setPtr  (2,  v);
    kern.setPtr  (3,  gCum);
    kern.setPtr  (4,  beta);
    kern.setPtr  (5,  a0);
    kern.setPtr  (6,  state);
    kern.setPtr  (7,  out);
    kern.setPtr  (8,  scratch.get());
    kern.setValue(9,  toInt32(T, "chunkfwdB T"));
    kern.setValue(10, toInt32(H, "chunkfwdB H"));
    kern.setValue(11, toInt32(S, "chunkfwdB S"));
    kern.setValue(12, toInt32(C, "chunkfwdB C"));
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(H),
                static_cast<std::uint32_t>(nSeq), 1,
                static_cast<std::uint32_t>(S), 1, 1);
    _ctx.stream().synchronize();
}

void GpuOps::deltanetKktSolveInverseAsync(const float* k_, const float* beta,
                                          float* a0, std::size_t T,
                                          std::size_t H, std::size_t S,
                                          std::size_t chunkSize) {
    if (T == 0 || H == 0 || S == 0) {
        return;
    }
    const std::size_t C       = chunkSize ? chunkSize : 64;
    const std::size_t nChunks = (T + C - 1) / C;
    const std::size_t nBlocks = nChunks * H;   // one block per (chunk, head)
    auto& kern = _pimpl->_deltanetKktSolveKernel;
    kern.setPtr  (0, k_);
    kern.setPtr  (1, beta);
    kern.setPtr  (2, a0);
    kern.setValue(3, toInt32(T, "kkt T"));
    kern.setValue(4, toInt32(H, "kkt H"));
    kern.setValue(5, toInt32(S, "kkt S"));
    kern.setValue(6, toInt32(C, "kkt C"));
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(nBlocks), 1, 1,
                static_cast<std::uint32_t>(C), 1, 1);
}

void GpuOps::moeTopKRouteDeviceAsync(const float* logits, std::int32_t* outIdx,
                                     float* outWeight, std::size_t T,
                                     std::size_t nExperts, std::size_t K,
                                     float wScale) {
    _moeTopKRoute.launch(logits, outIdx, outWeight, T, nExperts, K, wScale);
}

void GpuOps::sigmoidInPlaceAsync(float* y, std::size_t n) {
    if (n == 0) {
        return;
    }
    auto& k = _pimpl->_sigmoidInplaceKernel;
    k.setPtr  (0, y);
    k.setValue(1, toInt32(n, "sigmoidInplace n"));
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::gatherHeadsFromChannelsAsync(const float* src, float* dst,
                                          std::size_t T, std::size_t offset,
                                          std::size_t srcHeads,
                                          std::size_t dstHeads, std::size_t S,
                                          std::size_t convTotalWidth) {
    const std::size_t total = T * dstHeads * S;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_gatherHeadsKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, dst);
    k.setValue(2, toInt32(T,              "gather T"));
    k.setValue(3, toInt32(offset,         "gather offset"));
    k.setValue(4, toInt32(srcHeads,       "gather srcHeads"));
    k.setValue(5, toInt32(dstHeads,       "gather dstHeads"));
    k.setValue(6, toInt32(S,              "gather S"));
    k.setValue(7, toInt32(convTotalWidth, "gather convTotalWidth"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::ropeInPlaceWithFactorsAsync(void* xBase, const float* freqFactors,
                                         std::size_t seqLen,
                                         std::size_t numHeads,
                                         std::size_t headDim,
                                         std::size_t startPos, float base,
                                         std::size_t writeOffsetStride,
                                         runtime::KvDtype kvDtype) {
    if (seqLen == 0 || numHeads == 0 || headDim == 0) {
        return;
    }
    if (headDim % 2 != 0) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::ropeInPlaceWithFactors: headDim must be even");
    }
    // fp16-KV variant of freq-factors RoPE (`rope_inplace_ff_fp16.hip`)
    // still needs porting; Q8_0 is not a RoPE target (see
    // ropeInPlaceAsync). Both refuse loudly.
    if (kvDtype == runtime::KvDtype::FP16) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::ropeInPlaceWithFactorsAsync: FP16 KV "
            "path requires rope_inplace_ff_fp16.hip — not yet ported");
    }
    if (kvDtype == runtime::KvDtype::Q8_0) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::ropeInPlaceWithFactorsAsync: kvDtype=Q8_0 "
            "not supported — K-rope target buffer is fp32 workspace");
    }

    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    const std::int32_t startI = toInt32(startPos, "rope_ff startPos");
    stagedInt32ToDevice(_curLenSlotUsm, startI);

    auto& k = _pimpl->_ropeFfKernel;
    k.setPtr  (0, xBase);
    k.setPtr  (1, freqFactors);
    k.setValue(2, toInt32(seqLen,   "rope_ff seqLen"));
    k.setValue(3, toInt32(numHeads, "rope_ff numHeads"));
    k.setValue(4, toInt32(headDim,  "rope_ff headDim"));
    k.setPtr  (5, _curLenSlotUsm);
    k.setValue(6, base);
    k.setValue(7, toInt32(writeOffsetStride, "rope_ff writeOffsetStride"));
    k.launch(_ctx.stream(),
             groupsForN(total, kRopeLocalSize), 1, 1,
             kRopeLocalSize, 1, 1);
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
    stagedInt32ToDevice(_curLenSlotUsm, offI);

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
        stagedInt32ToDevice(_curLenSlotUsm, v);
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
             groupsForN(total, kQkvSplitLocalSize), 1, 1,
             kQkvSplitLocalSize, 1, 1);
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
            "compute::cuda::GpuOps::attentionAsync: nHeads (" +
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
                "compute::cuda::GpuOps::attentionAsync: flash path needs "
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
            "compute::cuda::GpuOps::attentionPlainAsync: T_k=" +
            std::to_string(T_k) + " exceeds compile-time bound "
            "ATTN_MAX_TK=" + std::to_string(kAttentionMaxTk) +
            " — the plain-attention kernel holds scores[ATTN_MAX_TK] in "
            "LDS. Re-enable the flash path (features.prefillFlash: true) "
            "or reduce runtime.maxContextTokens below " +
            std::to_string(kAttentionMaxTk));
    }
    (void)T_k;

    core::cuda::CudaKernel* kernelPtr = &_pimpl->_attentionKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        kernelPtr = &_pimpl->_attentionFp16Kernel;
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        kernelPtr = &_pimpl->_attentionQ8Kernel;
    }
    auto& kernel = *kernelPtr;

    const std::int32_t posI =
        toInt32(positionOffset, "attention positionOffset");
    stagedInt32ToDevice(_curLenSlotUsm, posI);

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
    // K-tile pick: 64 → smaller-SLM variant (higher occupancy on the
    // heavy per-Q-head register pressure); 128 → default M5i.J
    // streaming amortisation. Any other value was resolved / rejected
    // in the ctor.
    const std::size_t nQPerKv = nHeads / nKvHeads;
    const bool useQ8Gqa =
        (kvDtype == runtime::KvDtype::Q8_0) &&
        !_prefillFlashGqaQ8Disabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv);

    core::cuda::CudaKernel* kernelPtr = &_pimpl->_attentionPrefillFlashKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        kernelPtr = &_pimpl->_attentionPrefillFlashFp16Kernel;
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        if (useQ8Gqa) {
            kernelPtr = (_prefillFlashKTileQ8 == 64)
                ? &_pimpl->_attentionPrefillFlashQ8GqaKtile64Kernel
                : &_pimpl->_attentionPrefillFlashQ8GqaKernel;
        } else {
            kernelPtr = &_pimpl->_attentionPrefillFlashQ8Kernel;
        }
    }
    auto& kernel = *kernelPtr;

    const std::int32_t posI =
        toInt32(positionOffset, "prefill_flash positionOffset");
    stagedInt32ToDevice(_curLenSlotUsm, posI);

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
            "compute::cuda::GpuOps::attentionDecodeFlashAsync: nKTiles=" +
            std::to_string(nKTiles) + " out of [1, " +
            std::to_string(kFlashMaxKTiles) + "]");
    }
    (void)T_k;

    const std::int32_t posI =
        toInt32(positionOffset, "flash positionOffset");
    stagedInt32ToDevice(_curLenSlotUsm, posI);

    // Pass 1 — per-tile partial (m, l, o_unnorm) into persistent scratch.
    // Kernel by KV dtype; partial layout stays fp32 regardless so the
    // merge kernel is dtype-agnostic.
    core::cuda::CudaKernel* partialKernelPtr =
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

    // In graph capture/replay mode (staging off) the launch geometry must
    // cover the MAX KV length (_replayMaxKTiles) — the captured grid is
    // frozen, and the kernel early-exits K-tiles past curLen (read from the
    // slot). In immediate mode use the actual nKTiles (optimal). M-Q3N.5 K4.
    const std::uint32_t gridKTiles =
        (!_perKernelCurLenStaging && _replayMaxKTiles > 0)
            ? static_cast<std::uint32_t>(_replayMaxKTiles)
            : static_cast<std::uint32_t>(nKTiles);
    partialKernel.launch(_ctx.stream(),
                         static_cast<std::uint32_t>(nHeads),
                         gridKTiles,
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

void GpuOps::attentionDecodeFlashBatchedAsync(
        const float* q, const float* k, const float* v, float* partialScratch,
        float* out, std::size_t nSeq, std::size_t maxKTiles,
        std::size_t qSeqStride, std::size_t kvSeqStride,
        std::size_t partialSeqStride, std::size_t outSeqStride,
        std::size_t nHeads, std::size_t nKvHeads, std::size_t headDim,
        const std::int32_t* curLenDev, float scale, std::size_t slidingWindow,
        runtime::KvDtype kvDtype) {
    if (nSeq == 0 || nHeads == 0 || headDim == 0 || maxKTiles == 0) {
        return;
    }
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::attentionDecodeFlashBatchedAsync: only "
            "KvDtype::F32 batched today (M-Cuda.Batch Cat attention).");
    }
    // Pass 1 — per-tile partials. grid (nHeads, maxKTiles, nSeq); each
    // sequence uses its own q/kv/curLen, a uniform maxKTiles per-head
    // stride, and early-exits tiles past its own length. Provisional
    // per-seq strides — KV layout settled in Phase D. Byte-identical to
    // nSeq single attentionDecodeFlashAsync.
    auto& partialKernel = _pimpl->_attentionFlashPartialBatchedKernel;
    partialKernel.setPtr  (0, q);
    partialKernel.setPtr  (1, k);
    partialKernel.setPtr  (2, v);
    partialKernel.setPtr  (3, partialScratch);
    partialKernel.setValue(4, toInt32(nHeads,   "flashB nHeads"));
    partialKernel.setValue(5, toInt32(nKvHeads, "flashB nKvHeads"));
    partialKernel.setValue(6, toInt32(headDim,  "flashB headDim"));
    partialKernel.setPtr  (7, curLenDev);
    partialKernel.setValue(8, scale);
    partialKernel.setValue(9, toInt32(slidingWindow, "flashB slidingWindow"));
    partialKernel.setValue(10, toInt32(maxKTiles, "flashB kTilesStride"));
    partialKernel.setValue(11, toInt32(qSeqStride, "flashB qSeqStride"));
    partialKernel.setValue(12, toInt32(kvSeqStride, "flashB kvSeqStride"));
    partialKernel.setValue(13, toInt32(partialSeqStride, "flashB partialSeqStride"));
    partialKernel.launch(_ctx.stream(),
                         static_cast<std::uint32_t>(nHeads),
                         static_cast<std::uint32_t>(maxKTiles),
                         static_cast<std::uint32_t>(nSeq),
                         kAttentionLocalSize, 1, 1);

    // Pass 2 — merge. grid (nHeads, nSeq).
    auto& mergeKernel = _pimpl->_attentionFlashMergeBatchedKernel;
    mergeKernel.setPtr  (0, partialScratch);
    mergeKernel.setPtr  (1, out);
    mergeKernel.setValue(2, toInt32(nHeads,  "flashB_merge nHeads"));
    mergeKernel.setValue(3, toInt32(headDim, "flashB_merge headDim"));
    mergeKernel.setPtr  (4, curLenDev);
    mergeKernel.setValue(5, toInt32(maxKTiles, "flashB_merge kTilesStride"));
    mergeKernel.setValue(6, toInt32(partialSeqStride, "flashB_merge partialSeqStride"));
    mergeKernel.setValue(7, toInt32(outSeqStride, "flashB_merge outSeqStride"));
    mergeKernel.launch(_ctx.stream(),
                       static_cast<std::uint32_t>(nHeads),
                       static_cast<std::uint32_t>(nSeq),
                       1,
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

} // namespace mimirmind::compute::cuda
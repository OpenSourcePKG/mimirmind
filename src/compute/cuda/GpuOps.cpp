// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cuda/GpuOps.hpp"

#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#if MIMIRMIND_HAVE_CUDNN_SDPA
#include "compute/cuda/CudnnSdpaPrefill.hpp"
#endif
#include "core/gpu/cuda/CudaMemoryAllocator.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/log/Log.hpp"

#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
#include "MoeGroupedGemmNvfp4Tc.hpp" // E-d.4b: FP4-TC grouped GEMM (banks entry)
#endif

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <unordered_map>
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
// source under `kernels/hip/llm/` that corresponds to a `ComputeOps`
// entry point. HIP-only kernels used by `HipGpuMatmul` (matmul
// variants + moe_down) live on that class, not here. Kernels that
// exist in the L0 `GpuOps::Impl` but haven't been ported to HIP yet
// (rope_inplace_ff_fp16) are deliberately absent — the corresponding
// ComputeOps overrides throw a clear diagnostic at dispatch time.
struct GpuOps::Impl {
    core::cuda::CudaModule _rmsnormModule;
    core::cuda::CudaKernel _rmsnormKernel;
    core::cuda::CudaModule _layernormModule;
    core::cuda::CudaKernel _layernormKernel;
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
    core::cuda::CudaModule _siluMulSplitModule;
    core::cuda::CudaKernel _siluMulSplitKernel;
    core::cuda::CudaModule _geluMulModule;
    core::cuda::CudaKernel _geluMulKernel;
    core::cuda::CudaModule _geluErfModule;
    core::cuda::CudaKernel _geluErfKernel;
    core::cuda::CudaModule _encoderEmbedAddModule;
    core::cuda::CudaKernel _encoderEmbedAddKernel;
    core::cuda::CudaModule _tanhModule;
    core::cuda::CudaKernel _tanhKernel;
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
    // Interleaved (GPT-J / llama) RoPE — arch=llama; see ropeInPlaceInterleavedAsync.
    core::cuda::CudaModule _ropeInterleavedModule;
    core::cuda::CudaKernel _ropeInterleavedKernel;
    core::cuda::CudaModule _ropeFfInterleavedModule;
    core::cuda::CudaKernel _ropeFfInterleavedKernel;

    core::cuda::CudaModule _attentionModule;
    core::cuda::CudaKernel _attentionKernel;
    core::cuda::CudaModule _attentionEncoderModule;
    core::cuda::CudaKernel _attentionEncoderKernel;
    core::cuda::CudaModule _attentionEncoderBatchedModule;
    core::cuda::CudaKernel _attentionEncoderBatchedKernel;
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
    core::cuda::CudaModule _pagedAttentionV1Module;
    core::cuda::CudaKernel _pagedAttentionV1Kernel;
    core::cuda::CudaModule _pagedAttentionV2Module;
    core::cuda::CudaKernel _pagedAttentionV2Kernel;
    core::cuda::CudaKernel _pagedAttentionV2Fp16Kernel;   // fp16 KV (5.14 I1)
    core::cuda::CudaKernel _pagedAttentionV2ReduceKernel;
    // Split-K V2 per-partition workspace (grown on demand; RAII-freed).
    compute::ComputeBuffer _pagedV2TmpOut;      // [slots, headSize] fp32
    compute::ComputeBuffer _pagedV2ExpSums;     // [slots] fp32
    compute::ComputeBuffer _pagedV2MaxLogits;   // [slots] fp32
    std::size_t            _pagedV2SlotCap{0};   // slots = nSeq*nHeads*maxNumPartitions
    std::size_t            _pagedV2HeadDimCap{0};

    core::cuda::CudaModule _attentionPrefillFlashModule;
    core::cuda::CudaKernel _attentionPrefillFlashKernel;
    core::cuda::CudaModule _attentionPrefillFlashFp16Module;
    core::cuda::CudaKernel _attentionPrefillFlashFp16Kernel;
    // Step 3 — FP16 tensor-core FA-2 prefill (q-tiled, opt-in).
    core::cuda::CudaModule _attentionPrefillFlashFp16TcModule;
    core::cuda::CudaKernel _attentionPrefillFlashFp16TcKernel;
    // Step 3.2 — GQA-head-packed multi-warp FP16 tensor-core FA-2 (opt-in).
    core::cuda::CudaModule _attentionPrefillFlashFp16GqaTcModule;
    core::cuda::CudaKernel _attentionPrefillFlashFp16GqaTcKernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8Module;
    core::cuda::CudaKernel _attentionPrefillFlashQ8Kernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8GqaModule;
    core::cuda::CudaKernel _attentionPrefillFlashQ8GqaKernel;
    core::cuda::CudaModule _attentionPrefillFlashQ8GqaKtile64Module;
    core::cuda::CudaKernel _attentionPrefillFlashQ8GqaKtile64Kernel;
    // P3.a — GQA-head-packed F32 prefill flash (opt-in, F32 KV path).
    core::cuda::CudaModule _attentionPrefillFlashF32GqaModule;
    core::cuda::CudaKernel _attentionPrefillFlashF32GqaKernel;
    // P3.b — TF32 tensor-core GQA-head-packed F32 prefill flash (opt-in).
    core::cuda::CudaModule _attentionPrefillFlashF32GqaTcModule;
    core::cuda::CudaKernel _attentionPrefillFlashF32GqaTcKernel;
    // Multi-warp TF32 FA-2 for the F32 KV path (opt-in, fixes P3.b's flaws).
    core::cuda::CudaModule _attentionPrefillFlashF32GqaMwtcModule;
    core::cuda::CudaKernel _attentionPrefillFlashF32GqaMwtcKernel;

    core::cuda::CudaModule _qkvSplitModule;
    core::cuda::CudaKernel _qkvSplitKernel;
    core::cuda::CudaModule _qkvSplitFp16Module;
    core::cuda::CudaKernel _qkvSplitFp16Kernel;

    core::cuda::CudaModule _kvQuantCommitQ8Module;
    core::cuda::CudaKernel _kvQuantCommitQ8Kernel;
    // FP16-KV staging commit (fp32 scratch -> fp16 cache cast).
    core::cuda::CudaModule _kvCommitFp16Module;
    core::cuda::CudaKernel _kvCommitFp16Kernel;

    core::cuda::CudaModule _matmulQ8_0VecReorderModule;
    core::cuda::CudaKernel _matmulQ8_0VecReorderKernel;

    core::cuda::CudaModule _mropeModule;
    core::cuda::CudaKernel _mropeKernel;
    // FP16-KV IMRoPE (attn-FMHA track building block).
    core::cuda::CudaModule _mropeFp16Module;
    core::cuda::CudaKernel _mropeFp16Kernel;
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
    // P2.b — 2-way row-split smem-staged prefill recurrence (opt-in).
    core::cuda::CudaModule _gatedDeltaNetArR2Module;
    core::cuda::CudaKernel _gatedDeltaNetArR2Kernel;
    core::cuda::CudaModule _gatedDeltaNetArBatchedModule;
    core::cuda::CudaKernel _gatedDeltaNetArBatchedKernel;
    core::cuda::CudaModule _gatedDeltaNetArBatchedV2Module;
    core::cuda::CudaKernel _gatedDeltaNetArBatchedV2Kernel;
    core::cuda::CudaModule _gatedDeltaNetArBatchedV3Module;
    core::cuda::CudaKernel _gatedDeltaNetArBatchedV3Kernel;
    // GDN-Inc 2: v3 with the decay gate + beta-sigmoid folded in (from the same
    // module). Removes the deltanet_gate + sigmoid_inplace launches per layer.
    core::cuda::CudaKernel _gatedDeltaNetArBatchedV3GateFusedKernel;
    // MV-a: batched GDN verify (T-loop over K+1, per-position state export).
    core::cuda::CudaModule _gatedDeltaNetVerifyBatchedModule;
    core::cuda::CudaKernel _gatedDeltaNetVerifyBatchedKernel;
    // ReplaySSM: state-only fold of the accepted verify prefix (partial-accept
    // commit without a trunk re-forward).
    core::cuda::CudaModule _gatedDeltaNetFoldModule;
    core::cuda::CudaKernel _gatedDeltaNetFoldKernel;
    // MTP draft-side: per-row device argmax over vocab.
    core::cuda::CudaModule _argmaxRowsModule;
    core::cuda::CudaKernel _argmaxRowsKernel;
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
    // GDN-Inc 2b: fused post-conv prep (q/k/v gather + q/k L2-norm) in one launch.
    core::cuda::CudaKernel _fusedPostConvPrepKernel;

    // M-Cuda.MoeGroup — grouped-by-expert MoE prefill (token grouping build,
    // row gather, deterministic expert-output scatter).
    core::cuda::CudaModule _moeGroupBuildModule;
    core::cuda::CudaKernel _moeGroupBuildKernel;
    core::cuda::CudaModule _moeGatherRowsModule;
    core::cuda::CudaKernel _moeGatherRowsKernel;
    core::cuda::CudaModule _moeScatterExpertOutModule;
    core::cuda::CudaKernel _moeScatterExpertOutKernel;
    // Device KV scatter (graph-capturable paged-KV write by device index).
    core::cuda::CudaModule _kvWriteTokensModule;
    core::cuda::CudaKernel _kvWriteTokensKernel;
    core::cuda::CudaKernel _kvWriteTokensFp16Kernel;   // fp16 KV (5.14 I1)
    // M-Cuda.MoeGroup Sub-Step E — device-driven grouped GEMM (tile schedule
    // build + single grouped NVFP4 launch; no expOffset D2H).
    core::cuda::CudaModule _moeGroupTilesModule;
    core::cuda::CudaKernel _moeGroupTilesKernel;
    core::cuda::CudaModule _moeGroupedGemmNvfp4Module;
    core::cuda::CudaKernel _moeGroupedGemmNvfp4Kernel;
    // De-interleaved (uint4-coalesced) NVFP4 grouped decode + its de-interleave
    // helper, plus a by-pointer cache of the de-interleaved (nib, scale) banks.
    core::cuda::CudaModule _nvfp4DeintModule;
    core::cuda::CudaKernel _nvfp4DeinterleaveKernel;
    core::cuda::CudaKernel _moeGroupedGemmNvfp4DeintKernel;
    // Increment 1: de-interleaved uint4 weights + register-staged activation
    // (no shared memory). Selected inside moeGroupedGemmNvfp4DeintAsync when
    // MIMIRMIND_MOE_DEINT_REG=1, for the nSeq==1 decode A/B.
    core::cuda::CudaKernel _moeGroupedGemmNvfp4DeintRegKernel;
    struct DeintBank {
        compute::ComputeBuffer nib;
        compute::ComputeBuffer scale;
    };
    std::unordered_map<const void*, DeintBank> _deintCache;

    // MIMIRMIND_DECODE_PROFILE self-profiler.
    bool        _profOn{false};
    cudaEvent_t _profA{nullptr};
    cudaEvent_t _profB{nullptr};
    std::string _profPrev;
    int         _profSteps{0};
    std::vector<std::pair<std::string, double>> _profAcc;   // insertion order
    // GD-b: decode small-M variant (acc[4] + 4 KB shared) from the same module.
    core::cuda::CudaKernel _moeGroupedGemmNvfp4M4Kernel;
    // Decode single-user (M==1) register-staged variant: activation in registers
    // instead of shared memory, removing the MIO/short-scoreboard stall.
    core::cuda::CudaKernel _moeGroupedGemmNvfp4M1RegKernel;
    // E-d.4b: padding infra (one module, four kernels) + act-quant.
    core::cuda::CudaModule _moePadModule;
    core::cuda::CudaKernel _moePadOffsetsKernel;
    core::cuda::CudaKernel _moeContigToPadKernel;
    core::cuda::CudaKernel _moeRowsScatterKernel;
    core::cuda::CudaKernel _moeIndexGatherKernel;
    core::cuda::CudaModule _moeActQuantModule;
    core::cuda::CudaKernel _moeActQuantKernel;
    core::cuda::CudaKernel _moeActQuantRowsKernel;

    explicit Impl(core::cuda::CudaContext& ctx)
        : _rmsnormModule           {loadCudaModule(ctx, "rmsnorm")},
          _rmsnormKernel           {_rmsnormModule.getFunction("rmsnorm")},
          _layernormModule         {loadCudaModule(ctx, "layernorm")},
          _layernormKernel         {_layernormModule.getFunction("layernorm")},
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
          _siluMulSplitModule      {loadCudaModule(ctx, "silu_mul_split")},
          _siluMulSplitKernel      {_siluMulSplitModule.getFunction("silu_mul_split")},
          _geluMulModule           {loadCudaModule(ctx, "gelu_mul")},
          _geluMulKernel           {_geluMulModule.getFunction("gelu_mul")},
          _geluErfModule           {loadCudaModule(ctx, "gelu_erf")},
          _geluErfKernel           {_geluErfModule.getFunction("gelu_erf")},
          _encoderEmbedAddModule   {loadCudaModule(ctx, "encoder_embed_add")},
          _encoderEmbedAddKernel   {_encoderEmbedAddModule.getFunction("encoder_embed_add")},
          _tanhModule              {loadCudaModule(ctx, "tanh_inplace")},
          _tanhKernel              {_tanhModule.getFunction("tanh_inplace")},
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
          _ropeInterleavedModule   {loadCudaModule(ctx, "rope_inplace_interleaved")},
          _ropeInterleavedKernel   {_ropeInterleavedModule.getFunction("rope_inplace_interleaved")},
          _ropeFfInterleavedModule {loadCudaModule(ctx, "rope_inplace_ff_interleaved")},
          _ropeFfInterleavedKernel {_ropeFfInterleavedModule.getFunction("rope_inplace_ff_interleaved")},

          _attentionModule         {loadCudaModule(ctx, "attention")},
          _attentionKernel         {_attentionModule.getFunction("attention")},
          _attentionEncoderModule  {loadCudaModule(ctx, "attention_encoder")},
          _attentionEncoderKernel  {_attentionEncoderModule.getFunction("attention_encoder")},
          _attentionEncoderBatchedModule {loadCudaModule(ctx, "attention_encoder_batched")},
          _attentionEncoderBatchedKernel {_attentionEncoderBatchedModule.getFunction("attention_encoder_batched")},
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
          _pagedAttentionV1Module{loadCudaModule(ctx, "attention_paged_v1")},
          _pagedAttentionV1Kernel{
              _pagedAttentionV1Module.getFunction("paged_attention_v1")},
          _pagedAttentionV2Module{loadCudaModule(ctx, "attention_paged_v2")},
          _pagedAttentionV2Kernel{
              _pagedAttentionV2Module.getFunction("paged_attention_v2")},
          _pagedAttentionV2Fp16Kernel{
              _pagedAttentionV2Module.getFunction("paged_attention_v2_fp16")},
          _pagedAttentionV2ReduceKernel{
              _pagedAttentionV2Module.getFunction("paged_attention_v2_reduce")},

          _attentionPrefillFlashModule{loadCudaModule(ctx, "attention_prefill_flash")},
          _attentionPrefillFlashKernel{
              _attentionPrefillFlashModule.getFunction("attention_prefill_flash")},
          _attentionPrefillFlashFp16Module{
              loadCudaModule(ctx, "attention_prefill_flash_fp16")},
          _attentionPrefillFlashFp16Kernel{
              _attentionPrefillFlashFp16Module.getFunction("attention_prefill_flash_fp16")},
          _attentionPrefillFlashFp16TcModule{
              loadCudaModule(ctx, "attention_prefill_flash_fp16_tc")},
          _attentionPrefillFlashFp16TcKernel{
              _attentionPrefillFlashFp16TcModule.getFunction(
                  "attention_prefill_flash_fp16_tc")},
          _attentionPrefillFlashFp16GqaTcModule{
              loadCudaModule(ctx, "attention_prefill_flash_fp16_gqa_tc")},
          _attentionPrefillFlashFp16GqaTcKernel{
              _attentionPrefillFlashFp16GqaTcModule.getFunction(
                  "attention_prefill_flash_fp16_gqa_tc")},
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
          _attentionPrefillFlashF32GqaModule{
              loadCudaModule(ctx, "attention_prefill_flash_f32_gqa")},
          _attentionPrefillFlashF32GqaKernel{
              _attentionPrefillFlashF32GqaModule.getFunction(
                  "attention_prefill_flash_f32_gqa")},
          _attentionPrefillFlashF32GqaTcModule{
              loadCudaModule(ctx, "attention_prefill_flash_f32_gqa_tc")},
          _attentionPrefillFlashF32GqaTcKernel{
              _attentionPrefillFlashF32GqaTcModule.getFunction(
                  "attention_prefill_flash_f32_gqa_tc")},
          _attentionPrefillFlashF32GqaMwtcModule{
              loadCudaModule(ctx, "attention_prefill_flash_f32_gqa_mwtc")},
          _attentionPrefillFlashF32GqaMwtcKernel{
              _attentionPrefillFlashF32GqaMwtcModule.getFunction(
                  "attention_prefill_flash_f32_gqa_mwtc")},

          _qkvSplitModule          {loadCudaModule(ctx, "qkv_split")},
          _qkvSplitKernel          {_qkvSplitModule.getFunction("qkv_split")},
          _qkvSplitFp16Module      {loadCudaModule(ctx, "qkv_split_fp16")},
          _qkvSplitFp16Kernel      {_qkvSplitFp16Module.getFunction("qkv_split_fp16")},

          _kvQuantCommitQ8Module   {loadCudaModule(ctx, "kv_quant_commit_q8_0")},
          _kvQuantCommitQ8Kernel   {
              _kvQuantCommitQ8Module.getFunction("kv_quant_commit_q8_0")},
          _kvCommitFp16Module      {loadCudaModule(ctx, "kv_commit_fp16")},
          _kvCommitFp16Kernel      {
              _kvCommitFp16Module.getFunction("kv_commit_fp16")},

          _matmulQ8_0VecReorderModule{loadCudaModule(ctx, "matmul_q8_0_vec_reorder")},
          _matmulQ8_0VecReorderKernel{
              _matmulQ8_0VecReorderModule.getFunction("matmul_q8_0_vec_reorder")},
          _mropeModule             {loadCudaModule(ctx, "rope_mrope")},
          _mropeKernel             {_mropeModule.getFunction("rope_mrope")},
          _mropeFp16Module         {loadCudaModule(ctx, "rope_mrope_fp16")},
          _mropeFp16Kernel         {_mropeFp16Module.getFunction("rope_mrope_fp16")},
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
          _gatedDeltaNetArR2Module {loadCudaModule(ctx, "gated_deltanet_ar_r2")},
          _gatedDeltaNetArR2Kernel {
              _gatedDeltaNetArR2Module.getFunction("gated_deltanet_ar_r2")},
          _gatedDeltaNetArBatchedModule{loadCudaModule(ctx, "gated_deltanet_ar_batched")},
          _gatedDeltaNetArBatchedKernel{
              _gatedDeltaNetArBatchedModule.getFunction("gated_deltanet_ar_batched")},
          _gatedDeltaNetArBatchedV2Module{loadCudaModule(ctx, "gated_deltanet_ar_batched_v2")},
          _gatedDeltaNetArBatchedV2Kernel{
              _gatedDeltaNetArBatchedV2Module.getFunction("gated_deltanet_ar_batched_v2")},
          _gatedDeltaNetArBatchedV3Module{loadCudaModule(ctx, "gated_deltanet_ar_batched_v3")},
          _gatedDeltaNetArBatchedV3Kernel{
              _gatedDeltaNetArBatchedV3Module.getFunction("gated_deltanet_ar_batched_v3")},
          _gatedDeltaNetArBatchedV3GateFusedKernel{
              _gatedDeltaNetArBatchedV3Module.getFunction(
                  "gated_deltanet_ar_batched_v3_gatefused")},
          _gatedDeltaNetVerifyBatchedModule{loadCudaModule(ctx, "gated_deltanet_verify_batched")},
          _gatedDeltaNetVerifyBatchedKernel{
              _gatedDeltaNetVerifyBatchedModule.getFunction("gated_deltanet_verify_batched")},
          _gatedDeltaNetFoldModule{loadCudaModule(ctx, "gated_deltanet_fold")},
          _gatedDeltaNetFoldKernel{
              _gatedDeltaNetFoldModule.getFunction("gated_deltanet_fold")},
          _argmaxRowsModule{loadCudaModule(ctx, "argmax_rows")},
          _argmaxRowsKernel{_argmaxRowsModule.getFunction("argmax_rows")},
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
              _gatherHeadsModule.getFunction("gather_heads_from_channels")},
          _fusedPostConvPrepKernel {
              _gatherHeadsModule.getFunction("fused_post_conv_prep")},
          _moeGroupBuildModule     {loadCudaModule(ctx, "moe_group_build")},
          _moeGroupBuildKernel     {
              _moeGroupBuildModule.getFunction("moe_group_build")},
          _moeGatherRowsModule     {loadCudaModule(ctx, "moe_gather_rows")},
          _moeGatherRowsKernel     {
              _moeGatherRowsModule.getFunction("moe_gather_rows")},
          _moeScatterExpertOutModule{loadCudaModule(ctx, "moe_scatter_expert_out")},
          _moeScatterExpertOutKernel{
              _moeScatterExpertOutModule.getFunction("moe_scatter_expert_out")},
          _kvWriteTokensModule{loadCudaModule(ctx, "kv_write_tokens_batched")},
          _kvWriteTokensKernel{
              _kvWriteTokensModule.getFunction("kv_write_tokens_batched")},
          _kvWriteTokensFp16Kernel{
              _kvWriteTokensModule.getFunction("kv_write_tokens_batched_fp16")},
          _moeGroupTilesModule     {loadCudaModule(ctx, "moe_group_tiles")},
          _moeGroupTilesKernel     {
              _moeGroupTilesModule.getFunction("moe_group_tiles")},
          _moeGroupedGemmNvfp4Module{
              loadCudaModule(ctx, "moe_grouped_gemm_nvfp4blk")},
          _moeGroupedGemmNvfp4Kernel{
              _moeGroupedGemmNvfp4Module.getFunction("moe_grouped_gemm_nvfp4blk")},
          _moeGroupedGemmNvfp4M4Kernel{
              _moeGroupedGemmNvfp4Module.getFunction("moe_grouped_gemm_nvfp4blk_m4")},
          _moeGroupedGemmNvfp4M1RegKernel{
              _moeGroupedGemmNvfp4Module.getFunction("moe_grouped_gemm_nvfp4blk_m1reg")},
          _nvfp4DeintModule{loadCudaModule(ctx, "matmul_nvfp4blk_deint_vec")},
          _nvfp4DeinterleaveKernel{
              _nvfp4DeintModule.getFunction("nvfp4blk_deinterleave")},
          _moeGroupedGemmNvfp4DeintKernel{
              _nvfp4DeintModule.getFunction("moe_grouped_gemm_nvfp4blk_deint")},
          _moeGroupedGemmNvfp4DeintRegKernel{
              _nvfp4DeintModule.getFunction("moe_grouped_gemm_nvfp4blk_deint_m1reg")},
          _moePadModule            {loadCudaModule(ctx, "moe_pad")},
          _moePadOffsetsKernel     {_moePadModule.getFunction("moe_pad_offsets")},
          _moeContigToPadKernel    {_moePadModule.getFunction("moe_contig_to_pad")},
          _moeRowsScatterKernel    {_moePadModule.getFunction("moe_rows_scatter_f32")},
          _moeIndexGatherKernel    {_moePadModule.getFunction("moe_index_gather_i32")},
          _moeActQuantModule       {loadCudaModule(ctx, "moe_act_quant_nvfp4")},
          _moeActQuantKernel       {_moeActQuantModule.getFunction("moe_act_quant_nvfp4")},
          _moeActQuantRowsKernel   {_moeActQuantModule.getFunction("moe_act_quant_nvfp4_rows")}
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
    if (const char* pp = std::getenv("MIMIRMIND_DECODE_PROFILE")) {
        _pimpl->_profOn = (pp[0] == '1' && pp[1] == '\0');
    }
    if (_pimpl->_profOn) {
        cudaEventCreate(&_pimpl->_profA);
        cudaEventCreate(&_pimpl->_profB);
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

    // P3.a opt-in: GQA-head-packed F32 prefill flash (env, default off).
    if (const char* g = std::getenv("MIMIRMIND_ATTN_PREFILL_GQA")) {
        _prefillGqaF32Enabled = (g[0] != '\0' && !(g[0] == '0' && g[1] == '\0'));
    }
    if (_prefillGqaF32Enabled) {
        MM_LOG_INFO("hipgpuops",
                    "F32 prefill attention -> GQA-head-packed kernel enabled "
                    "(MIMIRMIND_ATTN_PREFILL_GQA=1) — each K/V row read once per "
                    "KV group instead of once per query head; bit-exact with the "
                    "plain flash kernel, plain kernel is the fallback");
    }

    // P3.b opt-in: TF32 tensor-core GQA-head-packed F32 prefill flash.
    if (const char* t = std::getenv("MIMIRMIND_ATTN_TC_PREFILL")) {
        _prefillTcF32Enabled = (t[0] != '\0' && !(t[0] == '0' && t[1] == '\0'));
    }
    if (_prefillTcF32Enabled) {
        MM_LOG_INFO("hipgpuops",
                    "F32 prefill attention -> TF32 tensor-core GQA kernel enabled "
                    "(MIMIRMIND_ATTN_TC_PREFILL=1) — QK^T and P.V on TF32 tensor "
                    "cores over the head-packed tiling; bit-near (TF32), "
                    "parity-gated; plain/GQA kernel is the fallback");
    }

    // Step 3 opt-in: FP16 tensor-core FA-2 prefill (requires fp16 KV cache).
    if (const char* ft = std::getenv("MIMIRMIND_ATTN_FP16_TC")) {
        _prefillFp16TcEnabled = (ft[0] != '\0' && !(ft[0] == '0' && ft[1] == '\0'));
    }
    if (_prefillFp16TcEnabled) {
        MM_LOG_INFO("hipgpuops",
                    "FP16 prefill attention -> tensor-core FA-2 kernel enabled "
                    "(MIMIRMIND_ATTN_FP16_TC=1) — q-tiled wmma m16n16k16 over the "
                    "fp16 KV cache; bit-near (fp16), parity-gated; scalar fp16 "
                    "kernel is the fallback");
    }

    // Step 3.2 opt-in: GQA-head-packed multi-warp FP16 tensor-core FA-2.
    if (const char* fg = std::getenv("MIMIRMIND_ATTN_FP16_GQA_TC")) {
        _prefillFp16GqaTcEnabled = (fg[0] != '\0' && !(fg[0] == '0' && fg[1] == '\0'));
    }
    if (_prefillFp16GqaTcEnabled) {
        MM_LOG_INFO("cudagpuops",
                    "FP16 prefill attention -> GQA-head-packed multi-warp "
                    "tensor-core FA-2 kernel enabled (MIMIRMIND_ATTN_FP16_GQA_TC=1) "
                    "— one CTA per (kv-head, q-tile), nQPerKv warps share the "
                    "fp16 K/V tile; M=16 real query rows; bit-near (fp16), "
                    "parity-gated; falls back to the scalar fp16 kernel");
    }

    // Multi-warp TF32 FA-2 for the F32 KV path (the path Qwen3-Next prefill
    // attention actually takes; fixes P3.b's single-warp + M=8-underfill).
    if (const char* fm = std::getenv("MIMIRMIND_ATTN_F32_MWTC")) {
        _prefillF32MwtcEnabled = (fm[0] != '\0' && !(fm[0] == '0' && fm[1] == '\0'));
    }
    if (_prefillF32MwtcEnabled) {
        MM_LOG_INFO("cudagpuops",
                    "F32 prefill attention -> multi-warp TF32 tensor-core FA-2 "
                    "kernel enabled (MIMIRMIND_ATTN_F32_MWTC=1) — one CTA per "
                    "(kv-head, q-tile, head-half), HPB warps share the F32 K/V "
                    "tile; M=16 real query rows; bit-near (TF32), parity-gated; "
                    "falls back to the F32-GQA/plain kernel");
    }

    // cuDNN 9 SDPA prefill attention (the FMHA-library path that beats the
    // hand-rolled kernels — cuDNN tiles head_dim=256 past the 99 KiB smem wall).
    if (const char* cd = std::getenv("MIMIRMIND_ATTN_CUDNN")) {
        _prefillCudnnEnabled = (cd[0] != '\0' && !(cd[0] == '0' && cd[1] == '\0'));
        _prefillCudnnEnvSet  = true;   // env is explicit — Layer-2 profile must not override
    }
#if MIMIRMIND_HAVE_CUDNN_SDPA
    if (_prefillCudnnEnabled) {
        MM_LOG_INFO("cudagpuops",
                    "F32 prefill attention -> cuDNN 9 SDPA (fused flash attention) "
                    "enabled (MIMIRMIND_ATTN_CUDNN=1) — single-forward causal GQA, "
                    "F32 staged to bf16; bit-near (bf16), parity-gated; hand kernel "
                    "is the fallback (posOff>0 / non-causal / cuDNN error)");
    }
#else
    if (_prefillCudnnEnabled) {
        MM_LOG_WARN("cudagpuops",
                    "MIMIRMIND_ATTN_CUDNN=1 set but this build was compiled without "
                    "cuDNN (MIMIRMIND_ENABLE_CUDNN=OFF) — ignored, using hand kernels");
        _prefillCudnnEnabled = false;
    }
#endif

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

void GpuOps::profileSection(const char* name) {
    if (!_pimpl->_profOn) {
        return;
    }
    cudaStream_t s = _ctx.stream().handle();
    if (!_pimpl->_profPrev.empty()) {
        cudaEventRecord(_pimpl->_profB, s);
        cudaEventSynchronize(_pimpl->_profB);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, _pimpl->_profA, _pimpl->_profB);
        bool found = false;
        for (auto& p : _pimpl->_profAcc) {
            if (p.first == _pimpl->_profPrev) { p.second += ms; found = true; break; }
        }
        if (!found) {
            _pimpl->_profAcc.emplace_back(_pimpl->_profPrev,
                                          static_cast<double>(ms));
        }
    }
    cudaEventRecord(_pimpl->_profA, s);
    _pimpl->_profPrev = name;
}

void GpuOps::profileStepEnd() {
    if (!_pimpl->_profOn) {
        return;
    }
    if (!_pimpl->_profPrev.empty()) {
        cudaStream_t s = _ctx.stream().handle();
        cudaEventRecord(_pimpl->_profB, s);
        cudaEventSynchronize(_pimpl->_profB);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, _pimpl->_profA, _pimpl->_profB);
        bool found = false;
        for (auto& p : _pimpl->_profAcc) {
            if (p.first == _pimpl->_profPrev) { p.second += ms; found = true; break; }
        }
        if (!found) {
            _pimpl->_profAcc.emplace_back(_pimpl->_profPrev,
                                          static_cast<double>(ms));
        }
        _pimpl->_profPrev.clear();
    }
    static const int kProfEvery = []() {
        const char* e = std::getenv("MIMIRMIND_PROFILE_EVERY");
        const int v = (e != nullptr) ? std::atoi(e) : 0;
        return (v >= 1) ? v : 32;   // default 32 (per decode step); lower for prefill
    }();
    if (++_pimpl->_profSteps >= kProfEvery) {
        double total = 0.0;
        for (auto& p : _pimpl->_profAcc) total += p.second;
        std::string line;
        for (auto& p : _pimpl->_profAcc) {
            const double perStep = p.second / static_cast<double>(kProfEvery);
            const double pct = (total > 0.0) ? (100.0 * p.second / total) : 0.0;
            line += " " + p.first + "="
                  + std::to_string(perStep).substr(0, 6) + "ms("
                  + std::to_string(static_cast<int>(pct + 0.5)) + "%)";
        }
        MM_LOG_INFO("decode-prof", "{}-step avg, ms/step:{} | total={}",
                    std::to_string(kProfEvery), line,
                    std::to_string(total / static_cast<double>(kProfEvery)).substr(0, 6));
        _pimpl->_profAcc.clear();
        _pimpl->_profSteps = 0;
    }
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

// Immutable-weight allocation. Same buffer as allocate() (Managed on unified
// GB10, Device on discrete), plus a read-mostly + device-preferred hint so the
// decode hot path reads weights device-resident without fault-driven migration.
// The hint is a no-op on discrete parts (guarded inside adviseReadMostly).
// Opt-in until A/B-validated on the target: MIMIRMIND_WEIGHT_ADVISE=1.
compute::ComputeBuffer GpuOps::allocateWeight(std::size_t bytes) {
    compute::ComputeBuffer buf = allocate(bytes);
    static const bool advise = std::getenv("MIMIRMIND_WEIGHT_ADVISE") != nullptr;
    if (advise && buf.get() != nullptr) {
        _ctx.allocator().adviseReadMostly(buf.get(), bytes);
    }
    return buf;
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

void GpuOps::layerNormAsync(const float* x, std::size_t M, std::size_t K,
                            const float* weight, const float* bias, float eps,
                            float* y) {
    if (M == 0 || K == 0) {
        return;
    }
    const std::int32_t Ki = toInt32(K, "layerNorm K");
    auto& k = _pimpl->_layernormKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, weight);
    k.setPtr  (2, bias);
    k.setPtr  (3, y);
    k.setValue(4, eps);
    k.setValue(5, Ki);
    // One workgroup per row (BERT/XLM-R encoder LayerNorm).
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(M), 1, 1,
             kLayernormLocalSize, 1, 1);
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

void GpuOps::siluMulSplitAsync(const float* w13, float* out,
                              std::size_t rows, std::size_t nff) {
    const std::size_t total = rows * nff;
    if (total == 0) {
        return;
    }
    const std::int32_t rowsI = toInt32(rows, "siluMulSplit rows");
    const std::int32_t nffI  = toInt32(nff,  "siluMulSplit nff");
    auto& k = _pimpl->_siluMulSplitKernel;
    k.setPtr  (0, w13);
    k.setPtr  (1, out);
    k.setValue(2, rowsI);
    k.setValue(3, nffI);
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
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

void GpuOps::geluErfAsync(float* x, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "geluErf n");
    auto& k = _pimpl->_geluErfKernel;
    k.setPtr  (0, x);
    k.setValue(1, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::tanhInPlaceAsync(float* x, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::int32_t ni = toInt32(n, "tanh n");
    auto& k = _pimpl->_tanhKernel;
    k.setPtr  (0, x);
    k.setValue(1, ni);
    k.launch(_ctx.stream(),
             groupsForN(n, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
}

void GpuOps::encoderEmbedAddAsync(float* x, const float* posTable,
                                  const float* typeVec, std::size_t T,
                                  std::size_t hidden, std::size_t posOffset) {
    const std::size_t total = T * hidden;
    if (total == 0) {
        return;
    }
    auto& k = _pimpl->_encoderEmbedAddKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, posTable);
    k.setPtr  (2, typeVec);
    k.setValue(3, toInt32(T,         "encEmbed T"));
    k.setValue(4, toInt32(hidden,    "encEmbed hidden"));
    k.setValue(5, toInt32(posOffset, "encEmbed posOffset"));
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
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
    if (kvDtype == runtime::KvDtype::Q8_0) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::mropeInPlaceAsync: kvDtype=Q8_0 not "
            "supported (no quantised IMRoPE path)");
    }
    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    const std::int32_t startI = toInt32(startPos, "mrope startPos");
    stagedInt32ToDevice(_curLenSlotUsm, startI);

    // F32 -> rope_mrope; FP16 -> rope_mrope_fp16 (identical arg layout, xBase
    // reinterpreted as __half* in-kernel). FP16 is the attn-FMHA KV path.
    auto& k = (kvDtype == runtime::KvDtype::FP16)
                  ? _pimpl->_mropeFp16Kernel
                  : _pimpl->_mropeKernel;
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

void GpuOps::fusedPostConvPrepAsync(const float* qkvMixed, float* qOut,
                                    float* kOut, float* vOut, std::size_t T,
                                    std::size_t srcHeadsKV, std::size_t dstHeads,
                                    std::size_t S, std::size_t convTotalWidth,
                                    std::size_t keyDim, float eps) {
    const std::size_t rows = T * dstHeads;
    if (rows == 0 || S == 0) {
        return;
    }
    auto& k = _pimpl->_fusedPostConvPrepKernel;
    k.setPtr  (0, qkvMixed);
    k.setPtr  (1, qOut);
    k.setPtr  (2, kOut);
    k.setPtr  (3, vOut);
    k.setValue(4,  toInt32(T,              "fpcp T"));
    k.setValue(5,  toInt32(srcHeadsKV,     "fpcp srcHeadsKV"));
    k.setValue(6,  toInt32(dstHeads,       "fpcp dstHeads"));
    k.setValue(7,  toInt32(S,              "fpcp S"));
    k.setValue(8,  toInt32(convTotalWidth, "fpcp convW"));
    k.setValue(9,  toInt32(keyDim,         "fpcp keyDim"));
    k.setValue(10, eps);
    k.launch(_ctx.stream(),
             groupsForN(rows, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
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
    // P2.a — opt-in smem-staged prefill recurrence. The batched v3 kernel with
    // grid (H, 1) is exactly the single-sequence smem-staged recurrence (seq=0
    // zeroes every per-seq stride), and it is bit-identical to the plain AR
    // kernel (proven byte-for-byte for decode). The plain prefill kernel streams
    // the [S,S] state 2R+2W/token from global; v3 stages it to dynamic smem once
    // (1R at start + 1W at end for the whole call). Opt-in MIMIRMIND_GDN_PREFILL_V3=1;
    // falls back to the plain kernel if the device rejects the >48 KiB smem opt-in.
    static const bool gdnPrefillV3Req = [] {
        const char* e = std::getenv("MIMIRMIND_GDN_PREFILL_V3");
        return e != nullptr && !(e[0] == '0' && e[1] == '\0') && e[0] != '\0';
    }();
    // P2.b — 2-way row-split r2 kernel (block = 2S, smem-staged). Halves the
    // per-thread reduction and doubles resident warps to hide the smem latency
    // the v3 kernel is left with. Bit-near (split-sum FP order); golden-parity
    // gated. Priority over v3 when set. Opt-in MIMIRMIND_GDN_PREFILL_R2=1.
    static const bool gdnPrefillR2Req = [] {
        const char* e = std::getenv("MIMIRMIND_GDN_PREFILL_R2");
        return e != nullptr && !(e[0] == '0' && e[1] == '\0') && e[0] != '\0';
    }();
    const std::size_t gdnSmemBytes =
        static_cast<std::size_t>(S) * S * sizeof(float);
    auto smemOptIn = [&](core::cuda::CudaKernel& kern, const char* tag) -> bool {
        try {
            kern.setMaxDynamicSharedBytes(gdnSmemBytes);
            MM_LOG_INFO("cudagpuops",
                        "GDN prefill {} (smem-staged) enabled: {} bytes dynamic smem",
                        tag, gdnSmemBytes);
            return true;
        } catch (const core::cuda::CudaDriverError& err) {
            MM_LOG_WARN("cudagpuops",
                        "GDN prefill {} opt-in for {} bytes smem rejected ({}); "
                        "using the plain AR kernel",
                        tag, gdnSmemBytes, err.what());
            return false;
        }
    };
    bool usePrefillR2 = false;
    bool usePrefillV3 = false;
    if (gdnPrefillR2Req) {
        static const bool r2Ready =
            smemOptIn(_pimpl->_gatedDeltaNetArR2Kernel,
                      "r2 (MIMIRMIND_GDN_PREFILL_R2=1)");
        usePrefillR2 = r2Ready;
    }
    if (!usePrefillR2 && gdnPrefillV3Req) {
        static const bool v3Ready =
            smemOptIn(_pimpl->_gatedDeltaNetArBatchedV3Kernel,
                      "v3 (MIMIRMIND_GDN_PREFILL_V3=1)");
        usePrefillV3 = v3Ready;
    }
    const bool useSmem = usePrefillR2 || usePrefillV3;
    // grid = H blocks (one per head). block = S (plain/v3, one thread/column) or
    // 2S (r2, two threads/column).
    auto& k = usePrefillR2 ? _pimpl->_gatedDeltaNetArR2Kernel
            : usePrefillV3 ? _pimpl->_gatedDeltaNetArBatchedV3Kernel
                           : _pimpl->_gatedDeltaNetArKernel;
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
    const std::uint32_t blockX = usePrefillR2
        ? static_cast<std::uint32_t>(2 * S)
        : static_cast<std::uint32_t>(S);
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H), 1, 1,
             blockX, 1, 1,
             useSmem ? gdnSmemBytes : 0);
}

void GpuOps::gatedDeltaNetRecurrentBatchedAsync(
        const float* q, const float* k_, const float* v, const float* gLog,
        const float* beta, float* state, float* out,
        const GdnBatchedShape& shape) {
    const std::size_t nSeq = shape.nSeq, T = shape.T, H = shape.H, S = shape.S;
    const std::uint8_t* const activeMask = shape.activeMask;
    if (nSeq == 0 || T == 0 || H == 0 || S == 0) {
        return;
    }
    // grid = (H, nSeq) blocks, block = S threads. Each (head, seq) block
    // owns one sequence [S,S] state; math is byte-identical to the
    // single-sequence gatedDeltaNetRecurrentAsync (M-Cuda.Batch Cat C-P0).
    // E-GDN.1: the latency-optimised v2 kernel (bit-identical math, fewer
    // global state passes + pipelined loads, +2.8% serving decode @nSeq16).
    // Default ON — it is bit-identical to v1, so there is no quality tradeoff;
    // MIMIRMIND_GDN_V2=0 rolls back to v1.
    static const bool gdnV2 = [] {
        const char* e = std::getenv("MIMIRMIND_GDN_V2");
        return e == nullptr || !(e[0] == '0' && e[1] == '\0');
    }();
    // E-GDN.2: the smem-staged v3 kernel (bit-identical math). Stages the whole
    // [S,S] state block into dynamic shared memory once so the latency-critical
    // recurrence reads hit smem, not the ~937-cycle global state loads. Needs a
    // >48 KiB dynamic-smem opt-in (S*S*4 = 64 KiB for S=128). Default ON — it is
    // bit-identical to v2 (verified byte-for-byte across 8 diverse prompts x 64
    // tokens) and +8.3% serving decode @nSeq16; MIMIRMIND_GDN_V3=0 rolls back to
    // v2. It silently falls back to v2/v1 if the device rejects the smem request
    // (e.g. a model whose S*S*4 exceeds the device opt-in cap).
    static const bool gdnV3Req = [] {
        const char* e = std::getenv("MIMIRMIND_GDN_V3");
        return e == nullptr || !(e[0] == '0' && e[1] == '\0');
    }();
    const std::size_t gdnSmemBytes =
        static_cast<std::size_t>(S) * S * sizeof(float);
    bool useV3 = false;
    if (gdnV3Req) {
        static const bool v3Ready = [&] {
            try {
                _pimpl->_gatedDeltaNetArBatchedV3Kernel
                    .setMaxDynamicSharedBytes(gdnSmemBytes);
                MM_LOG_INFO("cudagpuops",
                            "GDN v3 (smem-staged) enabled: {} bytes dynamic smem",
                            gdnSmemBytes);
                return true;
            } catch (const core::cuda::CudaDriverError& err) {
                MM_LOG_WARN("cudagpuops",
                            "GDN v3 opt-in for {} bytes smem rejected ({}); "
                            "falling back to v2",
                            gdnSmemBytes, err.what());
                return false;
            }
        }();
        useV3 = v3Ready;
    }
    auto& k = useV3 ? _pimpl->_gatedDeltaNetArBatchedV3Kernel
                    : (gdnV2 ? _pimpl->_gatedDeltaNetArBatchedV2Kernel
                             : _pimpl->_gatedDeltaNetArBatchedKernel);
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
    k.setPtr  (10, activeMask);   // 5.21-I: nullptr => all-active (bit-identical)
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H),
             static_cast<std::uint32_t>(nSeq), 1,
             static_cast<std::uint32_t>(S), 1, 1,
             useV3 ? gdnSmemBytes : 0);
}

void GpuOps::gatedDeltaNetRecurrentGateFusedBatchedAsync(
        const float* q, const float* k_, const float* v, const float* alpha,
        const float* beta, const float* ssmA, const float* ssmDt, float* state,
        float* out, const GdnBatchedShape& shape) {
    const std::size_t nSeq = shape.nSeq, T = shape.T, H = shape.H, S = shape.S;
    const std::uint8_t* const activeMask = shape.activeMask;
    if (nSeq == 0 || T == 0 || H == 0 || S == 0) {
        return;
    }
    // GDN-Inc 2: v3 with the decay gate (deltanet_gate) + beta sigmoid folded in
    // (vLLM fused_sigmoid_gating_delta_rule_update). Always smem-staged (same
    // >48 KiB opt-in as v3); requires the device to accept the S*S*4 request
    // (true wherever v3 runs). Bit-identical to v3 + the separate gate passes.
    const std::size_t gdnSmemBytes =
        static_cast<std::size_t>(S) * S * sizeof(float);
    static const bool ready = [&] {
        try {
            _pimpl->_gatedDeltaNetArBatchedV3GateFusedKernel
                .setMaxDynamicSharedBytes(gdnSmemBytes);
            return true;
        } catch (const core::cuda::CudaDriverError& err) {
            MM_LOG_WARN("cudagpuops",
                        "GDN gate-fused smem opt-in ({} bytes) rejected ({})",
                        gdnSmemBytes, err.what());
            return false;
        }
    }();
    if (!ready) {
        throw std::runtime_error(
            "gatedDeltaNetRecurrentGateFusedBatchedAsync: device rejected the "
            "dynamic-smem opt-in required by the fused GDN kernel");
    }
    auto& k = _pimpl->_gatedDeltaNetArBatchedV3GateFusedKernel;
    k.setPtr  (0, q);
    k.setPtr  (1, k_);
    k.setPtr  (2, v);
    k.setPtr  (3, alpha);
    k.setPtr  (4, beta);
    k.setPtr  (5, ssmA);
    k.setPtr  (6, ssmDt);
    k.setPtr  (7, state);
    k.setPtr  (8, out);
    k.setValue(9,  toInt32(T, "gdnGF T"));
    k.setValue(10, toInt32(H, "gdnGF H"));
    k.setValue(11, toInt32(S, "gdnGF S"));
    k.setPtr  (12, activeMask);   // 5.21-I: nullptr => all-active (bit-identical)
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H),
             static_cast<std::uint32_t>(nSeq), 1,
             static_cast<std::uint32_t>(S), 1, 1,
             gdnSmemBytes);
}

void GpuOps::gatedDeltaNetVerifyBatchedAsync(
        const float* q, const float* k_, const float* v, const float* gLog,
        const float* beta, const float* stateIn, float* stateOut, float* out,
        std::size_t nSeq, std::size_t T, std::size_t H, std::size_t S) {
    // MV-a: verify the T=K+1 window for nSeq slots in one launch, state resident
    // in smem across all T, exporting the per-position state to stateOut so a
    // partial accept can pick the accepted-prefix state (no per-step snapshot).
    // Byte-identical per-step math to gated_deltanet_ar_batched_v3. q/k/v/out
    // are time-major [T, nSeq, H, S] (verify layout); stateIn [nSeq,H,S,S];
    // stateOut [T, nSeq, H, S, S]. Always smem-staged (needs the >48 KiB opt-in).
    if (nSeq == 0 || T == 0 || H == 0 || S == 0) {
        return;
    }
    const std::size_t gdnSmemBytes =
        static_cast<std::size_t>(S) * S * sizeof(float);
    _pimpl->_gatedDeltaNetVerifyBatchedKernel.setMaxDynamicSharedBytes(gdnSmemBytes);
    auto& k = _pimpl->_gatedDeltaNetVerifyBatchedKernel;
    k.setPtr  (0, q);
    k.setPtr  (1, k_);
    k.setPtr  (2, v);
    k.setPtr  (3, gLog);
    k.setPtr  (4, beta);
    k.setPtr  (5, stateIn);
    k.setPtr  (6, stateOut);
    k.setPtr  (7, out);
    k.setValue(8,  toInt32(T, "gdnV T"));
    k.setValue(9,  toInt32(nSeq, "gdnV nSeq"));
    k.setValue(10, toInt32(H, "gdnV H"));
    k.setValue(11, toInt32(S, "gdnV S"));
    k.launch(_ctx.stream(),
             static_cast<std::uint32_t>(H),
             static_cast<std::uint32_t>(nSeq), 1,
             static_cast<std::uint32_t>(S), 1, 1,
             gdnSmemBytes);
}

void GpuOps::gatedDeltaNetFoldAsync(const float* k, const float* v,
                                    const float* gLog, const float* beta,
                                    float* state, std::size_t acceptLen,
                                    std::size_t H, std::size_t S) {
    // Replay timesteps [0, acceptLen) of the accepted verify window into `state`
    // in place (state-only gated delta-rule; byte-identical to gated_deltanet_ar
    // minus the output). k,v [acceptLen,H,S]; gLog,beta [acceptLen,H];
    // state [H,S,S]. Launch: grid=H, block=S.
    if (acceptLen == 0 || H == 0 || S == 0) {
        return;
    }
    auto& kern = _pimpl->_gatedDeltaNetFoldKernel;
    kern.setPtr  (0, k);
    kern.setPtr  (1, v);
    kern.setPtr  (2, gLog);
    kern.setPtr  (3, beta);
    kern.setPtr  (4, state);
    kern.setValue(5, toInt32(acceptLen, "fold acceptLen"));
    kern.setValue(6, toInt32(H, "fold H"));
    kern.setValue(7, toInt32(S, "fold S"));
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(H), 1, 1,
                static_cast<std::uint32_t>(S), 1, 1);
}

void GpuOps::argmaxRowsAsync(const float* logits, std::int32_t* out,
                             std::size_t nRows, std::size_t vocab) {
    if (nRows == 0 || vocab == 0) {
        return;
    }
    auto& k = _pimpl->_argmaxRowsKernel;
    k.setPtr  (0, logits);
    k.setPtr  (1, out);
    k.setValue(2, toInt32(nRows, "argmax nRows"));
    k.setValue(3, toInt32(vocab, "argmax vocab"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), 1, 1, 256, 1, 1);
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

void GpuOps::moeGroupBuildAsync(const std::int32_t* expIdx, const float* kw,
                                std::int32_t* expOffset, std::int32_t* rowSrcTok,
                                float* rowKw, std::int32_t* asnToRow,
                                std::size_t R, std::size_t nExperts,
                                std::size_t K) {
    if (R == 0 || nExperts == 0) {
        return;
    }
    auto& k = _pimpl->_moeGroupBuildKernel;
    k.setPtr  (0, expIdx);
    k.setPtr  (1, kw);
    k.setPtr  (2, expOffset);
    k.setPtr  (3, rowSrcTok);
    k.setPtr  (4, rowKw);
    k.setPtr  (5, asnToRow);
    k.setValue(6, toInt32(R, "moeGroupBuild R"));
    k.setValue(7, toInt32(nExperts, "moeGroupBuild nExperts"));
    k.setValue(8, toInt32(K, "moeGroupBuild K"));
    // v2: one block, 256 threads — parallel zero/publish + shared-mem histogram/
    // scan/scatter on thread 0 (bit-identical to the CPU golden; see kernel).
    k.launch(_ctx.stream(), 1, 1, 1, 256, 1, 1);
}

void GpuOps::moeGatherRowsAsync(const float* x, const std::int32_t* rowSrcTok,
                                float* xCompact, std::size_t dModel,
                                std::size_t R) {
    if (R == 0 || dModel == 0) {
        return;
    }
    auto& k = _pimpl->_moeGatherRowsKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, rowSrcTok);
    k.setPtr  (2, xCompact);
    k.setValue(3, toInt32(dModel, "moeGatherRows dModel"));
    k.setValue(4, toInt32(R, "moeGatherRows R"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(R), 1, 1, 256, 1, 1);
}

void GpuOps::moeScatterExpertOutAsync(const float* y, const std::int32_t* asnToRow,
                                      const float* kw, float* accum,
                                      std::size_t dModel, std::size_t T,
                                      std::size_t K) {
    if (T == 0 || dModel == 0) {
        return;
    }
    auto& k = _pimpl->_moeScatterExpertOutKernel;
    k.setPtr  (0, y);
    k.setPtr  (1, asnToRow);
    k.setPtr  (2, kw);
    k.setPtr  (3, accum);
    k.setValue(4, toInt32(dModel, "moeScatter dModel"));
    k.setValue(5, toInt32(T, "moeScatter T"));
    k.setValue(6, toInt32(K, "moeScatter K"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(T), 1, 1, 256, 1, 1);
}

void GpuOps::writeKvTokensBatchedAsync(const float* kProj, const float* vProj,
                                       const std::uint32_t* writeBlockIdDev,
                                       const std::int32_t* writeSlotDev,
                                       void* kPool, void* vPool,
                                       std::size_t nSeq, std::size_t blockSize,
                                       std::size_t width,
                                       runtime::KvDtype kvDtype,
                                       const std::uint8_t* activeMask) {
    if (nSeq == 0 || width == 0) {
        return;
    }
    // FP16 pool → cast-on-scatter variant; F32 → straight scatter (5.14 I1).
    auto& k = (kvDtype == runtime::KvDtype::FP16)
                  ? _pimpl->_kvWriteTokensFp16Kernel
                  : _pimpl->_kvWriteTokensKernel;
    k.setPtr  (0, kProj);
    k.setPtr  (1, vProj);
    k.setPtr  (2, writeBlockIdDev);
    k.setPtr  (3, writeSlotDev);
    k.setPtr  (4, kPool);
    k.setPtr  (5, vPool);
    k.setValue(6, toInt32(nSeq, "writeKv nSeq"));
    k.setValue(7, toInt32(blockSize, "writeKv blockSize"));
    k.setValue(8, toInt32(width, "writeKv width"));
    k.setPtr  (9, activeMask);   // 5.21-I: nullptr => all-active (bit-identical)
    const std::uint32_t blk = width < 256 ? static_cast<std::uint32_t>(width) : 256;
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nSeq), 1, 1, blk, 1, 1);
}

void GpuOps::moeGroupTilesAsync(const std::int32_t* expOffset,
                                std::int32_t* tileExpert, std::int32_t* tileRow0,
                                std::int32_t* tileRows, std::int32_t* nTiles,
                                std::size_t nExperts, std::size_t maxTiles,
                                std::size_t tileM) {
    if (nExperts == 0 || maxTiles == 0) {
        return;
    }
    auto& k = _pimpl->_moeGroupTilesKernel;
    k.setPtr  (0, expOffset);
    k.setPtr  (1, tileExpert);
    k.setPtr  (2, tileRow0);
    k.setPtr  (3, tileRows);
    k.setPtr  (4, nTiles);
    k.setValue(5, toInt32(nExperts, "moeGroupTiles nExperts"));
    k.setValue(6, toInt32(maxTiles, "moeGroupTiles maxTiles"));
    k.setValue(7, toInt32(tileM, "moeGroupTiles tileM"));
    // v2: one block, 256 threads — parallel sentinel-fill + thread-0 walk
    // (schedule bit-identical to the CPU golden; see kernel).
    k.launch(_ctx.stream(), 1, 1, 1, 256, 1, 1);
}

void GpuOps::moeGroupedGemmNvfp4Async(const float* x, const unsigned char* w,
                                      float* y, const std::int32_t* tileExpert,
                                      const std::int32_t* tileRow0,
                                      const std::int32_t* tileRows,
                                      std::size_t K, std::size_t N,
                                      std::size_t maxTiles, bool decodeSmallM) {
    if (N == 0 || K == 0 || maxTiles == 0) {
        return;
    }
    // Matches matmul_nvfp4blk_gemm's warp layout: 4 output columns per group,
    // 128 threads/block. grid.x tiles N, grid.y indexes the tile schedule.
    constexpr std::uint32_t kOutputsPerGroup = 4;
    constexpr std::uint32_t kLocal           = 128;
    const std::uint32_t nGroups = static_cast<std::uint32_t>(
        (N + kOutputsPerGroup - 1) / kOutputsPerGroup);
    // GD-b: decode uses the small-M (MAX_M=4) kernel — the schedule caps decode
    // tiles at tileM=4, and the smaller shared/register footprint lets the SM
    // run enough warps to hide the memory latency that bottlenecks decode-M.
    auto& k = decodeSmallM ? _pimpl->_moeGroupedGemmNvfp4M4Kernel
                           : _pimpl->_moeGroupedGemmNvfp4Kernel;
    k.setPtr  (0, x);
    k.setPtr  (1, w);
    k.setPtr  (2, y);
    k.setPtr  (3, tileExpert);
    k.setPtr  (4, tileRow0);
    k.setPtr  (5, tileRows);
    k.setValue(6, toInt32(K, "moeGroupedGemm K"));
    k.setValue(7, toInt32(N, "moeGroupedGemm N"));
    k.launch(_ctx.stream(), nGroups, static_cast<std::uint32_t>(maxTiles), 1,
             kLocal, 1, 1);
}

void GpuOps::moeGroupedGemmNvfp4M1NBAsync(const float* x, const unsigned char* w,
                                          float* y, const std::int32_t* tileExpert,
                                          const std::int32_t* tileRow0,
                                          const std::int32_t* tileRows,
                                          std::size_t K, std::size_t N,
                                          std::size_t maxTiles) {
    if (N == 0 || K == 0 || maxTiles == 0) {
        return;
    }
    // One output column per warp (4 warps/block), activation staged in registers.
    constexpr std::uint32_t kWarps = 4;
    constexpr std::uint32_t kLocal = 128;
    const std::uint32_t nGroups = static_cast<std::uint32_t>((N + kWarps - 1) / kWarps);
    auto& k = _pimpl->_moeGroupedGemmNvfp4M1RegKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, w);
    k.setPtr  (2, y);
    k.setPtr  (3, tileExpert);
    k.setPtr  (4, tileRow0);
    k.setPtr  (5, tileRows);
    k.setValue(6, toInt32(K, "moeGroupedGemmM1NB K"));
    k.setValue(7, toInt32(N, "moeGroupedGemmM1NB N"));
    k.launch(_ctx.stream(), nGroups, static_cast<std::uint32_t>(maxTiles), 1,
             kLocal, 1, 1);
}

void GpuOps::moeGroupedGemmNvfp4DeintAsync(
        const float* x, const unsigned char* w, float* y,
        const std::int32_t* tileExpert, const std::int32_t* tileRow0,
        const std::int32_t* tileRows, std::size_t K, std::size_t N,
        std::size_t nExperts, std::size_t maxTiles, bool /*decodeSmallM*/) {
    if (N == 0 || K == 0 || maxTiles == 0) {
        return;
    }
    constexpr std::uint32_t kLocal = 128;
    const std::size_t nSuper      = K / 32;
    const std::size_t totalSupers = nExperts * N * nSuper;

    // De-interleave the interleaved 20-byte blocked bank into a 16-byte-aligned
    // nibble bank + fp16 scale bank once; cache by the weight pointer.
    auto it = _pimpl->_deintCache.find(w);
    if (it == _pimpl->_deintCache.end()) {
        Impl::DeintBank bank;
        bank.nib   = allocate(totalSupers * 16);
        bank.scale = allocate(totalSupers * 4);
        auto& dk = _pimpl->_nvfp4DeinterleaveKernel;
        dk.setPtr  (0, w);
        dk.setPtr  (1, bank.nib.get());
        dk.setPtr  (2, bank.scale.get());
        dk.setValue(3, static_cast<std::int64_t>(totalSupers));
        const std::uint32_t g =
            static_cast<std::uint32_t>((totalSupers + 255) / 256);
        dk.launch(_ctx.stream(), g, 1, 1, 256, 1, 1);
        it = _pimpl->_deintCache.emplace(w, std::move(bank)).first;
    }

    // Increment 1 A/B: MIMIRMIND_MOE_DEINT_REG=1 selects the register-staged
    // variant (no shared memory: activation coalesced into registers, uint4
    // weight broadcast), isolating the alignment win from the smem-x conflict.
    static const bool useReg = []() {
        const char* r = std::getenv("MIMIRMIND_MOE_DEINT_REG");
        return r != nullptr && r[0] == '1' && r[1] == '\0';
    }();
    auto& k = useReg ? _pimpl->_moeGroupedGemmNvfp4DeintRegKernel
                     : _pimpl->_moeGroupedGemmNvfp4DeintKernel;
    k.setPtr  (0, x);
    k.setPtr  (1, it->second.nib.get());
    k.setPtr  (2, it->second.scale.get());
    k.setPtr  (3, y);
    k.setPtr  (4, tileExpert);
    k.setPtr  (5, tileRow0);
    k.setPtr  (6, tileRows);
    k.setValue(7, toInt32(K, "deint K"));
    k.setValue(8, toInt32(N, "deint N"));
    const std::uint32_t nGroups = static_cast<std::uint32_t>((N + 3) / 4);
    // The register variant uses no shared memory; the smem variant stages the
    // activation row (MAX_M(1) * K floats).
    const std::size_t smemBytes = useReg ? 0 : K * sizeof(float);
    k.launch(_ctx.stream(), nGroups, static_cast<std::uint32_t>(maxTiles), 1,
             kLocal, 1, 1, smemBytes);
}

// === E-d.4b FP4-tensor-core grouped MoE ===================================

bool GpuOps::moeGroupedGemmNvfp4TcAvailable() const noexcept {
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
    return kernels::cutlassmoe::nvfp4TcAvailable();
#else
    return false;
#endif
}

void GpuOps::moeZeroBytesAsync(void* dst, std::size_t bytes) {
    if (bytes == 0) return;
    const cudaError_t rc = cudaMemsetAsync(dst, 0, bytes, _ctx.stream().handle());
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string("moeZeroBytesAsync: cudaMemsetAsync failed: ")
                                 + cudaGetErrorString(rc));
    }
}

void GpuOps::moePadOffsetsAsync(const std::int32_t* expOffset,
                                std::int32_t* padOffset, std::size_t nExperts) {
    if (nExperts == 0) return;
    auto& k = _pimpl->_moePadOffsetsKernel;
    k.setPtr  (0, expOffset);
    k.setPtr  (1, padOffset);
    k.setValue(2, toInt32(nExperts, "moePadOffsets nExperts"));
    k.launch(_ctx.stream(), 1, 1, 1, 1, 1, 1);
}

void GpuOps::moeContigToPadAsync(const std::int32_t* expOffset,
                                 const std::int32_t* padOffset,
                                 std::int32_t* contigToPad,
                                 std::size_t nExperts, std::size_t R) {
    if (R == 0) return;
    auto& k = _pimpl->_moeContigToPadKernel;
    k.setPtr  (0, expOffset);
    k.setPtr  (1, padOffset);
    k.setPtr  (2, contigToPad);
    k.setValue(3, toInt32(nExperts, "moeContigToPad nExperts"));
    k.setValue(4, toInt32(R, "moeContigToPad R"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>((R + 127) / 128), 1, 1, 128, 1, 1);
}

void GpuOps::moeRowsScatterF32Async(const float* src, const std::int32_t* idxMap,
                                    float* dst, std::size_t nRows, std::size_t dim) {
    if (nRows == 0 || dim == 0) return;
    auto& k = _pimpl->_moeRowsScatterKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, idxMap);
    k.setPtr  (2, dst);
    k.setValue(3, toInt32(nRows, "moeRowsScatter nRows"));
    k.setValue(4, toInt32(dim, "moeRowsScatter dim"));
    const std::uint32_t gy = static_cast<std::uint32_t>((dim + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

void GpuOps::moeIndexGatherI32Async(const std::int32_t* src,
                                    const std::int32_t* idxMap,
                                    std::int32_t* dst, std::size_t n) {
    if (n == 0) return;
    auto& k = _pimpl->_moeIndexGatherKernel;
    k.setPtr  (0, src);
    k.setPtr  (1, idxMap);
    k.setPtr  (2, dst);
    k.setValue(3, toInt32(n, "moeIndexGather n"));
    k.launch(_ctx.stream(), static_cast<std::uint32_t>((n + 127) / 128), 1, 1, 128, 1, 1);
}

void GpuOps::moeActQuantNvfp4Async(const float* in, unsigned char* outNib,
                                   unsigned char* outSf, float gscale,
                                   std::size_t M, std::size_t K) {
    if (M == 0 || K == 0) return;
    auto& k = _pimpl->_moeActQuantKernel;
    k.setPtr  (0, in);
    k.setPtr  (1, outNib);
    k.setPtr  (2, outSf);
    k.setValue(3, gscale);
    k.setValue(4, toInt32(M, "moeActQuant M"));
    k.setValue(5, toInt32(K, "moeActQuant K"));
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(M), gy, 1, 256, 1, 1);
}

void GpuOps::moeActQuantNvfp4RowsAsync(const float* in, unsigned char* outNib,
                                       unsigned char* outSf, float gscale,
                                       const std::int32_t* rowMap,
                                       std::size_t nRows, std::size_t K) {
    if (nRows == 0 || K == 0) return;
    auto& k = _pimpl->_moeActQuantRowsKernel;
    k.setPtr  (0, in);
    k.setPtr  (1, outNib);
    k.setPtr  (2, outSf);
    k.setValue(3, gscale);
    k.setPtr  (4, rowMap);
    k.setValue(5, toInt32(nRows, "moeActQuantRows nRows"));
    k.setValue(6, toInt32(K, "moeActQuantRows K"));
    const std::uint32_t gy = static_cast<std::uint32_t>(((K / 16) + 255) / 256);
    k.launch(_ctx.stream(), static_cast<std::uint32_t>(nRows), gy, 1, 256, 1, 1);
}

std::size_t GpuOps::moeGroupedGemmNvfp4TcBanksScratchBytes(
    std::size_t nExperts) const noexcept {
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
    return kernels::cutlassmoe::groupedNvfp4TcBanksScratchBytes(
        static_cast<int>(nExperts));
#else
    (void)nExperts;
    return 0;
#endif
}

void GpuOps::moeGroupedGemmNvfp4TcBanksAsync(
    std::size_t nExperts, std::size_t N, std::size_t K,
    const std::int32_t* expOffset, const std::int32_t* padOffset,
    const void* aBank, const void* sfaBank,
    const void* bBank, const void* sfbBank,
    const float* globalsBank, void* dBank,
    void* scratch, std::size_t scratchBytes) {
#ifdef MIMIRMIND_HAVE_CUTLASS_MOE
    // Scratch is caller-owned (per-slot BlockBuffers) — no shared GpuOps state,
    // so concurrent prefills never collide on it.
    const int rc = kernels::cutlassmoe::runGroupedNvfp4TcF32Banks(
        static_cast<int>(nExperts), static_cast<int>(N), static_cast<int>(K),
        expOffset, padOffset, aBank, sfaBank, bBank, sfbBank, globalsBank, dBank,
        scratch, scratchBytes, _ctx.stream().handle());
    if (rc != 0) {
        throw std::runtime_error(
            "moeGroupedGemmNvfp4TcBanksAsync: CUTLASS grouped GEMM failed rc="
            + std::to_string(rc));
    }
#else
    (void)nExperts; (void)N; (void)K; (void)expOffset; (void)padOffset;
    (void)aBank; (void)sfaBank; (void)bBank; (void)sfbBank; (void)globalsBank;
    (void)dBank; (void)scratch; (void)scratchBytes;
    throw std::runtime_error(
        "moeGroupedGemmNvfp4TcBanksAsync: CUTLASS not linked in this build");
#endif
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

void GpuOps::ropeInPlaceInterleavedAsync(void* xBase, const float* freqFactors,
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
            "compute::cuda::GpuOps::ropeInPlaceInterleaved: headDim must be even");
    }
    // Interleaved kernels are F32-only (Q-rope + the llama K-rope fp32
    // workspace). FP16/Q8_0 here is a mis-wire — refuse loudly.
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cuda::GpuOps::ropeInPlaceInterleavedAsync: only F32 KV "
            "supported (interleaved RoPE runs on the fp32 workspace)");
    }

    const std::size_t halfDim = headDim / 2;
    const std::size_t total   = seqLen * numHeads * halfDim;

    const std::int32_t startI = toInt32(startPos, "rope_il startPos");
    stagedInt32ToDevice(_curLenSlotUsm, startI);

    // freqFactors==null -> plain interleaved; non-null -> llama3 factors.
    if (freqFactors == nullptr) {
        auto& k = _pimpl->_ropeInterleavedKernel;
        k.setPtr  (0, xBase);
        k.setValue(1, toInt32(seqLen,   "rope_il seqLen"));
        k.setValue(2, toInt32(numHeads, "rope_il numHeads"));
        k.setValue(3, toInt32(headDim,  "rope_il headDim"));
        k.setPtr  (4, _curLenSlotUsm);
        k.setValue(5, base);
        k.setValue(6, toInt32(writeOffsetStride, "rope_il writeOffsetStride"));
        k.launch(_ctx.stream(),
                 groupsForN(total, kRopeLocalSize), 1, 1,
                 kRopeLocalSize, 1, 1);
    } else {
        auto& k = _pimpl->_ropeFfInterleavedKernel;
        k.setPtr  (0, xBase);
        k.setPtr  (1, freqFactors);
        k.setValue(2, toInt32(seqLen,   "rope_ffil seqLen"));
        k.setValue(3, toInt32(numHeads, "rope_ffil numHeads"));
        k.setValue(4, toInt32(headDim,  "rope_ffil headDim"));
        k.setPtr  (5, _curLenSlotUsm);
        k.setValue(6, base);
        k.setValue(7, toInt32(writeOffsetStride, "rope_ffil writeOffsetStride"));
        k.launch(_ctx.stream(),
                 groupsForN(total, kRopeLocalSize), 1, 1,
                 kRopeLocalSize, 1, 1);
    }
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

void GpuOps::kvCommitFp16Async(const float* xSrc, void* kvDst,
                               std::size_t T, std::size_t kvDim,
                               std::size_t writeOffset) {
    if (T == 0 || kvDim == 0) {
        return;
    }
    // writeOffset (= curLen) goes through the shared curLen slot so kvDst stays
    // a stable layer-base pointer across replays; the kernel adds curLen*kvDim.
    const std::int32_t offI =
        toInt32(writeOffset, "kvCommitFp16 writeOffset");
    stagedInt32ToDevice(_curLenSlotUsm, offI);

    const std::size_t total = T * kvDim;
    auto& k = _pimpl->_kvCommitFp16Kernel;
    k.setPtr  (0, xSrc);
    k.setPtr  (1, kvDst);
    k.setValue(2, toInt32(T,     "kvCommitFp16 T"));
    k.setValue(3, toInt32(kvDim, "kvCommitFp16 kvDim"));
    k.setPtr  (4, _curLenSlotUsm);
    k.launch(_ctx.stream(),
             groupsForN(total, kElementwiseLocalSize), 1, 1,
             kElementwiseLocalSize, 1, 1);
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

void GpuOps::attentionEncoderAsync(const float* q, const float* k,
                                   const float* v, std::size_t T,
                                   std::size_t nHeads, std::size_t nKvHeads,
                                   std::size_t headDim, float scale,
                                   float* out) {
    if (T == 0 || nHeads == 0) {
        return;
    }
    auto& kernel = _pimpl->_attentionEncoderKernel;
    kernel.setPtr  (0, q);
    kernel.setPtr  (1, k);
    kernel.setPtr  (2, v);
    kernel.setPtr  (3, out);
    kernel.setValue(4, toInt32(T,        "attnEnc T_k"));
    kernel.setValue(5, toInt32(nHeads,   "attnEnc nHeads"));
    kernel.setValue(6, toInt32(nKvHeads, "attnEnc nKvHeads"));
    kernel.setValue(7, toInt32(headDim,  "attnEnc headDim"));
    kernel.setValue(8, scale);
    // One workgroup per (head, query-position); every query sees all keys.
    kernel.launch(_ctx.stream(),
                  static_cast<std::uint32_t>(nHeads),
                  static_cast<std::uint32_t>(T),
                  1,
                  kAttentionLocalSize, 1, 1);
}

void GpuOps::attentionEncoderCrossAsync(const float* q, const float* k,
                                        const float* v, std::size_t Tq,
                                        std::size_t Tk, std::size_t nHeads,
                                        std::size_t nKvHeads,
                                        std::size_t headDim, float scale,
                                        float* out) {
    if (Tq == 0 || Tk == 0 || nHeads == 0) {
        return;
    }
    // Same kernel as attentionEncoderAsync — it already reads T_k (arg 4)
    // independently of the query grid, so distinct Tq/Tk just means launching
    // grid.y = Tq while passing T_k = Tk. Query (blockIdx.y) attends to all Tk.
    auto& kernel = _pimpl->_attentionEncoderKernel;
    kernel.setPtr  (0, q);
    kernel.setPtr  (1, k);
    kernel.setPtr  (2, v);
    kernel.setPtr  (3, out);
    kernel.setValue(4, toInt32(Tk,       "attnEncCross T_k"));
    kernel.setValue(5, toInt32(nHeads,   "attnEncCross nHeads"));
    kernel.setValue(6, toInt32(nKvHeads, "attnEncCross nKvHeads"));
    kernel.setValue(7, toInt32(headDim,  "attnEncCross headDim"));
    kernel.setValue(8, scale);
    kernel.launch(_ctx.stream(),
                  static_cast<std::uint32_t>(nHeads),
                  static_cast<std::uint32_t>(Tq),
                  1,
                  kAttentionLocalSize, 1, 1);
}

void GpuOps::attentionEncoderBatchedAsync(const float* q, const float* k,
                                          const float* v, float* out,
                                          const std::int32_t* seqLens,
                                          std::size_t B, std::size_t Tmax,
                                          std::size_t nHeads,
                                          std::size_t nKvHeads,
                                          std::size_t headDim, float scale) {
    if (B == 0 || Tmax == 0 || nHeads == 0) {
        return;
    }
    auto& kernel = _pimpl->_attentionEncoderBatchedKernel;
    kernel.setPtr  (0, q);
    kernel.setPtr  (1, k);
    kernel.setPtr  (2, v);
    kernel.setPtr  (3, out);
    kernel.setPtr  (4, seqLens);
    kernel.setValue(5, toInt32(B,        "attnEncB B"));
    kernel.setValue(6, toInt32(Tmax,     "attnEncB Tmax"));
    kernel.setValue(7, toInt32(nHeads,   "attnEncB nHeads"));
    kernel.setValue(8, toInt32(nKvHeads, "attnEncB nKvHeads"));
    kernel.setValue(9, toInt32(headDim,  "attnEncB headDim"));
    kernel.setValue(10, scale);
    // grid (head, query-pos, batch).
    kernel.launch(_ctx.stream(),
                  static_cast<std::uint32_t>(nHeads),
                  static_cast<std::uint32_t>(Tmax),
                  static_cast<std::uint32_t>(B),
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

#if MIMIRMIND_HAVE_CUDNN_SDPA
    // cuDNN 9 SDPA — preferred F32 prefill-attn path when enabled and eligible.
    // Handles both the first chunk (positionOffset==0, plain causal) and
    // continuation chunks (positionOffset>0, bottom-right causal): the T_q
    // queries attend the full cached K/V range [0, positionOffset+T_q). k/v
    // already point to cache position 0 (the hand kernel reads them the same
    // way). No sliding-window support -> fall back for SWA. cuDNN casts
    // F32->bf16 internally; on any cuDNN error runF32Causal returns false.
    if (_prefillCudnnEnabled &&
        kvDtype == runtime::KvDtype::F32 &&
        slidingWindow == 0 &&
        nKvHeads > 0 && (nHeads % nKvHeads == 0)) {
        if (!_cudnnSdpa) _cudnnSdpa = std::make_unique<CudnnSdpaPrefill>();
        if (_cudnnSdpa->runF32Causal(
                _ctx.stream().handle(),
                q, static_cast<const float*>(k), static_cast<const float*>(v), out,
                toInt32(T_q,                 "cudnn_sdpa T_q"),
                toInt32(positionOffset + T_q, "cudnn_sdpa T_kv"),
                toInt32(nHeads,   "cudnn_sdpa nHeads"),
                toInt32(nKvHeads, "cudnn_sdpa nKvHeads"),
                toInt32(headDim,  "cudnn_sdpa headDim"),
                scale)) {
            return;   // cuDNN handled the prefill attention for this chunk
        }
        // else: fall through to the hand-written kernel selection below.
    }
#endif

    // Multi-warp TF32 FA-2 for the F32 KV path (Step 3.2's F32 sibling; the
    // path Qwen3-Next prefill attention actually takes). Preferred over the
    // scalar P3.a/P3.b F32 kernels when eligible. Needs a dynamic-smem opt-in
    // (qS+oRun dominate; sized by HPB=4 head-half, not nQPerKv).
    constexpr std::size_t kMwtcHpb = 2;   // == ATTN_MW_HPB in the kernel
                                          // (99 KiB sm_121 dyn-smem cap)
    bool useF32Mwtc =
        (kvDtype == runtime::KvDtype::F32) &&
        _prefillF32MwtcEnabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv) &&
        (headDim <= kFlashTcMaxHeadDim) &&
        (headDim % 16 == 0);
    if (useF32Mwtc) {
        constexpr std::size_t BQ = 16, BK = 16;
        auto a128 = [](std::size_t n) { return (n + 127u) & ~std::size_t(127u); };
        const std::size_t hp = kMwtcHpb;
        const std::size_t hd = static_cast<std::size_t>(headDim);
        std::size_t bytes = 0;
        bytes += a128(hp * BQ * hd * sizeof(float));    // qS
        bytes += a128(BK * hd * sizeof(float));         // kvS
        bytes += a128(hp * BQ * hd * sizeof(float));    // oRun
        bytes += a128(hp * BQ * BK * sizeof(float));    // sS
        bytes += a128(hp * BQ * BK * sizeof(float));    // pS
        bytes += a128(hp * BQ * BK * sizeof(float));    // oT
        bytes += a128(hp * BQ * sizeof(float));         // mSh
        bytes += a128(hp * BQ * sizeof(float));         // lSh
        bytes += a128(hp * BQ * sizeof(float));         // aSh
        if (!_prefillF32MwtcSmemAttempted) {
            _prefillF32MwtcSmemAttempted = true;
            try {
                _pimpl->_attentionPrefillFlashF32GqaMwtcKernel
                    .setMaxDynamicSharedBytes(bytes);
                _prefillF32MwtcSmemBytes = bytes;
                _prefillF32MwtcSmemOk    = true;
                MM_LOG_INFO("cudagpuops",
                            "F32 MWTC prefill: dynamic smem opt-in ok "
                            "({} bytes, HPB={}, headDim={})",
                            bytes, kMwtcHpb, headDim);
            } catch (const core::cuda::CudaDriverError& err) {
                _prefillF32MwtcSmemOk = false;
                MM_LOG_WARN("cudagpuops",
                            "F32 MWTC prefill: dynamic smem opt-in for {} bytes "
                            "rejected ({}); falling back to the F32-GQA/plain "
                            "kernel", bytes, err.what());
            }
        }
        useF32Mwtc = _prefillF32MwtcSmemOk &&
                     (bytes <= _prefillF32MwtcSmemBytes);
    }
    const bool useQ8Gqa =
        (kvDtype == runtime::KvDtype::Q8_0) &&
        !_prefillFlashGqaQ8Disabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv);
    // P3.b — opt-in TF32 tensor-core GQA kernel (bit-near, headDim-bounded).
    const bool useF32Tc =
        !useF32Mwtc &&
        (kvDtype == runtime::KvDtype::F32) &&
        _prefillTcF32Enabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv) &&
        (headDim <= kFlashTcMaxHeadDim) &&
        (headDim % 16 == 0);
    // P3.a — opt-in GQA-head-packed F32 kernel (bit-exact fast path).
    const bool useF32Gqa =
        !useF32Mwtc &&
        !useF32Tc &&
        (kvDtype == runtime::KvDtype::F32) &&
        _prefillGqaF32Enabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv);
    // Step 3.2 — opt-in GQA-head-packed multi-warp FP16 tensor-core FA-2.
    // Preferred over the q-tiled Step-3 kernel when eligible: it shares the
    // K/V tile across the GQA group (nQPerKv warps) instead of re-reading it
    // per query head. Needs the GQA shape and, on first use, a successful
    // dynamic-smem opt-in (dominant term oRun = nQPerKv*16*headDim*4).
    bool useFp16GqaTc =
        (kvDtype == runtime::KvDtype::FP16) &&
        _prefillFp16GqaTcEnabled &&
        (nQPerKv > 1) &&
        (nQPerKv <= kFlashPrefillGqaMaxQPerKv) &&
        (headDim <= kFlashTcMaxHeadDim) &&
        (headDim % 16 == 0);
    if (useFp16GqaTc) {
        // Compute the exact dynamic-smem footprint (must match the kernel's
        // carveSmem() region layout: each region padded up to 128 bytes).
        constexpr std::size_t BQ = 16, BK = 16;
        constexpr std::size_t kHalf = 2;   // sizeof(fp16); __half not in host TU
        auto a128 = [](std::size_t n) { return (n + 127u) & ~std::size_t(127u); };
        const std::size_t nW = static_cast<std::size_t>(nQPerKv);
        const std::size_t hd = static_cast<std::size_t>(headDim);
        std::size_t bytes = 0;
        bytes += a128(nW * BQ * hd * sizeof(float));    // oRun
        bytes += a128(BK * hd * kHalf);                 // kvS
        bytes += a128(nW * BQ * BK * kHalf);            // qStg
        bytes += a128(nW * BQ * BK * sizeof(float));    // sS
        bytes += a128(nW * BQ * BK * kHalf);            // pS
        bytes += a128(nW * BQ * BK * sizeof(float));    // oT
        bytes += a128(nW * BQ * sizeof(float));         // mSh
        bytes += a128(nW * BQ * sizeof(float));         // lSh
        bytes += a128(nW * BQ * sizeof(float));         // aSh
        if (!_prefillFp16GqaTcSmemAttempted) {
            _prefillFp16GqaTcSmemAttempted = true;
            try {
                _pimpl->_attentionPrefillFlashFp16GqaTcKernel
                    .setMaxDynamicSharedBytes(bytes);
                _prefillFp16GqaTcSmemBytes = bytes;
                _prefillFp16GqaTcSmemOk    = true;
                MM_LOG_INFO("cudagpuops",
                            "FP16 GQA-TC prefill: dynamic smem opt-in ok "
                            "({} bytes, nQPerKv={}, headDim={})",
                            bytes, nQPerKv, headDim);
            } catch (const core::cuda::CudaDriverError& err) {
                _prefillFp16GqaTcSmemOk = false;
                MM_LOG_WARN("cudagpuops",
                            "FP16 GQA-TC prefill: dynamic smem opt-in for {} "
                            "bytes rejected ({}); falling back to the scalar "
                            "fp16 kernel", bytes, err.what());
            }
        }
        // The opt-in is a one-shot per (headDim,nQPerKv). If a later dispatch
        // needs more bytes than the cached opt-in, disable for safety.
        useFp16GqaTc = _prefillFp16GqaTcSmemOk &&
                       (bytes <= _prefillFp16GqaTcSmemBytes);
    }
    // Step 3 — opt-in FP16 tensor-core FA-2 kernel (q-tiled, headDim-bounded).
    const bool useFp16Tc =
        !useFp16GqaTc &&
        (kvDtype == runtime::KvDtype::FP16) &&
        _prefillFp16TcEnabled &&
        (headDim <= kFlashTcMaxHeadDim) &&
        (headDim % 16 == 0);

    core::cuda::CudaKernel* kernelPtr = &_pimpl->_attentionPrefillFlashKernel;
    if (kvDtype == runtime::KvDtype::FP16) {
        if (useFp16GqaTc) {
            kernelPtr = &_pimpl->_attentionPrefillFlashFp16GqaTcKernel;
        } else if (useFp16Tc) {
            kernelPtr = &_pimpl->_attentionPrefillFlashFp16TcKernel;
        } else {
            kernelPtr = &_pimpl->_attentionPrefillFlashFp16Kernel;
        }
    } else if (kvDtype == runtime::KvDtype::Q8_0) {
        if (useQ8Gqa) {
            kernelPtr = (_prefillFlashKTileQ8 == 64)
                ? &_pimpl->_attentionPrefillFlashQ8GqaKtile64Kernel
                : &_pimpl->_attentionPrefillFlashQ8GqaKernel;
        } else {
            kernelPtr = &_pimpl->_attentionPrefillFlashQ8Kernel;
        }
    } else if (useF32Mwtc) {
        kernelPtr = &_pimpl->_attentionPrefillFlashF32GqaMwtcKernel;
    } else if (useF32Tc) {
        kernelPtr = &_pimpl->_attentionPrefillFlashF32GqaTcKernel;
    } else if (useF32Gqa) {
        kernelPtr = &_pimpl->_attentionPrefillFlashF32GqaKernel;
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
    // GQA-packed kernels (Q8_0 / F32 / F32-TC / F32-MWTC / FP16-GQA-TC): one
    // WG per (kv-head, query-position or q-tile).
    const std::uint32_t dim0 =
        (useQ8Gqa || useF32Gqa || useF32Tc || useF32Mwtc || useFp16GqaTc)
        ? static_cast<std::uint32_t>(nKvHeads)
        : static_cast<std::uint32_t>(nHeads);
    // Warp-collective tensor-core kernels use full-warp blocks. Step-3 q-tiled
    // is a single warp (32); the Step-3.2 GQA kernel runs one warp per query
    // head (32*nQPerKv); the F32-MWTC kernel runs HPB warps per head-half
    // (32*min(HPB,nQPerKv)). Scalar kernels use the 16-lane half-wave.
    const std::size_t mwtcWarps =
        (nQPerKv < kMwtcHpb) ? nQPerKv : kMwtcHpb;
    const std::uint32_t blockX =
        useF32Mwtc   ? static_cast<std::uint32_t>(32 * mwtcWarps)
      : useFp16GqaTc ? static_cast<std::uint32_t>(32 * nQPerKv)
      : (useF32Tc || useFp16Tc) ? 32u
      : static_cast<std::uint32_t>(kAttentionLocalSize);
    // The q-tiling kernels (Step 3 / 3.2 / F32-MWTC) take BQ=16 query positions
    // per CTA; every other kernel takes one WG per query position (dim1 = T_q).
    constexpr std::uint32_t kFp16TcBq = 16;
    const std::uint32_t dim1 = (useFp16Tc || useFp16GqaTc || useF32Mwtc)
        ? static_cast<std::uint32_t>((T_q + kFp16TcBq - 1) / kFp16TcBq)
        : static_cast<std::uint32_t>(T_q);
    // F32-MWTC splits the GQA group into ceil(nQPerKv/HPB) head-halves (grid.z).
    const std::uint32_t dim2 = useF32Mwtc
        ? static_cast<std::uint32_t>((nQPerKv + kMwtcHpb - 1) / kMwtcHpb)
        : 1u;
    const std::size_t smemBytes =
        useFp16GqaTc ? _prefillFp16GqaTcSmemBytes
      : useF32Mwtc  ? _prefillF32MwtcSmemBytes
      : std::size_t{0};
    kernel.launch(_ctx.stream(),
                  dim0,
                  dim1,
                  dim2,
                  blockX, 1, 1,
                  smemBytes);
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

void GpuOps::pagedAttentionDecodeV1Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs, std::size_t numHeads,
        std::size_t numKvHeads, std::size_t headSize, std::size_t blockSize,
        std::size_t maxNumBlocksPerSeq, float scale, float softcap) {
    if (numSeqs == 0 || numHeads == 0 || headSize == 0) {
        return;
    }
    // Baseline paged decode attention (fp32). Grid (numHeads, numSeqs); one
    // workgroup owns one (head, sequence). Dynamic SMEM holds the query row,
    // the per-dim accumulator and the reduction scratch:
    // (2*headSize + PAGED_ATTN_V1_LOCAL) floats. kLocal MUST match the
    // kernel's __launch_bounds__ (PagedAttentionV1::kBlockThreads).
    constexpr std::uint32_t kLocal = 128;   // == PAGED_ATTN_V1_LOCAL
    auto& kern = _pimpl->_pagedAttentionV1Kernel;
    kern.setPtr  (0, out);
    kern.setPtr  (1, query);
    kern.setPtr  (2, keyCache);
    kern.setPtr  (3, valueCache);
    kern.setPtr  (4, blockTables);
    kern.setPtr  (5, seqLens);
    kern.setValue(6,  toInt32(numSeqs,            "pagedV1 numSeqs"));
    kern.setValue(7,  toInt32(numHeads,           "pagedV1 numHeads"));
    kern.setValue(8,  toInt32(numKvHeads,         "pagedV1 numKvHeads"));
    kern.setValue(9,  toInt32(headSize,           "pagedV1 headSize"));
    kern.setValue(10, toInt32(blockSize,          "pagedV1 blockSize"));
    kern.setValue(11, toInt32(maxNumBlocksPerSeq, "pagedV1 maxBlocks"));
    kern.setValue(12, scale);
    kern.setValue(13, softcap);
    kern.setValue(14, static_cast<std::int32_t>(0));   // PAGED_ATTN_KV_DTYPE_FP32
    const std::size_t smemBytes = (2 * headSize + kLocal) * sizeof(float);
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(numHeads),
                static_cast<std::uint32_t>(numSeqs),
                1,
                kLocal, 1, 1,
                smemBytes);
}

void GpuOps::pagedAttentionDecodeV2Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs, std::size_t numHeads,
        std::size_t numKvHeads, std::size_t headSize, std::size_t blockSize,
        std::size_t maxNumBlocksPerSeq, std::size_t maxSeqLen, float scale,
        float softcap, runtime::KvDtype kvDtype) {
    if (numSeqs == 0 || numHeads == 0 || headSize == 0) {
        return;
    }
    // keyCache/valueCache are raw pool base addresses; when kvDtype==FP16 they
    // point at __half elements and the fp16 kernel variant reinterprets them.
    const bool fp16 = (kvDtype == runtime::KvDtype::FP16);
    // Split-K paged decode: partition the KV into kPartitionSize chunks so many
    // workgroups cover one (head, seq) in parallel (FlashDecoding / vLLM v2).
    // Pass 1 emits per-partition (acc, m, l); pass 2 merges via online-softmax.
    constexpr std::int32_t  kPartitionSize = 512;  // == PAGED_ATTN_V2_PARTITION_SIZE
    constexpr std::uint32_t kLocal         = 128;  // == PAGED_ATTN_V2_LOCAL
    const std::size_t maxNumPartitions =
        (maxSeqLen + kPartitionSize - 1) / static_cast<std::size_t>(kPartitionSize);
    // The split-K kernels are fp32, no-softcap (16-arg CudaKernel cap). Route
    // short/unsplittable contexts and any soft-capped call to the single-pass
    // V1 which handles both.
    // FP16 KV always takes the V2 path: V1 is F32-only, and all fp16 callers
    // (qwen35moe full-attn) run with softcap==0, so a single-partition V2 is
    // both correct and the only FP16-capable route. F32 keeps the V1 shortcut.
    if (!fp16 && (maxNumPartitions <= 1 || softcap > 0.0f)) {
        pagedAttentionDecodeV1Async(out, query, keyCache, valueCache,
                                    blockTables, seqLens, numSeqs, numHeads,
                                    numKvHeads, headSize, blockSize,
                                    maxNumBlocksPerSeq, scale, softcap);
        return;
    }

    // Grow the per-partition workspace on demand (RAII buffers in Impl).
    const std::size_t slots = numSeqs * numHeads * maxNumPartitions;
    if (slots > _pimpl->_pagedV2SlotCap
            || headSize > _pimpl->_pagedV2HeadDimCap) {
        _pimpl->_pagedV2TmpOut    = allocate(slots * headSize * sizeof(float));
        _pimpl->_pagedV2ExpSums   = allocate(slots * sizeof(float));
        _pimpl->_pagedV2MaxLogits = allocate(slots * sizeof(float));
        _pimpl->_pagedV2SlotCap    = slots;
        _pimpl->_pagedV2HeadDimCap = headSize;
    }
    float* tmpOut  = _pimpl->_pagedV2TmpOut.as<float>();
    float* expSums = _pimpl->_pagedV2ExpSums.as<float>();
    float* maxLog  = _pimpl->_pagedV2MaxLogits.as<float>();

    // --- Pass 1: per-partition partial attention -------------------------
    {
        auto& k = fp16 ? _pimpl->_pagedAttentionV2Fp16Kernel
                       : _pimpl->_pagedAttentionV2Kernel;
        k.setPtr  (0, tmpOut);
        k.setPtr  (1, expSums);
        k.setPtr  (2, maxLog);
        k.setPtr  (3, query);
        k.setPtr  (4, keyCache);
        k.setPtr  (5, valueCache);
        k.setPtr  (6, blockTables);
        k.setPtr  (7, seqLens);
        k.setValue(8,  toInt32(numSeqs,            "pagedV2 numSeqs"));
        k.setValue(9,  toInt32(numHeads,           "pagedV2 numHeads"));
        k.setValue(10, toInt32(numKvHeads,         "pagedV2 numKvHeads"));
        k.setValue(11, toInt32(headSize,           "pagedV2 headSize"));
        k.setValue(12, toInt32(blockSize,          "pagedV2 blockSize"));
        k.setValue(13, toInt32(maxNumBlocksPerSeq, "pagedV2 maxBlocks"));
        k.setValue(14, toInt32(maxNumPartitions,   "pagedV2 maxParts"));
        k.setValue(15, scale);   // partition_size / softcap / dtype are compile-time
        // smem = [nWarps*headSize (acc) | nWarps (m) | nWarps (l)].
        const std::size_t nWarps = kLocal / 32;
        const std::size_t smemBytes =
            (nWarps * headSize + 2 * nWarps) * sizeof(float);
        k.launch(_ctx.stream(),
                 static_cast<std::uint32_t>(numHeads),
                 static_cast<std::uint32_t>(numSeqs),
                 static_cast<std::uint32_t>(maxNumPartitions),
                 kLocal, 1, 1,
                 smemBytes);
    }
    // --- Pass 2: online-softmax reduce across partitions -----------------
    {
        auto& k = _pimpl->_pagedAttentionV2ReduceKernel;
        k.setPtr  (0, out);
        k.setPtr  (1, expSums);
        k.setPtr  (2, maxLog);
        k.setPtr  (3, tmpOut);
        k.setPtr  (4, seqLens);
        k.setValue(5, toInt32(numSeqs,          "pagedV2r numSeqs"));
        k.setValue(6, toInt32(numHeads,         "pagedV2r numHeads"));
        k.setValue(7, toInt32(headSize,         "pagedV2r headSize"));
        k.setValue(8, toInt32(maxNumPartitions, "pagedV2r maxParts"));
        const std::size_t smemBytes = kLocal * sizeof(float);
        k.launch(_ctx.stream(),
                 static_cast<std::uint32_t>(numHeads),
                 static_cast<std::uint32_t>(numSeqs),
                 1,
                 kLocal, 1, 1,
                 smemBytes);
    }
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
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 8.30.11.4 — the 3 paged (block-table) attention ops, carved out of the GpuOps
// god-class into their own IPagedAttentionOps implementation. Bodies are the
// former GpuOps::pagedAttention*Async methods verbatim (only GpuOps:: ->
// CudaPagedAttentionOps:: and _pimpl->_paged* -> _paged* member access changed).
// allocate() mirrors GpuOps::allocate; the module loader + toInt32 are copied
// from GpuOps.cpp (anon-namespace; small, self-contained). The cuDNN-SDPA
// query/fallback pair stays on GpuOps (consulted before this interface).

#include "compute/cuda/CudaPagedAttentionOps.hpp"

#include "compute/ComputeBuffer.hpp"
#include "compute/detail/LaunchGuards.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaMemoryAllocator.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/log/Log.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mimirmind::compute::cuda {

namespace {

constexpr const char* kDefaultPtxDir = "/usr/local/share/mimirmind/ptx";

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
        "cuda::PagedAttentionOps: cannot find " + filename +
        " — set MIMIRMIND_HSACO_DIR or install to " + kDefaultPtxDir);
}

core::cuda::CudaModule loadCudaModule(core::cuda::CudaContext& ctx,
                                      std::string_view       name) {
    const auto path = resolveHsacoPath(name);
    MM_LOG_INFO("cuda::PagedAttentionOps", "loading module '{}' from {}",
                std::string{name}, path.string());
    return core::cuda::CudaModule::fromFile(ctx, path.string());
}

std::int32_t toInt32(std::size_t v, const char* tag) {
    return detail::toInt32(v, tag, "cuda::PagedAttentionOps");
}

} // namespace

CudaPagedAttentionOps::CudaPagedAttentionOps(core::cuda::CudaComputeContext& ctx)
    : _ctx{ctx},
      _pagedAttentionV1Module{loadCudaModule(ctx.cudaContext(), "attention_paged_v1")},
      _pagedAttentionV1Kernel{
          _pagedAttentionV1Module.getFunction("paged_attention_v1")},
      _pagedAttentionPrefillCausalModule{
          loadCudaModule(ctx.cudaContext(), "attention_paged_prefill_causal")},
      _pagedAttentionPrefillCausalKernel{
          _pagedAttentionPrefillCausalModule.getFunction(
              "paged_attention_prefill_causal")},
      _pagedAttentionPrefillCausalFp16Kernel{
          _pagedAttentionPrefillCausalModule.getFunction(
              "paged_attention_prefill_causal_fp16")},
      _pagedAttentionPrefillCausalFp8Kernel{
          _pagedAttentionPrefillCausalModule.getFunction(
              "paged_attention_prefill_causal_fp8")},
      _pagedAttentionV2Module{loadCudaModule(ctx.cudaContext(), "attention_paged_v2")},
      _pagedAttentionV2Kernel{
          _pagedAttentionV2Module.getFunction("paged_attention_v2")},
      _pagedAttentionV2Fp16Kernel{
          _pagedAttentionV2Module.getFunction("paged_attention_v2_fp16")},
      _pagedAttentionV2Fp8Kernel{
          _pagedAttentionV2Module.getFunction("paged_attention_v2_fp8")},
      _pagedAttentionV2ReduceKernel{
          _pagedAttentionV2Module.getFunction("paged_attention_v2_reduce")} {}

// Mirror of GpuOps::allocate: unified GB10 needs Managed (host-reachable) so a
// captureless allocator deleter frees Device+Managed alike; discrete stays
// Device. Only the V2 per-partition workspace uses it here.
compute::ComputeBuffer CudaPagedAttentionOps::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    auto& alloc = _ctx.allocator();
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

void CudaPagedAttentionOps::pagedAttentionDecodeV1Async(
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
    auto& kern = _pagedAttentionV1Kernel;
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

void CudaPagedAttentionOps::pagedAttentionPrefillCausalAsync(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqT, const std::int32_t* queryOff,
        const std::int32_t* startPos, std::size_t numSeqs, std::size_t numHeads,
        std::size_t numKvHeads, std::size_t headSize, std::size_t blockSize,
        std::size_t maxNumBlocksPerSeq, std::size_t maxT, float scale,
        float softcap, runtime::KvDtype kvDtype) {
    if (numSeqs == 0 || numHeads == 0 || headSize == 0 || maxT == 0) {
        return;
    }
    // 5.21-II paged causal prefill attention. grid (numHeads, numSeqs, maxT);
    // block (pq >= seqT[seq]) early-out. Same smem + streaming-softmax as V1, so
    // pq's output == a V1 decode with seq_len = startPos[seq]+pq+1.
    // 5.16: pick the kernel variant by pool dtype so the mixed-step ragged
    // prefill read reinterprets the pool bytes correctly (fp16/fp8 pools would
    // otherwise be read as F32 and corrupt prefill attention).
    constexpr std::uint32_t kLocal = 128;   // == PAGED_ATTN_PREFILL_LOCAL
    auto& kern = (kvDtype == runtime::KvDtype::FP8_E4M3)
                     ? _pagedAttentionPrefillCausalFp8Kernel
                 : (kvDtype == runtime::KvDtype::FP16)
                     ? _pagedAttentionPrefillCausalFp16Kernel
                     : _pagedAttentionPrefillCausalKernel;
    kern.setPtr  (0, out);
    kern.setPtr  (1, query);
    kern.setPtr  (2, keyCache);
    kern.setPtr  (3, valueCache);
    kern.setPtr  (4, blockTables);
    kern.setPtr  (5, seqT);
    kern.setPtr  (6, queryOff);
    kern.setPtr  (7, startPos);
    kern.setValue(8,  toInt32(numSeqs,            "prefC numSeqs"));
    kern.setValue(9,  toInt32(numHeads,           "prefC numHeads"));
    kern.setValue(10, toInt32(numKvHeads,         "prefC numKvHeads"));
    kern.setValue(11, toInt32(headSize,           "prefC headSize"));
    kern.setValue(12, toInt32(blockSize,          "prefC blockSize"));
    kern.setValue(13, toInt32(maxNumBlocksPerSeq, "prefC maxBlocks"));
    kern.setValue(14, scale);
    kern.setValue(15, softcap);
    const std::size_t smemBytes = (2 * headSize + kLocal) * sizeof(float);
    kern.launch(_ctx.stream(),
                static_cast<std::uint32_t>(numHeads),
                static_cast<std::uint32_t>(numSeqs),
                static_cast<std::uint32_t>(maxT),
                kLocal, 1, 1,
                smemBytes);
}

void CudaPagedAttentionOps::pagedAttentionDecodeV2Async(
        float* out, const float* query, const float* keyCache,
        const float* valueCache, const std::int32_t* blockTables,
        const std::int32_t* seqLens, std::size_t numSeqs, std::size_t numHeads,
        std::size_t numKvHeads, std::size_t headSize, std::size_t blockSize,
        std::size_t maxNumBlocksPerSeq, std::size_t maxSeqLen, float scale,
        float softcap, runtime::KvDtype kvDtype) {
    if (numSeqs == 0 || numHeads == 0 || headSize == 0) {
        return;
    }
    // keyCache/valueCache are raw pool base addresses; when kvDtype is FP16/FP8
    // they point at __half / __nv_fp8_e4m3 elements and the matching kernel
    // variant reinterprets them (5.14 I1 / 5.16).
    const bool fp16 = (kvDtype == runtime::KvDtype::FP16);
    const bool fp8  = (kvDtype == runtime::KvDtype::FP8_E4M3);
    const bool nonF32 = fp16 || fp8;
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
    // FP16/FP8 KV always take the V2 path: V1 is F32-only, and all non-F32
    // callers (qwen35moe full-attn) run with softcap==0, so a single-partition
    // V2 is both correct and the only non-F32-capable route. F32 keeps the V1
    // shortcut.
    if (!nonF32 && (maxNumPartitions <= 1 || softcap > 0.0f)) {
        pagedAttentionDecodeV1Async(out, query, keyCache, valueCache,
                                    blockTables, seqLens, numSeqs, numHeads,
                                    numKvHeads, headSize, blockSize,
                                    maxNumBlocksPerSeq, scale, softcap);
        return;
    }

    // Grow the per-partition workspace on demand (RAII buffers in Impl).
    const std::size_t slots = numSeqs * numHeads * maxNumPartitions;
    if (slots > _pagedV2SlotCap
            || headSize > _pagedV2HeadDimCap) {
        _pagedV2TmpOut    = allocate(slots * headSize * sizeof(float));
        _pagedV2ExpSums   = allocate(slots * sizeof(float));
        _pagedV2MaxLogits = allocate(slots * sizeof(float));
        _pagedV2SlotCap    = slots;
        _pagedV2HeadDimCap = headSize;
    }
    float* tmpOut  = _pagedV2TmpOut.as<float>();
    float* expSums = _pagedV2ExpSums.as<float>();
    float* maxLog  = _pagedV2MaxLogits.as<float>();

    // --- Pass 1: per-partition partial attention -------------------------
    {
        auto& k = fp8  ? _pagedAttentionV2Fp8Kernel
                : fp16 ? _pagedAttentionV2Fp16Kernel
                       : _pagedAttentionV2Kernel;
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
        auto& k = _pagedAttentionV2ReduceKernel;
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

} // namespace mimirmind::compute::cuda

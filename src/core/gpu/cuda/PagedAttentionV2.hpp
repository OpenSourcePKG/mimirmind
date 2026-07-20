// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"
#include "core/gpu/cuda/PagedAttentionV1.hpp"    // PagedKvDtype enum

#include <cstddef>
#include <cstdint>
#include <string>

namespace mimirmind::core::cuda {

/**
 * C++ launch orchestrator for the `paged_attention_v2` +
 * `paged_attention_v2_reduce` kernel pair (M-Cuda.Batch Sub-Step B3).
 *
 * ---- When to prefer over V1 -----------------------------------------
 *
 * V1 processes each (head, sequence) with a single workgroup that
 * streams the full KV in one pass. V2 splits the KV into
 * `kPartitionSize`-token chunks so multiple workgroups can work on the
 * same (head, sequence) in parallel; a second reduce kernel merges the
 * per-partition partials via online-softmax rescale.
 *
 * Rule of thumb the Phase C scheduler should apply:
 *   seq_len <= kPartitionSize  → PagedAttentionV1  (no reduce overhead)
 *   seq_len >  kPartitionSize  → PagedAttentionV2  (partition-parallel)
 *
 * ---- IMPORTANT: skeleton status ------------------------------------
 *
 * Both kernel bodies are **not implemented yet** — they `__trap()` on
 * the device. This wrapper exists so downstream code has a real API
 * to bind against; runtime invocation will hard-fault the CUDA context
 * until the FA2-Ampere+PTX bodies ship (post-Blackwell + FlashInfer-
 * vault-read, same gating as V1).
 *
 * Callers must guard with `config.serving.paged_attention_enabled` in
 * scheduler code so the trap can never reach Prod paths.
 *
 * ---- Workspace requirements ----------------------------------------
 *
 * V2 needs per-partition workspace on the device, sized by
 * `maxNumPartitions = ceil(max_seq_len / kPartitionSize)`. The Phase
 * C scheduler owns these buffers (one set per model-instance, sized
 * for the max concurrent batch):
 *   tmp_out    fp32 [max_batch, num_heads, maxNumPartitions, head_size]
 *   exp_sums   fp32 [max_batch, num_heads, maxNumPartitions]
 *   max_logits fp32 [max_batch, num_heads, maxNumPartitions]
 *
 * `launch(...)` takes these as raw device pointers — allocation and
 * lifetime are the scheduler's responsibility, not this class's.
 *
 * ---- Threading -----------------------------------------------------
 *
 * NOT thread-safe (inherits CudaKernel arg-buffer semantics from BOTH
 * held kernel handles). The scheduler serialises `launch()` on its
 * single-thread event loop.
 */
class PagedAttentionV2 {
public:
    static constexpr const char* kPartialKernelSymbol = "paged_attention_v2";
    static constexpr const char* kReduceKernelSymbol  = "paged_attention_v2_reduce";

    /**
     * `PAGED_ATTN_V2_LOCAL` from the kernel header. Used as block-x
     * dim for the partial kernel launch. MUST stay in sync with the
     * `__launch_bounds__` in the `.cu` file.
     */
    static constexpr std::uint32_t kPartialBlockThreads = 128;

    /// `PAGED_ATTN_V2_REDUCE_LOCAL` — block-x dim for the reduce kernel.
    static constexpr std::uint32_t kReduceBlockThreads = 128;

    /**
     * `PAGED_ATTN_V2_PARTITION_SIZE` — the K-chunk granularity across
     * which V2 splits KV traversal. Must be a multiple of the paged-
     * block size (16, our Allocator default) so partitions align with
     * paged blocks (32 blocks per partition at block-size 16).
     * Value committed here for Bragi-v1; retunable during body
     * implementation once real sm_120 numbers exist.
     */
    static constexpr std::int32_t kPartitionSize = 512;

    /**
     * Resolve both kernel symbols from `module`. Throws
     * `CudaDriverError` if either is missing (probably means the
     * loaded .ptx was built without B3). The module must outlive
     * this object — no ownership transfer.
     */
    explicit PagedAttentionV2(CudaModule& module);

    ~PagedAttentionV2() = default;

    PagedAttentionV2(const PagedAttentionV2&)            = delete;
    PagedAttentionV2& operator=(const PagedAttentionV2&) = delete;
    PagedAttentionV2(PagedAttentionV2&&) noexcept        = default;
    PagedAttentionV2& operator=(PagedAttentionV2&&) noexcept = default;

    /**
     * Dispatch both kernels back-to-back on the same stream. The
     * reduce kernel is stream-ordered after the partial kernel — no
     * explicit host sync needed. `maxNumPartitions` MUST equal
     * `ceil(max_seq_len / kPartitionSize)` and MUST match the
     * allocation of tmp_out / exp_sums / max_logits.
     *
     * `partialSharedMemBytes` and `reduceSharedMemBytes` are dynamic
     * shared memory requests for the future bodies — placeholders (0
     * default); Phase-C wiring will provide the empirically-tuned
     * amounts once the bodies exist.
     *
     * WARNING: bodies currently `__trap()`. Do not call from any Prod
     * path until FA2-Ampere bodies land.
     */
    void launch(CudaStream&        stream,
                // Output + workspace (scheduler-owned)
                const void*        out,
                const void*        tmpOut,
                const void*        expSums,
                const void*        maxLogits,
                // Inputs (shared with V1 args)
                const void*        query,
                const void*        keyCache,
                const void*        valueCache,
                const void*        blockTables,
                const void*        seqLens,
                std::int32_t       numSeqs,
                std::int32_t       numHeads,
                std::int32_t       numKvHeads,
                std::int32_t       headSize,
                std::int32_t       blockSize,
                std::int32_t       maxNumBlocksPerSeq,
                std::int32_t       maxNumPartitions,
                float              scale,
                float              softcap,
                PagedKvDtype       kvDtype,
                std::size_t        partialSharedMemBytes = 0,
                std::size_t        reduceSharedMemBytes  = 0);

    [[nodiscard]] const std::string& partialKernelName() const noexcept {
        return _partial.name();
    }
    [[nodiscard]] const std::string& reduceKernelName() const noexcept {
        return _reduce.name();
    }

private:
    CudaKernel _partial;
    CudaKernel _reduce;
};

} // namespace mimirmind::core::cuda
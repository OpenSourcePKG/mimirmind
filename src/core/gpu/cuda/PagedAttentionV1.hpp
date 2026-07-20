// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"
#include "core/gpu/cuda/CudaStream.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mimirmind::core::cuda {

/**
 * KV-cache element format understood by the `paged_attention_v1` kernel.
 * Values MUST stay in sync with the `PAGED_ATTN_KV_DTYPE_*` macros in
 * `kernels_cuda/attention_paged_v1.cu` — the kernel switches on this
 * int argument, so 0/1/2 is part of the ABI.
 */
enum class PagedKvDtype : int {
    Fp32 = 0,
    Fp16 = 1,
    Q8_0 = 2,
};

/**
 * C++ launch wrapper for `paged_attention_v1` (M-Cuda.Batch Sub-Step
 * B2). Resolves the kernel symbol from a loaded `CudaModule` once,
 * then reuses the resulting `CudaKernel` handle for every dispatch.
 *
 * ---- IMPORTANT: skeleton status ----------------------------------
 *
 * The **kernel body is not implemented yet.** Invocation traps on the
 * device. This wrapper exists so downstream code (M-Cuda.Batch Phase C
 * scheduler + Phase D InferenceEngine wiring) can already bind to a
 * real API — link-time and compile-time surface is production-final,
 * runtime is not.
 *
 * Callers must assume `launch(...)` will hard-fault the CUDA context
 * on any input until the FA2-Ampere body ships. Use guard flags in
 * scheduler code:
 *   if (config.serving.paged_attention_enabled) { ... }
 * so the trap can never be reached from Prod paths until it's real.
 *
 * ---- Layout conventions (mirror kernel header) -------------------
 *
 * Every device pointer passed to `launch()` must satisfy:
 *   out          [num_seqs, num_heads, head_size]   fp32, output
 *   query        [num_seqs, num_heads, head_size]   fp32, single query
 *                                                    token per sequence
 *                                                    (decode pass)
 *   key_cache    paged pool, layout depends on kvDtype (see .cu
 *                header — subject to sm_120 empirical tuning)
 *   value_cache  paged pool, same convention as key_cache
 *   block_tables [num_seqs, max_num_blocks_per_seq]  int32,
 *                -1 sentinel for unused slots. Emit from
 *                PagedKvSequence::blockTable() via a per-launch
 *                staging copy.
 *   seq_lens     [num_seqs]                         int32, decoded
 *                length per sequence (drives causal-mask bounds)
 *
 * ---- Softcap ------------------------------------------------------
 *
 * `softcap = 0.0f` disables the final-logit soft-cap. Positive values
 * apply the Gemma-4 fix from commit `c69f012` on the CUDA side.
 * Non-Gemma models must pass 0.0f.
 *
 * ---- Threading ----------------------------------------------------
 *
 * `PagedAttentionV1` is NOT thread-safe (inherits `CudaKernel`'s
 * shared-arg-buffer non-safety). The M-Cuda.Batch scheduler owns one
 * instance per model and serialises `launch()` on its event loop.
 */
class PagedAttentionV1 {
public:
    /**
     * Symbol name of the kernel entry point. Public constant so tests
     * and diagnostic tooling can reference it without hard-coding a
     * string literal.
     */
    static constexpr const char* kKernelSymbol = "paged_attention_v1";

    /**
     * `PAGED_ATTN_V1_LOCAL` from the kernel header. `launch()` uses
     * this as the block-x dimension — the value MUST stay in sync
     * with the `__launch_bounds__` in the `.cu` file.
     */
    static constexpr std::uint32_t kBlockThreads = 128;

    /**
     * Resolve the kernel symbol from `module`. Throws
     * `CudaDriverError` if the symbol is missing (probably means the
     * loaded .ptx was built without B2). The passed module must
     * outlive this object — no ownership transfer.
     */
    explicit PagedAttentionV1(CudaModule& module);

    ~PagedAttentionV1() = default;

    PagedAttentionV1(const PagedAttentionV1&)            = delete;
    PagedAttentionV1& operator=(const PagedAttentionV1&) = delete;
    PagedAttentionV1(PagedAttentionV1&&) noexcept        = default;
    PagedAttentionV1& operator=(PagedAttentionV1&&) noexcept = default;

    /**
     * Dispatch the kernel. Ints are staged as-is; pointers are staged
     * via `setPtr`. Grid dim is `(num_heads, num_seqs, 1)` — one
     * workgroup per (head, sequence) tile.
     *
     * WARNING: the body currently `__trap()`s. Do not call from any
     * Prod path until the FA2-Ampere body lands.
     *
     * `sharedMemBytes` reserves dynamic-SMEM for the future body — the
     * value is a placeholder (0 default); Phase-C wiring will provide
     * the empirically-tuned amount once the body exists.
     */
    void launch(CudaStream&        stream,
                const void*        out,
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
                float              scale,
                float              softcap,
                PagedKvDtype       kvDtype,
                std::size_t        sharedMemBytes = 0);

    [[nodiscard]] const std::string& kernelName() const noexcept {
        return _kernel.name();
    }

private:
    CudaKernel _kernel;
};

} // namespace mimirmind::core::cuda
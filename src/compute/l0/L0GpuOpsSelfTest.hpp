// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::l0 {
class UsmAllocator;
}

namespace mimirmind::compute::l0 {

class GpuOps;

/**
 * Load-time GPU kernel self-test (SPV-regression guard). Extracted from the
 * former 270-line GpuOps::selfTest (roadmap 8.30.11.3). Runs three parity
 * checks against CPU references and throws std::runtime_error on the first
 * mismatch so a driver / ocloc miscompilation is caught before the first
 * block runs, with a targeted message instead of a mysterious later
 * divergence:
 *
 *   1. x_quant_i8       — per-row symmetric int8 quantisation (feeds DP4A).
 *   2. qkv_split        — full (hasV) and alt-attention (qk-only) paths.
 *   3. attention_prefill_flash — single-WG streaming FlashAttention prefill
 *                          vs compute::multiHeadAttention (skipped when the
 *                          rollback flag disables the flash-prefill kernel).
 *
 * Uses only GpuOps' public kernel + flush + capability API — no friend access
 * (keeps the 8.30.7 encapsulation direction). GpuOps::selfTest owns the
 * `_selfTestStatus` update + summary log around a `run()` call.
 */
class L0GpuOpsSelfTest {
public:
    explicit L0GpuOpsSelfTest(GpuOps& ops) noexcept : _ops{ops} {}

    /// Run all enabled stages. Throws on the first parity mismatch; returns
    /// normally when every stage passed.
    void run(core::l0::UsmAllocator& allocator);

private:
    void runQuantI8(core::l0::UsmAllocator& allocator);
    void runQkvSplit(core::l0::UsmAllocator& allocator);
    void runAttentionPrefillFlash(core::l0::UsmAllocator& allocator);

    GpuOps& _ops;
};

} // namespace mimirmind::compute::l0

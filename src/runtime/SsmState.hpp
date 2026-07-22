// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cstddef>

namespace mimirmind::compute { class ComputeOps; }

namespace mimirmind::runtime {

/**
 * Per-sequence recurrent state for hybrid linear-attention models
 * (Qwen3-Next / qwen35moe GatedDeltaNet). Holds, for every layer indexed
 * by blockIdx, the delta-net [S,S] recurrent state per v-head and the
 * rolling causal-conv tail. Full-attention layers keep an unused slot so
 * the blockIdx indexing stays trivial.
 *
 * Lifecycle mirrors KvCache, deliberately: one instance per sequence,
 * owned by InferenceEngine next to the KvCache. The size is
 * context-length-independent (state is a fixed [H_v,S,S] per layer, not
 * a per-token buffer), so it is allocated once and — unlike the transient
 * BlockBuffers scratch, which reallocates on prompt growth — persists
 * across forward calls AND across BlockBuffers reallocations. That is the
 * whole point: the recurrent state used to live inside BlockBuffers and
 * would be silently dropped when the scratch grew mid-conversation.
 *
 * Multi-tenant serving allocates one SsmState per concurrent sequence,
 * exactly like one KvCache per sequence (see PagedKvSequence for the
 * paged variant this will eventually mirror).
 *
 * Zeroing at sequence start is done lazily by the backend when
 * cache.length() == 0 (same signal that resets the KV write cursor);
 * the constructor zeroes once so the very first forward sees a defined
 * state, and reset() is available for an explicit clear.
 */
class SsmState {
public:
    SsmState(compute::ComputeOps& ops,
             std::size_t          blockCount,
             std::size_t          stateElemsPerLayer,
             std::size_t          convStateElemsPerLayer);

    SsmState(const SsmState&)            = delete;
    SsmState& operator=(const SsmState&) = delete;
    SsmState(SsmState&&)                 = delete;
    SsmState& operator=(SsmState&&)      = delete;

    /// Base of the recurrent delta-net state, [blockCount, H_v*S*S].
    /// Index layer L at `statePtr() + L * stateElemsPerLayer()`.
    [[nodiscard]] float* statePtr() noexcept { return _state.as<float>(); }
    /// Base of the rolling causal-conv tail, [blockCount, (d_conv-1)*conv_dim].
    /// Index layer L at `convStatePtr() + L * convStateElemsPerLayer()`.
    [[nodiscard]] float* convStatePtr() noexcept { return _convState.as<float>(); }

    [[nodiscard]] std::size_t stateElemsPerLayer()     const noexcept { return _stateElems; }
    [[nodiscard]] std::size_t convStateElemsPerLayer() const noexcept { return _convStateElems; }
    [[nodiscard]] std::size_t blockCount()             const noexcept { return _blockCount; }

    /// Zero the entire state (queued on the compute stream). Mirrors
    /// KvCache::reset(); used at true sequence start / eviction.
    void reset(compute::ComputeOps& ops);

private:
    std::size_t            _blockCount;
    std::size_t            _stateElems;
    std::size_t            _convStateElems;
    compute::ComputeBuffer _state;      // [blockCount, stateElemsPerLayer]
    compute::ComputeBuffer _convState;  // [blockCount, convStateElemsPerLayer]
};

} // namespace mimirmind::runtime
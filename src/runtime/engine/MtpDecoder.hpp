// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeOps.hpp"   // compute::ComputeBuffer

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
class KvCache;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::engine {

/**
 * Native multi-token-prediction (MTP) greedy decoder, extracted from
 * InferenceEngine. Owns the MTP-specific scratch (a private 1-layer KV cache
 * for the nextn module + per-step buffers + the GatedDeltaNet rollback
 * snapshot); everything else — the trunk forward (forwardVerify /
 * commitVerified), weights, backend, KV/SSM state — is reached through the
 * engine, of which this is a friend collaborator.
 *
 * Constructed lazily by InferenceEngine::generateMtp on first use. CUDA +
 * qwen35moe with a loaded nextn head only (the engine gates on mtpAvailable()).
 */
class MtpDecoder {
public:
    explicit MtpDecoder(InferenceEngine& engine) : _e{engine} {}

    /// See InferenceEngine::generateMtp — draft `mtpDepth`, verify in one trunk
    /// forward, accept the longest greedy prefix. Output is bit-identical to
    /// greedy generate() at temperature 0.
    [[nodiscard]] std::vector<std::int32_t>
    generate(std::span<const std::int32_t> promptIds,
             std::size_t                   maxNew,
             std::size_t                   mtpDepth,
             std::int32_t                  eosId,
             std::size_t*                  draftedOut,
             std::size_t*                  acceptedOut);

private:
    InferenceEngine& _e;

    // MTP-specific scratch (lazily allocated on the first generate()).
    std::unique_ptr<KvCache> _kv;      // 1-layer nextn self-attn cache
    compute::ComputeBuffer   _emb;     // [d_model] embed(prevTok)
    compute::ComputeBuffer   _cat;     // [2*d_model] concat(norms)
    compute::ComputeBuffer   _eh;      // [d_model] eh_proj / block out
    compute::ComputeBuffer   _logits;  // [vocab] draft logits
    compute::ComputeBuffer   _hidden;  // [d_model] stable trunk-hidden copy
    compute::ComputeBuffer   _ssmBak;  // GatedDeltaNet state snapshot (verify rollback)
    compute::ComputeBuffer   _convBak; // rolling conv-tail snapshot
};

} // namespace mimirmind::runtime::engine

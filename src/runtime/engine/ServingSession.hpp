// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"   // InferenceEngine + ServingSlotStep

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mimirmind::runtime::engine {

struct ServingState;   // persistent continuous-batch substrate (defined in .cpp)

/**
 * Batched / serving-class decode substrate, extracted from InferenceEngine.
 * Bundles the one-shot batched-generation harnesses (generateBatch /
 * generateServingParity) and the persistent continuous-batching state
 * (ensureServingState / stepServing) that ContinuousBatcher drives. Owns the
 * persistent ServingState; everything else — backend, weights, ops, matmul —
 * is reached through the engine, of which this is a friend collaborator.
 *
 * Constructed lazily by InferenceEngine on first serving call. CUDA +
 * qwen35moe only (the methods throw otherwise).
 */
class ServingSession {
public:
    explicit ServingSession(InferenceEngine& engine);  // out-of-line (ServingState
    ~ServingSession();                                 // complete only in the .cpp)

    ServingSession(const ServingSession&)            = delete;
    ServingSession& operator=(const ServingSession&) = delete;

    /// See InferenceEngine::generateBatch.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateBatch(const std::vector<std::vector<std::int32_t>>& prompts,
                  std::size_t maxNew, std::int32_t eosId);

    /// See InferenceEngine::generateServingParity.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateServingParity(std::span<const std::int32_t> promptIds,
                          std::size_t nSeq, std::size_t maxNew);

    /// See InferenceEngine::ensureServingState.
    void ensureServingState(std::size_t maxBatch, std::size_t maxContext);

    /// See InferenceEngine::stepServing.
    void stepServing(std::span<const InferenceEngine::ServingSlotStep> steps,
                     std::span<std::int32_t>                           outTokens);

    [[nodiscard]] std::size_t maxBatch() const noexcept;
    [[nodiscard]] std::size_t maxContext() const noexcept;

private:
    InferenceEngine&              _e;
    std::unique_ptr<ServingState> _state;
};

} // namespace mimirmind::runtime::engine

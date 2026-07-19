// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/SpeculativeSampler.hpp"
#include "runtime/spec/Drafter.hpp"

namespace mimirmind::runtime {

class InferenceEngine;

/**
 * `Drafter` backed by a second, smaller `InferenceEngine` — the classic
 * M9.11 naive spec-dec path. Preserves the prefix KV warm-up that
 * `SpeculativeDecoder` used to do inline, so behaviour for existing
 * `drafter="model"` deployments stays byte-identical.
 */
class ModelDrafter final : public Drafter {
public:
    explicit ModelDrafter(InferenceEngine& draft);

    ModelDrafter(const ModelDrafter&)            = delete;
    ModelDrafter& operator=(const ModelDrafter&) = delete;
    ModelDrafter(ModelDrafter&&)                 = delete;
    ModelDrafter& operator=(ModelDrafter&&)      = delete;

    compute::SpeculativeBatch propose(
        std::span<const std::int32_t>  context,
        std::size_t                    N,
        const compute::SamplingParams& sampling) override;

    void warmPrefix(std::span<const std::int32_t> promptPlusFirstToken) override;

    [[nodiscard]] std::string_view kind() const noexcept override { return "model"; }

    /// Exposed so status endpoints can report the backing engine's arch
    /// / block count / d_model without a downcast through the vtable.
    [[nodiscard]] InferenceEngine& engine() noexcept { return _draft; }

private:
    InferenceEngine&            _draft;
    compute::SpeculativeSampler _sampler;
};

} // namespace mimirmind::runtime
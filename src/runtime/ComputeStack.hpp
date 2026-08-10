// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeContext.hpp"

#include <memory>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::core::config {
struct Config;
} // namespace mimirmind::core::config

namespace mimirmind::runtime {

/**
 * A standalone compute stack — context + ops + matmul — for a backend kind,
 * owned as one unit. Mirrors the per-engine stack InferenceEngine builds in
 * its ctor, but usable by non-decoder consumers (the cross-encoder reranker /
 * EncoderRunner) that need their own isolated GPU context + kernels without
 * spinning up a full InferenceEngine.
 *
 * Field order = construction/destruction order: context first, then ops (holds
 * a context ref), then matmul (holds an ops ref).
 */
struct ComputeStack {
    std::unique_ptr<core::backend::ComputeContext> context;
    std::unique_ptr<compute::ComputeOps>           ops;
    std::unique_ptr<compute::ComputeMatmul>        matmul;
};

/// Build a compute stack for `kind`. Throws if the backend isn't compiled in.
[[nodiscard]] ComputeStack
makeComputeStack(const core::config::Config& cfg,
                 core::backend::BackendKind  kind);

} // namespace mimirmind::runtime

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <memory>

namespace mimirmind::runtime {
class InferenceEngine;
class Drafter;
} // namespace mimirmind::runtime

namespace mimirmind::core::config {
struct Config;
} // namespace mimirmind::core::config

namespace mimirmind::cli {

/**
 * Boot-time construction helpers for `mimirmind serve`. Each method builds
 * one cohesive slice of the server's long-lived state and RETURNS the owning
 * objects, so `runServe` keeps them as locals and their RAII lifetime /
 * destruction order stay anchored in the serve frame (the monitors and the
 * drafter must outlive `ApiServer::run`). This keeps `runServe` a readable
 * orchestrator instead of a multi-thousand-line boot body.
 */
class ServerBootstrap {
public:
    /**
     * Optional speculative-decoding drafter, built from `cfg.speculative`
     * against the already-loaded default/target engine.
     *
     * `draftEngine` MUST outlive `drafter` — a `ModelDrafter` holds a
     * reference into `*draftEngine`, so the member order here is load-bearing:
     * on destruction `drafter` (declared last) is torn down first, then
     * `draftEngine`. Both are null when speculative decoding is disabled or a
     * drafter could not be constructed (vocab mismatch / load failure); the
     * method logs the reason and degrades to no speculation.
     */
    struct SpeculativeSetup {
        std::unique_ptr<runtime::InferenceEngine> draftEngine;
        std::unique_ptr<runtime::Drafter>         drafter;
    };

    [[nodiscard]] static SpeculativeSetup buildSpeculative(
        const core::config::Config&    cfg,
        const runtime::InferenceEngine& targetEngine);
};

} // namespace mimirmind::cli

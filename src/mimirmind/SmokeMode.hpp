// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind smoke` mode. Prints the banner, boots one
 * InferenceEngine, and runs the M1-M5 diagnostic suite via
 * `mimirmind::diagnostics::run*`. When `args.modelPath` is empty
 * the model-dependent stages (M3 summary, Q4_K/Q6_K matvec parity,
 * embed+lm_head, chat-template, generate) are skipped and only the
 * device/USM/rmsNorm probes run — that's the intended behaviour when
 * an operator wants a quick "is Level Zero + the allocator alive?"
 * check without a model on hand.
 *
 * Returns 0 on success. Any thrown exception is caught in `main` —
 * this function does not swallow them.
 */
[[nodiscard]] int runSmoke(const CliArgs& args,
                           const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/log/Log.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

/**
 * Fail-fast CUDA-context-poison guard (roadmap 5.21.2 / gemma4-SWA incident
 * 2026-08-26).
 *
 * An irrecoverable CUDA fault — an illegal memory access, an unspecified launch
 * failure, a misaligned address, a launch timeout, or a device-side assert —
 * leaves the WHOLE CUDA context dead. Every subsequent GPU op on that context
 * fails, so a serve that keeps running after such a fault returns corrupt output
 * (or 500s) for EVERY co-resident model — e.g. a long-context gemma4 request
 * poisons the context and qwen then serves garbage until the process finally
 * dies. The only safe recovery is to exit and let the `--restart` supervisor
 * recreate the process on a fresh context.
 *
 * `isCudaContextPoisoned` recognises those fault strings in an exception's
 * what(); `hardExitOnCudaPoison` logs and `std::_Exit(70)`s. Both the batched
 * (ContinuousBatcher) and single-session (generate()) forward-failure catch
 * sites route irrecoverable faults through here.
 */
[[nodiscard]] inline bool isCudaContextPoisoned(std::string_view what) noexcept {
    return what.find("illegal memory access")     != std::string_view::npos ||
           what.find("unspecified launch failure") != std::string_view::npos ||
           what.find("misaligned address")         != std::string_view::npos ||
           what.find("launch timed out")           != std::string_view::npos ||
           what.find("device-side assert")         != std::string_view::npos;
}

[[noreturn]] inline void hardExitOnCudaPoison(std::string_view where,
                                              const std::string& what) {
    MM_LOG_ERROR("poison-guard",
                 "CUDA context POISONED in {} — hard-exiting (_Exit 70) for a "
                 "clean supervisor restart so no co-resident model serves "
                 "corrupt output: {}",
                 where, what);
    std::fflush(nullptr);   // _Exit skips buffer flush; force it here
    std::_Exit(70);         // EX_SOFTWARE; --restart recreates the process
}

/// Convenience: if `what` names an irrecoverable CUDA fault, hard-exit; else
/// return (the caller handles the (recoverable) error normally). Optional env
/// fault-injection (MIMIRMIND_POISON_FAULT_TEST) treats the next call from
/// `where` as poisoned, to prove the guard end-to-end without a real fault.
inline void guardCudaPoison(std::string_view where, const std::string& what) {
    if (const char* ft = std::getenv("MIMIRMIND_POISON_FAULT_TEST");
        ft != nullptr && ft[0] != '\0' && ft[0] != '0') {
        hardExitOnCudaPoison(where,
                             "SIMULATED (MIMIRMIND_POISON_FAULT_TEST): " + what);
    }
    if (isCudaContextPoisoned(what)) {
        hardExitOnCudaPoison(where, what);
    }
}

} // namespace mimirmind::runtime

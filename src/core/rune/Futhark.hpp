// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/rune/RuneApplier.hpp"

#include <cstddef>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::gguf {
class WeightsMap;
struct GgufTensor;
}

namespace mimirmind::core::rune {

/// Device transfer hooks. The engine binds them to
/// ComputeOps::readbackToHost / uploadHostBytes; tests bind plain memcpy.
struct FutharkDeviceIo {
    std::function<void(void* hostDst, const void* deviceSrc, std::size_t bytes)> readback;
    std::function<void(void* deviceDst, const void* hostSrc, std::size_t bytes)> upload;
};

/**
 * Futhark — applies a model's configured Runes to its loaded weights.
 *
 * Runs once at load time, after the weights are on the device and BEFORE any
 * derived copy (fused QKV, requantised variants, arch-backend caches) is built
 * from them, so every consumer sees the patched values.
 *
 * Staging model: each touched tensor is read back into a host buffer once;
 * all runes apply on those buffers in configured order (stacked runes chain
 * through the per-op tensor hashes). Only when EVERY rune applied cleanly are
 * the touched buffers uploaded. Any failure leaves the device weights
 * byte-identical to the base model.
 *
 * Refuses targets that the loader already shadowed with a derived copy the
 * kernels read instead (sibling `<name>.nv` / `<name>.fp8`, FP4-TC side
 * banks) — patching the primary would be a silent no-op.
 */
class Futhark {
public:
    struct Result {
        std::vector<RuneApplyReport> reports;   ///< one per rune, in order
        std::size_t                  tensorsUploaded{0};
        std::size_t                  bytesUploaded{0};
    };

    explicit Futhark(FutharkDeviceIo io) : _io(std::move(io)) {}

    /// Load + apply every rune file in order; all-or-nothing over the list.
    [[nodiscard]] std::expected<Result, std::string>
    applyFiles(std::span<const std::string> runePaths, std::string_view baseFingerprint,
               const gguf::WeightsMap& weights);

    /// Same, for already-parsed runes.
    [[nodiscard]] std::expected<Result, std::string>
    apply(std::span<const RuneFile> runes, std::string_view baseFingerprint,
          const gguf::WeightsMap& weights);

    /// Sibling-name suffixes the loader uses for derived weight variants.
    static constexpr std::string_view kDerivedSuffixes[] = {".nv", ".fp8"};

    /// True when the engine keeps a derived copy of `t` that kernels may read.
    [[nodiscard]] static bool hasDerivedCopies(const gguf::GgufTensor& t,
                                               const gguf::WeightsMap& weights);

private:
    FutharkDeviceIo _io;
};

} // namespace mimirmind::core::rune

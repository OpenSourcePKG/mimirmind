// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/rune/RuneFormat.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::rune {

/**
 * An in-memory Rune: parsed manifest plus owned F32 payload blobs.
 *
 * `parse` validates everything that can be checked without the base model:
 * container well-formedness, manifest schema/version, op-kind spelling,
 * payload existence + dtype (F32 only) + shape plausibility per op. Checks
 * that need the target tensor (existence, dtype, dims, content hash) belong
 * to `RuneApplier`.
 *
 * `serialize` is the inverse and is what Hugin uses to emit Runes; a
 * parse(serialize(x)) round trip is exact.
 *
 * Parse errors throw std::runtime_error prefixed with "rune: " (same
 * convention as the safetensors parser this builds on).
 */
class RuneFile {
public:
    RuneFile() = default;
    RuneFile(RuneManifest manifest, std::map<std::string, std::vector<float>> payloads);

    /// Parse a whole Rune file buffer.
    [[nodiscard]] static RuneFile parse(std::span<const std::uint8_t> file);

    /// Read and parse a Rune from disk.
    [[nodiscard]] static RuneFile load(const std::string& path);

    /// Serialize to the safetensors container layout.
    [[nodiscard]] std::vector<std::uint8_t> serialize() const;

    [[nodiscard]] const RuneManifest& manifest() const noexcept { return _manifest; }

    /// Payload values by name; throws if absent.
    [[nodiscard]] std::span<const float> payload(std::string_view name) const;

    [[nodiscard]] const std::map<std::string, std::vector<float>>& payloads() const noexcept {
        return _payloads;
    }

private:
    /// Schema checks shared by the ctor and `parse`.
    void validate() const;

    RuneManifest                               _manifest;
    std::map<std::string, std::vector<float>>  _payloads;
};

} // namespace mimirmind::core::rune

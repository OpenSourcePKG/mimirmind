// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/rune/RuneFormat.hpp"

#include <array>
#include <cstdio>

namespace mimirmind::core::rune {

namespace {

struct KindName {
    RuneOpKind       kind;
    std::string_view name;
};

constexpr std::array<KindName, 8> kKindNames{{
    {RuneOpKind::WriteSlice,       "write_slice"},
    {RuneOpKind::Scale,            "scale"},
    {RuneOpKind::Mask,             "mask"},
    {RuneOpKind::LowRank,          "lowrank"},
    {RuneOpKind::Steer,            "steer"},
    {RuneOpKind::WriteQuantBlocks, "write_quant_blocks"},
    {RuneOpKind::SteerProject,     "steer_project"},
    {RuneOpKind::SteerIf,          "steer_if"},
}};

} // namespace

std::string_view opKindName(RuneOpKind k) noexcept {
    for (const auto& e : kKindNames) {
        if (e.kind == k) {
            return e.name;
        }
    }
    return "unknown";
}

std::optional<RuneOpKind> opKindFromName(std::string_view s) noexcept {
    for (const auto& e : kKindNames) {
        if (e.name == s) {
            return e.kind;
        }
    }
    return std::nullopt;
}

bool opKindApplicable(RuneOpKind k) noexcept {
    return k == RuneOpKind::WriteSlice || k == RuneOpKind::Scale || opKindIsSteer(k) ||
           k == RuneOpKind::Mask;
}

bool opKindIsOverlay(RuneOpKind k) noexcept {
    return opKindIsSteer(k) || k == RuneOpKind::Mask;
}

bool opKindIsSteer(RuneOpKind k) noexcept {
    return k == RuneOpKind::Steer || k == RuneOpKind::SteerProject || k == RuneOpKind::SteerIf;
}

std::string_view steerModeName(RuneSteerMode m) noexcept {
    return m == RuneSteerMode::Project ? "project" : "add";
}

std::optional<RuneSteerMode> steerModeFromName(std::string_view s) noexcept {
    if (s == "add") {
        return RuneSteerMode::Add;
    }
    if (s == "project") {
        return RuneSteerMode::Project;
    }
    return std::nullopt;
}

std::optional<std::uint32_t> expertMaskBlockOf(std::string_view target) noexcept {
    constexpr std::string_view kSuffix = ".ffn_gate_inp.weight";
    if (target.size() <= kSuffix.size() ||
        target.substr(target.size() - kSuffix.size()) != kSuffix) {
        return std::nullopt;
    }
    return steerBlockOf(target.substr(0, target.size() - kSuffix.size()));
}

std::optional<std::uint32_t> steerBlockOf(std::string_view target) noexcept {
    constexpr std::string_view kPrefix = "blk.";
    if (target.size() <= kPrefix.size() || target.substr(0, kPrefix.size()) != kPrefix ||
        target.size() > kPrefix.size() + 9) {
        return std::nullopt;
    }
    std::uint32_t v = 0;
    for (const char c : target.substr(kPrefix.size())) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
        v = v * 10U + static_cast<std::uint32_t>(c - '0');
    }
    return v;
}

std::string tensorContentHash(const void* bytes, std::size_t n) noexcept {
    constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kPrime  = 0x100000001b3ULL;
    std::uint64_t h = kOffset;
    const auto* p = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kPrime;
    }
    char buf[8 + 16 + 1];
    std::snprintf(buf, sizeof(buf), "fnv1a64:%016llx",
                  static_cast<unsigned long long>(h));
    return std::string{buf};
}

} // namespace mimirmind::core::rune

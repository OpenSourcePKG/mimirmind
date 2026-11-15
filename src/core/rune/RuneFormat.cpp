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

constexpr std::array<KindName, 6> kKindNames{{
    {RuneOpKind::WriteSlice,       "write_slice"},
    {RuneOpKind::Scale,            "scale"},
    {RuneOpKind::Mask,             "mask"},
    {RuneOpKind::LowRank,          "lowrank"},
    {RuneOpKind::Steer,            "steer"},
    {RuneOpKind::WriteQuantBlocks, "write_quant_blocks"},
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
    return k == RuneOpKind::WriteSlice || k == RuneOpKind::Scale;
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

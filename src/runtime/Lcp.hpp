#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace mimirmind::runtime {

/// Length of the longest common prefix between two int32_t token sequences.
/// Caps at min(a.size(), b.size()).
///
/// Used by the M9.1 prefix-cache: comparing the new request's prompt ids
/// against the engine's cached-tokens vector to decide how many leading
/// tokens of prefill can be skipped because their KV state is still valid.
[[nodiscard]] inline std::size_t
longestCommonPrefix(std::span<const std::int32_t> a,
                    std::span<const std::int32_t> b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

} // namespace mimirmind::runtime
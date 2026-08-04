// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <string_view>

namespace mimirmind::core::security {

/**
 * Constant-time equality for secrets (API keys / bearer tokens).
 *
 * A naive `==` / `std::equal` short-circuits on the first mismatching
 * byte, which leaks the length of the matching prefix through timing and
 * lets an attacker recover a key byte-by-byte. This helper always walks
 * the full length of both inputs and folds every byte into an XOR
 * accumulator, so the running time depends only on the input sizes, never
 * on the position of the first difference.
 *
 * Length mismatch still returns false, but only after touching every byte
 * of the longer input, so the timing signal carries no per-byte structure.
 */
[[nodiscard]] inline bool constantTimeEquals(std::string_view a,
                                             std::string_view b) noexcept {
    // Fold the length difference into the accumulator so unequal lengths
    // never take a shorter path. `diff` starts non-zero when sizes differ.
    unsigned int  diff = static_cast<unsigned int>(a.size() ^ b.size());
    const std::size_t m = a.size() > b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < m; ++i) {
        // Read from the shorter buffer's byte 0 once we run past its end
        // — the value is irrelevant (diff is already non-zero via the
        // size mismatch above), the point is to keep the loop's length a
        // function of `m`, not of the first-difference position.
        const unsigned char ca =
            static_cast<unsigned char>(a[i < a.size() ? i : 0]);
        const unsigned char cb =
            static_cast<unsigned char>(b[i < b.size() ? i : 0]);
        diff |= static_cast<unsigned int>(ca ^ cb);
    }
    return diff == 0;
}

} // namespace mimirmind::core::security

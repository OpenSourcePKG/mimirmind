#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

/**
 * Per-host temperature limits. Describes what THIS specific machine can
 * handle thermally — not a generic preset. Loaded from a JSON file the
 * operator hand-tunes (see examples/thermal-profile-*.json).
 *
 * Scope is intentionally narrow: only package temperature thresholds
 * live here. RAM / GPU-frequency / load are surfaced through the
 * /v1/system/status endpoint as observability, but they are not part
 * of the throttle/admission decision — those concerns belong in their
 * own configuration if they ever land.
 *
 * Decision semantics for the package temperature pair:
 *   reading <= soft         → no throttle
 *   soft < reading < hard   → linear pace ramp from 0 to throttle_max_ms
 *   reading >= hard         → max pace AND new requests refused (503)
 *
 * Leaving the soft/hard pair absent in the JSON disables monitoring
 * entirely — the guard becomes a no-op for that profile.
 */
struct ThermalProfile {
    std::string name{};            // required, free-form identifier
    std::string description{};     // optional, free-form

    // Both fields must be present together; absence disables monitoring.
    std::optional<float>       package_temp_soft_c;
    std::optional<float>       package_temp_hard_c;
    // Cap on the pacing pause when the package is at or beyond hard.
    int                        package_throttle_max_ms{200};

    [[nodiscard]] bool hasPackageLimits() const noexcept {
        return package_temp_soft_c.has_value() && package_temp_hard_c.has_value();
    }
};

/**
 * Parse a thermal-profile JSON file from disk. Throws with a precise
 * error message on:
 *   - missing file / unreadable file
 *   - invalid JSON
 *   - missing `name` field
 *   - soft >= hard for package temp
 *   - only one of soft/hard present (must be all-or-nothing)
 */
[[nodiscard]] ThermalProfile loadThermalProfile(std::string_view path);

} // namespace mimirmind::runtime
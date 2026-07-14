// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <string>

namespace mimirmind::core::l0 {
struct DeviceInfo;
}

namespace mimirmind::diagnostics {

/// Human-readable byte-count formatter. Picks the largest suitable unit
/// (GiB/MiB/KiB/B) and prints two decimal places. Used by every M1-M5
/// smoke printer so alloc-size logs stay consistent across the boot
/// output.
[[nodiscard]] std::string formatBytes(std::size_t bytes);

/// Print one device row from a Level-Zero enumeration. `selected`
/// marks the device the L0Context chose as the compute target (leading
/// `*` marker). Output goes to std::cout; caller controls surrounding
/// blank lines.
void printDevice(const ::mimirmind::core::l0::DeviceInfo& info, bool selected);

} // namespace mimirmind::diagnostics
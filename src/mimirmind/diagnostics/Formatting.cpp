// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/diagnostics/Formatting.hpp"

#include "core/l0/L0Context.hpp"

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

namespace mimirmind::diagnostics {

std::string formatBytes(std::size_t bytes) {
    constexpr double kKi = 1024.0;
    constexpr double kMi = kKi * 1024.0;
    constexpr double kGi = kMi * 1024.0;

    const double b = static_cast<double>(bytes);
    char buf[64];
    if (b >= kGi) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", b / kGi);
    } else if (b >= kMi) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB", b / kMi);
    } else if (b >= kKi) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB", b / kKi);
    } else {
        std::snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return std::string{buf};
}

void printDevice(const ::mimirmind::core::l0::DeviceInfo& info, bool selected) {
    using ::mimirmind::core::l0::L0Context;

    std::cout << (selected ? "  * " : "    ")
              << info.name
              << "  [" << L0Context::typeToString(info.type) << "]\n";
    std::cout << "      vendor   : 0x" << std::hex << std::setw(4) << std::setfill('0')
              << info.vendorId << std::dec << std::setfill(' ') << "\n";
    std::cout << "      deviceId : 0x" << std::hex << std::setw(4) << std::setfill('0')
              << info.deviceId << std::dec << std::setfill(' ') << "\n";
    std::cout << "      uuid     : " << info.uuid << "\n";
    std::cout << "      compute  : " << info.numComputeUnits << " threads, "
              << info.coreClockRate << " MHz\n";
    std::cout << "      local mem: " << formatBytes(info.totalLocalMem) << "\n";
    std::cout << "      sub-devs : " << info.numSubDevices << "\n";
}

} // namespace mimirmind::diagnostics
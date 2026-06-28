#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kBanner =
    "+------------------------------------------------------------+\n"
    "|                          Mimirmind                         |\n"
    "|   Project Well startup smoke test (M1+M2 ctx/USM/pool)     |\n"
    "+------------------------------------------------------------+\n";

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

void printDevice(const mimirmind::runtime::DeviceInfo& info, bool selected) {
    using mimirmind::runtime::L0Context;

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

} // namespace

int main() {
    mimirmind::runtime::Log::initFromEnv();

    std::cout << kBanner;
    std::cout.flush();

    MM_LOG_INFO("main", "mimirmind smoke test starting (M1 ctx + M2 USM probe)");

    try {
        // --- [M1] Level Zero context + device enumeration --------------------

        MM_LOG_INFO("main", "[M1] constructing L0Context");
        mimirmind::runtime::L0Context ctx;

        std::cout << "[M1] Level Zero enumeration\n";
        const auto& all = ctx.allDevices();
        std::cout << "  Found " << all.size() << " Level-Zero device(s):\n\n";

        for (const auto& d : all) {
            const bool selected = (d.uuid == ctx.info().uuid);
            printDevice(d, selected);
            std::cout << "\n";
        }

        std::cout << "  Selected target device : " << ctx.info().name << "\n";
        std::cout << "  Context created OK     : "
                  << (ctx.context() != nullptr ? "yes" : "no") << "\n";

        // --- [M2] USM allocation probe ---------------------------------------

        std::cout << "\n[M2] USM allocation probe (this may take a moment)\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M2] starting USM allocation probe");

        mimirmind::runtime::UsmAllocator allocator{ctx};
        allocator.probeLimits();
        const auto& lim = allocator.limits();

        std::cout << "  per-alloc max     : " << formatBytes(lim.perAllocMaxBytes) << "\n";
        std::cout << "  total allocatable : " << formatBytes(lim.totalAllocatableBytes)
                  << " (" << lim.probeBlocksGranted << " x 256 MiB blocks)\n";

        // --- [M2b] Allocator + free-list smoke test --------------------------

        std::cout << "\n[M2b] Allocator + free-list smoke test\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M2b] exercising allocator with mixed sizes");

        // Mixed sizes that touch several buckets, including one exact and one
        // off-by-one to force rounding. 64 MiB is the largest — well below
        // the per-alloc ceiling but big enough to be interesting.
        constexpr std::array<std::size_t, 6> kSmokeSizes{
            8ULL  << 10,     // 8 KiB     -> bucket 8 KiB
            128ULL << 10,    // 128 KiB   -> bucket 128 KiB
            (4ULL << 20) + 1,// 4 MiB + 1 -> bucket 8 MiB (off-by-one demo)
            64ULL << 20,     // 64 MiB    -> bucket 64 MiB
            4ULL  << 20,     // 4 MiB     -> bucket 4 MiB
            1ULL  << 20,     // 1 MiB     -> bucket 1 MiB
        };

        auto exercise = [&](const char* label) {
            MM_LOG_INFO("main", "[M2b] round '{}' — allocate {} chunks",
                        label, kSmokeSizes.size());
            std::vector<std::pair<void*, std::size_t>> live;
            live.reserve(kSmokeSizes.size());
            for (auto s : kSmokeSizes) {
                void* p = allocator.allocate(s);
                // Touch the first and last cache line to make sure the
                // mapping is real and CPU-addressable.
                std::memset(p, 0xA5, 64);
                std::memset(static_cast<char*>(p) + (s > 64 ? s - 64 : 0), 0x5A, 64);
                live.emplace_back(p, s);
            }
            MM_LOG_INFO("main", "[M2b] round '{}' — deallocate (reverse order)",
                        label);
            while (!live.empty()) {
                auto [p, s] = live.back();
                live.pop_back();
                allocator.deallocate(p, s);
            }
        };

        exercise("cold");   // free-list empty: all misses
        exercise("warm");   // same sizes: should hit free-list

        allocator.logStats(mimirmind::runtime::LogLevel::Info);

        const auto st = allocator.stats();
        std::cout << "  total alloc/free  : " << st.totalAllocations
                  << " / " << st.totalDeallocations << "\n";
        std::cout << "  ze calls          : " << st.zeAllocCalls
                  << " alloc / " << st.zeFreeCalls << " free\n";
        std::cout << "  free-list         : " << st.freeListHits
                  << " hits / " << st.freeListMisses << " misses";
        if (st.totalAllocations > 0) {
            std::cout << " ("
                      << (100ULL * st.freeListHits / st.totalAllocations)
                      << "% hit rate)";
        }
        std::cout << "\n";
        std::cout << "  peak live bytes   : " << formatBytes(st.peakBytes) << "\n";
        std::cout << "  live at end       : " << st.liveAllocations
                  << " allocs / " << formatBytes(st.liveBytes) << "\n";

        MM_LOG_INFO("main",
                    "smoke test passed — perAllocMax={} bytes totalAllocatable={} bytes "
                    "freeListHits={} / {} totalAllocs",
                    lim.perAllocMaxBytes,
                    lim.totalAllocatableBytes,
                    st.freeListHits,
                    st.totalAllocations);
        std::cout << "\nProject Well startup smoke test passed.\n";
        return 0;
    } catch (const mimirmind::runtime::L0Error& e) {
        MM_LOG_ERROR("main", "Level Zero error: {}", e.what());
        std::cerr << "Level Zero error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        MM_LOG_ERROR("main", "fatal: {}", e.what());
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
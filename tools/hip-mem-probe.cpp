// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_mem_probe — round-trip sanity for HipMemoryAllocator + HipBuffer.
//
// For each of the three allocation kinds (Device, HostPinned, Managed):
//   1) allocate a 4 MiB buffer via HipBuffer
//   2) fill a host pattern
//   3) copy H2D → copy D2H into a fresh host block
//   4) byte-compare the round-trip
// Then verify the stats bookkeeping matches what we did, and print a
// one-line summary. Exit code 0 on success, non-zero on any mismatch.
//
// Not installed in any runtime image — bringup diagnostic, analog to
// hip_probe / backend_probe. First proof that the memory layer of the
// HIP backend is live end-to-end before we build kernels on top.
//
// Run via:  cmake --build build --target hip_mem_probe && ./build/hip_mem_probe

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/log/Log.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string_view>
#include <vector>

namespace {

using mimirmind::core::hip::HipAllocKind;
using mimirmind::core::hip::HipBuffer;
using mimirmind::core::hip::HipContext;
using mimirmind::core::hip::HipMemoryAllocator;

constexpr std::size_t kBufBytes = 4u << 20;  // 4 MiB

std::string_view kindLabel(HipAllocKind k) noexcept {
    switch (k) {
        case HipAllocKind::Device:     return "Device";
        case HipAllocKind::HostPinned: return "HostPinned";
        case HipAllocKind::Managed:    return "Managed";
    }
    return "Unknown";
}

// Fill with a deterministic pattern so byte-compare gives a concrete
// mismatch index if anything is wrong. u8 pattern is enough — we're
// not stressing endianness here.
void fillPattern(std::vector<std::uint8_t>& v, std::uint32_t seed) {
    std::uint32_t x = seed;
    for (auto& b : v) {
        x = x * 1664525u + 1013904223u;   // LCG (numerical recipes)
        b = static_cast<std::uint8_t>(x >> 24);
    }
}

// Returns first-mismatch index, or SIZE_MAX on full match.
std::size_t firstDiff(const std::vector<std::uint8_t>& a,
                      const std::vector<std::uint8_t>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return i;
    }
    return (a.size() == b.size()) ? SIZE_MAX : n;
}

// Outcome of one round-trip attempt. SKIPPED is a soft-fail — used
// for `Managed` on RDNA consumer where hipMallocManaged is known-flaky.
enum class ProbeOutcome { Ok, Failed, Skipped };

// Run one kind's round-trip test. `soft` decides whether an allocation
// OOM counts as Failed (hard) or Skipped (soft — printed but doesn't
// fail the tool). Data-mismatches always count as Failed.
ProbeOutcome roundTrip(HipMemoryAllocator& alloc, HipAllocKind kind, bool soft) {
    std::vector<std::uint8_t> src(kBufBytes);
    std::vector<std::uint8_t> dst(kBufBytes, 0);
    fillPattern(src, /*seed=*/0xC0FFEEu ^ static_cast<std::uint32_t>(kind));

    try {
        HipBuffer buf{alloc, kBufBytes, kind};

        if (kind == HipAllocKind::HostPinned) {
            // HostPinned pointer is directly host-writable — skip the
            // H2D and use it in-place. This is exactly the pattern the
            // real inference loop will use for staging: write from CPU
            // into a pinned buffer, then a fast H↔D DMA later.
            std::memcpy(buf.data(), src.data(), kBufBytes);
            std::memcpy(dst.data(), buf.data(), kBufBytes);
        } else {
            // Device / Managed: opaque device pointer, use the memcpy API.
            alloc.copyH2D(buf.data(), src.data(), kBufBytes);
            alloc.copyD2H(dst.data(), buf.data(), kBufBytes);
        }
    } catch (const std::exception& e) {
        const char* verdict = soft ? "SKIPPED" : "FAIL";
        std::fprintf(stderr, "  [%-10.*s] %s — %s\n",
                     static_cast<int>(kindLabel(kind).size()),
                     kindLabel(kind).data(),
                     verdict,
                     e.what());
        return soft ? ProbeOutcome::Skipped : ProbeOutcome::Failed;
    }

    const std::size_t diff = firstDiff(src, dst);
    if (diff != SIZE_MAX) {
        std::fprintf(stderr,
                     "  [%-10.*s] FAIL — byte mismatch at %zu (src=0x%02x dst=0x%02x)\n",
                     static_cast<int>(kindLabel(kind).size()),
                     kindLabel(kind).data(),
                     diff, src[diff], dst[diff]);
        return ProbeOutcome::Failed;
    }
    std::printf("  [%-10.*s] OK — %zu bytes round-trip clean\n",
                static_cast<int>(kindLabel(kind).size()),
                kindLabel(kind).data(),
                kBufBytes);
    return ProbeOutcome::Ok;
}

} // namespace

int main() {
    HipContext ctx{};
    HipMemoryAllocator alloc{ctx};

    std::printf("HipMemoryAllocator on %s (arch=%s):\n\n",
                ctx.hipDeviceInfo().name.c_str(),
                ctx.hipDeviceInfo().gfxArch.c_str());

    // Device + HostPinned are the two kinds mimirmind will actually rely
    // on for the kernel port. Managed is exposed in the API for API
    // symmetry with the L0 `UsmAllocKind::Shared` and for future UMA-APU
    // hardware, but AMD's `hipMallocManaged` is known-flaky on consumer
    // RDNA (RX 7000 series) — OOM on small allocs, driver-version-
    // dependent. Treat Managed as soft: SKIPPED on failure doesn't fail
    // the tool.
    struct Case { HipAllocKind kind; bool soft; };
    const Case cases[] = {
        { HipAllocKind::Device,     /*soft=*/false },
        { HipAllocKind::HostPinned, /*soft=*/false },
        { HipAllocKind::Managed,    /*soft=*/true  },
    };

    int okCount = 0, failCount = 0, skipCount = 0, hardMemcpy = 0;
    for (const auto& c : cases) {
        const ProbeOutcome r = roundTrip(alloc, c.kind, c.soft);
        if (r == ProbeOutcome::Ok) {
            ++okCount;
            if (c.kind != HipAllocKind::HostPinned) ++hardMemcpy;
        }
        else if (r == ProbeOutcome::Failed)  ++failCount;
        else                                 ++skipCount;
    }

    std::printf("\nStats:\n");
    alloc.logStats(::mimirmind::core::log::LogLevel::Info);

    // Sanity on the numbers: one alloc/free per OK case, H2D+D2H per
    // OK case that wasn't HostPinned (Device / Managed use the memcpy
    // API — HostPinned uses in-place std::memcpy).
    const auto& s = alloc.stats();
    const bool statsOk =
        s.totalAllocations   == static_cast<std::uint64_t>(okCount) &&
        s.totalDeallocations == static_cast<std::uint64_t>(okCount) &&
        s.liveAllocations    == 0 &&
        s.liveBytes          == 0 &&
        s.bytesCopiedH2D     == static_cast<std::uint64_t>(hardMemcpy) * kBufBytes &&
        s.bytesCopiedD2H     == static_cast<std::uint64_t>(hardMemcpy) * kBufBytes;

    if (!statsOk) {
        std::fprintf(stderr,
                     "\nstats bookkeeping mismatch — expected allocs=%d "
                     "frees=%d H2D=%zu D2H=%zu, got allocs=%llu frees=%llu "
                     "H2D=%llu D2H=%llu\n",
                     okCount, okCount,
                     static_cast<std::size_t>(hardMemcpy) * kBufBytes,
                     static_cast<std::size_t>(hardMemcpy) * kBufBytes,
                     static_cast<unsigned long long>(s.totalAllocations),
                     static_cast<unsigned long long>(s.totalDeallocations),
                     static_cast<unsigned long long>(s.bytesCopiedH2D),
                     static_cast<unsigned long long>(s.bytesCopiedD2H));
    }

    const bool allOk = failCount == 0 && statsOk;
    std::printf("\nhip_mem_probe: %s (%d ok / %d skipped / %d failed)\n",
                allOk ? "OK" : "FAIL", okCount, skipCount, failCount);
    return allOk ? 0 : 1;
}
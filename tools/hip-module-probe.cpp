// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_module_probe — end-to-end sanity for HipModule + HipKernel:
//   1) load kernels_hip/hip_probe_kernel.hsaco via HipModule
//   2) resolve write_pattern via HipModule::getKernel
//   3) allocate a device buffer, set kernel args, launch on a stream
//   4) copy back to host and byte-compare against the reference
//   5) time the kernel with HipEvent (exercises the event / timing path
//      alongside the module + kernel path)
//
// Argv[1] optionally overrides the hsaco path — defaults to the layout
// CMake writes to ${CMAKE_BINARY_DIR}/hsaco/hip_probe_kernel.hsaco
// which lives next to the binary at ./hsaco/hip_probe_kernel.hsaco.

#include "core/gpu/hip/HipContext.hpp"
#include "core/gpu/hip/HipEvent.hpp"
#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipMemoryAllocator.hpp"
#include "core/gpu/hip/HipModule.hpp"
#include "core/gpu/hip/HipStream.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kNumElements = 1u << 20;   // 1M ints = 4 MiB

std::string defaultHsacoPath(const char* argv0) {
    // Layout the CMake custom-command lands in:
    //   ${CMAKE_BINARY_DIR}/hsaco/hip_probe_kernel.hsaco
    // Test binary lives at ${CMAKE_BINARY_DIR}/hip_module_probe, so the
    // hsaco is at ./hsaco/hip_probe_kernel.hsaco relative to the exe.
    std::filesystem::path exe = std::filesystem::canonical(argv0);
    return (exe.parent_path() / "hsaco" / "hip_probe_kernel.hsaco").string();
}

std::uint32_t reference(std::uint32_t i) noexcept {
    return i * 7u + 3u;    // must match the kernel
}

} // namespace

int main(int argc, char** argv) {
    using namespace mimirmind::core::hip;

    const std::string hsacoPath =
        (argc > 1) ? std::string{argv[1]} : defaultHsacoPath(argv[0]);

    std::printf("hip_module_probe:\n  hsaco: %s\n", hsacoPath.c_str());

    try {
        HipContext          ctx{};
        HipMemoryAllocator  alloc{ctx};
        HipStream           stream{ctx, HipStreamKind::BlockingDefault};
        HipEvent            evStart{ctx};
        HipEvent            evEnd  {ctx};

        // ---- module + kernel ---------------------------------------------
        HipModule mod = HipModule::fromFile(ctx, hsacoPath);
        HipKernel kernel = mod.getKernel("write_pattern");

        // ---- device buffer ----------------------------------------------
        const std::size_t bytes = kNumElements * sizeof(std::uint32_t);
        HipBuffer devOut{alloc, bytes, HipAllocKind::Device};

        kernel.setPtr  (0, devOut.data());
        kernel.setValue(1, kNumElements);

        // Launch geometry: one warp per block (32 threads on RDNA3,
        // gfx1101 warpSize=32 — see lesson_rdna3_wave32.md), enough
        // blocks to cover kNumElements.
        constexpr std::uint32_t kBlock = 256;
        const std::uint32_t kGrid = (kNumElements + kBlock - 1u) / kBlock;

        evStart.record(stream);
        kernel.launch(stream, kGrid, 1, 1, kBlock, 1, 1, /*shared=*/0);
        evEnd.record(stream);
        stream.synchronize();

        const float kernelMs = evEnd.elapsedMs(evStart);

        // ---- verify ------------------------------------------------------
        std::vector<std::uint32_t> host(kNumElements, 0u);
        alloc.copyD2H(host.data(), devOut.data(), bytes);

        std::size_t mismatches = 0;
        std::uint32_t firstBadIdx = 0;
        std::uint32_t firstBadGot = 0;
        std::uint32_t firstBadExp = 0;
        for (std::uint32_t i = 0; i < kNumElements; ++i) {
            const std::uint32_t exp = reference(i);
            if (host[i] != exp) {
                if (mismatches == 0) {
                    firstBadIdx = i;
                    firstBadGot = host[i];
                    firstBadExp = exp;
                }
                ++mismatches;
            }
        }

        std::printf("  elements: %u\n", kNumElements);
        std::printf("  kernel:   %.3f ms\n", static_cast<double>(kernelMs));
        std::printf("  mismatches: %zu\n", mismatches);
        if (mismatches != 0) {
            std::printf("  first bad @ %u: got=%u exp=%u\n",
                        firstBadIdx, firstBadGot, firstBadExp);
        }

        std::printf("\nStats:\n");
        alloc.logStats(::mimirmind::core::log::LogLevel::Info);

        const bool ok = (mismatches == 0);
        std::printf("\nhip_module_probe: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_module_probe: threw: %s\n", e.what());
        return 2;
    }
}
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// backend_select_probe — end-to-end demonstration of runtime backend
// selection. Reads MIMIRMIND_BACKEND (falling back to LevelZero), then
// actually constructs the corresponding `ComputeContext` and prints
// the concrete device info from it.
//
// Distinct from `backend_probe`, which only *probes* what's compiled
// in / runtime-available without instantiating anything. This tool
// closes the loop: probe → resolve → construct → observe.
//
// Fails with exit 2 (and a clear message) when the selected backend
// is compiled-in but the device init throws (no dev/dri permission,
// no GPU present, half-installed driver, ...). Fails with exit 3 when
// the selected backend is not compiled in.
//
// Run via:  cmake --build build --target backend_select_probe
//           MIMIRMIND_BACKEND=hip ./build/backend_select_probe
//           MIMIRMIND_BACKEND=l0  ./build/backend_select_probe

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"
#include "core/backend/ComputeContext.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

namespace {

const char* deviceKindName(mimirmind::core::backend::DeviceKind k) {
    using DK = mimirmind::core::backend::DeviceKind;
    switch (k) {
        case DK::GpuIntegrated: return "GpuIntegrated";
        case DK::GpuDiscrete:   return "GpuDiscrete";
        case DK::Npu:           return "Npu";
        case DK::Cpu:           return "Cpu";
        case DK::Unknown:       return "Unknown";
    }
    return "?";
}

} // namespace

int main() {
    using namespace mimirmind::core::backend;

    // Step 1 — the probe picture, so operators see what's available
    // before we try to construct anything.
    std::printf("== BackendRegistry::probeAll() ==\n");
    const auto probes = BackendRegistry::probeAll();
    for (const auto& p : probes) {
        std::printf("  %-10s compiled=%s available=%s  (%s)\n",
                    BackendRegistry::name(p.kind),
                    p.compiledIn ? "yes" : "no",
                    p.available  ? "yes" : "no",
                    p.detail.c_str());
    }

    // Step 2 — resolve the requested backend. Uses autoSelect() so a
    // build compiled with both L0 and HIP picks whichever is actually
    // reachable on this box (env override still wins if set).
    const char* envVal = std::getenv("MIMIRMIND_BACKEND");
    const BackendKind requested = BackendRegistry::autoSelect();
    std::printf("\n== Selection ==\n");
    std::printf("  MIMIRMIND_BACKEND env: %s\n",
                envVal ? envVal : "(unset — autoselect)");
    std::printf("  resolved kind:         %s\n",
                BackendRegistry::name(requested));

    // Step 3 — construct + observe.
    std::unique_ptr<ComputeContext> ctx;
    try {
        ctx = BackendRegistry::createContext(requested);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[FAIL] createContext(%s) threw: %s\n",
                     BackendRegistry::name(requested), e.what());
        // exit 3 for "not compiled in", exit 2 for "compiled but init
        // threw". Distinction lives in the message; here we just pick
        // an exit code by string content — cheap enough for a diag tool.
        return (std::string_view{e.what()}.find("not compiled in")
                != std::string_view::npos) ? 3 : 2;
    }

    const ComputeBackend& backend       = ctx->backend();
    const BackendDeviceInfo& info       = backend.deviceInfo();

    std::printf("\n== ComputeContext initialised ==\n");
    std::printf("  kind:              %s\n", BackendRegistry::name(backend.kind()));
    std::printf("  device name:       %s\n", info.name.c_str());
    std::printf("  device uuid:       %s\n",
                info.uuid.empty() ? "(none)" : info.uuid.c_str());
    std::printf("  pci vendor:device: 0x%04x:0x%04x\n",
                info.vendorId, info.deviceId);
    std::printf("  device kind:       %s\n", deviceKindName(info.kind));
    std::printf("  compute units:     %u\n", info.numComputeUnits);
    std::printf("  total local mem:   %.2f GiB\n",
                static_cast<double>(info.totalLocalMem)
                / (1024.0 * 1024.0 * 1024.0));
    std::printf("  max alloc size:    %.2f GiB\n",
                static_cast<double>(info.maxMemAllocSize)
                / (1024.0 * 1024.0 * 1024.0));
    std::printf("  core clock:        %u MHz\n", info.coreClockRate);

    std::printf("\n== Features ==\n");
    const BackendFeature features[] = {
        BackendFeature::MutableCommandLists,
        BackendFeature::IntegerDotProduct,
        BackendFeature::IpcHandleExport,
        BackendFeature::UnifiedMemoryHost,
        BackendFeature::MatrixEngine,
    };
    const char* names[] = {
        "MutableCommandLists",
        "IntegerDotProduct",
        "IpcHandleExport",
        "UnifiedMemoryHost",
        "MatrixEngine",
    };
    for (std::size_t i = 0; i < 5; ++i) {
        std::printf("  %-22s %s\n",
                    names[i],
                    backend.hasFeature(features[i]) ? "yes" : "no");
    }

    std::printf("\nbackend_select_probe: OK\n");
    return 0;
}
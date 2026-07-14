// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// backend_probe — enumerate every compute backend mimirmind knows about,
// probe each one for a usable device, and print a table. Symmetric to
// hip_probe and l0_ipc_testrig but at the neutral level — no L0- or HIP-
// specific detail, just the BackendRegistry snapshot that later
// runtime-selection logic will consume.
//
// Never fails: even if zero backends are available, the process still
// exits 0 and prints the honest "nothing found" table. Callers that need
// a non-zero exit on "no compute" can grep the output.
//
// Run via:  cmake --build build --target backend_probe && ./build/backend_probe

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeBackend.hpp"

#include <cstdio>

int main() {
    using namespace mimirmind::core::backend;

    const auto probes = BackendRegistry::probeAll();

    std::printf("%-12s  %-11s  %-9s  %s\n",
                "backend", "compiled", "available", "detail");
    std::printf("%-12s  %-11s  %-9s  %s\n",
                "-------", "--------", "---------", "------");

    int availableCount = 0;
    for (const auto& p : probes) {
        std::printf("%-12s  %-11s  %-9s  %s\n",
                    BackendRegistry::name(p.kind),
                    p.compiledIn ? "yes" : "no",
                    p.available  ? "yes" : "no",
                    p.detail.c_str());
        if (p.available) ++availableCount;
    }
    std::printf("\n%d backend(s) available at runtime.\n", availableCount);
    return 0;
}
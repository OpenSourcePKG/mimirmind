// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/SmokeMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"
#include "mimirmind/diagnostics/SmokeSuite.hpp"

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeContext.hpp"
#include "core/config/Config.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"
#include "core/log/Log.hpp"
#include "runtime/InferenceEngine.hpp"

#include <iostream>

namespace mimirmind::cli {

using ::mimirmind::diagnostics::printM1M2;
using ::mimirmind::diagnostics::printM3Summary;
using ::mimirmind::diagnostics::runM2bAllocatorSmoke;
using ::mimirmind::diagnostics::runM5RmsNormParity;
using ::mimirmind::diagnostics::runM5bQ4KParity;
using ::mimirmind::diagnostics::runM5cQ6KParity;
using ::mimirmind::diagnostics::runM4aEmbedAndM4bLmHead;
using ::mimirmind::diagnostics::runM7cChatTemplate;
using ::mimirmind::diagnostics::runM4deGenerate;

int runSmoke(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    std::cout << kBanner;
    std::cout.flush();

    MM_LOG_INFO("main", "mimirmind smoke starting (M1-M5 + engine.generate)");

    ::mimirmind::runtime::InferenceEngine engine{cfg};

    // Schicht 6.0 — SmokeMode's M1-M5 diagnostics all reach into
    // L0-native handles (UsmAllocator, L0Context, UsmHandle). On HIP
    // the accessors throw, so we skip them and drive straight from
    // loadModel to runM4deGenerate (which is backend-agnostic — only
    // touches engine.tokenizer / engine.config / engine.generate).
    const bool isL0 = engine.computeContextKind() ==
                      core::backend::BackendKind::LevelZero;

    if (isL0) {
        printM1M2(engine);
        runM2bAllocatorSmoke(engine.allocator());
        runM5RmsNormParity(engine.ctx(), engine.allocator());
    } else {
        std::cout << "\n[M1-M2b/M5] skipped (L0-native diagnostics; "
                  << "runtime bound to "
                  << core::backend::BackendRegistry::name(
                         engine.computeContextKind())
                  << ")\n";
        MM_LOG_INFO("main",
                    "non-L0 backend — skipping M1/M2b/M5 L0-native diagnostics");
    }

    if (args.modelPath.empty()) {
        std::cout << "\n[M3] GGUF reader — skipped "
                     "(pass --model PATH or set models[].path in config.json)\n";
        MM_LOG_INFO("main", "[M3] no model path; skipping load");
    } else {
        std::cout << "\n[M3] GGUF reader (" << args.modelPath << ")\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M3] opening model '{}'", args.modelPath);

        engine.loadModel(args.modelPath);

        if (isL0) {
            printM3Summary(engine);
            runM5bQ4KParity(engine.ctx(), engine.allocator(), engine.weights());
            runM5cQ6KParity(engine.ctx(), engine.allocator(), engine.weights());
            runM4aEmbedAndM4bLmHead(engine);
            runM7cChatTemplate(engine);
        } else {
            std::cout << "\n[M3/M5b-c/M4a-b/M7c] skipped (L0-native)\n";
        }

        runM4deGenerate(engine, args);
    }

    if (isL0) {
        const auto lim = engine.allocator().limits();
        const auto st  = engine.allocator().stats();
        MM_LOG_INFO("main",
                    "smoke test passed — perAllocMax={} bytes totalAllocatable={} bytes "
                    "freeListHits={} / {} totalAllocs",
                    lim.perAllocMaxBytes,
                    lim.totalAllocatableBytes,
                    st.freeListHits,
                    st.totalAllocations);
    }
    std::cout << "\nProject Well + Envoy smoke test passed.\n";
    return 0;
}

} // namespace mimirmind::cli
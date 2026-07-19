// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/SmokeMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"
#include "mimirmind/diagnostics/SmokeSuite.hpp"

#include "core/backend/BackendRegistry.hpp"
#include "core/backend/ComputeContext.hpp"
#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "runtime/InferenceEngine.hpp"

#ifdef MIMIRMIND_HAVE_L0
#include "core/gpu/l0/UsmAllocator.hpp"
#endif

#include <iostream>

namespace mimirmind::cli {

#ifdef MIMIRMIND_HAVE_L0
using ::mimirmind::diagnostics::printM1M2;
using ::mimirmind::diagnostics::printM3Summary;
using ::mimirmind::diagnostics::runM2bAllocatorSmoke;
using ::mimirmind::diagnostics::runM5RmsNormParity;
using ::mimirmind::diagnostics::runM5bQ4KParity;
using ::mimirmind::diagnostics::runM5cQ6KParity;
using ::mimirmind::diagnostics::runM4aEmbedAndM4bLmHead;
#endif
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
    // HIP-only builds skip the entire L0-native chain at compile-time
    // — the accessors are guarded behind `#ifdef MIMIRMIND_HAVE_L0`
    // and would fail to compile otherwise.
#ifdef MIMIRMIND_HAVE_L0
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
#else
    constexpr bool isL0 = false;
    std::cout << "\n[M1-M2b/M5] skipped (no L0 compiled in)\n";
#endif

    if (args.modelPath.empty()) {
        std::cout << "\n[M3] GGUF reader — skipped "
                     "(pass --model PATH or set models[].path in config.json)\n";
        MM_LOG_INFO("main", "[M3] no model path; skipping load");
    } else {
        std::cout << "\n[M3] GGUF reader (" << args.modelPath << ")\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M3] opening model '{}'", args.modelPath);

        engine.loadModel(args.modelPath);

        // Apply runtime.kvDtype / runtime.maxContextTokens from
        // config.json — mirrors ServeMode.cpp:197-200 (setKvDtype +
        // setMaxContextTokens inspect loaded model state for
        // fused-QKV / attn_k-v.bias coverage per block, so overrides
        // must land AFTER loadModel).
        {
            const auto& rt = cfg.runtime;
            if (rt.maxContextTokens.has_value() && *rt.maxContextTokens > 0) {
                engine.setMaxContextTokens(*rt.maxContextTokens);
            }
            if (rt.kvDtype.has_value()) {
                const std::string_view v{*rt.kvDtype};
                if      (v == "fp16") engine.setKvDtype(::mimirmind::runtime::KvDtype::FP16);
                else if (v == "q8_0") engine.setKvDtype(::mimirmind::runtime::KvDtype::Q8_0);
                else if (v == "f32" || v.empty())
                                      engine.setKvDtype(::mimirmind::runtime::KvDtype::F32);
                else {
                    MM_LOG_WARN("main",
                                "runtime.kvDtype='{}' unrecognised — falling "
                                "back to f32", v);
                }
            }
        }

#ifdef MIMIRMIND_HAVE_L0
        if (isL0) {
            printM3Summary(engine);
            runM5bQ4KParity(engine.ctx(), engine.allocator(), engine.weights());
            runM5cQ6KParity(engine.ctx(), engine.allocator(), engine.weights());
            runM4aEmbedAndM4bLmHead(engine);
            runM7cChatTemplate(engine);
        } else {
            std::cout << "\n[M3/M5b-c/M4a-b/M7c] skipped (L0-native)\n";
        }
#else
        std::cout << "\n[M3/M5b-c/M4a-b/M7c] skipped (no L0 compiled in)\n";
#endif

        runM4deGenerate(engine, args);
    }

#ifdef MIMIRMIND_HAVE_L0
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
#endif
    std::cout << "\nProject Well + Envoy smoke test passed.\n";
    return 0;
}

} // namespace mimirmind::cli
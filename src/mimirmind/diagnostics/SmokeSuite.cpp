// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/diagnostics/SmokeSuite.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/diagnostics/Formatting.hpp"

#include "compute/Embedding.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "core/config/Config.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/log/Log.hpp"
#include "model/ChatTemplate.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"

// L0-native single-kernel parity harnesses (M1/M2/M5*) reach into
// UsmAllocator, L0Context, GpuModule/Kernel, CommandQueue directly.
// They can only be defined when L0 is compiled in; the neutral
// harnesses (printM3Summary, runM4aEmbedAndM4bLmHead, runM7cChatTemplate,
// runM4deGenerate) use only the backend-neutral surface on
// `InferenceEngine` and stay always-compiled.
#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuMatmul.hpp"
#include "core/gpu/l0/CommandQueue.hpp"
#include "core/gpu/l0/GpuKernel.hpp"
#include "core/gpu/l0/GpuModule.hpp"
#include "core/gpu/l0/L0Context.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"
#include "core/gpu/l0/UsmHandle.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mimirmind::diagnostics {

// ---- L0-native diagnostics (M1/M2/M5/M4a-b) ---------------------------------
//
// Every function in this block reaches into UsmAllocator or L0Context
// directly and can only be compiled + linked when the L0 backend is on.
// Callers guard their invocations with the same `#ifdef MIMIRMIND_HAVE_L0`
// so the neutral M7c + M4de generate paths below stay reachable in HIP-
// only or CPU-only builds.
#ifdef MIMIRMIND_HAVE_L0

// ---- M1+M2 device + USM probe summary ---------------------------------------

void printM1M2(mimirmind::runtime::InferenceEngine& engine) {
    using mimirmind::core::l0::L0Context;

    auto& ctx       = engine.ctx();
    auto& allocator = engine.allocator();

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

    const auto& lim = allocator.limits();
    std::cout << "\n[M2] USM allocation probe\n";
    std::cout << "  per-alloc max     : " << formatBytes(lim.perAllocMaxBytes) << "\n";
    std::cout << "  total allocatable : " << formatBytes(lim.totalAllocatableBytes)
              << " (" << lim.probeBlocksGranted << " x 256 MiB blocks)\n";
}

// ---- M2b allocator + free-list smoke test ----------------------------------

void runM2bAllocatorSmoke(mimirmind::core::l0::UsmAllocator& allocator) {
    std::cout << "\n[M2b] Allocator + free-list smoke test\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M2b] exercising allocator with mixed sizes");

    constexpr std::array<std::size_t, 6> kSmokeSizes{
        8ULL  << 10,
        128ULL << 10,
        (4ULL << 20) + 1,
        64ULL << 20,
        4ULL  << 20,
        1ULL  << 20,
    };

    auto exercise = [&](const char* label) {
        MM_LOG_INFO("main", "[M2b] round '{}' — allocate {} chunks",
                    label, kSmokeSizes.size());
        std::vector<std::pair<void*, std::size_t>> live;
        live.reserve(kSmokeSizes.size());
        for (auto s : kSmokeSizes) {
            void* p = allocator.allocate(s);
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

    exercise("cold");
    exercise("warm");

    allocator.logStats(mimirmind::core::log::LogLevel::Info);

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
}

// ---- M5 GPU RMSNorm parity --------------------------------------------------

void runM5RmsNormParity(mimirmind::core::l0::L0Context&    ctx,
                        mimirmind::core::l0::UsmAllocator& allocator) {
    std::cout << "\n[M5] GPU RMSNorm kernel (SPIR-V via Level Zero)\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M5] starting GPU RMSNorm parity test");
    try {
        mimirmind::runtime::GpuModule modRms{ctx, "rmsnorm"};
        mimirmind::runtime::GpuKernel knRms{modRms.kernel("rmsnorm")};
        mimirmind::runtime::CommandQueue queue{ctx};

        constexpr std::uint32_t kLocalSize = 128;
        constexpr int           kK         = 3584;
        const std::size_t       bytesK     = static_cast<std::size_t>(kK) * sizeof(float);

        mimirmind::core::l0::UsmHandle xH{allocator, bytesK};
        mimirmind::core::l0::UsmHandle wH{allocator, bytesK};
        mimirmind::core::l0::UsmHandle yH{allocator, bytesK};

        float* x = xH.as<float>();
        float* w = wH.as<float>();
        for (int i = 0; i < kK; ++i) {
            x[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.5F;
            w[i] = 1.0F + std::cos(static_cast<float>(i) * 0.007F) * 0.1F;
        }

        knRms.setPtr(0, xH.get());
        knRms.setPtr(1, wH.get());
        knRms.setPtr(2, yH.get());
        knRms.setValue<float>(3, 1e-6F);
        knRms.setValue<std::int32_t>(4, kK);
        knRms.setGroupSize(kLocalSize, 1, 1);

        const auto g0 = std::chrono::steady_clock::now();
        queue.dispatch(knRms, /* groupCountX = M = 1 row */ 1, 1, 1);
        const auto g1 = std::chrono::steady_clock::now();

        std::vector<float> cpuRef(kK);
        const auto c0 = std::chrono::steady_clock::now();
        mimirmind::compute::rmsNorm(x, 1, kK, w, 1e-6F, cpuRef.data());
        const auto c1 = std::chrono::steady_clock::now();

        const float* y = yH.as<float>();
        float maxAbsDiff = 0.0F;
        float maxAbsCpu  = 0.0F;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kK); ++i) {
            const float d = std::fabs(y[i] - cpuRef[i]);
            if (d > maxAbsDiff) {
                maxAbsDiff = d;
            }
            const float a = std::fabs(cpuRef[i]);
            if (a > maxAbsCpu) {
                maxAbsCpu = a;
            }
        }

        const double gpuMs = std::chrono::duration<double, std::milli>(g1 - g0).count();
        const double cpuMs = std::chrono::duration<double, std::milli>(c1 - c0).count();

        std::cout << "  shape         : [1, " << kK
                  << "]  local_size=" << kLocalSize << "\n";
        std::cout << "  gpu time      : " << gpuMs << " ms (incl. queue sync)\n";
        std::cout << "  cpu time      : " << cpuMs << " ms\n";
        std::cout << "  max |GPU-CPU| : " << maxAbsDiff
                  << "  (max |CPU| = " << maxAbsCpu << ")\n";
        std::cout << "  first 5 GPU   :  " << y[0] << "  " << y[1]
                  << "  " << y[2] << "  " << y[3] << "  " << y[4] << "\n";
        std::cout << "  first 5 CPU   :  " << cpuRef[0] << "  " << cpuRef[1]
                  << "  " << cpuRef[2] << "  " << cpuRef[3] << "  "
                  << cpuRef[4] << "\n";

        MM_LOG_INFO("main",
                    "[M5] gpu={:.3f}ms cpu={:.3f}ms max_diff={:.3e} max_cpu={:.3e}",
                    gpuMs, cpuMs,
                    static_cast<double>(maxAbsDiff),
                    static_cast<double>(maxAbsCpu));
    } catch (const std::exception& e) {
        std::cout << "  GPU RMSNorm failed: " << e.what() << "\n";
        MM_LOG_ERROR("main", "[M5] failed: {}", e.what());
    }
}

// ---- M3 model summary + tokenizer round-trip --------------------------------

void printM3Summary(const mimirmind::runtime::InferenceEngine& engine) {
    const auto& reader = engine.reader();
    const auto& config = engine.config();
    const auto& tok    = engine.tokenizer();

    std::cout << "  version       : " << reader.version() << "\n";
    std::cout << "  metadata      : " << reader.metadataCount() << " entries\n";
    std::cout << "  tensors       : " << reader.tensorCount() << " entries\n";
    std::cout << "  alignment     : " << reader.alignment() << " bytes\n";
    std::cout << "  data offset   : " << reader.tensorDataOffset() << " bytes\n";
    std::cout << "  payload total : " << formatBytes(reader.totalTensorBytes()) << "\n";

    static constexpr std::array<const char*, 9> kProbeKeys = {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "llama.context_length",
        "gemma.context_length",
        "gemma3.context_length",
        "gemma4.context_length",
    };
    std::cout << "  known meta    :\n";
    for (const char* k : kProbeKeys) {
        if (reader.findMetadata(k) != nullptr) {
            std::cout << "    " << k << "\n";
        }
    }

    std::cout << "  first tensors :\n";
    const auto& ts = reader.tensors();
    for (std::size_t i = 0; i < std::min<std::size_t>(3, ts.size()); ++i) {
        const auto& t = ts[i];
        std::cout << "    " << t.name
                  << "  type=" << mimirmind::core::gguf::typeInfo(t.type).name
                  << "  elems=" << t.nelements
                  << "  bytes=" << formatBytes(t.nbytes) << "\n";
    }

    std::cout << "  config:\n";
    std::cout << "    architecture     : " << config.architecture << "\n";
    std::cout << "    blocks           : " << config.blockCount << "\n";
    std::cout << "    context_length   : " << config.contextLength << "\n";
    std::cout << "    embedding_length : " << config.embeddingLength << "\n";
    std::cout << "    ffn_length       : " << config.feedForwardLength << "\n";
    std::cout << "    heads            : " << config.headCount
              << " (kv " << config.headCountKv
              << ", head_dim " << config.headDim() << ")\n";
    std::cout << "    rope_freq_base   : " << config.ropeFreqBase << "\n";
    std::cout << "    rms_norm_eps     : " << config.rmsNormEps << "\n";
    if (config.slidingWindow > 0) {
        std::cout << "    sliding_window   : " << config.slidingWindow << "\n";
    }

    std::cout << "  tokenizer:\n";
    std::cout << "    model      : " << tok.modelType() << "\n";
    std::cout << "    vocab_size : " << tok.vocabSize() << "\n";
    std::cout << "    bos/eos    : " << tok.bosId() << " / " << tok.eosId() << "\n";
    std::cout << "    unk/pad    : " << tok.unknownId() << " / " << tok.padId() << "\n";

    const std::string sample = "Hello, world!";
    const auto ids = tok.encode(sample, true);
    std::cout << "    encode('" << sample << "') = [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "] (" << ids.size() << " tokens)\n";

    std::cout << "    pieces     = [";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "'" << tok.tokenText(ids[i]) << "'";
    }
    std::cout << "]\n";

    const std::string round = tok.decode(ids, true);
    std::cout << "    decode     = '" << round << "'\n";

    const auto st3 = engine.allocator().stats();
    std::cout << "  loaded        : " << reader.tensorCount()
              << " tensors, " << formatBytes(st3.liveBytes) << " live in USM\n";
    std::cout << "  ze alloc/free : " << st3.zeAllocCalls
              << " / " << st3.zeFreeCalls << "\n";
}

// ---- M5b GPU Q4_K matvec parity --------------------------------------------

void runM5bQ4KParity(mimirmind::core::l0::L0Context&    ctx,
                     mimirmind::core::l0::UsmAllocator& allocator,
                     const mimirmind::core::gguf::WeightsMap& weights) {
    std::cout << "\n[M5b] GPU Q4_K matvec kernel parity\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M5b] starting GPU Q4_K matvec parity test");
    try {
        const auto* qW = weights.find("blk.0.attn_q.weight");
        if (qW == nullptr ||
            qW->type != mimirmind::core::gguf::GgmlType::Q4_K ||
            qW->dimensions.size() < 2) {
            std::cout << "  blk.0.attn_q.weight not Q4_K or missing — skipping\n";
            return;
        }
        const std::size_t K = qW->dimensions[0];
        const std::size_t N = qW->dimensions[1];
        std::cout << "  weight        : " << qW->name
                  << "  type=Q4_K  K=" << K << "  N=" << N << "\n";

        mimirmind::runtime::GpuModule modMm{ctx, "matmul_q4k_vec"};
        mimirmind::runtime::GpuKernel knMm{modMm.kernel("matmul_q4k_vec")};
        mimirmind::runtime::CommandQueue queueMm{ctx};

        const std::size_t xBytes = K * sizeof(float);
        const std::size_t yBytes = N * sizeof(float);
        mimirmind::core::l0::UsmHandle xH{allocator, xBytes};
        mimirmind::core::l0::UsmHandle yH{allocator, yBytes};
        std::vector<float> yCpuRef(N);
        std::vector<float> scratch(K);

        float* x = xH.as<float>();
        for (std::size_t i = 0; i < K; ++i) {
            x[i] = std::sin(static_cast<float>(i) * 0.017F) * 0.3F;
        }

        // M5h: kernel has subgroup_size=16 inside a 64-thread workgroup
        // and emits 4 outputs per workgroup via sub_group_reduce_add.
        constexpr std::uint32_t kLocalN          = 64;
        constexpr std::uint32_t kOutputsPerGroup = 4;
        const std::uint32_t groups = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup - 1) / kOutputsPerGroup);

        knMm.setPtr(0, xH.get());
        knMm.setPtr(1, qW->usmPtr);
        knMm.setPtr(2, yH.get());
        knMm.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        knMm.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));
        knMm.setGroupSize(kLocalN, 1, 1);

        using clock = std::chrono::steady_clock;
        const auto g0 = clock::now();
        queueMm.dispatch(knMm, groups, 1, 1);
        const auto g1 = clock::now();

        const auto c0 = clock::now();
        mimirmind::compute::matmul(
            qW->type, qW->usmPtr, N, K,
            xH.as<float>(), 1,
            yCpuRef.data(), scratch.data());
        const auto c1 = clock::now();

        const float* yGpu = yH.as<float>();
        float maxDiff = 0.0F;
        float maxRef  = 0.0F;
        for (std::size_t i = 0; i < N; ++i) {
            const float d = std::fabs(yGpu[i] - yCpuRef[i]);
            if (d > maxDiff) maxDiff = d;
            const float a = std::fabs(yCpuRef[i]);
            if (a > maxRef)  maxRef = a;
        }
        const double gpuMs = std::chrono::duration<double, std::milli>(g1 - g0).count();
        const double cpuMs = std::chrono::duration<double, std::milli>(c1 - c0).count();

        std::cout << "  launch        : groups=" << groups
                  << " x local=" << kLocalN << " => " << (groups * kLocalN)
                  << " threads (N=" << N << ")\n";
        std::cout << "  gpu time      : " << gpuMs << " ms\n";
        std::cout << "  cpu time      : " << cpuMs << " ms\n";
        std::cout << "  speedup       : " << (cpuMs / gpuMs) << "x\n";
        std::cout << "  max |GPU-CPU| : " << maxDiff
                  << "  (max |CPU| = " << maxRef
                  << ", rel = "
                  << (maxRef > 0.0F ? (maxDiff / maxRef) : 0.0F) << ")\n";

        MM_LOG_INFO("main",
                    "[M5b] gpu={:.2f}ms cpu={:.2f}ms speedup={:.2f}x "
                    "max_diff={:.3e} max_cpu={:.3e}",
                    gpuMs, cpuMs, cpuMs / gpuMs,
                    static_cast<double>(maxDiff),
                    static_cast<double>(maxRef));
    } catch (const std::exception& e) {
        std::cout << "  Q4_K matmul failed: " << e.what() << "\n";
        MM_LOG_ERROR("main", "[M5b] failed: {}", e.what());
    }
}

// ---- M5c GPU Q6_K matvec parity --------------------------------------------

void runM5cQ6KParity(mimirmind::core::l0::L0Context&    ctx,
                     mimirmind::core::l0::UsmAllocator& allocator,
                     const mimirmind::core::gguf::WeightsMap& weights) {
    std::cout << "\n[M5c] GPU Q6_K matvec kernel parity (lm_head)\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M5c] starting GPU Q6_K matvec parity test");
    try {
        const auto* outW = weights.find("output.weight");
        if (outW == nullptr ||
            outW->type != mimirmind::core::gguf::GgmlType::Q6_K ||
            outW->dimensions.size() < 2) {
            std::cout << "  output.weight not Q6_K or missing — skipping\n";
            return;
        }
        const std::size_t K = outW->dimensions[0];
        const std::size_t N = outW->dimensions[1];
        std::cout << "  weight        : " << outW->name
                  << "  type=Q6_K  K=" << K << "  N=" << N << "\n";

        mimirmind::runtime::GpuModule modMm6{ctx, "matmul_q6k_vec"};
        mimirmind::runtime::GpuKernel knMm6{modMm6.kernel("matmul_q6k_vec")};
        mimirmind::runtime::CommandQueue queueMm6{ctx};

        const std::size_t xBytes6 = K * sizeof(float);
        const std::size_t yBytes6 = N * sizeof(float);
        mimirmind::core::l0::UsmHandle xH{allocator, xBytes6};
        mimirmind::core::l0::UsmHandle yH{allocator, yBytes6};
        std::vector<float> yCpu6(N);
        std::vector<float> scratch6(K);

        float* x6 = xH.as<float>();
        for (std::size_t i = 0; i < K; ++i) {
            x6[i] = std::sin(static_cast<float>(i) * 0.011F) * 0.4F;
        }

        // M5h: 4 outputs per workgroup via sub_group_reduce_add (sg=16).
        constexpr std::uint32_t kLocalN6          = 64;
        constexpr std::uint32_t kOutputsPerGroup6 = 4;
        const std::uint32_t groups6 = static_cast<std::uint32_t>(
            (N + kOutputsPerGroup6 - 1) / kOutputsPerGroup6);

        knMm6.setPtr(0, xH.get());
        knMm6.setPtr(1, outW->usmPtr);
        knMm6.setPtr(2, yH.get());
        knMm6.setValue<std::int32_t>(3, static_cast<std::int32_t>(K));
        knMm6.setValue<std::int32_t>(4, static_cast<std::int32_t>(N));
        knMm6.setGroupSize(kLocalN6, 1, 1);

        using clock = std::chrono::steady_clock;
        const auto g0 = clock::now();
        queueMm6.dispatch(knMm6, groups6, 1, 1);
        const auto g1 = clock::now();

        const auto c0 = clock::now();
        mimirmind::compute::matmul(
            outW->type, outW->usmPtr, N, K,
            xH.as<float>(), 1,
            yCpu6.data(), scratch6.data());
        const auto c1 = clock::now();

        const float* yG = yH.as<float>();
        float maxDiff6 = 0.0F;
        float maxRef6  = 0.0F;
        for (std::size_t i = 0; i < N; ++i) {
            const float d = std::fabs(yG[i] - yCpu6[i]);
            if (d > maxDiff6) maxDiff6 = d;
            const float a = std::fabs(yCpu6[i]);
            if (a > maxRef6)  maxRef6 = a;
        }
        const double gpuMs6 = std::chrono::duration<double, std::milli>(g1 - g0).count();
        const double cpuMs6 = std::chrono::duration<double, std::milli>(c1 - c0).count();

        std::cout << "  launch        : groups=" << groups6
                  << " x local=" << kLocalN6 << " => "
                  << (groups6 * kLocalN6) << " threads (N=" << N << ")\n";
        std::cout << "  gpu time      : " << gpuMs6 << " ms\n";
        std::cout << "  cpu time      : " << cpuMs6 << " ms\n";
        std::cout << "  speedup       : " << (cpuMs6 / gpuMs6) << "x\n";
        std::cout << "  max |GPU-CPU| : " << maxDiff6
                  << "  (max |CPU| = " << maxRef6
                  << ", rel = "
                  << (maxRef6 > 0.0F ? (maxDiff6 / maxRef6) : 0.0F) << ")\n";

        MM_LOG_INFO("main",
                    "[M5c] gpu={:.2f}ms cpu={:.2f}ms speedup={:.2f}x "
                    "max_diff={:.3e} max_cpu={:.3e}",
                    gpuMs6, cpuMs6, cpuMs6 / gpuMs6,
                    static_cast<double>(maxDiff6),
                    static_cast<double>(maxRef6));
    } catch (const std::exception& e) {
        std::cout << "  Q6_K matmul failed: " << e.what() << "\n";
        MM_LOG_ERROR("main", "[M5c] failed: {}", e.what());
    }
}

// ---- M4a/M4b embedding lookup + final norm + lm_head -----------------------

void runM4aEmbedAndM4bLmHead(mimirmind::runtime::InferenceEngine& engine) {
    namespace mm = mimirmind;

    const auto& weights = engine.weights();
    const auto& tok     = engine.tokenizer();
    const auto& config  = engine.config();
    auto& allocator     = engine.allocator();
    auto& gmm           = engine.gpuMatmul();

    std::cout << "\n[M4a] Embedding lookup smoke test\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M4a] starting embedding lookup");

    const auto* tokEmb = weights.find("token_embd.weight");
    if (tokEmb == nullptr) {
        tokEmb = weights.find("tok_embeddings.weight");
    }
    if (tokEmb == nullptr) {
        std::cout << "  no token-embedding tensor found, skipping\n";
        MM_LOG_WARN("main",
                    "[M4a] no token_embd.weight / tok_embeddings.weight");
        return;
    }
    std::cout << "  embed tensor  : " << tokEmb->name
              << "  type=" << mm::core::gguf::typeInfo(tokEmb->type).name
              << "  dims=[";
    for (std::size_t i = 0; i < tokEmb->dimensions.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << tokEmb->dimensions[i];
    }
    std::cout << "]\n";

    const auto sampleIds = tok.encode("Hello", false);
    std::cout << "  sample tokens : [";
    for (std::size_t i = 0; i < sampleIds.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << sampleIds[i];
    }
    std::cout << "]\n";

    const std::size_t d_model    = config.embeddingLength;
    const std::size_t vocab_size = tokEmb->dimensions.size() >= 2
                                    ? tokEmb->dimensions[1]
                                    : tok.vocabSize();
    const std::size_t seqLen     = sampleIds.size();
    if (seqLen == 0) {
        std::cout << "  encode returned 0 tokens, nothing to embed\n";
        return;
    }

    const std::size_t outBytes = seqLen * d_model * sizeof(float);
    mm::core::l0::UsmHandle embH{allocator, outBytes};
    mm::compute::embeddingLookup(
        tokEmb->type, tokEmb->usmPtr,
        d_model, vocab_size,
        sampleIds, embH.as<float>());

    const float* emb = embH.as<float>();
    std::cout << "  d_model       : " << d_model << "\n";
    std::cout << "  output bytes  : " << formatBytes(outBytes) << "\n";
    std::cout << "  first id      : " << sampleIds[0]
              << "  piece='" << tok.tokenText(sampleIds[0]) << "'\n";

    std::cout << "  first 10 f32  :";
    std::cout << std::setprecision(6) << std::fixed;
    for (std::size_t i = 0; i < std::min<std::size_t>(10, d_model); ++i) {
        std::cout << "  " << emb[i];
    }
    std::cout << "\n";
    std::cout.unsetf(std::ios::floatfield);

    double sumSq = 0.0;
    float  vMin  = emb[0];
    float  vMax  = emb[0];
    for (std::size_t i = 0; i < d_model; ++i) {
        const float v = emb[i];
        sumSq += static_cast<double>(v) * static_cast<double>(v);
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
    }
    std::cout << "  L2 norm       : " << std::sqrt(sumSq) << "\n";
    std::cout << "  min / max     : " << vMin << " / " << vMax << "\n";

    MM_LOG_INFO("main",
                "[M4a] first token id={} L2={:.6f} min={:.6f} max={:.6f}",
                sampleIds[0], std::sqrt(sumSq),
                static_cast<double>(vMin),
                static_cast<double>(vMax));

    // -- M4b: final norm + lm_head -> top-5 ----------------------------------

    std::cout << "\n[M4b] Final norm + lm_head matmul -> top-5\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M4b] starting final-norm + lm_head");

    const auto* outNorm = weights.find("output_norm.weight");
    const auto* lmHead  = weights.find("output.weight");
    if (lmHead == nullptr) {
        lmHead = weights.find("token_embd.weight");
    }

    if (outNorm == nullptr) {
        std::cout << "  output_norm.weight missing, skipping\n";
        return;
    }
    if (lmHead == nullptr) {
        std::cout << "  no lm_head tensor found, skipping\n";
        return;
    }

    const std::size_t vocab_lm = lmHead->dimensions.size() >= 2
        ? lmHead->dimensions[1] : vocab_size;

    std::cout << "  output_norm   : " << outNorm->name
              << "  type=" << mm::core::gguf::typeInfo(outNorm->type).name << "\n";
    std::cout << "  lm_head       : " << lmHead->name
              << "  type=" << mm::core::gguf::typeInfo(lmHead->type).name
              << "  dims=[" << lmHead->dimensions[0]
              << "," << vocab_lm << "]\n";

    const std::size_t normBytes    = d_model * sizeof(float);
    const std::size_t logitsBytes  = vocab_lm * sizeof(float);
    const std::size_t scratchBytes = d_model * sizeof(float);

    mm::core::l0::UsmHandle normedH {allocator, normBytes};
    mm::core::l0::UsmHandle logitsH {allocator, logitsBytes};
    mm::core::l0::UsmHandle scratchH{allocator, scratchBytes};

    if (outNorm->type != mm::core::gguf::GgmlType::F32) {
        std::cout << "  (output_norm weight is "
                  << mm::core::gguf::typeInfo(outNorm->type).name
                  << ", not F32 — unsupported for now, skipping)\n";
        return;
    }
    const auto* normWeight = static_cast<const float*>(outNorm->usmPtr);

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    mm::compute::rmsNorm(
        emb, seqLen, d_model,
        normWeight, config.rmsNormEps,
        normedH.as<float>());
    auto t1 = clock::now();
    gmm.matmul(
        lmHead->type, lmHead->usmPtr,
        vocab_lm, d_model,
        normedH.as<float>(), seqLen,
        logitsH.as<float>(),
        scratchH.as<float>());
    auto t2 = clock::now();

    const double normMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mmMs   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "  rmsNorm time  : " << normMs << " ms\n";
    std::cout << "  matmul time   : " << mmMs   << " ms\n";

    const float* logitsRow = logitsH.as<float>()
                            + (seqLen - 1) * vocab_lm;

    std::vector<std::int32_t> idx(vocab_lm);
    std::iota(idx.begin(), idx.end(), 0);
    const std::size_t topK = std::min<std::size_t>(5, vocab_lm);
    std::partial_sort(
        idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(topK),
        idx.end(),
        [logitsRow](std::int32_t a, std::int32_t b) {
            return logitsRow[a] > logitsRow[b];
        });

    std::cout << "  top-5 (no attention/FFN — embed -> norm -> lm_head):\n";
    for (std::size_t i = 0; i < topK; ++i) {
        const std::int32_t v = idx[i];
        std::cout << "    " << i << ": id=" << v
                  << "  logit=" << logitsRow[v]
                  << "  piece='" << tok.tokenText(v) << "'\n";
    }

    MM_LOG_INFO("main",
                "[M4b] norm={:.1f}ms matmul={:.1f}ms top1_id={} logit={:.4f}",
                normMs, mmMs,
                idx[0], static_cast<double>(logitsRow[idx[0]]));
}

#endif // MIMIRMIND_HAVE_L0

// ---- Backend-neutral diagnostics (M7c + M4d/e) ------------------------------
//
// Below this line: functions that only touch the neutral
// `engine.tokenizer()` / `engine.config()` / `engine.generate()` surface
// and stay compiled regardless of backend.

// ---- M7c chat-template smoke -----------------------------------------------

void runM7cChatTemplate(const mimirmind::runtime::InferenceEngine& engine) {
    namespace mm = mimirmind;

    std::cout << "\n[M7c] Chat template (Qwen ChatML)\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M7c] starting chat-template diagnostics");

    const auto& tok    = engine.tokenizer();
    const auto& reader = engine.reader();

    // Log the embedded Jinja chat_template (first ~160 chars) for
    // divergence-spotting. We do NOT interpret it — M7c hardcodes Qwen.
    if (const auto* v = reader.findMetadata("tokenizer.chat_template")) {
        if (std::holds_alternative<std::string>(*v)) {
            const auto& s = std::get<std::string>(*v);
            std::cout << "  embedded jinja (first 160 chars):\n    "
                      << s.substr(0, 160) << "\n";
        }
    } else {
        std::cout << "  embedded jinja: <none in GGUF>\n";
    }

    mm::model::ChatTemplate::Style style;
    try {
        style = mm::model::ChatTemplate::detectFromArch(
            engine.config().architecture);
    } catch (const std::exception& e) {
        std::cout << "  detect: " << e.what() << "\n";
        MM_LOG_WARN("main", "[M7c] detect failed: {}", e.what());
        return;
    }
    std::cout << "  style    : Qwen ChatML (arch="
              << engine.config().architecture << ")\n";

    const auto imStart = tok.findToken("<|im_start|>");
    const auto imEnd   = tok.findToken("<|im_end|>");
    std::cout << "  specials : <|im_start|>=" << imStart
              << "  <|im_end|>=" << imEnd << "\n";
    if (imStart < 0 || imEnd < 0) {
        std::cout << "  -> Qwen specials missing from vocab; chat won't work.\n";
        MM_LOG_WARN("main",
                    "[M7c] Qwen special tokens not in vocab "
                    "(im_start={}, im_end={})", imStart, imEnd);
        return;
    }

    std::vector<mm::model::ChatMessage> sample{
        {mm::model::ChatRole::User, "Hello, world!"},
    };
    const auto ids = mm::model::ChatTemplate::encode(
        style, tok, sample, /*addGenerationPrompt=*/true);

    std::cout << "  rendered : " << ids.size() << " tokens, first 24:\n   ";
    for (std::size_t i = 0; i < std::min<std::size_t>(24, ids.size()); ++i) {
        std::cout << " " << ids[i];
    }
    std::cout << "\n";
    std::cout << "  decode (with specials): '"
              << tok.decode(ids, /*skipSpecial=*/false) << "'\n";

    const auto stops = mm::model::ChatTemplate::stopIds(style, tok);
    std::cout << "  stop ids : [";
    for (std::size_t i = 0; i < stops.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << stops[i];
    }
    std::cout << "] (in addition to eos=" << tok.eosId() << ")\n";

    MM_LOG_INFO("main",
                "[M7c] qwen template rendered {} tokens, stop_ids[0]={}, eos={}",
                ids.size(),
                stops.empty() ? -1 : stops[0],
                tok.eosId());
}

// ---- M4d/M4e prefill + autoregressive generation ---------------------------

void runM4deGenerate(mimirmind::runtime::InferenceEngine& engine,
                     const mimirmind::cli::CliArgs&       args) {
    namespace mm = mimirmind;

    std::cout << "\n[M4d/M4e] Prefill + autoregressive generation\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M4d/M4e] starting full forward + decode");

    const auto& tok = engine.tokenizer();

    std::vector<std::int32_t>  promptIds;
    std::vector<std::int32_t>  chatStops;
    std::string                displayPrompt = args.prompt;

    if (args.chat) {
        mm::model::ChatTemplate::Style style;
        try {
            style = mm::model::ChatTemplate::detectFromArch(
                engine.config().architecture);
        } catch (const std::exception& e) {
            std::cout << "  --chat: " << e.what() << "\n";
            MM_LOG_ERROR("main", "[M4d/M4e] chat template detect failed: {}",
                         e.what());
            return;
        }

        std::vector<mm::model::ChatMessage> messages;
        if (!args.systemMessage.empty()) {
            messages.push_back({mm::model::ChatRole::System, args.systemMessage});
        }
        messages.push_back({mm::model::ChatRole::User, args.prompt});

        promptIds = mm::model::ChatTemplate::encode(
            style, tok, messages, /*addGenerationPrompt=*/true);
        chatStops = mm::model::ChatTemplate::stopIds(style, tok);
        displayPrompt = tok.decode(promptIds, /*skipSpecial=*/false);
    } else {
        promptIds = tok.encode(args.prompt, /*addBos=*/false);
        displayPrompt = tok.decode(promptIds, /*skipSpecial=*/false);
    }

    if (promptIds.empty()) {
        std::cout << "  empty prompt, skipping\n";
        return;
    }

    std::cout << "  prompt   : \"" << args.prompt
              << "\" -> " << promptIds.size() << " tokens (max_gen="
              << args.maxNew << ")\n";
    if (args.chat) {
        std::cout << "  chat     : on (stop ids beyond EOS: " << chatStops.size()
                  << ")\n";
    }
    if (args.temperature > 0.0F) {
        std::cout << "  sampling : temp=" << args.temperature
                  << " top_k=" << args.topK
                  << " top_p=" << args.topP
                  << " seed=" << args.seed << "\n";
    } else {
        std::cout << "  sampling : greedy (argmax)\n";
    }
    std::cout << "  text     : '" << displayPrompt << "' >>>" << std::flush;

    mm::runtime::GenerateParams params{};
    params.maxNewTokens         = args.maxNew;
    params.sampling.temperature = args.temperature;
    params.sampling.topK        = args.topK;
    params.sampling.topP        = args.topP;
    params.sampling.seed        = args.seed;
    params.stopIds              = chatStops;

    auto onToken = [&](std::int32_t id) -> bool {
        std::cout << tok.decode(std::span<const std::int32_t>(&id, 1), true)
                  << std::flush;
        return true;
    };

    mm::runtime::GenerateStats stats{};
    try {
        engine.generate(promptIds, params, onToken, &stats);
    } catch (const std::exception& e) {
        std::cout << "<<<\n  generate failed: " << e.what() << "\n";
        MM_LOG_ERROR("main", "[M4d/M4e] failed: {}", e.what());
        return;
    }

    const double perTok = stats.generatedTokens > 1
        ? stats.decodeMs / static_cast<double>(stats.generatedTokens - 1)
        : 0.0;

    std::cout << "<<<\n";
    std::cout << "  generated: " << stats.generatedTokens << " token(s)"
              << (stats.hitStop ? " (hit stop)" : "") << "\n";
    std::cout << "  prefill  : " << stats.prefillMs << " ms ("
              << stats.promptTokens << " tokens, "
              << engine.config().blockCount << " blocks)\n";
    std::cout << "  decode   : " << stats.decodeMs << " ms ("
              << perTok << " ms/token avg)\n";

    MM_LOG_INFO("main",
                "[M4d/M4e] prefill={:.1f}ms decode={:.1f}ms "
                "({} new tokens, {:.1f}ms/token)",
                stats.prefillMs, stats.decodeMs,
                stats.generatedTokens, perTok);
}

} // namespace mimirmind::diagnostics
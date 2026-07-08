#include "compute/Embedding.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "model/ChatTemplate.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"
#include "runtime/FanController.hpp"
#include "runtime/GpuClockGovernor.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/PerfRegressionDetector.hpp"
#include "runtime/PowerMonitor.hpp"
#include "runtime/SystemMonitor.hpp"
#include "runtime/ThermalGuard.hpp"
#include "runtime/ThermalProfile.hpp"
#include "runtime/UsmAllocator.hpp"
#include "runtime/UsmHandle.hpp"
#include "server/ApiServer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr const char* kBanner =
    "+------------------------------------------------------------+\n"
    "|                          Mimirmind                         |\n"
    "|   M1-M5 GPU matmul + M7a InferenceEngine (Envoy -> Mimir)  |\n"
    "+------------------------------------------------------------+\n";

constexpr const char* kUsage =
    "Usage:\n"
    "  mimirmind [smoke|serve|parity] [options]\n"
    "\n"
    "Modes:\n"
    "  smoke              Run M1-M5 diagnostics + end-to-end generate (default)\n"
    "  serve              Start the OpenAI-compatible HTTP server (M7d, stub)\n"
    "  parity             Tensor-parity test: run llama.cpp + mimirmind on\n"
    "                     the same prompt, dump per-block hidden state, diff\n"
    "\n"
    "Options:\n"
    "  --model PATH       GGUF model file (overrides MIMIRMIND_MODEL_PATH)\n"
    "  --port N           HTTP port for serve mode (default 8080)\n"
    "  --prompt TEXT      Prompt for smoke generate (default \"Hello, world!\")\n"
    "  --max-new N        Max new tokens for smoke generate (default 20)\n"
    "  --temperature F    Sampling temperature, 0 = greedy (default 0)\n"
    "  --top-k N          Top-K cutoff, 0 = disabled (default 0)\n"
    "  --top-p F          Top-P (nucleus) cutoff, 1.0 = disabled (default 1.0)\n"
    "  --seed N           RNG seed, 0 = random_device (default 0)\n"
    "  --chat             Wrap --prompt as a user message via the chat template\n"
    "  --system TEXT      System message for --chat (default: Qwen2.5 default)\n"
    "  --thermal-profile PATH\n"
    "                     JSON profile with per-host throttle limits\n"
    "                     (or set MIMIRMIND_THERMAL_PROFILE). Without it the\n"
    "                     engine runs unprotected — only set this for serve.\n"
    "  -h, --help         Show this help and exit\n";

enum class Mode {
    Smoke,
    Serve,
    Parity,
};

struct CliArgs {
    Mode          mode{Mode::Smoke};
    std::string   modelPath;
    std::string   prompt{"Hello, world!"};
    std::size_t   maxNew{20};
    std::uint16_t port{8080};
    float         temperature{0.0F};
    std::size_t   topK{0};
    float         topP{1.0F};
    std::uint64_t seed{0};
    bool          chat{false};
    std::string   systemMessage{};
    std::string   thermalProfilePath{};
};

[[nodiscard]] bool parseArgs(int argc, char** argv, CliArgs& out) {
    int idx = 1;

    if (argc > 1 && argv[1][0] != '-') {
        const std::string_view m{argv[1]};
        if (m == "smoke") {
            out.mode = Mode::Smoke;
        } else if (m == "serve") {
            out.mode = Mode::Serve;
        } else if (m == "parity") {
            out.mode = Mode::Parity;
        } else if (m == "-h" || m == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else {
            std::cerr << "unknown mode '" << m << "'\n" << kUsage;
            return false;
        }
        idx = 2;
    }

    auto needValue = [&](const char* flag) -> const char* {
        if (idx + 1 >= argc) {
            std::cerr << flag << " requires a value\n";
            return nullptr;
        }
        return argv[++idx];
    };

    for (; idx < argc; ++idx) {
        const std::string_view a{argv[idx]};
        if (a == "-h" || a == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else if (a == "--model") {
            const char* v = needValue("--model");
            if (v == nullptr) return false;
            out.modelPath = v;
        } else if (a == "--prompt") {
            const char* v = needValue("--prompt");
            if (v == nullptr) return false;
            out.prompt = v;
        } else if (a == "--max-new") {
            const char* v = needValue("--max-new");
            if (v == nullptr) return false;
            out.maxNew = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--port") {
            const char* v = needValue("--port");
            if (v == nullptr) return false;
            out.port = static_cast<std::uint16_t>(std::strtoul(v, nullptr, 10));
        } else if (a == "--temperature") {
            const char* v = needValue("--temperature");
            if (v == nullptr) return false;
            out.temperature = std::strtof(v, nullptr);
        } else if (a == "--top-k") {
            const char* v = needValue("--top-k");
            if (v == nullptr) return false;
            out.topK = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--top-p") {
            const char* v = needValue("--top-p");
            if (v == nullptr) return false;
            out.topP = std::strtof(v, nullptr);
        } else if (a == "--seed") {
            const char* v = needValue("--seed");
            if (v == nullptr) return false;
            out.seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--chat") {
            out.chat = true;
        } else if (a == "--system") {
            const char* v = needValue("--system");
            if (v == nullptr) return false;
            out.systemMessage = v;
        } else if (a == "--thermal-profile") {
            const char* v = needValue("--thermal-profile");
            if (v == nullptr) return false;
            out.thermalProfilePath = v;
        } else {
            std::cerr << "unknown argument '" << a << "'\n" << kUsage;
            return false;
        }
    }

    if (out.modelPath.empty()) {
        if (const char* env = std::getenv("MIMIRMIND_MODEL_PATH")) {
            out.modelPath = env;
        }
    }

    return true;
}

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

// ---- M1+M2 device + USM probe summary ---------------------------------------

void printM1M2(mimirmind::runtime::InferenceEngine& engine) {
    using mimirmind::runtime::L0Context;

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

void runM2bAllocatorSmoke(mimirmind::runtime::UsmAllocator& allocator) {
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
}

// ---- M5 GPU RMSNorm parity --------------------------------------------------

void runM5RmsNormParity(mimirmind::runtime::L0Context&    ctx,
                        mimirmind::runtime::UsmAllocator& allocator) {
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

        mimirmind::runtime::UsmHandle xH{allocator, bytesK};
        mimirmind::runtime::UsmHandle wH{allocator, bytesK};
        mimirmind::runtime::UsmHandle yH{allocator, bytesK};

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
                  << "  type=" << mimirmind::model::typeInfo(t.type).name
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

void runM5bQ4KParity(mimirmind::runtime::L0Context&    ctx,
                     mimirmind::runtime::UsmAllocator& allocator,
                     const mimirmind::model::WeightsMap& weights) {
    std::cout << "\n[M5b] GPU Q4_K matvec kernel parity\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M5b] starting GPU Q4_K matvec parity test");
    try {
        const auto* qW = weights.find("blk.0.attn_q.weight");
        if (qW == nullptr ||
            qW->type != mimirmind::model::GgmlType::Q4_K ||
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
        mimirmind::runtime::UsmHandle xH{allocator, xBytes};
        mimirmind::runtime::UsmHandle yH{allocator, yBytes};
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

void runM5cQ6KParity(mimirmind::runtime::L0Context&    ctx,
                     mimirmind::runtime::UsmAllocator& allocator,
                     const mimirmind::model::WeightsMap& weights) {
    std::cout << "\n[M5c] GPU Q6_K matvec kernel parity (lm_head)\n";
    std::cout.flush();
    MM_LOG_INFO("main", "[M5c] starting GPU Q6_K matvec parity test");
    try {
        const auto* outW = weights.find("output.weight");
        if (outW == nullptr ||
            outW->type != mimirmind::model::GgmlType::Q6_K ||
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
        mimirmind::runtime::UsmHandle xH{allocator, xBytes6};
        mimirmind::runtime::UsmHandle yH{allocator, yBytes6};
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
              << "  type=" << mm::model::typeInfo(tokEmb->type).name
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
    mm::runtime::UsmHandle embH{allocator, outBytes};
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
              << "  type=" << mm::model::typeInfo(outNorm->type).name << "\n";
    std::cout << "  lm_head       : " << lmHead->name
              << "  type=" << mm::model::typeInfo(lmHead->type).name
              << "  dims=[" << lmHead->dimensions[0]
              << "," << vocab_lm << "]\n";

    const std::size_t normBytes    = d_model * sizeof(float);
    const std::size_t logitsBytes  = vocab_lm * sizeof(float);
    const std::size_t scratchBytes = d_model * sizeof(float);

    mm::runtime::UsmHandle normedH {allocator, normBytes};
    mm::runtime::UsmHandle logitsH {allocator, logitsBytes};
    mm::runtime::UsmHandle scratchH{allocator, scratchBytes};

    if (outNorm->type != mm::model::GgmlType::F32) {
        std::cout << "  (output_norm weight is "
                  << mm::model::typeInfo(outNorm->type).name
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
                     const CliArgs&                       args) {
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

// ---- Smoke driver ----------------------------------------------------------

int runSmoke(const CliArgs& args) {
    std::cout << kBanner;
    std::cout.flush();

    MM_LOG_INFO("main", "mimirmind smoke starting (M1-M5 + engine.generate)");

    mimirmind::runtime::InferenceEngine engine;

    printM1M2(engine);
    runM2bAllocatorSmoke(engine.allocator());
    runM5RmsNormParity(engine.ctx(), engine.allocator());

    if (args.modelPath.empty()) {
        std::cout << "\n[M3] GGUF reader — skipped "
                     "(pass --model PATH or set MIMIRMIND_MODEL_PATH)\n";
        MM_LOG_INFO("main", "[M3] no model path; skipping load");
    } else {
        std::cout << "\n[M3] GGUF reader (" << args.modelPath << ")\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M3] opening model '{}'", args.modelPath);

        engine.loadModel(args.modelPath);
        printM3Summary(engine);

        runM5bQ4KParity(engine.ctx(), engine.allocator(), engine.weights());
        runM5cQ6KParity(engine.ctx(), engine.allocator(), engine.weights());

        runM4aEmbedAndM4bLmHead(engine);

        runM7cChatTemplate(engine);

        runM4deGenerate(engine, args);
    }

    const auto lim = engine.allocator().limits();
    const auto st  = engine.allocator().stats();
    MM_LOG_INFO("main",
                "smoke test passed — perAllocMax={} bytes totalAllocatable={} bytes "
                "freeListHits={} / {} totalAllocs",
                lim.perAllocMaxBytes,
                lim.totalAllocatableBytes,
                st.freeListHits,
                st.totalAllocations);
    std::cout << "\nProject Well + Envoy smoke test passed.\n";
    return 0;
}

// ---- Serve driver (M7d) ----------------------------------------------------

// Held in the SIGINT handler so a Ctrl-C asks the listener to drain.
std::atomic<mimirmind::server::ApiServer*> g_runningServer{nullptr};

extern "C" void signalStop(int /*sig*/) {
    if (auto* s = g_runningServer.load(std::memory_order_acquire)) {
        s->stop();
    }
}

int runServe(const CliArgs& args) {
    std::cout << kBanner;
    std::cout.flush();

    if (args.modelPath.empty()) {
        std::cerr << "serve: --model PATH is required (or set MIMIRMIND_MODEL_PATH)\n";
        return 2;
    }

    MM_LOG_INFO("main", "serve: loading model '{}'", args.modelPath);
    mimirmind::runtime::InferenceEngine engine;
    engine.loadModel(args.modelPath);

    // Refuse architectures without a working forward path so the user
    // finds out at serve startup, not on the first chat request.
    // Qwen2 has been production since M7; Gemma 4 is in M8 verification
    // (forward runs end-to-end but generation quality is still being
    // dialled in — see Memory/mimirmind/research/m8-gemma4-staging.md).
    const auto& arch = engine.config().architecture;
    if (arch != "qwen2" && arch != "gemma4") {
        const std::string msg =
            "serve: architecture '" + arch +
            "' is not implemented yet. The model loaded fine and the "
            "tokenizer is ready, but the transformer block needs "
            "architecture-specific code. See "
            "Memory/mimirmind/research/m8-gemma4-staging.md.";
        MM_LOG_ERROR("main", "{}", msg);
        std::cerr << msg << "\n";
        return 2;
    }
    // Optional KV-cache pre-allocation size. The cache is sized once
    // and never reallocated for request growth, so this is the upper
    // bound on (prompt + max_new) any single generate() can hold.
    // Larger = bigger upfront memory commitment but multi-turn prefix
    // reuse keeps working as conversations grow.
    if (const char* env = std::getenv("MIMIRMIND_MAX_CONTEXT_TOKENS")) {
        if (const auto n = std::strtoull(env, nullptr, 10); n > 0) {
            engine.setMaxContextTokens(static_cast<std::size_t>(n));
        }
    }

    // M10.2 Phase 0 / 1a — KV cache element dtype. Default F32
    // (bit-identical to pre-M10.2). `fp16` opts in to the 2× bandwidth
    // / RAM win; `q8_0` layers on the block-quantised 4× win once the
    // Q8_0 kernels land (M10.2.0 Commits 3-5). Other values log a
    // warning and fall back to F32. Must be set before the first
    // generate() so the lazy KvCache construction picks it up.
    if (const char* env = std::getenv("MIMIRMIND_KV_DTYPE")) {
        const std::string_view v{env};
        if (v == "fp16") {
            engine.setKvDtype(mimirmind::runtime::KvDtype::FP16);
        } else if (v == "q8_0") {
            engine.setKvDtype(mimirmind::runtime::KvDtype::Q8_0);
        } else if (v == "f32" || v.empty()) {
            engine.setKvDtype(mimirmind::runtime::KvDtype::F32);
        } else {
            MM_LOG_WARN("main",
                        "MIMIRMIND_KV_DTYPE='{}' unrecognised — falling "
                        "back to f32", env);
        }
    }
    {
        // M10.2 Phase 1a — for uniform dtypes the "B/elem" wording
        // still fits (block_elements=1); Q8_0 has 32-element blocks
        // so the log line reports the block footprint instead.
        const auto d = engine.kvDtype();
        const char* dName = (d == mimirmind::runtime::KvDtype::FP16 ? "fp16"
                           : d == mimirmind::runtime::KvDtype::Q8_0 ? "q8_0"
                                                                    : "f32");
        MM_LOG_INFO("main",
                    "KV cache dtype: {} (block {} B × {} elem)",
                    dName,
                    mimirmind::runtime::kvBlockBytes(d),
                    mimirmind::runtime::kvBlockElements(d));
    }

    // M9.11.1 — Optional speculative-decoding draft engine. Opt-in via
    // MIMIRMIND_DRAFT_MODEL_PATH. Fully independent InferenceEngine so
    // the target's L0 context / autotune / KV cache stay untouched;
    // costs a second autotune pass at startup (~30 s) plus draft
    // weights in USM. The speculation loop that actually calls the
    // draft lands in M9.11.2+ — this step only loads and vocab-checks.
    std::unique_ptr<mimirmind::runtime::InferenceEngine> draftEngine;
    if (const char* draftPathEnv = std::getenv("MIMIRMIND_DRAFT_MODEL_PATH");
        draftPathEnv != nullptr && draftPathEnv[0] != '\0') {
        MM_LOG_INFO("main",
                    "serve: loading draft model '{}'", draftPathEnv);
        try {
            draftEngine = std::make_unique<mimirmind::runtime::InferenceEngine>();
            draftEngine->loadModel(draftPathEnv);

            // Vocab compatibility. Modified rejection sampling only
            // works when draft token-id N and target token-id N mean
            // the same subword. vocabSize alone doesn't guarantee it,
            // but a mismatch there is a hard disqualification. bos/eos
            // must match too because we replay the same prompt-id stream
            // through both engines.
            const auto& tTok = engine.tokenizer();
            const auto& dTok = draftEngine->tokenizer();
            const bool sizeMatch = tTok.vocabSize() == dTok.vocabSize();
            const bool bosMatch  = tTok.bosId()     == dTok.bosId();
            const bool eosMatch  = tTok.eosId()     == dTok.eosId();
            if (!sizeMatch || !bosMatch || !eosMatch) {
                MM_LOG_WARN("main",
                            "serve: draft model vocab incompatible with "
                            "target — disabling speculative decoding. "
                            "target(vocab={}, bos={}, eos={}) vs "
                            "draft(vocab={}, bos={}, eos={})",
                            tTok.vocabSize(), tTok.bosId(), tTok.eosId(),
                            dTok.vocabSize(), dTok.bosId(), dTok.eosId());
                draftEngine.reset();
            } else {
                MM_LOG_INFO("main",
                            "serve: speculative decoding ready — "
                            "target arch={} d_model={}, draft arch={} d_model={} "
                            "(shared vocab_size={}, bos={}, eos={})",
                            engine.config().architecture,
                            engine.config().embeddingLength,
                            draftEngine->config().architecture,
                            draftEngine->config().embeddingLength,
                            tTok.vocabSize(), tTok.bosId(), tTok.eosId());
            }
        } catch (const std::exception& e) {
            MM_LOG_WARN("main",
                        "serve: draft model load failed ({}) — "
                        "speculative decoding disabled", e.what());
            draftEngine.reset();
        }
    }

    mimirmind::server::ServerConfig cfg{};
    cfg.host    = "0.0.0.0";
    cfg.port    = args.port;
    cfg.modelId = std::filesystem::path{args.modelPath}.stem().string();

    if (const char* env = std::getenv("MIMIRMIND_PRESERVE_THINKING")) {
        std::string_view v{env};
        if (!v.empty() && v != "0" && v != "false" && v != "off") {
            cfg.preserveThinking = true;
        }
    }

    // Resolve thermal profile path from CLI > env. Empty means
    // unprotected — we'll log a loud warning below.
    std::string thermalProfilePath = args.thermalProfilePath;
    if (thermalProfilePath.empty()) {
        if (const char* env = std::getenv("MIMIRMIND_THERMAL_PROFILE")) {
            thermalProfilePath = env;
        }
    }

    std::unique_ptr<mimirmind::runtime::SystemMonitor> monitor;
    std::unique_ptr<mimirmind::runtime::ThermalGuard>  guard;
    if (!thermalProfilePath.empty()) {
        mimirmind::runtime::ThermalProfile profile;
        try {
            profile = mimirmind::runtime::loadThermalProfile(thermalProfilePath);
        } catch (const std::exception& e) {
            std::cerr << "serve: failed to load thermal profile: "
                      << e.what() << "\n";
            return 1;
        }
        try {
            monitor = std::make_unique<mimirmind::runtime::SystemMonitor>(
                /*requirePackageTemp=*/profile.hasPackageLimits(),
                /*requireRam=*/        false);
        } catch (const std::exception& e) {
            std::cerr << "serve: profile '" << profile.name
                      << "' requires sensors the host does not expose: "
                      << e.what() << "\n";
            return 1;
        }
        guard = std::make_unique<mimirmind::runtime::ThermalGuard>(
            profile, *monitor);
        engine.setThermalGuard(guard.get());

        // GPU clock governor lives in the same profile (field
        // gpu_target_temp_c). If present AND the iGPU sysfs is
        // writable, we install it. Otherwise we move on without one —
        // the per-token thermal pace still runs as a safety net.
        //
        // MIMIRMIND_GPU_CLOCK_PIN pins the software cap for the whole
        // session and suppresses the P-controller tick. Meant for
        // perf-bench runs where the M9.6.5 asymmetric gains would
        // otherwise clock down aggressively and confound the
        // measurement. Package thermal safety still runs via the
        // ThermalGuard admission check + per-token pace. Do NOT ship
        // this to sustained workloads on a passively-cooled chassis.
        //
        // M9.11.a accepted values:
        //   rp0            → hardware max (RP0)
        //   rpn            → hardware min (RPn, ~800 MHz on Xe-LPG)
        //   <MHz integer>  → arbitrary cap, clamped to [RPn, RP0]
        //   unset / 0 / off / false / no → no pin (governor ticks as normal)
        enum class ClockPinIntent { None, Rp0, Rpn, Numeric };
        struct ClockPinRequest {
            ClockPinIntent intent = ClockPinIntent::None;
            std::uint32_t  mhz    = 0;
            std::string    rawEnv;
            bool           malformed = false;
        };
        const auto parseClockPin = [](const char* env) {
            ClockPinRequest req;
            if (env == nullptr || env[0] == '\0') {
                return req;
            }
            std::string_view sv{env};
            if (sv == "0" || sv == "off" || sv == "false" || sv == "no") {
                return req; // treat as unset
            }
            req.rawEnv = env;
            if (sv == "rp0" || sv == "RP0") {
                req.intent = ClockPinIntent::Rp0;
                return req;
            }
            if (sv == "rpn" || sv == "RPn" || sv == "RPN") {
                req.intent = ClockPinIntent::Rpn;
                return req;
            }
            char* end = nullptr;
            const unsigned long v = std::strtoul(env, &end, 10);
            if (end != env && *end == '\0' && v > 0 && v < 100000) {
                req.intent = ClockPinIntent::Numeric;
                req.mhz    = static_cast<std::uint32_t>(v);
                return req;
            }
            req.malformed = true;
            return req;
        };

        static std::unique_ptr<mimirmind::runtime::GpuClockGovernor> governor;
        const auto pinReq = parseClockPin(std::getenv("MIMIRMIND_GPU_CLOCK_PIN"));
        const bool clockPinRequested = pinReq.intent != ClockPinIntent::None;

        if (profile.hasGpuClockTarget()) {
            governor = std::make_unique<mimirmind::runtime::GpuClockGovernor>();
            governor->setTargetTempC(*profile.gpu_target_temp_c);
            if (!governor->available()) {
                MM_LOG_WARN("main",
                            "thermal profile asks for GPU clock governor "
                            "(gpu_target_temp_c={:.1f}) but it is not "
                            "available: {}",
                            *profile.gpu_target_temp_c,
                            governor->unavailableReason());
                governor.reset();
            } else if (clockPinRequested) {
                std::uint32_t   requestedMhz = 0;
                std::string_view intentName  = "";
                switch (pinReq.intent) {
                    case ClockPinIntent::Rp0:
                        requestedMhz = governor->rp0Mhz();
                        intentName   = "rp0";
                        break;
                    case ClockPinIntent::Rpn:
                        requestedMhz = governor->rpnMhz();
                        intentName   = "rpn";
                        break;
                    case ClockPinIntent::Numeric:
                        requestedMhz = pinReq.mhz;
                        intentName   = "numeric";
                        break;
                    case ClockPinIntent::None:
                        break;
                }
                const auto pinned = governor->pin(
                    requestedMhz, intentName, pinReq.rawEnv);
                MM_LOG_WARN("main",
                            "MIMIRMIND_GPU_CLOCK_PIN={} — cap pinned to "
                            "{} MHz (intent={}, envelope [{},{}]). "
                            "P-controller tick suppressed. Bench mode. "
                            "Thermal safety still via ThermalGuard.",
                            pinReq.rawEnv, pinned, intentName,
                            governor->rpnMhz(), governor->rp0Mhz());
                // Install the governor anyway so ApiServer can report
                // the pin state via /system/info + /system/status. The
                // engine's tick loop consults governor->pinned() and
                // skips its adjust call, so the pin survives the run.
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            } else {
                if (pinReq.malformed) {
                    MM_LOG_WARN("main",
                                "MIMIRMIND_GPU_CLOCK_PIN={} not recognised — "
                                "expected rp0 / rpn / <MHz> / 0 / off. "
                                "Installing governor as if unset.",
                                pinReq.rawEnv);
                }
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            }
        } else if (clockPinRequested) {
            MM_LOG_WARN("main",
                        "MIMIRMIND_GPU_CLOCK_PIN={} ignored — thermal "
                        "profile has no gpu_target_temp_c so no governor "
                        "was going to be installed anyway.",
                        pinReq.rawEnv);
        }

        // M9.6.6.0 tick sink. Opt-in per env — when set, the governor
        // writes one NDJSON line per tick to the file, which we then
        // consume for M9.6.6 adaptive-gain baseline analysis.
        if (governor != nullptr) {
            if (const char* tickLog =
                    std::getenv("MIMIRMIND_GOVERNOR_TICK_LOG");
                tickLog != nullptr && tickLog[0] != '\0') {
                if (governor->setTickLogPath(tickLog)) {
                    MM_LOG_INFO("main",
                                "GovernorTickSink open — writing NDJSON to '{}'",
                                tickLog);
                } else {
                    MM_LOG_WARN("main",
                                "MIMIRMIND_GOVERNOR_TICK_LOG='{}' — could "
                                "not open for append. Sink stays off.",
                                tickLog);
                }
            }
        }
    }

    // Power telemetry — always-on attempt, never fatal. If RAPL is
    // masked (Docker / unprivileged LXC without explicit mount) the
    // monitor reports unavailable and /v1/system/status shows the
    // reason; the engine still runs.
    auto powerMonitor = std::make_unique<mimirmind::runtime::PowerMonitor>();
    engine.setPowerMonitor(powerMonitor.get());

    // M9.11.b chassis fan controller. Probes /sys/class/hwmon/* at
    // construction; if a writable pwm/pwm_enable pair is found, the
    // engine boosts the fan at the start of each generate() and
    // releases to auto at the end. Original BIOS values captured for
    // RAII restore on process exit.
    //
    // Env vars:
    //   MIMIRMIND_FAN_BOOST=off        → do not install (kill switch)
    //   MIMIRMIND_FAN_PWM_BOOST=<0-255>→ override boost target
    //   MIMIRMIND_FAN_PWM_MIN=<0-255>  → override safety floor
    // Kill switch is checked first so we can disable the whole feature
    // without touching sysfs at all — useful when the BIOS refuses
    // manual mode and hwmon writes are throwing kernel warnings.
    static std::unique_ptr<mimirmind::runtime::FanController> fanController;
    {
        const char* boostEnv = std::getenv("MIMIRMIND_FAN_BOOST");
        const bool  disabled =
            boostEnv != nullptr && std::string_view{boostEnv} == "off";
        if (!disabled) {
            fanController = std::make_unique<mimirmind::runtime::FanController>();
            if (!fanController->available()) {
                MM_LOG_WARN("main",
                            "FanController unavailable — no proactive fan "
                            "boost. Reason: {}",
                            fanController->unavailableReason());
                fanController.reset();
            } else {
                if (const char* v = std::getenv("MIMIRMIND_FAN_PWM_BOOST");
                    v != nullptr && v[0] != '\0') {
                    char* end = nullptr;
                    const unsigned long parsed = std::strtoul(v, &end, 10);
                    if (end != v && *end == '\0' && parsed <= 255) {
                        fanController->setBoostPwm(
                            static_cast<std::uint8_t>(parsed));
                    }
                }
                if (const char* v = std::getenv("MIMIRMIND_FAN_PWM_MIN");
                    v != nullptr && v[0] != '\0') {
                    char* end = nullptr;
                    const unsigned long parsed = std::strtoul(v, &end, 10);
                    if (end != v && *end == '\0' && parsed <= 255) {
                        fanController->setMinSafePwm(
                            static_cast<std::uint8_t>(parsed));
                    }
                }
                MM_LOG_INFO("main",
                            "FanController ready — chip='{}' pwm='{}' "
                            "fan_input='{}' orig_pwm={} orig_enable={} "
                            "boost={} min_safe={}",
                            fanController->chipName(),
                            fanController->pwmPath(),
                            fanController->fanInputPath(),
                            fanController->originalPwm(),
                            fanController->originalEnableMode(),
                            fanController->boostPwm(),
                            fanController->minSafePwm());
                engine.setFanController(fanController.get());
            }
        }
    }

    // Thermal-safety cross-check: MIMIRMIND_GPU_CLOCK_PIN=rp0 disables
    // the P-controller entirely, so the FanController is the only
    // active thermal regulator during sustained decode. Warn loudly
    // when the operator asks for rp0 without a functioning fan-boost
    // path — this is the exact 2026-07-01 shutdown scenario.
    {
        const char* pinEnv = std::getenv("MIMIRMIND_GPU_CLOCK_PIN");
        const bool  wantsRp0 =
            pinEnv != nullptr
            && (std::string_view{pinEnv} == "rp0"
                || std::string_view{pinEnv} == "RP0");
        const bool fanActive =
            fanController != nullptr && fanController->available();
        if (wantsRp0 && !fanActive) {
            MM_LOG_WARN("main",
                        "MIMIRMIND_GPU_CLOCK_PIN=rp0 is set but the "
                        "FanController is not active — the P-controller "
                        "is disabled AND no proactive cooling is "
                        "installed. Sustained decode on a passively "
                        "cooled chassis can trigger a hardware thermal "
                        "shutdown (see 2026-07-01 incident). Consider "
                        "MIMIRMIND_GPU_CLOCK_PIN=<numeric MHz> as a "
                        "safer bench mode.");
        }
    }

    // In-process perf-regression detector. Feeds off the same per-token
    // wall-time the NDJSON sink already computes, so it costs a couple
    // of doubles per token and one median at end-of-run. Kill-switch:
    // MIMIRMIND_REGRESSION_ALERT=off skips installation entirely for the
    // case where the detector itself misbehaves and needs to be silenced
    // without a redeploy.
    std::unique_ptr<mimirmind::runtime::PerfRegressionDetector> perfDetector;
    {
        const char* alertEnv = std::getenv("MIMIRMIND_REGRESSION_ALERT");
        const bool  disabled =
            alertEnv != nullptr && std::string_view{alertEnv} == "off";
        if (!disabled) {
            std::string baselinePath;
            if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0') {
                baselinePath = std::string{h} +
                               "/.cache/mimirmind/perf-baseline.json";
            } else {
                baselinePath = "/tmp/mimirmind-perf-baseline.json";
            }
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path{baselinePath}.parent_path(), ec);
            // create_directories failure is not fatal — the detector
            // logs a warning on the first write and keeps running.
            perfDetector =
                std::make_unique<mimirmind::runtime::PerfRegressionDetector>(
                    baselinePath);
            engine.setPerfRegressionDetector(perfDetector.get());
        } else {
            MM_LOG_WARN("main",
                        "MIMIRMIND_REGRESSION_ALERT=off — perf-regression "
                        "detector not installed for this session");
        }
    }

    mimirmind::server::ApiServer server{engine, cfg, draftEngine.get()};

    g_runningServer.store(&server, std::memory_order_release);
    std::signal(SIGINT,  signalStop);
    std::signal(SIGTERM, signalStop);

    std::cout << "\n[M7d/M7e] OpenAI-compatible HTTP API listening on "
              << cfg.host << ":" << cfg.port
              << "\n  GET  /health\n"
                 "  GET  /v1/models\n"
                 "  GET  /v1/system/info\n"
                 "  GET  /v1/system/status\n"
                 "  POST /v1/chat/completions  (stream=true supported)\n"
                 "  model id:           " << cfg.modelId << "\n"
                 "  preserve-thinking:  "
              << (cfg.preserveThinking ? "on (raw deltas, KV-cache friendly)"
                                       : "off (cleaned text, channel-wrapper stripped)")
              << "\n  thermal profile:    ";
    if (guard) {
        std::cout << "'" << guard->profile().name
                  << "' (package=" << monitor->packageTempSource() << ")";
    } else {
        std::cout << "\033[1;33mNOT CONFIGURED — engine is unprotected\033[0m";
    }
    std::cout << "\n  power telemetry:    ";
    if (powerMonitor->available()) {
        std::cout << "on (" << powerMonitor->domainNames().size()
                  << " RAPL domain(s))";
    } else {
        std::cout << "off (" << powerMonitor->unavailableReason() << ")";
    }
    std::cout << "\n  gpu clock governor: ";
    if (auto* gov = engine.gpuClockGovernor()) {
        std::cout << "on (target=" << gov->targetTempC()
                  << "°C, " << gov->rpnMhz() << ".."
                  << gov->rp0Mhz() << " MHz on "
                  << gov->cardPath() << ")";
    } else {
        std::cout << "off";
    }
    std::cout << "\n  perf regression:    ";
    if (auto* det = engine.perfRegressionDetector()) {
        std::cout << "on (baseline=" << det->baselineSampleCount()
                  << " samples, threshold="
                  << mimirmind::runtime::PerfRegressionDetector::kAlertThreshold
                  << "x)";
    } else {
        std::cout << "off (MIMIRMIND_REGRESSION_ALERT=off)";
    }
    std::cout << "\n  spec decoding:      ";
    if (draftEngine != nullptr) {
        std::cout << "ready (draft arch="
                  << draftEngine->config().architecture
                  << ", d_model=" << draftEngine->config().embeddingLength
                  << ")";
    } else if (std::getenv("MIMIRMIND_DRAFT_MODEL_PATH") != nullptr) {
        std::cout << "disabled (draft load or vocab check failed — see log)";
    } else {
        std::cout << "off (set MIMIRMIND_DRAFT_MODEL_PATH to enable)";
    }
    std::cout << "\n  max context tokens: " << engine.maxContextTokens()
              << "\n  Ctrl-C to stop.\n";
    std::cout.flush();

    if (!guard) {
        MM_LOG_WARN("main",
                    "serve: no thermal profile configured. The engine will "
                    "not throttle decode on temperature/RAM limits. Pass "
                    "--thermal-profile PATH or set MIMIRMIND_THERMAL_PROFILE "
                    "to protect the host.");
    }

    try {
        server.run();
    } catch (const std::exception& e) {
        g_runningServer.store(nullptr, std::memory_order_release);
        MM_LOG_ERROR("main", "serve: {}", e.what());
        std::cerr << "serve failed: " << e.what() << "\n";
        return 1;
    }
    g_runningServer.store(nullptr, std::memory_order_release);

    MM_LOG_INFO("main", "serve: stopped cleanly");
    return 0;
}

/**
 * Tensor parity-test orchestrator. Drives three sub-steps:
 *   1. llama-parity-dump (CPU reference)   → /tmp/dumps/llama/blk{N}.bin
 *   2. mimirmind own prefill (this binary) → /tmp/dumps/mimir-blk{N}.bin
 *   3. parity-diff (Python)                → prints first divergence
 *
 * All three live in the Docker runtime image (see Dockerfile stage 2b
 * "llamacpp" + runtime COPY). Container exits with the diff's return
 * code so callers can scripts on it (0 = parity, 1 = divergence).
 */
[[nodiscard]] int runParity(const CliArgs& argsIn) {
    if (argsIn.modelPath.empty()) {
        std::cerr << "parity: --model PATH is required "
                     "(or set MIMIRMIND_MODEL_PATH)\n";
        return 2;
    }

    const std::string dumpRoot = "/tmp/dumps";
    const std::string llamaDir = dumpRoot + "/llama";
    const std::string mimirPfx = dumpRoot + "/mimir";

    // Wipe + recreate dump dirs.
    std::filesystem::remove_all(dumpRoot);
    std::filesystem::create_directories(llamaDir);

    auto shellQuote = [](const std::string& s) {
        std::string out{"'"};
        for (char c : s) {
            if (c == '\'') { out += "'\\''"; } else { out += c; }
        }
        out += "'";
        return out;
    };

    // --- 1. llama-parity-dump --------------------------------------------
    {
        std::cout << "\n=== [1/3] llama-parity-dump (reference oracle) ===\n";
        std::string cmd = "llama-parity-dump";
        cmd += " --model "    + shellQuote(argsIn.modelPath);
        cmd += " --prompt "   + shellQuote(argsIn.prompt);
        cmd += " --dump-dir " + shellQuote(llamaDir);
        cmd += " --n-predict 0";  // prefill-only, no generation
        std::cout << "$ " << cmd << "\n" << std::flush;
        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            std::cerr << "parity: llama-parity-dump exited " << rc << "\n";
            return 1;
        }
    }

    // --- 2. mimirmind own forward dump -----------------------------------
    {
        std::cout << "\n=== [2/3] mimirmind dump ===\n" << std::flush;
        setenv("MIMIRMIND_PARITY_DUMP", mimirPfx.c_str(), 1);

        mimirmind::runtime::InferenceEngine engine;
        engine.loadModel(argsIn.modelPath);

        const auto promptIds = engine.tokenizer().encode(argsIn.prompt, true);
        mimirmind::runtime::GenerateParams gp{};
        gp.maxNewTokens = 1;  // only need prefill; one token is fine
        engine.generate(std::span<const std::int32_t>{promptIds.data(),
                                                       promptIds.size()},
                        gp);
    }

    // --- 3. parity-diff ---------------------------------------------------
    {
        std::cout << "\n=== [3/3] parity-diff ===\n" << std::flush;
        const std::string cmd = "parity-diff " +
            shellQuote(llamaDir) + " " + shellQuote(mimirPfx);
        std::cout << "$ " << cmd << "\n" << std::flush;
        return std::system(cmd.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    mimirmind::runtime::Log::initFromEnv();

    CliArgs args;
    if (!parseArgs(argc, argv, args)) {
        return 2;
    }

    if (args.modelPath.empty()) {
        if (const char* env = std::getenv("MIMIRMIND_MODEL_PATH")) {
            args.modelPath = env;
        }
    }

    try {
        switch (args.mode) {
            case Mode::Smoke:  return runSmoke(args);
            case Mode::Serve:  return runServe(args);
            case Mode::Parity: return runParity(args);
        }
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
#include "compute/Embedding.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr const char* kBanner =
    "+------------------------------------------------------------+\n"
    "|                          Mimirmind                         |\n"
    "|   Project Well smoke (M1 ctx | M2 USM | M3 GGUF | M4 embed)|\n"
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

        // --- [M3] GGUF reader smoke test ------------------------------------

        const char* modelPath = std::getenv("MIMIRMIND_MODEL_PATH");
        if (modelPath == nullptr || modelPath[0] == '\0') {
            std::cout << "\n[M3] GGUF reader — skipped "
                         "(set MIMIRMIND_MODEL_PATH to a .gguf file to enable)\n";
            MM_LOG_INFO("main", "[M3] MIMIRMIND_MODEL_PATH not set; skipping");
        } else {
            std::cout << "\n[M3] GGUF reader (" << modelPath << ")\n";
            std::cout.flush();
            MM_LOG_INFO("main", "[M3] opening model '{}'", modelPath);

            mimirmind::model::GgufReader reader;
            reader.open(modelPath);

            std::cout << "  version       : " << reader.version() << "\n";
            std::cout << "  metadata      : " << reader.metadataCount() << " entries\n";
            std::cout << "  tensors       : " << reader.tensorCount() << " entries\n";
            std::cout << "  alignment     : " << reader.alignment() << " bytes\n";
            std::cout << "  data offset   : " << reader.tensorDataOffset() << " bytes\n";
            std::cout << "  payload total : " << formatBytes(reader.totalTensorBytes())
                      << "\n";

            // A few well-known metadata keys — print whichever are present.
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

            // --- [M3b] LlmConfig + Tokenizer ----------------------------

            mimirmind::model::LlmConfig config;
            config.parseFromGguf(reader);

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

            mimirmind::model::Tokenizer tok;
            tok.loadFromGguf(reader);

            std::cout << "  tokenizer:\n";
            std::cout << "    model      : " << tok.modelType() << "\n";
            std::cout << "    vocab_size : " << tok.vocabSize() << "\n";
            std::cout << "    bos/eos    : " << tok.bosId() << " / " << tok.eosId() << "\n";
            std::cout << "    unk/pad    : " << tok.unknownId() << " / " << tok.padId() << "\n";

            const std::string sample = "Hello, world!";
            const auto ids = tok.encode(sample, true);
            std::cout << "    encode('" << sample << "') = [";
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << ids[i];
            }
            std::cout << "] (" << ids.size() << " tokens)\n";

            std::cout << "    pieces     = [";
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << "'" << tok.tokenText(ids[i]) << "'";
            }
            std::cout << "]\n";

            const std::string round = tok.decode(ids, true);
            std::cout << "    decode     = '" << round << "'\n";

            // --- [M3c] Tensor payload load --------------------------------

            MM_LOG_INFO("main", "[M3] loading tensors into USM");
            reader.loadTensors(allocator);

            const auto st3 = allocator.stats();
            std::cout << "  loaded        : " << reader.tensorCount()
                      << " tensors, " << formatBytes(st3.liveBytes) << " live in USM\n";
            std::cout << "  ze alloc/free : " << st3.zeAllocCalls
                      << " / " << st3.zeFreeCalls << "\n";

            allocator.logStats(mimirmind::runtime::LogLevel::Info);

            // --- [M4a] Embedding lookup ----------------------------------

            std::cout << "\n[M4a] Embedding lookup smoke test\n";
            std::cout.flush();
            MM_LOG_INFO("main", "[M4a] starting embedding lookup");

            mimirmind::model::WeightsMap weights{reader};

            const auto* tokEmb = weights.find("token_embd.weight");
            if (tokEmb == nullptr) {
                tokEmb = weights.find("tok_embeddings.weight");
            }
            if (tokEmb == nullptr) {
                std::cout << "  no token-embedding tensor found, skipping\n";
                MM_LOG_WARN("main",
                            "[M4a] no token_embd.weight / tok_embeddings.weight");
            } else {
                std::cout << "  embed tensor  : " << tokEmb->name
                          << "  type=" << mimirmind::model::typeInfo(tokEmb->type).name
                          << "  dims=[";
                for (std::size_t i = 0; i < tokEmb->dimensions.size(); ++i) {
                    if (i > 0) {
                        std::cout << ",";
                    }
                    std::cout << tokEmb->dimensions[i];
                }
                std::cout << "]\n";

                const auto sampleIds = tok.encode("Hello", false);
                std::cout << "  sample tokens : [";
                for (std::size_t i = 0; i < sampleIds.size(); ++i) {
                    if (i > 0) {
                        std::cout << ", ";
                    }
                    std::cout << sampleIds[i];
                }
                std::cout << "]\n";

                const std::size_t d_model    = config.embeddingLength;
                const std::size_t vocab_size = tokEmb->dimensions.size() >= 2
                                                ? tokEmb->dimensions[1]
                                                : tok.vocabSize();
                const std::size_t seqLen     = sampleIds.size();
                const std::size_t outBytes   = seqLen * d_model * sizeof(float);

                if (seqLen == 0) {
                    std::cout << "  encode returned 0 tokens, nothing to embed\n";
                } else {
                    void* embPtr = allocator.allocate(outBytes);
                    mimirmind::compute::embeddingLookup(
                        tokEmb->type, tokEmb->usmPtr,
                        d_model, vocab_size,
                        sampleIds,
                        static_cast<float*>(embPtr));

                    const float* emb = static_cast<const float*>(embPtr);

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
                        if (v < vMin) {
                            vMin = v;
                        }
                        if (v > vMax) {
                            vMax = v;
                        }
                    }
                    std::cout << "  L2 norm       : " << std::sqrt(sumSq) << "\n";
                    std::cout << "  min / max     : " << vMin << " / " << vMax << "\n";

                    MM_LOG_INFO("main",
                                "[M4a] first token id={} L2={:.6f} min={:.6f} max={:.6f}",
                                sampleIds[0], std::sqrt(sumSq),
                                static_cast<double>(vMin),
                                static_cast<double>(vMax));

                    allocator.deallocate(embPtr, outBytes);
                }
            }
        }

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
#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/Embedding.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "compute/Rope.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/L0Context.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr const char* kBanner =
    "+------------------------------------------------------------+\n"
    "|                          Mimirmind                         |\n"
    "|   smoke: M1 ctx | M2 USM | M3 GGUF | M4 full forward CPU   |\n"
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

// ---- Transformer-block scratch + forward (used by [M4d]) -----------------

struct BlockBuffers {
    std::size_t T{};
    std::size_t d_model{};
    std::size_t q_dim{};
    std::size_t kv_dim{};
    std::size_t ff_dim{};

    void* qBuf{};
    void* kBuf{};
    void* vBuf{};
    void* normBuf{};
    void* attnOut{};
    void* projOut{};
    void* gateOut{};
    void* upOut{};
    void* matmulScratch{};
    void* scoreScratch{};

    std::size_t qBytes{};
    std::size_t kvBytes{};
    std::size_t normBytes{};
    std::size_t attnOutBytes{};
    std::size_t projOutBytes{};
    std::size_t gateOutBytes{};
    std::size_t upOutBytes{};
    std::size_t matmulScratchBytes{};
    std::size_t scoreScratchBytes{};
};

BlockBuffers allocBlockBuffers(mimirmind::runtime::UsmAllocator& alloc,
                               std::size_t T,
                               const mimirmind::model::LlmConfig& cfg) {
    BlockBuffers b{};
    b.T       = T;
    b.d_model = cfg.embeddingLength;
    b.q_dim   = cfg.headCount   * cfg.headDim();
    b.kv_dim  = cfg.headCountKv * cfg.headDim();
    b.ff_dim  = cfg.feedForwardLength;

    b.qBytes             = T * b.q_dim   * sizeof(float);
    b.kvBytes            = T * b.kv_dim  * sizeof(float);
    b.normBytes          = T * b.d_model * sizeof(float);
    b.attnOutBytes       = T * b.q_dim   * sizeof(float);
    b.projOutBytes       = T * b.d_model * sizeof(float);
    b.gateOutBytes       = T * b.ff_dim  * sizeof(float);
    b.upOutBytes         = T * b.ff_dim  * sizeof(float);
    b.scoreScratchBytes  = T             * sizeof(float);
    b.matmulScratchBytes =
        std::max({b.d_model, b.q_dim, b.ff_dim}) * sizeof(float);

    b.qBuf          = alloc.allocate(b.qBytes);
    b.kBuf          = alloc.allocate(b.kvBytes);
    b.vBuf          = alloc.allocate(b.kvBytes);
    b.normBuf       = alloc.allocate(b.normBytes);
    b.attnOut       = alloc.allocate(b.attnOutBytes);
    b.projOut       = alloc.allocate(b.projOutBytes);
    b.gateOut       = alloc.allocate(b.gateOutBytes);
    b.upOut         = alloc.allocate(b.upOutBytes);
    b.matmulScratch = alloc.allocate(b.matmulScratchBytes);
    b.scoreScratch  = alloc.allocate(b.scoreScratchBytes);
    return b;
}

void freeBlockBuffers(mimirmind::runtime::UsmAllocator& alloc, BlockBuffers& b) {
    alloc.deallocate(b.scoreScratch,  b.scoreScratchBytes);
    alloc.deallocate(b.matmulScratch, b.matmulScratchBytes);
    alloc.deallocate(b.upOut,         b.upOutBytes);
    alloc.deallocate(b.gateOut,       b.gateOutBytes);
    alloc.deallocate(b.projOut,       b.projOutBytes);
    alloc.deallocate(b.attnOut,       b.attnOutBytes);
    alloc.deallocate(b.normBuf,       b.normBytes);
    alloc.deallocate(b.vBuf,          b.kvBytes);
    alloc.deallocate(b.kBuf,          b.kvBytes);
    alloc.deallocate(b.qBuf,          b.qBytes);
}

void runTransformerBlock(std::size_t blockIdx,
                         const mimirmind::model::LlmConfig&  cfg,
                         const mimirmind::model::WeightsMap& weights,
                         float*                              x,
                         BlockBuffers&                       s) {
    namespace mm = mimirmind;

    const auto* attnNorm = weights.findBlock(blockIdx, "attn_norm.weight");
    const auto* qW = weights.findBlock(blockIdx, "attn_q.weight");
    const auto* qB = weights.findBlock(blockIdx, "attn_q.bias");
    const auto* kW = weights.findBlock(blockIdx, "attn_k.weight");
    const auto* kB = weights.findBlock(blockIdx, "attn_k.bias");
    const auto* vW = weights.findBlock(blockIdx, "attn_v.weight");
    const auto* vB = weights.findBlock(blockIdx, "attn_v.bias");
    const auto* oW = weights.findBlock(blockIdx, "attn_output.weight");

    const auto* ffnNorm = weights.findBlock(blockIdx, "ffn_norm.weight");
    const auto* ffnGate = weights.findBlock(blockIdx, "ffn_gate.weight");
    const auto* ffnUp   = weights.findBlock(blockIdx, "ffn_up.weight");
    const auto* ffnDown = weights.findBlock(blockIdx, "ffn_down.weight");

    if (attnNorm == nullptr || qW == nullptr || kW == nullptr ||
        vW == nullptr || oW == nullptr ||
        ffnNorm == nullptr || ffnGate == nullptr ||
        ffnUp == nullptr || ffnDown == nullptr) {
        throw std::runtime_error(
            "transformer block " + std::to_string(blockIdx) + " missing a tensor");
    }

    const std::size_t T        = s.T;
    const std::size_t d_model  = s.d_model;
    const std::size_t q_dim    = s.q_dim;
    const std::size_t kv_dim   = s.kv_dim;
    const std::size_t ff_dim   = s.ff_dim;
    const std::size_t head_dim = cfg.headDim();

    // attn_norm(x) -> normBuf
    mm::compute::rmsNorm(x, T, d_model,
                         static_cast<const float*>(attnNorm->usmPtr),
                         cfg.rmsNormEps,
                         static_cast<float*>(s.normBuf));

    auto project = [&](const mm::model::GgufTensor* W,
                       const mm::model::GgufTensor* B,
                       std::size_t N, void* dst) {
        mm::compute::matmul(W->type, W->usmPtr, N, d_model,
                            static_cast<const float*>(s.normBuf), T,
                            static_cast<float*>(dst),
                            static_cast<float*>(s.matmulScratch));
        if (B != nullptr && B->type == mm::model::GgmlType::F32) {
            mm::compute::addBias(static_cast<float*>(dst), T, N,
                                 static_cast<const float*>(B->usmPtr));
        }
    };

    project(qW, qB, q_dim,  s.qBuf);
    project(kW, kB, kv_dim, s.kBuf);
    project(vW, vB, kv_dim, s.vBuf);

    mm::compute::applyRopeInPlace(static_cast<float*>(s.qBuf), T,
                                  cfg.headCount,   head_dim, 0, cfg.ropeFreqBase);
    mm::compute::applyRopeInPlace(static_cast<float*>(s.kBuf), T,
                                  cfg.headCountKv, head_dim, 0, cfg.ropeFreqBase);

    mm::compute::multiHeadAttentionPrefill(
        static_cast<const float*>(s.qBuf),
        static_cast<const float*>(s.kBuf),
        static_cast<const float*>(s.vBuf),
        T, cfg.headCount, cfg.headCountKv, head_dim,
        static_cast<float*>(s.scoreScratch),
        static_cast<float*>(s.attnOut));

    mm::compute::matmul(oW->type, oW->usmPtr, d_model, q_dim,
                        static_cast<const float*>(s.attnOut), T,
                        static_cast<float*>(s.projOut),
                        static_cast<float*>(s.matmulScratch));

    mm::compute::addResidual(x, static_cast<const float*>(s.projOut), T * d_model);

    // ffn_norm(x) -> normBuf
    mm::compute::rmsNorm(x, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         cfg.rmsNormEps,
                         static_cast<float*>(s.normBuf));

    mm::compute::matmul(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                        static_cast<const float*>(s.normBuf), T,
                        static_cast<float*>(s.gateOut),
                        static_cast<float*>(s.matmulScratch));

    mm::compute::matmul(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                        static_cast<const float*>(s.normBuf), T,
                        static_cast<float*>(s.upOut),
                        static_cast<float*>(s.matmulScratch));

    mm::compute::siluInPlace(static_cast<float*>(s.gateOut), T * ff_dim);
    mm::compute::mulInPlace(static_cast<float*>(s.gateOut),
                            static_cast<const float*>(s.upOut), T * ff_dim);

    mm::compute::matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                        static_cast<const float*>(s.gateOut), T,
                        static_cast<float*>(s.projOut),
                        static_cast<float*>(s.matmulScratch));

    mm::compute::addResidual(x, static_cast<const float*>(s.projOut), T * d_model);
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

                    // --- [M4b] output_norm + lm_head matmul -> top-K ----

                    std::cout << "\n[M4b] Final norm + lm_head matmul -> top-5\n";
                    std::cout.flush();
                    MM_LOG_INFO("main", "[M4b] starting final-norm + lm_head");

                    const auto* outNorm = weights.find("output_norm.weight");
                    const auto* lmHead  = weights.find("output.weight");
                    if (lmHead == nullptr) {
                        // Many models tie the lm_head to the embedding.
                        lmHead = weights.find("token_embd.weight");
                    }

                    if (outNorm == nullptr) {
                        std::cout << "  output_norm.weight missing, skipping\n";
                    } else if (lmHead == nullptr) {
                        std::cout << "  no lm_head tensor found, skipping\n";
                    } else {
                        const std::size_t vocab_lm = lmHead->dimensions.size() >= 2
                            ? lmHead->dimensions[1] : vocab_size;

                        std::cout << "  output_norm   : " << outNorm->name
                                  << "  type=" << mimirmind::model::typeInfo(outNorm->type).name << "\n";
                        std::cout << "  lm_head       : " << lmHead->name
                                  << "  type=" << mimirmind::model::typeInfo(lmHead->type).name
                                  << "  dims=[" << lmHead->dimensions[0]
                                  << "," << vocab_lm << "]\n";

                        const std::size_t normBytes    = d_model * sizeof(float);
                        const std::size_t logitsBytes  = vocab_lm * sizeof(float);
                        const std::size_t scratchBytes = d_model * sizeof(float);

                        void* normedPtr  = allocator.allocate(normBytes);
                        void* logitsPtr  = allocator.allocate(logitsBytes);
                        void* scratchPtr = allocator.allocate(scratchBytes);

                        // Every production Llama/Qwen/Gemma stores norm weights
                        // as F32; we'd add dequant-on-load here if that ever
                        // stops being true.
                        if (outNorm->type != mimirmind::model::GgmlType::F32) {
                            std::cout << "  (output_norm weight is "
                                      << mimirmind::model::typeInfo(outNorm->type).name
                                      << ", not F32 — unsupported for now, skipping)\n";
                            allocator.deallocate(scratchPtr, scratchBytes);
                            allocator.deallocate(logitsPtr,  logitsBytes);
                            allocator.deallocate(normedPtr,  normBytes);
                            allocator.deallocate(embPtr,     outBytes);
                            std::cout << "\nProject Well startup smoke test passed.\n";
                            return 0;
                        }
                        const auto* normWeight = static_cast<const float*>(outNorm->usmPtr);

                        using clock = std::chrono::steady_clock;
                        auto t0 = clock::now();

                        mimirmind::compute::rmsNorm(
                            emb, seqLen, d_model,
                            normWeight, config.rmsNormEps,
                            static_cast<float*>(normedPtr));

                        auto t1 = clock::now();
                        mimirmind::compute::matmul(
                            lmHead->type, lmHead->usmPtr,
                            vocab_lm, d_model,
                            static_cast<const float*>(normedPtr), seqLen,
                            static_cast<float*>(logitsPtr),
                            static_cast<float*>(scratchPtr));
                        auto t2 = clock::now();

                        const double normMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        const double mmMs   = std::chrono::duration<double, std::milli>(t2 - t1).count();
                        std::cout << "  rmsNorm time  : " << normMs << " ms\n";
                        std::cout << "  matmul time   : " << mmMs   << " ms\n";

                        // top-5 across the last row of logits (seqLen=1 -> row 0)
                        const float* logitsRow = static_cast<const float*>(logitsPtr)
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

                        allocator.deallocate(scratchPtr, scratchBytes);
                        allocator.deallocate(logitsPtr,  logitsBytes);
                        allocator.deallocate(normedPtr,  normBytes);
                    }

                    // --- [M4c] Block-0 self-attention (prefill, GQA) ----

                    std::cout << "\n[M4c] Block 0 self-attention smoke (prefill)\n";
                    std::cout.flush();
                    MM_LOG_INFO("main", "[M4c] starting block-0 attention");

                    const std::string promptStr = "Hello, world!";
                    const auto promptIds = tok.encode(promptStr, false);
                    const std::size_t T = promptIds.size();

                    const auto* attnNorm = weights.findBlock(0, "attn_norm.weight");
                    const auto* qW = weights.findBlock(0, "attn_q.weight");
                    const auto* qB = weights.findBlock(0, "attn_q.bias");
                    const auto* kW = weights.findBlock(0, "attn_k.weight");
                    const auto* kB = weights.findBlock(0, "attn_k.bias");
                    const auto* vW = weights.findBlock(0, "attn_v.weight");
                    const auto* vB = weights.findBlock(0, "attn_v.bias");
                    const auto* oW = weights.findBlock(0, "attn_output.weight");

                    if (T == 0 || attnNorm == nullptr || qW == nullptr ||
                        kW == nullptr || vW == nullptr || oW == nullptr) {
                        std::cout << "  prerequisites missing, skipping\n";
                    } else {
                        const std::size_t n_heads    = config.headCount;
                        const std::size_t n_kv_heads = config.headCountKv;
                        const std::size_t head_dim   = config.headDim();
                        const std::size_t kv_dim     = n_kv_heads * head_dim;
                        const std::size_t q_dim      = n_heads    * head_dim;

                        std::cout << "  prompt        : \"" << promptStr
                                  << "\" -> " << T << " tokens\n";
                        std::cout << "  shapes        : Q=[" << T << "," << q_dim
                                  << "] KV=[" << T << "," << kv_dim
                                  << "] heads=" << n_heads
                                  << " (kv=" << n_kv_heads
                                  << ", head_dim=" << head_dim << ")\n";

                        const std::size_t xBytes        = T * d_model * sizeof(float);
                        const std::size_t qBufBytes     = T * q_dim   * sizeof(float);
                        const std::size_t kvBufBytes    = T * kv_dim  * sizeof(float);
                        const std::size_t normedC_Bytes = T * d_model * sizeof(float);
                        const std::size_t attnOutBytes  = T * q_dim   * sizeof(float);
                        const std::size_t oProjBytes    = T * d_model * sizeof(float);
                        const std::size_t scoreBytes    = T           * sizeof(float);
                        // Matmul row scratch: needs K floats, where K is the
                        // input feature dim (d_model for Q/K/V/Onorm, q_dim for o_proj).
                        const std::size_t scratchC_Bytes =
                            std::max(d_model, q_dim) * sizeof(float);

                        void* xBuf       = allocator.allocate(xBytes);
                        void* qBuf       = allocator.allocate(qBufBytes);
                        void* kBuf       = allocator.allocate(kvBufBytes);
                        void* vBuf       = allocator.allocate(kvBufBytes);
                        void* normedC    = allocator.allocate(normedC_Bytes);
                        void* attnOutBuf = allocator.allocate(attnOutBytes);
                        void* oProjBuf   = allocator.allocate(oProjBytes);
                        void* scratchC   = allocator.allocate(scratchC_Bytes);
                        void* scoreBuf   = allocator.allocate(scoreBytes);

                        using clock = std::chrono::steady_clock;
                        const auto t0 = clock::now();

                        // 1. Embedding lookup -> xBuf
                        mimirmind::compute::embeddingLookup(
                            tokEmb->type, tokEmb->usmPtr, d_model, vocab_size,
                            promptIds, static_cast<float*>(xBuf));

                        // 2. attn_norm -> normedC
                        mimirmind::compute::rmsNorm(
                            static_cast<const float*>(xBuf), T, d_model,
                            static_cast<const float*>(attnNorm->usmPtr),
                            config.rmsNormEps,
                            static_cast<float*>(normedC));

                        auto projectAndBias = [&](
                            const mimirmind::model::GgufTensor* W,
                            const mimirmind::model::GgufTensor* B,
                            std::size_t N, void* dst)
                        {
                            mimirmind::compute::matmul(
                                W->type, W->usmPtr,
                                N, d_model,
                                static_cast<const float*>(normedC), T,
                                static_cast<float*>(dst),
                                static_cast<float*>(scratchC));
                            if (B != nullptr &&
                                B->type == mimirmind::model::GgmlType::F32) {
                                mimirmind::compute::addBias(
                                    static_cast<float*>(dst), T, N,
                                    static_cast<const float*>(B->usmPtr));
                            }
                        };

                        // 3-5. Q, K, V projections
                        projectAndBias(qW, qB, q_dim,  qBuf);
                        projectAndBias(kW, kB, kv_dim, kBuf);
                        projectAndBias(vW, vB, kv_dim, vBuf);

                        // 6. RoPE on Q and K (V untouched)
                        mimirmind::compute::applyRopeInPlace(
                            static_cast<float*>(qBuf), T, n_heads,
                            head_dim, 0, config.ropeFreqBase);
                        mimirmind::compute::applyRopeInPlace(
                            static_cast<float*>(kBuf), T, n_kv_heads,
                            head_dim, 0, config.ropeFreqBase);

                        // 7. Multi-head attention (causal, GQA)
                        mimirmind::compute::multiHeadAttentionPrefill(
                            static_cast<const float*>(qBuf),
                            static_cast<const float*>(kBuf),
                            static_cast<const float*>(vBuf),
                            T, n_heads, n_kv_heads, head_dim,
                            static_cast<float*>(scoreBuf),
                            static_cast<float*>(attnOutBuf));

                        // 8. attn_output projection (no bias in Qwen2)
                        mimirmind::compute::matmul(
                            oW->type, oW->usmPtr,
                            d_model, q_dim,
                            static_cast<const float*>(attnOutBuf), T,
                            static_cast<float*>(oProjBuf),
                            static_cast<float*>(scratchC));

                        // 9. Residual: xBuf += oProjBuf
                        mimirmind::compute::addResidual(
                            static_cast<float*>(xBuf),
                            static_cast<const float*>(oProjBuf),
                            T * d_model);

                        const auto t1 = clock::now();
                        const double ms =
                            std::chrono::duration<double, std::milli>(t1 - t0).count();
                        std::cout << "  block-0 time  : " << ms << " ms (incl. embed + 4 matmuls)\n";

                        // Last-token hidden state
                        const float* last = static_cast<const float*>(xBuf) +
                                            (T - 1) * d_model;
                        std::cout << "  last token id : " << promptIds.back()
                                  << "  piece='" << tok.tokenText(promptIds.back()) << "'\n";

                        std::cout << "  first 10 f32  :";
                        std::cout << std::setprecision(6) << std::fixed;
                        for (std::size_t i = 0; i < 10; ++i) {
                            std::cout << "  " << last[i];
                        }
                        std::cout << "\n";
                        std::cout.unsetf(std::ios::floatfield);

                        double sumSqC = 0.0;
                        float vMinL = last[0];
                        float vMaxL = last[0];
                        for (std::size_t i = 0; i < d_model; ++i) {
                            const float v = last[i];
                            sumSqC += static_cast<double>(v) * static_cast<double>(v);
                            if (v < vMinL) {
                                vMinL = v;
                            }
                            if (v > vMaxL) {
                                vMaxL = v;
                            }
                        }
                        std::cout << "  L2 norm       : " << std::sqrt(sumSqC) << "\n";
                        std::cout << "  min / max     : " << vMinL << " / " << vMaxL << "\n";

                        MM_LOG_INFO("main",
                                    "[M4c] block-0 attn done in {:.1f}ms — "
                                    "last-token L2={:.4f} min={:.4f} max={:.4f}",
                                    ms, std::sqrt(sumSqC),
                                    static_cast<double>(vMinL),
                                    static_cast<double>(vMaxL));

                        allocator.deallocate(scoreBuf,   scoreBytes);
                        allocator.deallocate(scratchC,   scratchC_Bytes);
                        allocator.deallocate(oProjBuf,   oProjBytes);
                        allocator.deallocate(attnOutBuf, attnOutBytes);
                        allocator.deallocate(normedC,    normedC_Bytes);
                        allocator.deallocate(vBuf,       kvBufBytes);
                        allocator.deallocate(kBuf,       kvBufBytes);
                        allocator.deallocate(qBuf,       qBufBytes);
                        allocator.deallocate(xBuf,       xBytes);
                    }

                    // --- [M4d] Full 28-block forward + lm_head -> sample ---

                    std::cout << "\n[M4d] Full forward (28 blocks + norm + lm_head) -> top-5\n";
                    std::cout.flush();
                    MM_LOG_INFO("main", "[M4d] starting full forward");

                    const std::string fwdPrompt = "Hello, world!";
                    const auto fwdIds = tok.encode(fwdPrompt, false);
                    const std::size_t Tf = fwdIds.size();

                    const auto* outNormD = weights.find("output_norm.weight");
                    const auto* lmHeadD  = weights.find("output.weight");
                    if (lmHeadD == nullptr) {
                        lmHeadD = weights.find("token_embd.weight");
                    }

                    if (Tf == 0 || outNormD == nullptr || lmHeadD == nullptr) {
                        std::cout << "  prerequisites missing, skipping\n";
                    } else if (outNormD->type != mimirmind::model::GgmlType::F32) {
                        std::cout << "  output_norm not F32, skipping\n";
                    } else {
                        const std::size_t vocab_lm =
                            lmHeadD->dimensions.size() >= 2
                            ? lmHeadD->dimensions[1]
                            : vocab_size;

                        std::cout << "  prompt        : \"" << fwdPrompt
                                  << "\" -> " << Tf << " tokens\n";
                        std::cout << "  blocks        : " << config.blockCount << "\n";

                        BlockBuffers buffersD =
                            allocBlockBuffers(allocator, Tf, config);
                        const std::size_t xBytesD =
                            Tf * config.embeddingLength * sizeof(float);
                        const std::size_t normFinalBytes =
                            Tf * config.embeddingLength * sizeof(float);
                        const std::size_t logitsBytesD =
                            vocab_lm * sizeof(float);
                        const std::size_t logitsScratchBytes =
                            config.embeddingLength * sizeof(float);

                        void* xBufD     = allocator.allocate(xBytesD);
                        void* normFinal = allocator.allocate(normFinalBytes);
                        void* logitsD   = allocator.allocate(logitsBytesD);
                        void* logitsSc  = allocator.allocate(logitsScratchBytes);

                        using clock = std::chrono::steady_clock;
                        const auto fwdT0 = clock::now();

                        mimirmind::compute::embeddingLookup(
                            tokEmb->type, tokEmb->usmPtr,
                            config.embeddingLength, vocab_size,
                            fwdIds, static_cast<float*>(xBufD));

                        for (std::uint32_t b = 0; b < config.blockCount; ++b) {
                            const auto bT0 = clock::now();
                            runTransformerBlock(b, config, weights,
                                                static_cast<float*>(xBufD),
                                                buffersD);
                            const auto bT1 = clock::now();
                            const double bMs =
                                std::chrono::duration<double, std::milli>(bT1 - bT0).count();
                            MM_LOG_DEBUG("main", "[M4d] block {} done in {:.1f}ms", b, bMs);
                        }

                        const auto fwdT1 = clock::now();

                        // output_norm on last token only
                        const float* lastX =
                            static_cast<const float*>(xBufD) +
                            (Tf - 1) * config.embeddingLength;
                        mimirmind::compute::rmsNorm(
                            lastX, 1, config.embeddingLength,
                            static_cast<const float*>(outNormD->usmPtr),
                            config.rmsNormEps,
                            static_cast<float*>(normFinal));

                        // lm_head matmul -> logits (vocab)
                        mimirmind::compute::matmul(
                            lmHeadD->type, lmHeadD->usmPtr,
                            vocab_lm, config.embeddingLength,
                            static_cast<const float*>(normFinal), 1,
                            static_cast<float*>(logitsD),
                            static_cast<float*>(logitsSc));

                        const auto fwdT2 = clock::now();

                        const double blocksMs = std::chrono::duration<double, std::milli>(fwdT1 - fwdT0).count();
                        const double headMs   = std::chrono::duration<double, std::milli>(fwdT2 - fwdT1).count();
                        const double totalMs  = std::chrono::duration<double, std::milli>(fwdT2 - fwdT0).count();
                        std::cout << "  28 blocks     : " << blocksMs << " ms\n";
                        std::cout << "  norm+lm_head  : " << headMs   << " ms\n";
                        std::cout << "  total forward : " << totalMs  << " ms\n";

                        // Greedy argmax / top-5
                        const float* lg = static_cast<const float*>(logitsD);
                        std::vector<std::int32_t> idxD(vocab_lm);
                        std::iota(idxD.begin(), idxD.end(), 0);
                        const std::size_t kTop = std::min<std::size_t>(5, vocab_lm);
                        std::partial_sort(
                            idxD.begin(),
                            idxD.begin() + static_cast<std::ptrdiff_t>(kTop),
                            idxD.end(),
                            [lg](std::int32_t a, std::int32_t b) {
                                return lg[a] > lg[b];
                            });

                        std::cout << "  prompt decoded: '" << tok.decode(fwdIds, false) << "'\n";
                        std::cout << "  next-token top-5:\n";
                        for (std::size_t i = 0; i < kTop; ++i) {
                            const std::int32_t v = idxD[i];
                            std::cout << "    " << i << ": id=" << v
                                      << "  logit=" << lg[v]
                                      << "  piece='" << tok.tokenText(v) << "'\n";
                        }

                        MM_LOG_INFO("main",
                                    "[M4d] forward done in {:.1f}ms — top1 id={} logit={:.4f} piece='{}'",
                                    totalMs, idxD[0],
                                    static_cast<double>(lg[idxD[0]]),
                                    tok.tokenText(idxD[0]));

                        allocator.deallocate(logitsSc,  logitsScratchBytes);
                        allocator.deallocate(logitsD,   logitsBytesD);
                        allocator.deallocate(normFinal, normFinalBytes);
                        allocator.deallocate(xBufD,     xBytesD);
                        freeBlockBuffers(allocator, buffersD);
                    }

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
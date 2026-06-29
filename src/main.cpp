#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/Embedding.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "compute/Rope.hpp"
#include "model/GgufReader.hpp"
#include "model/LlmConfig.hpp"
#include "model/Tokenizer.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"
#include "runtime/KvCache.hpp"
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
    "|   smoke: M1-M4 + M5 GPU matmul integrated (Envoy in-path)  |\n"
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

// ---- Diagnostics --------------------------------------------------------

/// Log every tensor whose name starts with `blk.<idx>.`. Truth for the
/// architecture handler: shows the exact suffix names, quant types, and
/// dimensions of one transformer block. Cheap one-shot — call only at
/// startup, only for block 0.
void logBlockTensorInventory(const mimirmind::model::GgufReader& reader,
                             std::size_t blockIdx) {
    const std::string prefix = "blk." + std::to_string(blockIdx) + ".";
    std::size_t hits = 0;
    for (const auto& t : reader.tensors()) {
        if (t.name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        ++hits;
        std::string dims;
        for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
            if (i > 0) {
                dims += ',';
            }
            dims += std::to_string(t.dimensions[i]);
        }
        MM_LOG_INFO("inventory",
                    "  {} type={} dims=[{}] bytes={}",
                    t.name, mimirmind::model::typeInfo(t.type).name,
                    dims, t.nbytes);
    }
    MM_LOG_INFO("inventory", "block {} has {} tensor(s)", blockIdx, hits);
}

// ---- USM RAII handle ----------------------------------------------------

/// Scope-bound USM allocation. Frees on destruction so an exception
/// thrown mid-pipeline (e.g. unsupported dequant type) does not leak.
class UsmHandle {
public:
    UsmHandle() = default;

    UsmHandle(mimirmind::runtime::UsmAllocator& alloc, std::size_t bytes)
        : _alloc{&alloc}, _ptr{alloc.allocate(bytes)}, _bytes{bytes} {}

    ~UsmHandle() {
        if (_alloc != nullptr && _ptr != nullptr) {
            _alloc->deallocate(_ptr, _bytes);
        }
    }

    UsmHandle(const UsmHandle&)            = delete;
    UsmHandle& operator=(const UsmHandle&) = delete;

    UsmHandle(UsmHandle&& other) noexcept
        : _alloc{other._alloc}, _ptr{other._ptr}, _bytes{other._bytes} {
        other._alloc = nullptr;
        other._ptr   = nullptr;
        other._bytes = 0;
    }

    UsmHandle& operator=(UsmHandle&& other) noexcept {
        if (this != &other) {
            if (_alloc != nullptr && _ptr != nullptr) {
                _alloc->deallocate(_ptr, _bytes);
            }
            _alloc = other._alloc;
            _ptr   = other._ptr;
            _bytes = other._bytes;
            other._alloc = nullptr;
            other._ptr   = nullptr;
            other._bytes = 0;
        }
        return *this;
    }

    [[nodiscard]] void*       get()       noexcept { return _ptr; }
    [[nodiscard]] const void* get() const noexcept { return _ptr; }
    template <typename T> [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(_ptr);
    }
    template <typename T> [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(_ptr);
    }

private:
    mimirmind::runtime::UsmAllocator* _alloc{nullptr};
    void*                             _ptr{nullptr};
    std::size_t                       _bytes{0};
};

// ---- Transformer-block scratch + forward (shared by M4c / M4d / M4e) -----

struct BlockBuffers {
    std::size_t maxT{};
    std::size_t maxSeq{};
    std::size_t d_model{};
    std::size_t q_dim{};
    std::size_t ff_dim{};

    UsmHandle qBuf;            // maxT * q_dim   (K and V go straight into the KV cache)
    UsmHandle normBuf;         // maxT * d_model
    UsmHandle attnOut;         // maxT * q_dim
    UsmHandle projOut;         // maxT * d_model
    UsmHandle gateOut;         // maxT * ff_dim
    UsmHandle upOut;           // maxT * ff_dim
    UsmHandle matmulScratch;   // max(d_model, q_dim, ff_dim)
    UsmHandle scoreScratch;    // maxSeq   (full attention row including KV history)
};

BlockBuffers allocBlockBuffers(mimirmind::runtime::UsmAllocator& alloc,
                               std::size_t maxT,
                               std::size_t maxSeq,
                               const mimirmind::model::LlmConfig& cfg) {
    BlockBuffers b{};
    b.maxT    = maxT;
    b.maxSeq  = maxSeq;
    b.d_model = cfg.embeddingLength;
    b.q_dim   = cfg.headCount * cfg.headDim();
    b.ff_dim  = cfg.feedForwardLength;

    const std::size_t qBytes            = maxT * b.q_dim   * sizeof(float);
    const std::size_t normBytes         = maxT * b.d_model * sizeof(float);
    const std::size_t attnOutBytes      = maxT * b.q_dim   * sizeof(float);
    const std::size_t projOutBytes      = maxT * b.d_model * sizeof(float);
    const std::size_t gateOutBytes      = maxT * b.ff_dim  * sizeof(float);
    const std::size_t upOutBytes        = maxT * b.ff_dim  * sizeof(float);
    const std::size_t scoreScratchBytes = maxSeq           * sizeof(float);
    const std::size_t matmulScratchBytes =
        std::max({b.d_model, b.q_dim, b.ff_dim}) * sizeof(float);

    b.qBuf          = UsmHandle{alloc, qBytes};
    b.normBuf       = UsmHandle{alloc, normBytes};
    b.attnOut       = UsmHandle{alloc, attnOutBytes};
    b.projOut       = UsmHandle{alloc, projOutBytes};
    b.gateOut       = UsmHandle{alloc, gateOutBytes};
    b.upOut         = UsmHandle{alloc, upOutBytes};
    b.matmulScratch = UsmHandle{alloc, matmulScratchBytes};
    b.scoreScratch  = UsmHandle{alloc, scoreScratchBytes};
    return b;
}

/**
 * One transformer block forward — prefill OR decode.
 *
 *   x:     [T, d_model] in/out residual stream — overwritten with x + attn + ffn
 *   T:     new tokens this call (T <= s.maxT)
 *   cache: K/V cache. This block writes layer `blockIdx`'s new K/V at
 *          cache.length(); attention reads cache.baseK/V with
 *          T_k = cache.length() + T, positionOffset = cache.length().
 *          Caller must commit(T) once per forward pass (after all layers).
 */
void runTransformerBlock(std::size_t                         blockIdx,
                         const mimirmind::model::LlmConfig&  cfg,
                         const mimirmind::model::WeightsMap& weights,
                         float*                              x,
                         std::size_t                         T,
                         mimirmind::runtime::KvCache&        cache,
                         BlockBuffers&                       s,
                         mimirmind::compute::GpuMatmul&      gmm) {
    namespace mm = mimirmind;

    // INFO-level trace only for the very first block in the very first
    // forward — enough to localise a crash to one operation. Higher
    // blocks would just spam.
    const bool diag = (blockIdx == 0 && cache.length() == 0);
    auto trace = [&](const char* tag) {
        if (diag) {
            MM_LOG_INFO("blkdiag", "blk0 {}", tag);
        }
    };

    trace("enter");

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

    if (diag) {
        auto t = [](const mm::model::GgufTensor* p) {
            return p == nullptr ? "MISSING" : mm::model::typeInfo(p->type).name.data();
        };
        MM_LOG_INFO("blkdiag",
                    "blk0 lookups: attn_norm={} qW={} kW={} vW={} oW={} "
                    "ffn_norm={} ffn_gate={} ffn_up={} ffn_down={} qB={} kB={} vB={}",
                    t(attnNorm), t(qW), t(kW), t(vW), t(oW),
                    t(ffnNorm), t(ffnGate), t(ffnUp), t(ffnDown),
                    t(qB), t(kB), t(vB));
    }

    if (attnNorm == nullptr || qW == nullptr || kW == nullptr ||
        vW == nullptr || oW == nullptr ||
        ffnNorm == nullptr || ffnGate == nullptr ||
        ffnUp == nullptr || ffnDown == nullptr) {
        throw std::runtime_error(
            "transformer block " + std::to_string(blockIdx) + " missing a tensor");
    }

    const std::size_t d_model  = s.d_model;
    const std::size_t q_dim    = s.q_dim;
    const std::size_t kv_dim   = cfg.headCountKv * cfg.headDim();
    const std::size_t ff_dim   = s.ff_dim;
    const std::size_t head_dim = cfg.headDim();
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const scoreScratch  = s.scoreScratch.as<float>();

    trace("attn rmsNorm");
    // attn_norm(x) -> normBuf
    mm::compute::rmsNorm(x, T, d_model,
                         static_cast<const float*>(attnNorm->usmPtr),
                         cfg.rmsNormEps,
                         normBuf);

    auto projectAsync = [&](const mm::model::GgufTensor* W,
                            std::size_t N, float* dst) {
        gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                        normBuf, T, dst, matmulScratch);
    };
    auto addBiasIf = [&](const mm::model::GgufTensor* B,
                         std::size_t N, float* dst) {
        if (B != nullptr && B->type == mm::model::GgmlType::F32) {
            mm::compute::addBias(dst, T, N,
                                 static_cast<const float*>(B->usmPtr));
        }
    };

    // K and V project straight into the KV cache at offset curLen.
    float* kSlot = cache.writeSlotK(blockIdx);
    float* vSlot = cache.writeSlotV(blockIdx);

    // Q/K/V are independent and all needed before RoPE — batch into one
    // GPU submission, sync once, then run the CPU bias adds.
    trace("Q projection (matmulAsync)");
    projectAsync(qW, q_dim,  qBuf);
    trace("K projection (matmulAsync)");
    projectAsync(kW, kv_dim, kSlot);
    trace("V projection (matmulAsync)");
    projectAsync(vW, kv_dim, vSlot);
    trace("QKV sync");
    gmm.sync();
    trace("QKV bias add");
    addBiasIf(qB, q_dim,  qBuf);
    addBiasIf(kB, kv_dim, kSlot);
    addBiasIf(vB, kv_dim, vSlot);

    trace("RoPE Q+K");
    // RoPE on Q and the freshly-written K rows; startPos == curLen so the
    // absolute positions in decode mode line up with what's already cached.
    mm::compute::applyRopeInPlace(qBuf, T,
                                  cfg.headCount,   head_dim, curLen,
                                  cfg.ropeFreqBase);
    mm::compute::applyRopeInPlace(kSlot, T,
                                  cfg.headCountKv, head_dim, curLen,
                                  cfg.ropeFreqBase);

    trace("attention");
    // Attention: T_q = T, T_k = totalLen (cached + new), causal w/ position offset.
    mm::compute::multiHeadAttention(
        qBuf,
        cache.baseK(blockIdx),
        cache.baseV(blockIdx),
        T, totalLen,
        cfg.headCount, cfg.headCountKv, head_dim,
        curLen,
        scoreScratch,
        attnOutBuf);

    trace("O projection (matmul)");
    gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
               attnOutBuf, T,
               projOutBuf, matmulScratch);

    trace("attn residual");
    mm::compute::addResidual(x, projOutBuf, T * d_model);

    trace("ffn rmsNorm");
    // ffn_norm(x) -> normBuf
    mm::compute::rmsNorm(x, T, d_model,
                         static_cast<const float*>(ffnNorm->usmPtr),
                         cfg.rmsNormEps,
                         normBuf);

    trace("FFN gate+up (matmulAsync)");
    // gate and up read the same normBuf and are independent — batch.
    gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                    normBuf, T, gateOutBuf, matmulScratch);
    gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                    normBuf, T, upOutBuf, matmulScratch);
    trace("FFN gate+up sync");
    gmm.sync();

    trace("FFN silu+mul");
    mm::compute::siluInPlace(gateOutBuf, T * ff_dim);
    mm::compute::mulInPlace(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down (matmul)");
    gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
               gateOutBuf, T,
               projOutBuf, matmulScratch);

    trace("ffn residual + exit");
    mm::compute::addResidual(x, projOutBuf, T * d_model);
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

        // --- [M5] GPU RMSNorm kernel — parity vs CPU ------------------------

        std::cout << "\n[M5] GPU RMSNorm kernel (SPIR-V via Level Zero)\n";
        std::cout.flush();
        MM_LOG_INFO("main", "[M5] starting GPU RMSNorm parity test");
        try {
            mimirmind::runtime::GpuModule modRms{ctx, "rmsnorm"};
            mimirmind::runtime::GpuKernel knRms{modRms.kernel("rmsnorm")};
            mimirmind::runtime::CommandQueue queue{ctx};

            // Single row of length K = 3584 (Qwen2.5-7B d_model). Same K
            // we will actually use during forward — exercises the same
            // workgroup layout.
            constexpr std::uint32_t kLocalSize = 128;
            constexpr int           kK         = 3584;
            const std::size_t       bytesK     = static_cast<std::size_t>(kK) * sizeof(float);

            void* xUsm = allocator.allocate(bytesK);
            void* wUsm = allocator.allocate(bytesK);
            void* yUsm = allocator.allocate(bytesK);

            float* x = static_cast<float*>(xUsm);
            float* w = static_cast<float*>(wUsm);
            for (int i = 0; i < kK; ++i) {
                // Bounded pseudo-random; deterministic across runs.
                x[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.5F;
                w[i] = 1.0F + std::cos(static_cast<float>(i) * 0.007F) * 0.1F;
            }

            // GPU launch
            knRms.setPtr(0, xUsm);
            knRms.setPtr(1, wUsm);
            knRms.setPtr(2, yUsm);
            knRms.setValue<float>(3, 1e-6F);
            knRms.setValue<std::int32_t>(4, kK);
            knRms.setGroupSize(kLocalSize, 1, 1);

            const auto g0 = std::chrono::steady_clock::now();
            queue.dispatch(knRms, /* groupCountX = M = 1 row */ 1, 1, 1);
            const auto g1 = std::chrono::steady_clock::now();

            // CPU reference (same numerical recipe)
            std::vector<float> cpuRef(kK);
            const auto c0 = std::chrono::steady_clock::now();
            mimirmind::compute::rmsNorm(x, 1, kK, w, 1e-6F, cpuRef.data());
            const auto c1 = std::chrono::steady_clock::now();

            const float* y = static_cast<const float*>(yUsm);
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
            std::cout << "  gpu time      : " << gpuMs
                      << " ms (incl. queue sync)\n";
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

            allocator.deallocate(yUsm, bytesK);
            allocator.deallocate(wUsm, bytesK);
            allocator.deallocate(xUsm, bytesK);
        } catch (const std::exception& e) {
            std::cout << "  GPU RMSNorm failed: " << e.what() << "\n";
            MM_LOG_ERROR("main", "[M5] failed: {}", e.what());
        }

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

            // GPU matmul dispatcher: owns the SPIR-V modules + command
            // queue and is reused by [M4b] lm_head, all transformer-block
            // matmuls in [M4d/M4e], and the in-decode lm_head sampler.
            // Constructed here (after tensors are in USM) so we have a
            // single instance to thread through everything.
            mimirmind::compute::GpuMatmul gmm{ctx};

            // --- [M5b] GPU Q4_K matmul (vec) — parity vs CPU ---------------

            mimirmind::model::WeightsMap weights{reader};

            // One-shot diagnostic so the architecture-handler design is
            // grounded in what the GGUF actually contains for block 0
            // (suffix names, quants, dims) — not in assumptions.
            logBlockTensorInventory(reader, 0);

            std::cout << "\n[M5b] GPU Q4_K matvec kernel parity\n";
            std::cout.flush();
            MM_LOG_INFO("main", "[M5b] starting GPU Q4_K matvec parity test");
            try {
                const auto* qW = weights.find("blk.0.attn_q.weight");
                if (qW == nullptr ||
                    qW->type != mimirmind::model::GgmlType::Q4_K ||
                    qW->dimensions.size() < 2) {
                    std::cout << "  blk.0.attn_q.weight not Q4_K or missing — skipping\n";
                } else {
                    const std::size_t K = qW->dimensions[0];
                    const std::size_t N = qW->dimensions[1];
                    std::cout << "  weight        : " << qW->name
                              << "  type=Q4_K  K=" << K << "  N=" << N << "\n";

                    mimirmind::runtime::GpuModule modMm{ctx, "matmul_q4k_vec"};
                    mimirmind::runtime::GpuKernel knMm{modMm.kernel("matmul_q4k_vec")};
                    mimirmind::runtime::CommandQueue queueMm{ctx};

                    const std::size_t xBytes = K * sizeof(float);
                    const std::size_t yBytes = N * sizeof(float);
                    void* xUsm = allocator.allocate(xBytes);
                    void* yUsmGpu = allocator.allocate(yBytes);
                    std::vector<float> yCpuRef(N);
                    std::vector<float> scratch(K);

                    // Deterministic pseudo-random X vector.
                    float* x = static_cast<float*>(xUsm);
                    for (std::size_t i = 0; i < K; ++i) {
                        x[i] = std::sin(static_cast<float>(i) * 0.017F) * 0.3F;
                    }

                    constexpr std::uint32_t kLocalN = 64;
                    const std::uint32_t groups =
                        static_cast<std::uint32_t>((N + kLocalN - 1) / kLocalN);

                    knMm.setPtr(0, xUsm);
                    knMm.setPtr(1, qW->usmPtr);
                    knMm.setPtr(2, yUsmGpu);
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
                        static_cast<const float*>(xUsm), 1,
                        yCpuRef.data(), scratch.data());
                    const auto c1 = clock::now();

                    const float* yGpu = static_cast<const float*>(yUsmGpu);
                    float maxDiff = 0.0F;
                    float maxRef  = 0.0F;
                    for (std::size_t i = 0; i < N; ++i) {
                        const float d = std::fabs(yGpu[i] - yCpuRef[i]);
                        if (d > maxDiff) {
                            maxDiff = d;
                        }
                        const float a = std::fabs(yCpuRef[i]);
                        if (a > maxRef) {
                            maxRef = a;
                        }
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
                    std::cout << "  first 5 GPU   :  " << yGpu[0] << "  " << yGpu[1]
                              << "  " << yGpu[2] << "  " << yGpu[3] << "  " << yGpu[4] << "\n";
                    std::cout << "  first 5 CPU   :  " << yCpuRef[0] << "  " << yCpuRef[1]
                              << "  " << yCpuRef[2] << "  " << yCpuRef[3] << "  "
                              << yCpuRef[4] << "\n";

                    MM_LOG_INFO("main",
                                "[M5b] gpu={:.2f}ms cpu={:.2f}ms speedup={:.2f}x "
                                "max_diff={:.3e} max_cpu={:.3e}",
                                gpuMs, cpuMs,
                                cpuMs / gpuMs,
                                static_cast<double>(maxDiff),
                                static_cast<double>(maxRef));

                    allocator.deallocate(yUsmGpu, yBytes);
                    allocator.deallocate(xUsm,    xBytes);
                }
            } catch (const std::exception& e) {
                std::cout << "  Q4_K matmul failed: " << e.what() << "\n";
                MM_LOG_ERROR("main", "[M5b] failed: {}", e.what());
            }

            // --- [M5c] GPU Q6_K matmul (vec) — parity vs CPU (lm_head sized) -

            std::cout << "\n[M5c] GPU Q6_K matvec kernel parity (lm_head)\n";
            std::cout.flush();
            MM_LOG_INFO("main", "[M5c] starting GPU Q6_K matvec parity test");
            try {
                const auto* outW = weights.find("output.weight");
                if (outW == nullptr ||
                    outW->type != mimirmind::model::GgmlType::Q6_K ||
                    outW->dimensions.size() < 2) {
                    std::cout << "  output.weight not Q6_K or missing — skipping\n";
                } else {
                    const std::size_t K = outW->dimensions[0];
                    const std::size_t N = outW->dimensions[1];
                    std::cout << "  weight        : " << outW->name
                              << "  type=Q6_K  K=" << K << "  N=" << N << "\n";

                    mimirmind::runtime::GpuModule modMm6{ctx, "matmul_q6k_vec"};
                    mimirmind::runtime::GpuKernel knMm6{modMm6.kernel("matmul_q6k_vec")};
                    mimirmind::runtime::CommandQueue queueMm6{ctx};

                    const std::size_t xBytes6 = K * sizeof(float);
                    const std::size_t yBytes6 = N * sizeof(float);
                    void* xUsm6 = allocator.allocate(xBytes6);
                    void* yUsm6 = allocator.allocate(yBytes6);
                    std::vector<float> yCpu6(N);
                    std::vector<float> scratch6(K);

                    float* x6 = static_cast<float*>(xUsm6);
                    for (std::size_t i = 0; i < K; ++i) {
                        x6[i] = std::sin(static_cast<float>(i) * 0.011F) * 0.4F;
                    }

                    constexpr std::uint32_t kLocalN6 = 64;
                    const std::uint32_t groups6 =
                        static_cast<std::uint32_t>((N + kLocalN6 - 1) / kLocalN6);

                    knMm6.setPtr(0, xUsm6);
                    knMm6.setPtr(1, outW->usmPtr);
                    knMm6.setPtr(2, yUsm6);
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
                        static_cast<const float*>(xUsm6), 1,
                        yCpu6.data(), scratch6.data());
                    const auto c1 = clock::now();

                    const float* yG = static_cast<const float*>(yUsm6);
                    float maxDiff6 = 0.0F;
                    float maxRef6  = 0.0F;
                    for (std::size_t i = 0; i < N; ++i) {
                        const float d = std::fabs(yG[i] - yCpu6[i]);
                        if (d > maxDiff6) {
                            maxDiff6 = d;
                        }
                        const float a = std::fabs(yCpu6[i]);
                        if (a > maxRef6) {
                            maxRef6 = a;
                        }
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
                    std::cout << "  first 5 GPU   :  " << yG[0] << "  " << yG[1]
                              << "  " << yG[2] << "  " << yG[3] << "  " << yG[4] << "\n";
                    std::cout << "  first 5 CPU   :  " << yCpu6[0] << "  " << yCpu6[1]
                              << "  " << yCpu6[2] << "  " << yCpu6[3] << "  "
                              << yCpu6[4] << "\n";

                    MM_LOG_INFO("main",
                                "[M5c] gpu={:.2f}ms cpu={:.2f}ms speedup={:.2f}x "
                                "max_diff={:.3e} max_cpu={:.3e}",
                                gpuMs6, cpuMs6,
                                cpuMs6 / gpuMs6,
                                static_cast<double>(maxDiff6),
                                static_cast<double>(maxRef6));

                    allocator.deallocate(yUsm6, yBytes6);
                    allocator.deallocate(xUsm6, xBytes6);
                }
            } catch (const std::exception& e) {
                std::cout << "  Q6_K matmul failed: " << e.what() << "\n";
                MM_LOG_ERROR("main", "[M5c] failed: {}", e.what());
            }

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
                    UsmHandle embHandle{allocator, outBytes};
                    mimirmind::compute::embeddingLookup(
                        tokEmb->type, tokEmb->usmPtr,
                        d_model, vocab_size,
                        sampleIds,
                        embHandle.as<float>());

                    const float* emb = embHandle.as<float>();

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

                        UsmHandle normedHandle {allocator, normBytes};
                        UsmHandle logitsHandle {allocator, logitsBytes};
                        UsmHandle scratchHandle{allocator, scratchBytes};

                        // Every production Llama/Qwen/Gemma stores norm weights
                        // as F32; we'd add dequant-on-load here if that ever
                        // stops being true.
                        if (outNorm->type != mimirmind::model::GgmlType::F32) {
                            std::cout << "  (output_norm weight is "
                                      << mimirmind::model::typeInfo(outNorm->type).name
                                      << ", not F32 — unsupported for now, skipping)\n";
                            std::cout << "\nProject Well startup smoke test passed.\n";
                            return 0;
                        }
                        const auto* normWeight = static_cast<const float*>(outNorm->usmPtr);

                        using clock = std::chrono::steady_clock;
                        auto t0 = clock::now();

                        mimirmind::compute::rmsNorm(
                            emb, seqLen, d_model,
                            normWeight, config.rmsNormEps,
                            normedHandle.as<float>());

                        auto t1 = clock::now();
                        gmm.matmul(
                            lmHead->type, lmHead->usmPtr,
                            vocab_lm, d_model,
                            normedHandle.as<const float>(), seqLen,
                            logitsHandle.as<float>(),
                            scratchHandle.as<float>());
                        auto t2 = clock::now();

                        const double normMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        const double mmMs   = std::chrono::duration<double, std::milli>(t2 - t1).count();
                        std::cout << "  rmsNorm time  : " << normMs << " ms\n";
                        std::cout << "  matmul time   : " << mmMs   << " ms\n";

                        // top-5 across the last row of logits (seqLen=1 -> row 0)
                        const float* logitsRow = logitsHandle.as<const float>()
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

                    // --- [M4d/M4e] Prefill + autoregressive generation ---
                    //
                    // Single combined pipeline: prefill the prompt (28-block
                    // forward populates the KV cache), greedy-sample the
                    // first new token, then loop decode-mode forwards (T=1)
                    // for N more tokens. Streams the generated text token by
                    // token to stdout.

                    namespace mm = mimirmind;
                    using clock = std::chrono::steady_clock;

                    std::cout << "\n[M4d/M4e] Prefill + autoregressive generation\n";
                    std::cout.flush();
                    MM_LOG_INFO("main", "[M4d/M4e] starting full forward + decode");

                    const std::string fwdPrompt = "Hello, world!";
                    const auto fwdIds = tok.encode(fwdPrompt, false);
                    const std::size_t Tp = fwdIds.size();

                    constexpr std::size_t kMaxGen = 20;

                    const auto* outNormD = weights.find("output_norm.weight");
                    const auto* lmHeadD  = weights.find("output.weight");
                    if (lmHeadD == nullptr) {
                        lmHeadD = weights.find("token_embd.weight");
                    }

                    if (Tp == 0 || outNormD == nullptr || lmHeadD == nullptr ||
                        outNormD->type != mm::model::GgmlType::F32) {
                        std::cout << "  prerequisites missing, skipping\n";
                    } else {
                        const std::size_t vocab_lm =
                            lmHeadD->dimensions.size() >= 2
                            ? lmHeadD->dimensions[1] : vocab_size;
                        const std::size_t cacheMax = Tp + kMaxGen + 4;
                        const std::size_t maxT    = std::max<std::size_t>(Tp, 1);

                        std::cout << "  prompt   : \"" << fwdPrompt
                                  << "\" -> " << Tp << " tokens (max_gen=" << kMaxGen << ")\n";

                        mm::runtime::KvCache cache(
                            allocator, config.blockCount, cacheMax,
                            config.headCountKv, config.headDim());

                        BlockBuffers buffersD =
                            allocBlockBuffers(allocator, maxT, cacheMax, config);

                        const std::size_t xBytesD       = maxT * config.embeddingLength * sizeof(float);
                        const std::size_t normFinalBytes = config.embeddingLength * sizeof(float);
                        const std::size_t logitsBytesD  = vocab_lm * sizeof(float);
                        const std::size_t logitsScBytes = config.embeddingLength * sizeof(float);

                        UsmHandle xBufDH    {allocator, xBytesD};
                        UsmHandle normFinalH{allocator, normFinalBytes};
                        UsmHandle logitsDH  {allocator, logitsBytesD};
                        UsmHandle logitsScH {allocator, logitsScBytes};

                        float* const xBufD     = xBufDH.as<float>();
                        float* const normFinal = normFinalH.as<float>();
                        float* const logitsD   = logitsDH.as<float>();
                        float* const logitsSc  = logitsScH.as<float>();

                        auto sampleArgmax = [&](const float* hidden) -> std::int32_t {
                            mm::compute::rmsNorm(
                                hidden, 1, config.embeddingLength,
                                static_cast<const float*>(outNormD->usmPtr),
                                config.rmsNormEps,
                                normFinal);
                            gmm.matmul(
                                lmHeadD->type, lmHeadD->usmPtr,
                                vocab_lm, config.embeddingLength,
                                normFinal, 1,
                                logitsD, logitsSc);
                            std::int32_t best = 0;
                            float bestV = logitsD[0];
                            for (std::size_t i = 1; i < vocab_lm; ++i) {
                                if (logitsD[i] > bestV) {
                                    bestV = logitsD[i];
                                    best  = static_cast<std::int32_t>(i);
                                }
                            }
                            return best;
                        };

                        // -- Prefill -----------------------------------------

                        const auto preT0 = clock::now();
                        mm::compute::embeddingLookup(
                            tokEmb->type, tokEmb->usmPtr,
                            config.embeddingLength, vocab_size,
                            fwdIds, xBufD);
                        for (std::uint32_t b = 0; b < config.blockCount; ++b) {
                            runTransformerBlock(b, config, weights,
                                                xBufD, Tp,
                                                cache, buffersD, gmm);
                        }
                        cache.commit(Tp);
                        const auto preT1 = clock::now();
                        const double preMs =
                            std::chrono::duration<double, std::milli>(preT1 - preT0).count();

                        // Sample first new token from the last prefill row.
                        const float* lastRow = xBufD + (Tp - 1) * config.embeddingLength;
                        std::int32_t nextId = sampleArgmax(lastRow);

                        std::cout << "  prefill  : " << preMs << " ms ("
                                  << Tp << " tokens, " << config.blockCount << " blocks)\n";
                        std::cout << "  text     : '" << tok.decode(fwdIds, false)
                                  << "' >>>" << std::flush;

                        std::vector<std::int32_t> genIds;
                        genIds.reserve(kMaxGen);
                        genIds.push_back(nextId);
                        std::cout << tok.decode(std::span<const std::int32_t>(&nextId, 1), true)
                                  << std::flush;

                        // -- Decode loop -------------------------------------

                        const auto decT0 = clock::now();
                        std::size_t generated = 1;
                        bool hitEos = false;
                        for (std::size_t step = 1;
                             step < kMaxGen && cache.length() < cacheMax;
                             ++step)
                        {
                            if (nextId == tok.eosId()) {
                                hitEos = true;
                                break;
                            }

                            std::array<std::int32_t, 1> oneId{nextId};
                            mm::compute::embeddingLookup(
                                tokEmb->type, tokEmb->usmPtr,
                                config.embeddingLength, vocab_size,
                                oneId, xBufD);

                            for (std::uint32_t b = 0; b < config.blockCount; ++b) {
                                runTransformerBlock(b, config, weights,
                                                    xBufD, 1,
                                                    cache, buffersD, gmm);
                            }
                            cache.commit(1);

                            nextId = sampleArgmax(xBufD);
                            genIds.push_back(nextId);
                            std::cout << tok.decode(std::span<const std::int32_t>(&nextId, 1), true)
                                      << std::flush;
                            ++generated;
                        }
                        const auto decT1 = clock::now();
                        const double decMs =
                            std::chrono::duration<double, std::milli>(decT1 - decT0).count();
                        const double perTok = generated > 1
                            ? decMs / static_cast<double>(generated - 1)
                            : 0.0;

                        std::cout << "<<<\n";
                        std::cout << "  generated: " << generated << " token(s)"
                                  << (hitEos ? " (hit EOS)" : "") << "\n";
                        std::cout << "  decode   : " << decMs << " ms ("
                                  << perTok << " ms/token avg)\n";

                        MM_LOG_INFO("main",
                                    "[M4d/M4e] prefill={:.1f}ms decode={:.1f}ms "
                                    "({} new tokens, {:.1f}ms/token)",
                                    preMs, decMs, generated, perTok);
                    }
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
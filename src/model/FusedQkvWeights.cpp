#include "model/FusedQkvWeights.hpp"

#include "model/GgufReader.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <cstdlib>
#include <cstring>
#include <string_view>

namespace mimirmind::model {

namespace {

bool disabledByEnv() noexcept {
    const char* v = std::getenv("MIMIRMIND_DISABLE_FUSED_QKV");
    if (v == nullptr) return false;
    const std::string_view s{v};
    return !s.empty() && s != "0" && s != "false" && s != "off";
}

// GGUF stores weight tensor dimensions as [K, N] — dimensions[0] is
// the fastest-varying axis (input dim), dimensions[1] is the row count.
std::size_t nRows(const GgufTensor& t) noexcept {
    return t.dimensions.size() >= 2
        ? static_cast<std::size_t>(t.dimensions[1])
        : 0;
}

std::size_t nCols(const GgufTensor& t) noexcept {
    return t.dimensions.size() >= 1
        ? static_cast<std::size_t>(t.dimensions[0])
        : 0;
}

} // namespace

FusedQkvWeights::FusedQkvWeights(const WeightsMap&      weights,
                                 runtime::UsmAllocator& allocator,
                                 std::size_t            numBlocks)
    : _alloc{allocator}
{
    _blocks.resize(numBlocks);
    _disabledByEnv = ::mimirmind::model::disabledByEnv();

    if (_disabledByEnv) {
        MM_LOG_INFO("qkvfuse",
                    "FusedQkvWeights: skipped — MIMIRMIND_DISABLE_FUSED_QKV set");
        return;
    }

    std::size_t& fusedCount   = _fusedCount;
    std::size_t& skippedCount = _skippedCount;
    std::size_t& totalBytes   = _totalBytes;

    // Per-reason skip tallies for the summary log line at the end.
    std::size_t missingQ_or_K   = 0;
    std::size_t typeMismatch_QK = 0;
    std::size_t typeMismatch_QV = 0;
    std::size_t Kmismatch       = 0;
    std::size_t Nmismatch       = 0;

    for (std::size_t b = 0; b < numBlocks; ++b) {
        const auto* qT = weights.findBlock(b, "attn_q.weight");
        const auto* kT = weights.findBlock(b, "attn_k.weight");
        const auto* vT = weights.findBlock(b, "attn_v.weight");

        auto typeName = [](const GgufTensor* t) {
            return t == nullptr ? "MISSING" : typeInfo(t->type).name.data();
        };

        // Both Q and K are required; V is optional (altAttention).
        if (qT == nullptr || kT == nullptr) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: attn_q={} attn_k={}",
                        b, typeName(qT), typeName(kT));
            ++skippedCount;
            ++missingQ_or_K;
            continue;
        }
        if (qT->type != kT->type) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: attn_q.type={} != attn_k.type={}",
                        b, typeName(qT), typeName(kT));
            ++skippedCount;
            ++typeMismatch_QK;
            continue;
        }
        const bool hasV = (vT != nullptr);
        if (hasV && vT->type != qT->type) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: attn_v.type={} != attn_q.type={}",
                        b, typeName(vT), typeName(qT));
            ++skippedCount;
            ++typeMismatch_QV;
            continue;
        }

        // Consistency: all tensors must share input dim K.
        const std::size_t Kq  = nCols(*qT);
        const std::size_t Kk  = nCols(*kT);
        const std::size_t Kv  = hasV ? nCols(*vT) : Kq;
        if (Kq == 0 || Kq != Kk || Kq != Kv) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: K mismatch — "
                        "Kq={} Kk={} Kv={} (hasV={})",
                        b, Kq, Kk, Kv, hasV);
            ++skippedCount;
            ++Kmismatch;
            continue;
        }
        // KV rows must match between K and V.
        const std::size_t Nq  = nRows(*qT);
        const std::size_t Nk  = nRows(*kT);
        const std::size_t Nv  = hasV ? nRows(*vT) : Nk;
        if (Nq == 0 || Nk == 0 || (hasV && Nv != Nk)) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: N check failed — "
                        "Nq={} Nk={} Nv={} (hasV={})",
                        b, Nq, Nk, Nv, hasV);
            ++skippedCount;
            ++Nmismatch;
            continue;
        }

        const std::size_t bytes =
            qT->nbytes + kT->nbytes + (hasV ? vT->nbytes : 0);

        void* dst = allocator.allocate(bytes);
        std::uint8_t* dstBytes = static_cast<std::uint8_t*>(dst);

        std::memcpy(dstBytes,                        qT->usmPtr, qT->nbytes);
        std::memcpy(dstBytes + qT->nbytes,           kT->usmPtr, kT->nbytes);
        if (hasV) {
            std::memcpy(dstBytes + qT->nbytes + kT->nbytes,
                        vT->usmPtr, vT->nbytes);
        }

        Block blk;
        blk.usmPtr = dst;
        blk.type   = qT->type;
        blk.Nq     = Nq;
        blk.Nkv    = Nk;
        blk.K      = Kq;
        blk.hasV   = hasV;
        blk.nbytes = bytes;

        _blocks[b] = blk;
        ++fusedCount;
        totalBytes += bytes;
    }

    _anyFused = (fusedCount > 0);

    MM_LOG_INFO("qkvfuse",
                "FusedQkvWeights: {} block(s) fused ({} skipped), "
                "{} MiB extra USM",
                fusedCount, skippedCount,
                (totalBytes + (1ULL << 20) - 1) >> 20);
    if (skippedCount > 0) {
        MM_LOG_INFO("qkvfuse",
                    "skip breakdown: missing_qk={} type_qk={} type_qv={} "
                    "K_mismatch={} N_mismatch={}",
                    missingQ_or_K, typeMismatch_QK, typeMismatch_QV,
                    Kmismatch, Nmismatch);
    }
}

FusedQkvWeights::~FusedQkvWeights() {
    for (auto& maybe : _blocks) {
        if (!maybe.has_value()) continue;
        if (maybe->usmPtr != nullptr) {
            _alloc.deallocate(maybe->usmPtr, maybe->nbytes);
        }
    }
}

const FusedQkvWeights::Block*
FusedQkvWeights::find(std::size_t blockIdx) const noexcept {
    if (blockIdx >= _blocks.size()) return nullptr;
    const auto& maybe = _blocks[blockIdx];
    return maybe.has_value() ? &(*maybe) : nullptr;
}

} // namespace mimirmind::model
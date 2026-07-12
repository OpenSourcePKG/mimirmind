#include "model/FusedQkvWeights.hpp"

#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "compute/quant/Q8_0.hpp"
#include "model/GgufReader.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <cstring>
#include <string_view>
#include <vector>

namespace mimirmind::model {

namespace {

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
                                 std::size_t            numBlocks,
                                 bool                   enabled,
                                 std::size_t            sharedKvLayers,
                                 bool                   requantMismatchToQ8_0,
                                 bool                   q8_0ReorderEnabled)
    : _alloc{allocator}
{
    _blocks.resize(numBlocks);
    _disabled = !enabled;

    // Layers past this index reuse an earlier layer's K/V and never
    // touch the QKV projection, so fusion is a no-op for them.
    const std::size_t ownKvEnd = sharedKvLayers < numBlocks
                                     ? numBlocks - sharedKvLayers
                                     : numBlocks;

    if (_disabled) {
        MM_LOG_INFO("qkvfuse",
                    "FusedQkvWeights: skipped — features.fusedQkv=false");
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
        // Shared-KV layers don't project K/V, so fusion is pointless
        // for them — attention will read from the source layer's cache.
        if (b >= ownKvEnd) {
            ++skippedCount;
            continue;
        }

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

        const bool hasV = (vT != nullptr);
        const bool qkMismatch = (qT->type != kT->type);
        const bool qvMismatch = hasV && (vT->type != qT->type);
        const bool typeMismatch = qkMismatch || qvMismatch;

        // Requant path: dequant Q/K/V to f32, then quantize each to
        // Q8_0. Only kicks in when the caller opts in AND every source
        // type has a registered dequant path. Otherwise fall through
        // to the pre-existing skip behaviour so old models keep the
        // same log output.
        const auto* qQt = compute::quantType(qT->type);
        const auto* kQt = compute::quantType(kT->type);
        const auto* vQt = hasV ? compute::quantType(vT->type) : nullptr;
        const bool haveDequantAll =
            (qQt != nullptr) && (kQt != nullptr) && (!hasV || vQt != nullptr);
        const bool doRequant =
            requantMismatchToQ8_0 && typeMismatch && haveDequantAll;

        if (!doRequant && qkMismatch) {
            MM_LOG_WARN("qkvfuse",
                        "block {} skipped: attn_q.type={} != attn_k.type={}",
                        b, typeName(qT), typeName(kT));
            ++skippedCount;
            ++typeMismatch_QK;
            continue;
        }
        if (!doRequant && qvMismatch) {
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

        GgmlType fusedType = qT->type;
        std::size_t bytes = 0;
        void* dst = nullptr;

        if (doRequant) {
            // Requantize all three tensors row-by-row into Q8_0. Row K
            // must be a Q8_0 block multiple (32) — every attention shape
            // we care about (2560, 2048, 4096, 512, 1024) satisfies this.
            if (Kq % 32 != 0) {
                MM_LOG_WARN("qkvfuse",
                            "block {} skipped: K={} not divisible by 32 "
                            "(Q8_0 requant impossible)", b, Kq);
                ++skippedCount;
                continue;
            }
            const std::size_t rowBytesQ8 = (Kq / 32) * 34;
            const std::size_t qBytes = Nq * rowBytesQ8;
            const std::size_t kBytes = Nk * rowBytesQ8;
            const std::size_t vBytes = hasV ? Nv * rowBytesQ8 : 0;
            bytes = qBytes + kBytes + vBytes;

            dst = allocator.allocate(bytes);
            auto* dstBytes = static_cast<std::uint8_t*>(dst);

            std::vector<float> rowF32(Kq);
            auto requantRows = [&](const GgufTensor& src,
                                   const compute::QuantType& qt,
                                   std::size_t rowCount,
                                   std::uint8_t* out) {
                const std::size_t srcRowBytes =
                    qt.blockElements() > 1
                        ? (Kq / qt.blockElements()) * qt.blockBytes()
                        : Kq * qt.blockBytes();
                const auto* base = static_cast<const std::uint8_t*>(src.usmPtr);
                for (std::size_t n = 0; n < rowCount; ++n) {
                    qt.dequantToF32(base + n * srcRowBytes, Kq, rowF32.data());
                    compute::quant::Q8_0::quantizeRow(
                        rowF32.data(), Kq, out + n * rowBytesQ8);
                }
            };

            requantRows(*qT, *qQt, Nq, dstBytes);
            requantRows(*kT, *kQt, Nk, dstBytes + qBytes);
            if (hasV) {
                requantRows(*vT, *vQt, Nv, dstBytes + qBytes + kBytes);
            }
            fusedType = GgmlType::Q8_0;
            ++_requantCount;

            MM_LOG_INFO("qkvfuse",
                        "block {} requantized: attn_q={} attn_k={} attn_v={} "
                        "→ Q8_0 fused ({:.2f} MiB)",
                        b, typeName(qT), typeName(kT),
                        hasV ? typeName(vT) : "-",
                        static_cast<double>(bytes) / (1024.0 * 1024.0));
        } else {
            // Fast path (types already match): raw memcpy of the source
            // bytes into a single contiguous buffer.
            bytes = qT->nbytes + kT->nbytes + (hasV ? vT->nbytes : 0);
            dst = allocator.allocate(bytes);
            auto* dstBytes = static_cast<std::uint8_t*>(dst);
            std::memcpy(dstBytes,                        qT->usmPtr, qT->nbytes);
            std::memcpy(dstBytes + qT->nbytes,           kT->usmPtr, kT->nbytes);
            if (hasV) {
                std::memcpy(dstBytes + qT->nbytes + kT->nbytes,
                            vT->usmPtr, vT->nbytes);
            }
        }

        Block blk;
        blk.usmPtr = dst;
        blk.type   = fusedType;
        blk.Nq     = Nq;
        blk.Nkv    = Nk;
        blk.K      = Kq;
        blk.hasV   = hasV;
        blk.nbytes = bytes;

        // M8.K.Q8_0-Reorder — allocate a parallel USM buffer holding
        // the same fused bytes in scales-then-quants layout so decode
        // (T==1) at the QKV projection can dispatch through
        // matmul_q8_0_vec_reorder. Native `dst` stays untouched so
        // prefill (T>1) keeps hitting GpuMatmul's Q8_0 GEMM path.
        // Reorder is skipped entirely unless the block emitted Q8_0
        // AND the operator opted in via features.q8_0Reorder.
        if (q8_0ReorderEnabled && fusedType == GgmlType::Q8_0) {
            const std::size_t Nfused = Nq + Nk + (hasV ? Nv : 0);
            try {
                void* rdst = allocator.allocate(bytes);
                std::memcpy(rdst, dst, bytes);
                std::vector<std::uint8_t> scratch(
                    (Kq / 32) * 34);
                compute::quant::Q8_0::reorderMatrixInPlace(
                    rdst, Nfused, Kq, scratch.data());
                blk.reorderUsmPtr = rdst;
                blk.reorderBytes  = bytes;
                ++_reorderCount;
                _reorderTotalBytes += bytes;
                MM_LOG_INFO("qkvfuse",
                            "block {} Q8_0-reorder copy: {} bytes "
                            "({:.2f} MiB, {} rows × K={})",
                            b, bytes,
                            static_cast<double>(bytes) / (1024.0 * 1024.0),
                            Nfused, Kq);
            } catch (const std::exception& e) {
                MM_LOG_WARN("qkvfuse",
                            "block {} Q8_0-reorder copy failed ({}) — "
                            "decode falls back to native mmvq for this "
                            "block", b, e.what());
                if (blk.reorderUsmPtr != nullptr) {
                    allocator.deallocate(blk.reorderUsmPtr,
                                         blk.reorderBytes);
                    blk.reorderUsmPtr = nullptr;
                    blk.reorderBytes  = 0;
                }
            }
        }

        _blocks[b] = blk;
        ++fusedCount;
        totalBytes += bytes;
    }

    _anyFused = (fusedCount > 0);

    MM_LOG_INFO("qkvfuse",
                "FusedQkvWeights: {} block(s) fused ({} skipped, "
                "{} requantized to Q8_0), {} MiB extra USM",
                fusedCount, skippedCount, _requantCount,
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
        // M8.K.Q8_0-Reorder — Phase 5b parallel USM buffer.
        if (maybe->reorderUsmPtr != nullptr) {
            _alloc.deallocate(maybe->reorderUsmPtr,
                              maybe->reorderBytes);
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
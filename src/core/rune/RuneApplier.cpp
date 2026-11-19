// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/rune/RuneApplier.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace mimirmind::core::rune {

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

float halfBitsToFloat(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fu;
    const std::uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        const float v = std::ldexp(static_cast<float>(mant), -24);
        return sign != 0 ? -v : v;
    }
    std::uint32_t bits = 0;
    if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::uint16_t floatToHalfBits(float f) noexcept {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const auto          sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t absx = x & 0x7fffffffu;
    if (absx >= 0x7f800000u) {                       // inf / nan
        return static_cast<std::uint16_t>(sign | 0x7c00u | (absx > 0x7f800000u ? 0x200u : 0u));
    }
    if (absx >= 0x477ff000u) {                       // >= 65520 rounds to inf
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (absx < 0x38800000u) {                        // below 2^-14: subnormal / zero
        float af = 0.0f;
        std::memcpy(&af, &absx, sizeof(af));
        const auto m = static_cast<std::uint32_t>(std::nearbyint(af * 16777216.0f));
        return static_cast<std::uint16_t>(sign | m);
    }
    const std::uint32_t mant = absx & 0x7fffffu;
    const std::uint32_t exp  = (absx >> 23) - 127u + 15u;
    std::uint32_t       h    = (exp << 10) | (mant >> 13);
    const std::uint32_t rem  = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0)) {
        ++h;
    }
    return static_cast<std::uint16_t>(sign | h);
}

float bf16BitsToFloat(std::uint16_t b) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::uint16_t floatToBf16Bits(float f) noexcept {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u) {          // nan: keep it quiet
        return static_cast<std::uint16_t>((x >> 16) | 0x40u);
    }
    x += 0x7fffu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>(x >> 16);
}

namespace {

std::size_t elemSize(RuneElemType t) noexcept {
    switch (t) {
        case RuneElemType::F32:  return 4;
        case RuneElemType::F16:  return 2;
        case RuneElemType::BF16: return 2;
        default:                 return 0;
    }
}

// Block-scaled encodings (Q8_0, FP8_E4M3_B32): f16 scale + 32 one-byte values
// per block along ne0.
constexpr std::size_t kQ8Block      = 32;
constexpr std::size_t kQ8BlockBytes = 34;

bool isBlockScaled(RuneElemType t) noexcept {
    return t == RuneElemType::Q8_0 || t == RuneElemType::FP8_E4M3_B32;
}

const char* typeName(RuneElemType t) noexcept {
    return t == RuneElemType::Q8_0 ? "Q8_0" : "FP8_E4M3";
}

/// Block-scaled Scale: finite non-negative factors, for axis=cols uniform
/// within each 32-column block (one factor per block scale). Exactness of
/// factors other than 0/1 is checked per block when applied.
std::string checkBlockScale(RuneElemType t, std::span<const float> p, RuneScaleAxis axis) {
    for (const float f : p) {
        if (!(f >= 0.0F)) {
            return std::string{typeName(t)} + " targets accept only non-negative scale factors";
        }
    }
    if (axis == RuneScaleAxis::Cols) {
        for (std::size_t b = 0; b < p.size(); b += kQ8Block) {
            for (std::size_t k = 1; k < kQ8Block && b + k < p.size(); ++k) {
                if (p[b + k] != p[b]) {
                    return std::string{typeName(t)} +
                           " column scale must be uniform within each 32-column block";
                }
            }
        }
    }
    return {};
}

/// Expert mask (Mask op on a router): every factor 0 or 1, at least one expert
/// disabled and one kept, one factor per expert = per router row. The router
/// must exist in the loaded model (that also proves the block is MoE); its
/// encoding does not matter since no byte of it changes.
std::string checkExpertMask(const RuneOp& op, std::span<const float> p,
                            const RuneTensorResolver& resolve) {
    std::size_t kept = 0;
    for (const float f : p) {
        if (f != 0.0F && f != 1.0F) {
            return "expert mask factors must be 0 (disabled) or 1 (kept)";
        }
        kept += f != 0.0F ? 1 : 0;
    }
    if (kept == p.size()) {
        return "expert mask disables no expert";
    }
    if (kept == 0) {
        return "expert mask disables every expert";
    }
    const std::optional<RuneTensorView> v = resolve(op.target);
    if (!v) {
        return "router tensor not found in the loaded model (not a MoE block?)";
    }
    std::uint64_t rows = 1;
    for (std::size_t i = 1; i < v->dims.size(); ++i) {
        rows *= v->dims[i];
    }
    if (v->dims.empty() || rows != p.size()) {
        return "expert mask has " + std::to_string(p.size()) + " factors, router has " +
               std::to_string(rows) + " experts";
    }
    return {};
}

float readElem(const std::byte* base, RuneElemType t, std::size_t i) noexcept {
    if (t == RuneElemType::F32) {
        float v = 0.0f;
        std::memcpy(&v, base + i * 4, 4);
        return v;
    }
    std::uint16_t b = 0;
    std::memcpy(&b, base + i * 2, 2);
    return t == RuneElemType::F16 ? halfBitsToFloat(b) : bf16BitsToFloat(b);
}

void writeElem(std::byte* base, RuneElemType t, std::size_t i, float v) noexcept {
    if (t == RuneElemType::F32) {
        std::memcpy(base + i * 4, &v, 4);
        return;
    }
    const std::uint16_t b = t == RuneElemType::F16 ? floatToHalfBits(v) : floatToBf16Bits(v);
    std::memcpy(base + i * 2, &b, 2);
}

struct Geometry {
    std::uint64_t rows{0};
    std::uint64_t cols{0};
};

/// Shape/type checks for one target. Returns an error string or empty.
std::string checkTarget(const RuneTensorView& v, Geometry& g) {
    if (v.type == RuneElemType::Unsupported) {
        return "target '" + v.name +
               "' has a quantised/packed encoding — only F32/F16/BF16 targets (and exact "
               "block scales on Q8_0 / FP8_E4M3) are patchable in this build";
    }
    if (v.hasDerivedCopies) {
        return "target '" + v.name +
               "' has derived copies (fused/repacked/side-bank) that the kernels read; "
               "patching only the primary bytes would have no effect";
    }
    if (v.dims.empty() || v.dims[0] == 0) {
        return "target '" + v.name + "' has empty dims";
    }
    std::uint64_t n = 1;
    for (const auto d : v.dims) {
        n *= d;
    }
    if (isBlockScaled(v.type)) {
        if (v.dims[0] % kQ8Block != 0 || v.bytes.size() != n / kQ8Block * kQ8BlockBytes) {
            return "target '" + v.name + "' " + typeName(v.type) + " byte size does not match dims";
        }
        g.cols = v.dims[0];
        g.rows = n / g.cols;
        return {};
    }
    if (v.bytes.size() != n * elemSize(v.type)) {
        return "target '" + v.name + "' byte size does not match dims x element size";
    }
    g.cols = v.dims[0];
    g.rows = n / g.cols;
    return {};
}

bool allFinite(std::span<const float> p) noexcept {
    for (const float x : p) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

double l2Norm(std::span<const float> p) noexcept {
    double s = 0.0;
    for (const float x : p) {
        s += static_cast<double>(x) * static_cast<double>(x);
    }
    return std::sqrt(s);
}

std::vector<float> unitVector(std::span<const float> p) {
    const double n = l2Norm(p);
    std::vector<float> out(p.size());
    for (std::size_t k = 0; k < p.size(); ++k) {
        out[k] = static_cast<float>(static_cast<double>(p[k]) / n);
    }
    return out;
}

/// SteerProject / SteerIf -> engine form (phase 1 checked norms + finiteness).
RuneCondSteer resolveCondSteer(const RuneOp& op, const RuneFile& rune) {
    const std::span<const float> p = rune.payload(op.payload);
    RuneCondSteer st;
    st.block = *steerBlockOf(op.target);
    st.mode  = op.kind == RuneOpKind::SteerProject ? RuneSteerMode::Project : op.steerMode;
    if (st.mode == RuneSteerMode::Project) {
        st.values = unitVector(p);
        st.alpha  = static_cast<float>(op.alpha);
    } else {
        st.values.resize(p.size());
        for (std::size_t k = 0; k < p.size(); ++k) {
            st.values[k] = static_cast<float>(static_cast<double>(p[k]) * op.alpha);
        }
    }
    if (op.kind == RuneOpKind::SteerIf) {
        st.hasCondition   = true;
        st.conditionBlock = op.conditionBlock;
        st.condition      = unitVector(rune.payload(op.condition));
        st.threshold      = static_cast<float>(op.threshold);
    }
    return st;
}

} // namespace

// ---------------------------------------------------------------------------
// RuneApplier
// ---------------------------------------------------------------------------

void RuneApplier::restore(BackupSet& set) noexcept {
    for (auto it = set.rbegin(); it != set.rend(); ++it) {
        std::memcpy(it->where.data(), it->original.data(), it->original.size());
    }
}

std::expected<RuneApplyReport, std::string>
RuneApplier::apply(const RuneFile& rune, std::string_view baseFingerprint,
                   const RuneTensorResolver& resolve) {
    const RuneManifest& man = rune.manifest();
    const std::string   who = "rune '" + man.name + "'";

    if (man.baseFingerprint != baseFingerprint) {
        return std::unexpected(who + ": built for base '" + man.baseFingerprint +
                               "' but the loaded model is '" + std::string{baseFingerprint} +
                               "' — refusing");
    }

    // Phase 1: everything checkable without mutating anything.
    std::unordered_map<std::string, RuneTensorView> views;
    for (std::size_t i = 0; i < man.ops.size(); ++i) {
        const RuneOp&     op  = man.ops[i];
        const std::string ctx = who + " op[" + std::to_string(i) + "] (" +
                                std::string{opKindName(op.kind)} + " on '" + op.target + "')";
        if (!opKindApplicable(op.kind)) {
            return std::unexpected(ctx + ": op kind not applicable in this build");
        }
        if (opKindIsSteer(op.kind)) {
            const std::span<const float> p = rune.payload(op.payload);
            if (!allFinite(p)) {
                return std::unexpected(ctx + ": payload contains NaN/Inf");
            }
            const bool project = op.kind == RuneOpKind::SteerProject ||
                                 (op.kind == RuneOpKind::SteerIf &&
                                  op.steerMode == RuneSteerMode::Project);
            if (project && l2Norm(p) == 0.0) {
                return std::unexpected(ctx + ": projection direction is zero");
            }
            if (op.kind == RuneOpKind::SteerIf) {
                const std::span<const float> c = rune.payload(op.condition);
                if (!allFinite(c)) {
                    return std::unexpected(ctx + ": condition contains NaN/Inf");
                }
                if (c.size() != p.size()) {
                    return std::unexpected(ctx + ": condition has " + std::to_string(c.size()) +
                                           " values, payload " + std::to_string(p.size()));
                }
                if (l2Norm(c) == 0.0) {
                    return std::unexpected(ctx + ": condition direction is zero");
                }
            }
            continue;   // activation overlay — no target tensor
        }
        if (op.kind == RuneOpKind::LowRank) {
            const std::optional<RuneTensorView> v = resolve(op.target);
            if (!v) {
                return std::unexpected(ctx + ": target tensor not found in the loaded model");
            }
            if (v->dims.empty() || v->dims[0] == 0) {
                return std::unexpected(ctx + ": target has empty dims");
            }
            std::uint64_t n = 1;
            for (const auto d : v->dims) n *= d;
            const std::uint64_t in = v->dims[0], out = n / in;
            const std::span<const float> pa = rune.payload(op.payload);
            const std::span<const float> pb = rune.payload(op.payloadB);
            if (pa.size() != op.rank * in || pb.size() != out * op.rank) {
                return std::unexpected(ctx + ": A has " + std::to_string(pa.size()) + " and B " +
                                       std::to_string(pb.size()) + " values; the target [" +
                                       std::to_string(out) + " x " + std::to_string(in) +
                                       "] at rank " + std::to_string(op.rank) + " needs " +
                                       std::to_string(op.rank * in) + " and " +
                                       std::to_string(out * op.rank));
            }
            if (!allFinite(pa) || !allFinite(pb) || !std::isfinite(op.alpha)) {
                return std::unexpected(ctx + ": payload or alpha contains NaN/Inf");
            }
            continue;   // projection overlay — no weight changes
        }
        if (op.kind == RuneOpKind::Mask) {
            if (std::string err = checkExpertMask(op, rune.payload(op.payload), resolve);
                !err.empty()) {
                return std::unexpected(ctx + ": " + err);
            }
            continue;   // router overlay — no weight changes
        }
        if (views.find(op.target) == views.end()) {
            std::optional<RuneTensorView> v = resolve(op.target);
            if (!v) {
                return std::unexpected(ctx + ": target tensor not found in the loaded model");
            }
            views.emplace(op.target, std::move(*v));
        }
        const RuneTensorView& v = views.at(op.target);
        Geometry g;
        if (std::string err = checkTarget(v, g); !err.empty()) {
            return std::unexpected(ctx + ": " + err);
        }
        const std::span<const float> p = rune.payload(op.payload);
        if (!allFinite(p)) {
            return std::unexpected(ctx + ": payload contains NaN/Inf");
        }
        if (isBlockScaled(v.type) && op.kind != RuneOpKind::Scale) {
            return std::unexpected(ctx + ": " + typeName(v.type) +
                                   " targets accept only scale (exact block scales)");
        }
        if (op.kind == RuneOpKind::WriteSlice) {
            if (op.rowEnd > g.rows) {
                return std::unexpected(ctx + ": row_end " + std::to_string(op.rowEnd) +
                                       " exceeds rows " + std::to_string(g.rows));
            }
            if (p.size() != (op.rowEnd - op.rowBegin) * g.cols) {
                return std::unexpected(ctx + ": payload has " + std::to_string(p.size()) +
                                       " values, expected (row_end-row_begin)*cols = " +
                                       std::to_string((op.rowEnd - op.rowBegin) * g.cols));
            }
        } else {   // Scale
            const std::uint64_t want = op.axis == RuneScaleAxis::Rows ? g.rows : g.cols;
            if (p.size() != want) {
                return std::unexpected(ctx + ": payload has " + std::to_string(p.size()) +
                                       " factors, expected " + std::to_string(want));
            }
            if (isBlockScaled(v.type)) {
                if (std::string err = checkBlockScale(v.type, p, op.axis); !err.empty()) {
                    return std::unexpected(ctx + ": " + err);
                }
            }
        }
    }

    // Phase 2: mutate with backups; all-or-nothing.
    BackupSet                        backups;
    std::unordered_map<std::string, std::size_t> touchedIdx;
    RuneApplyReport                  report;
    report.runeName = man.name;

    for (std::size_t i = 0; i < man.ops.size(); ++i) {
        const RuneOp& op = man.ops[i];
        if (op.kind == RuneOpKind::Mask) {
            const std::span<const float> p = rune.payload(op.payload);
            RuneExpertMask m;
            m.block = *expertMaskBlockOf(op.target);
            m.keep.resize(p.size());
            for (std::size_t e = 0; e < p.size(); ++e) {
                m.keep[e] = p[e] != 0.0F ? 1 : 0;
            }
            report.expertMasks.push_back(std::move(m));
            ++report.opsApplied;
            continue;
        }
        if (op.kind == RuneOpKind::LowRank) {
            const std::span<const float> pa = rune.payload(op.payload);
            const std::span<const float> pb = rune.payload(op.payloadB);
            RuneLowRank lr;
            lr.target = op.target;
            lr.rank   = op.rank;
            lr.in     = pa.size() / op.rank;
            lr.out    = pb.size() / op.rank;
            lr.a.assign(pa.begin(), pa.end());
            lr.b.resize(pb.size());
            for (std::size_t k = 0; k < pb.size(); ++k) {
                lr.b[k] = static_cast<float>(static_cast<double>(pb[k]) * op.alpha);
            }
            report.lowRanks.push_back(std::move(lr));
            ++report.opsApplied;
            continue;
        }
        if (op.kind == RuneOpKind::Steer) {
            const std::span<const float> p = rune.payload(op.payload);
            RuneSteer st;
            st.block = *steerBlockOf(op.target);
            st.values.resize(p.size());
            for (std::size_t k = 0; k < p.size(); ++k) {
                st.values[k] = static_cast<float>(static_cast<double>(p[k]) * op.alpha);
            }
            report.steers.push_back(std::move(st));
            ++report.opsApplied;
            continue;
        }
        if (op.kind == RuneOpKind::SteerProject || op.kind == RuneOpKind::SteerIf) {
            report.condSteers.push_back(resolveCondSteer(op, rune));
            ++report.opsApplied;
            continue;
        }
        const RuneTensorView& v  = views.at(op.target);
        const std::string     ctx = who + " op[" + std::to_string(i) + "] (" +
                                    std::string{opKindName(op.kind)} + " on '" + op.target + "')";

        const std::string current = tensorContentHash(v.bytes.data(), v.bytes.size());
        if (current != op.tensorHash) {
            restore(backups);
            return std::unexpected(ctx + ": tensor content hash " + current +
                                   " != expected " + op.tensorHash +
                                   " (rune authored against different weights or wrong stack order)");
        }
        if (touchedIdx.find(op.target) == touchedIdx.end()) {
            backups.push_back({v.bytes, std::vector<std::byte>(v.bytes.begin(), v.bytes.end())});
            touchedIdx.emplace(op.target, report.touched.size());
            report.touched.push_back({op.target, current, {}});
        }

        Geometry g;
        (void)checkTarget(v, g);
        const std::span<const float> p    = rune.payload(op.payload);
        std::byte*                   base = v.bytes.data();
        if (isBlockScaled(v.type)) {
            // Exact block scaling: a 0 factor zeroes the whole 32-element block
            // (scale and values); 1 leaves it; any other factor multiplies the
            // block's f16 scale and must be exact in f16, else the rune is refused.
            const std::uint64_t blocksPerRow = g.cols / kQ8Block;
            for (std::uint64_t r = 0; r < g.rows; ++r) {
                for (std::uint64_t b = 0; b < blocksPerRow; ++b) {
                    const float f = op.axis == RuneScaleAxis::Rows
                                        ? p[r] : p[static_cast<std::size_t>(b * kQ8Block)];
                    std::byte* blk = base + static_cast<std::size_t>(
                                                (r * blocksPerRow + b) * kQ8BlockBytes);
                    if (f == 0.0F) {
                        std::fill_n(blk, kQ8BlockBytes, std::byte{0});
                        continue;
                    }
                    if (f == 1.0F) {
                        continue;
                    }
                    std::uint16_t sBits = 0;
                    std::memcpy(&sBits, blk, sizeof(sBits));
                    const float         sOld  = halfBitsToFloat(sBits);
                    const float         sNew  = sOld * f;
                    const std::uint16_t nBits = floatToHalfBits(sNew);
                    if (!std::isfinite(sNew) || halfBitsToFloat(nBits) != sNew ||
                        static_cast<double>(sOld) * static_cast<double>(f) !=
                            static_cast<double>(sNew)) {
                        restore(backups);
                        return std::unexpected(
                            ctx + ": block scale " + std::to_string(sOld) + " x " +
                            std::to_string(f) + " (row " + std::to_string(r) + ", block " +
                            std::to_string(b) + ") is not exact in f16 — use a power-of-two factor");
                    }
                    std::memcpy(blk, &nBits, sizeof(nBits));
                }
            }
        } else if (op.kind == RuneOpKind::WriteSlice) {
            const std::size_t first = static_cast<std::size_t>(op.rowBegin * g.cols);
            for (std::size_t k = 0; k < p.size(); ++k) {
                writeElem(base, v.type, first + k, p[k]);
            }
        } else {
            for (std::uint64_t r = 0; r < g.rows; ++r) {
                for (std::uint64_t c = 0; c < g.cols; ++c) {
                    const std::size_t idx = static_cast<std::size_t>(r * g.cols + c);
                    const float       s   = op.axis == RuneScaleAxis::Rows ? p[r] : p[c];
                    writeElem(base, v.type, idx, readElem(base, v.type, idx) * s);
                }
            }
        }
        ++report.opsApplied;
    }

    for (auto& t : report.touched) {
        const RuneTensorView& v = views.at(t.name);
        t.hashAfter = tensorContentHash(v.bytes.data(), v.bytes.size());
    }
    _history.push_back(std::move(backups));
    return report;
}

std::size_t RuneApplier::rollback() noexcept {
    std::size_t restored = 0;
    for (auto it = _history.rbegin(); it != _history.rend(); ++it) {
        restore(*it);
        restored += it->size();
    }
    _history.clear();
    return restored;
}

} // namespace mimirmind::core::rune

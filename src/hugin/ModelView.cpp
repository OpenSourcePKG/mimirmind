// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ModelView.hpp"

#include "compute/Dequant.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/TensorFingerprint.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <variant>

namespace mimirmind::hugin {

namespace gg = core::gguf;

ModelView::ModelView(const std::string& path) : _path(path) {
    _reader.open(path);
    _fingerprint = gg::tensorFingerprint(_reader);
}

std::string ModelView::architecture() const {
    const auto* v = _reader.findMetadata("general.architecture");
    if (v == nullptr) {
        return {};
    }
    const auto* s = std::get_if<std::string>(v);
    return s != nullptr ? *s : std::string{};
}

const gg::GgufTensor* ModelView::find(std::string_view name) const noexcept {
    return _reader.findTensor(name);
}

const gg::GgufTensor& ModelView::require(std::string_view name) const {
    const auto* t = find(name);
    if (t == nullptr) {
        throw std::runtime_error("no tensor named '" + std::string{name} + "'");
    }
    return *t;
}

std::span<const std::uint8_t> ModelView::bytes(const gg::GgufTensor& t) const {
    const auto b = _reader.tensorFileBytes(t);
    if (b.size() != t.nbytes) {
        throw std::runtime_error("tensor '" + t.name + "' lies outside the file");
    }
    return b;
}

std::uint64_t ModelView::colsOf(const gg::GgufTensor& t) noexcept {
    return t.dimensions.empty() ? 0 : t.dimensions[0];
}

std::uint64_t ModelView::rowsOf(const gg::GgufTensor& t) noexcept {
    const std::uint64_t c = colsOf(t);
    return c == 0 ? 0 : t.nelements / c;
}

std::vector<float> ModelView::rowsF32(const gg::GgufTensor& t, std::uint64_t rowBegin,
                                      std::uint64_t rowEnd) const {
    const std::uint64_t cols = colsOf(t);
    const std::uint64_t rows = rowsOf(t);
    if (cols == 0 || rowEnd > rows || rowBegin >= rowEnd) {
        throw std::runtime_error("row range [" + std::to_string(rowBegin) + "," +
                                 std::to_string(rowEnd) + ") invalid for '" + t.name +
                                 "' with " + std::to_string(rows) + " rows");
    }
    const gg::GgmlTypeInfo info = gg::typeInfo(t.type);
    if (info.blockSize == 0 || info.typeSize == 0 || cols % info.blockSize != 0) {
        throw std::runtime_error("tensor '" + t.name + "' type " + std::string{info.name} +
                                 " is not row-addressable");
    }
    const std::size_t rowBytes = static_cast<std::size_t>(cols / info.blockSize) * info.typeSize;
    const auto        all      = bytes(t);
    std::vector<float> out(static_cast<std::size_t>((rowEnd - rowBegin) * cols));
    compute::dequantToF32(t.type, all.data() + rowBegin * rowBytes, out.size(), out.data());
    return out;
}

TensorStats ModelView::stats(const gg::GgufTensor& t, std::optional<std::uint64_t> rowBegin,
                             std::optional<std::uint64_t> rowEnd) const {
    TensorStats s;
    s.cols = colsOf(t);
    const std::uint64_t r0 = rowBegin.value_or(0);
    const std::uint64_t r1 = rowEnd.value_or(rowsOf(t));
    s.rows = r1 > r0 ? r1 - r0 : 0;

    double sum = 0, sumSq = 0, rnSum = 0;
    s.min        = std::numeric_limits<double>::infinity();
    s.max        = -std::numeric_limits<double>::infinity();
    s.rowNormMin = std::numeric_limits<double>::infinity();
    constexpr std::uint64_t kChunkElems = 1u << 22;   // ~16 MiB of F32 per chunk
    const std::uint64_t rowsPerChunk = std::max<std::uint64_t>(1, kChunkElems / std::max<std::uint64_t>(1, s.cols));

    for (std::uint64_t r = r0; r < r1; r += rowsPerChunk) {
        const std::uint64_t re = std::min(r1, r + rowsPerChunk);
        const std::vector<float> v = rowsF32(t, r, re);
        for (std::uint64_t i = 0; i < re - r; ++i) {
            double rowSq = 0;
            for (std::uint64_t c = 0; c < s.cols; ++c) {
                const double x = v[static_cast<std::size_t>(i * s.cols + c)];
                if (std::isnan(x)) { ++s.nanCount; continue; }
                if (std::isinf(x)) { ++s.infCount; continue; }
                sum += x; sumSq += x * x; rowSq += x * x;
                s.min = std::min(s.min, x);
                s.max = std::max(s.max, x);
                ++s.elements;
            }
            const double rn = std::sqrt(rowSq);
            rnSum += rn;
            s.rowNormMin = std::min(s.rowNormMin, rn);
            s.rowNormMax = std::max(s.rowNormMax, rn);
        }
    }
    if (s.elements > 0) {
        const double n = static_cast<double>(s.elements);
        s.mean   = sum / n;
        s.stddev = std::sqrt(std::max(0.0, sumSq / n - s.mean * s.mean));
        s.l2     = std::sqrt(sumSq);
        s.absMax = std::max(std::fabs(s.min), std::fabs(s.max));
    } else {
        s.min = s.max = 0;
    }
    if (s.rows > 0) {
        s.rowNormMean = rnSum / static_cast<double>(s.rows);
    } else {
        s.rowNormMin = 0;
    }
    return s;
}

gg::WeightsMap ModelView::readOnlyWeights() const {
    std::vector<gg::GgufTensor> owned;
    owned.reserve(_reader.tensors().size());
    for (const auto& t : _reader.tensors()) {
        gg::GgufTensor c = t;
        // The view is never written: Futhark's dry run only reads (readback)
        // and the caller binds upload to a no-op.
        c.usmPtr = const_cast<std::uint8_t*>(_reader.tensorFileBytes(t).data());
        owned.push_back(std::move(c));
    }
    return gg::WeightsMap::fromOwnedTensors(std::move(owned));
}

} // namespace mimirmind::hugin

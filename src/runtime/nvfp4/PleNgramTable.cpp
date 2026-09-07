// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/PleNgramTable.hpp"

#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mimirmind::runtime::nvfp4 {

namespace st = core::safetensors;

namespace {

// FP8 E4M3 (OCP/NVIDIA float8_e4m3fn) -> float. bias 7; no inf; 0x7F/0xFF = NaN.
inline float e4m3ToFloat(std::uint8_t b) {
    const int s = (b >> 7) & 0x1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    const float sign = s ? -1.0F : 1.0F;
    if (e == 0) {
        // subnormal: (-1)^s * 2^-6 * (m/8) = (-1)^s * m * 2^-9
        return sign * static_cast<float>(m) * 0.001953125F;  // 2^-9
    }
    if (e == 0xF && m == 0x7) {
        return 0.0F;  // NaN sentinel — never expected in the table; treat as 0
    }
    return sign * std::ldexp(1.0F + static_cast<float>(m) * 0.125F, e - 7);
}

} // namespace

void PleNgramTable::open(const std::string& checkpointDir,
                         const std::string& hfLayerPrefix, std::size_t cols) {
    _sm.open(checkpointDir);
    _cols = cols;
    _shardData.clear();
    _shardRowStart.clear();
    _shardRowStart.push_back(0);
    _rowsTotal = 0;

    const std::string base = hfLayerPrefix + ".ple.ple_embedding.ngram_embedding.";
    for (std::size_t i = 0;; ++i) {
        const std::string name = base + "shard_" + std::to_string(i) + ".weight";
        const st::SafetensorsTensor* t = _sm.find(name);
        if (t == nullptr) {
            break;
        }
        if (t->dtype != st::SafetensorsDtype::F8_E4M3 || t->shape.size() != 2
            || t->shape[1] != cols) {
            throw std::runtime_error("PleNgramTable: shard '" + name +
                                     "' is not 2-D F8_E4M3 [rows," +
                                     std::to_string(cols) + "]");
        }
        const auto bytes = _sm.tensorBytes(name);
        if (bytes.size() != t->shape[0] * cols) {
            throw std::runtime_error("PleNgramTable: shard '" + name +
                                     "' byte span size mismatch");
        }
        _shardData.push_back(bytes.data());
        _rowsTotal += static_cast<std::int64_t>(t->shape[0]);
        _shardRowStart.push_back(_rowsTotal);
    }
    if (_shardData.empty()) {
        throw std::runtime_error("PleNgramTable: no shards under '" + base + "shard_*'");
    }

    const std::string scaleName = base + "weight_scale";
    const st::SafetensorsTensor* sc = _sm.find(scaleName);
    if (sc == nullptr) {
        throw std::runtime_error("PleNgramTable: missing '" + scaleName + "'");
    }
    const auto scBytes = _sm.tensorBytes(scaleName);
    if (sc->dtype == st::SafetensorsDtype::BF16) {
        if (scBytes.size() < 2) {
            throw std::runtime_error("PleNgramTable: '" + scaleName + "' too small for BF16");
        }
        std::uint16_t bf;
        std::memcpy(&bf, scBytes.data(), 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(bf) << 16;
        std::memcpy(&_scale, &bits, sizeof(float));
    } else {
        if (scBytes.size() < sizeof(float)) {
            throw std::runtime_error("PleNgramTable: '" + scaleName + "' too small for F32");
        }
        std::memcpy(&_scale, scBytes.data(), sizeof(float));
    }

    // Stored layer_multipliers (int64 [ngram_size]) — read verbatim.
    const std::string multName = hfLayerPrefix + ".ple.ple_embedding.layer_multipliers";
    const st::SafetensorsTensor* mt = _sm.find(multName);
    if (mt == nullptr) {
        throw std::runtime_error("PleNgramTable: missing '" + multName + "'");
    }
    const auto multBytes = _sm.tensorBytes(multName);
    _mult.resize(multBytes.size() / sizeof(std::int64_t));
    std::memcpy(_mult.data(), multBytes.data(), _mult.size() * sizeof(std::int64_t));
}

void PleNgramTable::gatherDequantF32(std::span<const std::int64_t> ids,
                                     float* outF32) const {
    const std::size_t nShards = _shardData.size();
    for (std::size_t r = 0; r < ids.size(); ++r) {
        const std::int64_t id = ids[r];
        if (id < 0 || id >= _rowsTotal) {
            throw std::runtime_error("PleNgramTable: row id out of range");
        }
        // Locate the shard: _shardRowStart is ascending with a trailing total,
        // so the shard is the last start <= id.
        std::size_t lo = 0, hi = nShards;   // search in [0, nShards)
        while (lo + 1 < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (_shardRowStart[mid] <= id) lo = mid; else hi = mid;
        }
        const std::int64_t within = id - _shardRowStart[lo];
        const std::uint8_t* p =
            _shardData[lo] + static_cast<std::size_t>(within) * _cols;
        float* out = outF32 + r * _cols;
        for (std::size_t k = 0; k < _cols; ++k) {
            out[k] = e4m3ToFloat(p[k]) * _scale;
        }
    }
}

} // namespace mimirmind::runtime::nvfp4

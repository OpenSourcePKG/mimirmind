// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/SafetensorsHeader.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::core::safetensors {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("safetensors: " + msg);
}

/// Read a JSON number as a bounded unsigned integer, rejecting negatives,
/// non-integers, and anything that is not a number.
std::uint64_t asUint(const json& v, const char* what) {
    if (!v.is_number_unsigned()) {
        fail(std::string(what) + " is not an unsigned integer");
    }
    return v.get<std::uint64_t>();
}

} // namespace

ParsedSafetensorsHeader parseSafetensorsHeader(std::span<const std::uint8_t> file) {
    // --- 8-byte little-endian header length -------------------------------
    if (file.size() < sizeof(std::uint64_t)) {
        fail("file smaller than 8-byte header-length prefix");
    }
    std::uint64_t headerLen = 0;
    std::memcpy(&headerLen, file.data(), sizeof(headerLen)); // host is little-endian

    const std::size_t dataOffset = sizeof(std::uint64_t) + headerLen;
    if (headerLen == 0) {
        fail("header length is zero");
    }
    if (dataOffset > file.size()) {
        fail("header length " + std::to_string(headerLen)
             + " exceeds file size " + std::to_string(file.size()));
    }

    // --- JSON header ------------------------------------------------------
    const char* jsonBegin =
        reinterpret_cast<const char*>(file.data()) + sizeof(std::uint64_t);
    json hdr = json::parse(jsonBegin, jsonBegin + headerLen, nullptr,
                           /*allow_exceptions=*/false);
    if (hdr.is_discarded()) {
        fail("header is not valid JSON");
    }
    if (!hdr.is_object()) {
        fail("header JSON is not an object");
    }

    const std::size_t dataRegionSize = file.size() - dataOffset;

    ParsedSafetensorsHeader out;
    out.dataOffset = dataOffset;

    for (const auto& [name, spec] : hdr.items()) {
        if (name == "__metadata__") {
            if (!spec.is_object()) {
                fail("__metadata__ is not an object");
            }
            for (const auto& [mk, mv] : spec.items()) {
                if (!mv.is_string()) {
                    fail("__metadata__[" + mk + "] is not a string");
                }
                out.metadata.emplace(mk, mv.get<std::string>());
            }
            continue;
        }

        if (!spec.is_object()) {
            fail("tensor '" + name + "' entry is not an object");
        }
        if (!spec.contains("dtype") || !spec.contains("shape")
            || !spec.contains("data_offsets")) {
            fail("tensor '" + name + "' missing dtype/shape/data_offsets");
        }

        const auto& dtypeVal = spec.at("dtype");
        if (!dtypeVal.is_string()) {
            fail("tensor '" + name + "' dtype is not a string");
        }
        const SafetensorsDtype dtype =
            dtypeFromString(dtypeVal.get<std::string>());
        if (dtype == SafetensorsDtype::Unknown) {
            fail("tensor '" + name + "' has unsupported dtype '"
                 + dtypeVal.get<std::string>() + "'");
        }

        const auto& shapeVal = spec.at("shape");
        if (!shapeVal.is_array()) {
            fail("tensor '" + name + "' shape is not an array");
        }
        SafetensorsTensor t;
        t.name  = name;
        t.dtype = dtype;
        t.nelements = 1;
        for (const auto& dim : shapeVal) {
            const std::uint64_t d = asUint(dim, "shape dimension");
            t.shape.push_back(d);
            t.nelements *= d; // a 0-dim yields 0 elements, which we allow
        }

        const auto& off = spec.at("data_offsets");
        if (!off.is_array() || off.size() != 2) {
            fail("tensor '" + name + "' data_offsets is not a [begin,end] pair");
        }
        const std::uint64_t begin = asUint(off[0], "data_offsets begin");
        const std::uint64_t end   = asUint(off[1], "data_offsets end");
        if (begin > end) {
            fail("tensor '" + name + "' data_offsets begin > end");
        }
        if (end > dataRegionSize) {
            fail("tensor '" + name + "' data_offsets end " + std::to_string(end)
                 + " exceeds data region " + std::to_string(dataRegionSize));
        }

        t.nbytes    = static_cast<std::size_t>(end - begin);
        t.dataBegin = dataOffset + static_cast<std::size_t>(begin);
        t.dataEnd   = dataOffset + static_cast<std::size_t>(end);

        const std::size_t expect =
            static_cast<std::size_t>(t.nelements) * dtypeWidth(dtype);
        if (t.nbytes != expect) {
            fail("tensor '" + name + "' byte length " + std::to_string(t.nbytes)
                 + " != dtype-width x elements " + std::to_string(expect));
        }

        out.tensors.push_back(std::move(t));
    }

    // nlohmann's default object preserves ascending key order, so the vector
    // is already name-sorted; the caller can rely on that for a stable index.
    // Guard against a duplicate name having silently collapsed in the JSON
    // object (nlohmann keeps the last on a repeat) by checking neighbours.
    for (std::size_t i = 1; i < out.tensors.size(); ++i) {
        if (out.tensors[i].name == out.tensors[i - 1].name) {
            fail("duplicate tensor name '" + out.tensors[i].name + "'");
        }
    }

    return out;
}

} // namespace mimirmind::core::safetensors
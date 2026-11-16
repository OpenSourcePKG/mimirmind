// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/rune/RuneFile.hpp"

#include "core/safetensors/SafetensorsHeader.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace mimirmind::core::rune {

namespace {

using nlohmann::json;
namespace st = ::mimirmind::core::safetensors;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("rune: " + msg);
}

std::string_view axisName(RuneScaleAxis a) noexcept {
    return a == RuneScaleAxis::Rows ? "rows" : "cols";
}

const json& requireField(const json& obj, const char* key, const std::string& ctx) {
    const auto it = obj.find(key);
    if (it == obj.end()) {
        fail(ctx + ": missing field '" + key + "'");
    }
    return *it;
}

std::string requireString(const json& obj, const char* key, const std::string& ctx) {
    const json& v = requireField(obj, key, ctx);
    if (!v.is_string()) {
        fail(ctx + ": field '" + key + "' is not a string");
    }
    return v.get<std::string>();
}

std::uint64_t requireUint(const json& obj, const char* key, const std::string& ctx) {
    const json& v = requireField(obj, key, ctx);
    if (!v.is_number_unsigned()) {
        fail(ctx + ": field '" + key + "' is not an unsigned integer");
    }
    return v.get<std::uint64_t>();
}

RuneOp parseOp(const json& j, std::size_t idx) {
    const std::string ctx = "op[" + std::to_string(idx) + "]";
    if (!j.is_object()) {
        fail(ctx + " is not an object");
    }
    RuneOp op;
    const std::string kind = requireString(j, "op", ctx);
    const auto k = opKindFromName(kind);
    if (!k) {
        fail(ctx + ": unknown op '" + kind + "'");
    }
    op.kind       = *k;
    op.target = requireString(j, "target", ctx);
    if (op.kind == RuneOpKind::Steer) {
        // Activation overlay: no weight is bound, the base fingerprint is.
        if (j.contains("tensor_hash")) {
            op.tensorHash = requireString(j, "tensor_hash", ctx);
        }
        if (j.contains("alpha")) {
            if (!j["alpha"].is_number()) {
                fail(ctx + ": field 'alpha' is not a number");
            }
            op.alpha = j["alpha"].get<double>();
        }
    } else {
        op.tensorHash = requireString(j, "tensor_hash", ctx);
    }
    if (j.contains("payload")) {
        op.payload = requireString(j, "payload", ctx);
    }
    if (op.kind == RuneOpKind::WriteSlice) {
        op.rowBegin = requireUint(j, "row_begin", ctx);
        op.rowEnd   = requireUint(j, "row_end", ctx);
    } else if (op.kind == RuneOpKind::Scale) {
        const std::string axis = requireString(j, "axis", ctx);
        if (axis == "rows") {
            op.axis = RuneScaleAxis::Rows;
        } else if (axis == "cols") {
            op.axis = RuneScaleAxis::Cols;
        } else {
            fail(ctx + ": axis must be 'rows' or 'cols', got '" + axis + "'");
        }
    }
    return op;
}

json opToJson(const RuneOp& op) {
    json j;
    j["op"]          = std::string{opKindName(op.kind)};
    j["target"]      = op.target;
    if (op.kind != RuneOpKind::Steer || !op.tensorHash.empty()) {
        j["tensor_hash"] = op.tensorHash;
    }
    if (op.kind == RuneOpKind::Steer) {
        j["alpha"] = op.alpha;
    }
    if (!op.payload.empty()) {
        j["payload"] = op.payload;
    }
    if (op.kind == RuneOpKind::WriteSlice) {
        j["row_begin"] = op.rowBegin;
        j["row_end"]   = op.rowEnd;
    } else if (op.kind == RuneOpKind::Scale) {
        j["axis"] = std::string{axisName(op.axis)};
    }
    return j;
}

bool wellFormedHash(const std::string& h) {
    constexpr std::string_view kPrefix = "fnv1a64:";
    if (h.size() != kPrefix.size() + 16 || h.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    for (std::size_t i = kPrefix.size(); i < h.size(); ++i) {
        const char c = h[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

} // namespace

RuneFile::RuneFile(RuneManifest manifest, std::map<std::string, std::vector<float>> payloads)
    : _manifest(std::move(manifest)), _payloads(std::move(payloads)) {
    validate();
}

void RuneFile::validate() const {
    if (_manifest.version == 0 || _manifest.version > kRuneFormatVersion) {
        fail("unsupported manifest version " + std::to_string(_manifest.version) +
             " (this build understands <= " + std::to_string(kRuneFormatVersion) + ")");
    }
    if (_manifest.baseFingerprint.empty()) {
        fail("manifest has an empty base_fingerprint — a rune must be bound to one base model");
    }
    if (_manifest.ops.empty()) {
        fail("manifest has no ops");
    }
    for (std::size_t i = 0; i < _manifest.ops.size(); ++i) {
        const RuneOp&     op  = _manifest.ops[i];
        const std::string ctx = "op[" + std::to_string(i) + "] (" +
                                std::string{opKindName(op.kind)} + ")";
        if (op.target.empty()) {
            fail(ctx + ": empty target");
        }
        const bool hashOptional = op.kind == RuneOpKind::Steer && op.tensorHash.empty();
        if (!hashOptional && !wellFormedHash(op.tensorHash)) {
            fail(ctx + ": tensor_hash '" + op.tensorHash +
                 "' is not of the form fnv1a64:<16 hex>");
        }
        if (op.kind == RuneOpKind::Steer) {
            if (!steerBlockOf(op.target)) {
                fail(ctx + ": steer target must be 'blk.<i>', got '" + op.target + "'");
            }
            if (!std::isfinite(op.alpha)) {
                fail(ctx + ": alpha is not finite");
            }
        }
        const bool needsPayload = op.kind == RuneOpKind::WriteSlice ||
                                  op.kind == RuneOpKind::Scale || op.kind == RuneOpKind::Steer;
        if (!needsPayload) {
            continue;
        }
        const auto it = _payloads.find(op.payload);
        if (op.payload.empty() || it == _payloads.end()) {
            fail(ctx + ": payload '" + op.payload + "' not present in the container");
        }
        if (it->second.empty()) {
            fail(ctx + ": payload '" + op.payload + "' is empty");
        }
        if (op.kind == RuneOpKind::WriteSlice && op.rowEnd <= op.rowBegin) {
            fail(ctx + ": row_end must be > row_begin");
        }
    }
}

RuneFile RuneFile::parse(std::span<const std::uint8_t> file) {
    const st::ParsedSafetensorsHeader hdr = st::parseSafetensorsHeader(file);

    const auto mit = hdr.metadata.find(std::string{kRuneMetadataKey});
    if (mit == hdr.metadata.end()) {
        fail("no '" + std::string{kRuneMetadataKey} +
             "' entry in __metadata__ — not a rune file");
    }
    json m;
    try {
        m = json::parse(mit->second);
    } catch (const json::exception& e) {
        fail(std::string{"manifest is not valid JSON: "} + e.what());
    }
    if (!m.is_object()) {
        fail("manifest is not a JSON object");
    }
    if (requireString(m, "format", "manifest") != kRuneFormatName) {
        fail("manifest format is not '" + std::string{kRuneFormatName} + "'");
    }

    RuneManifest man;
    const std::uint64_t ver = requireUint(m, "version", "manifest");
    if (ver == 0 || ver > kRuneFormatVersion) {
        fail("unsupported manifest version " + std::to_string(ver));
    }
    man.version         = static_cast<std::uint32_t>(ver);
    man.name            = m.contains("name") ? requireString(m, "name", "manifest") : "";
    man.baseFingerprint = requireString(m, "base_fingerprint", "manifest");
    if (m.contains("provenance")) {
        if (!m["provenance"].is_object()) {
            fail("manifest provenance is not a JSON object");
        }
        man.provenanceJson = m["provenance"].dump();
    }
    if (!m.contains("ops")) {
        fail("manifest: missing field 'ops'");
    }
    const json& ops = m.at("ops");
    if (!ops.is_array()) {
        fail("manifest ops is not an array");
    }
    for (std::size_t i = 0; i < ops.size(); ++i) {
        man.ops.push_back(parseOp(ops[i], i));
    }

    std::map<std::string, std::vector<float>> payloads;
    for (const auto& t : hdr.tensors) {
        if (t.dtype != st::SafetensorsDtype::F32) {
            fail("payload '" + t.name + "' has dtype " +
                 std::string{st::dtypeName(t.dtype)} + ", only F32 payloads are allowed");
        }
        std::vector<float> v(static_cast<std::size_t>(t.nelements));
        if (t.nbytes != v.size() * sizeof(float)) {
            fail("payload '" + t.name + "' byte size does not match its element count");
        }
        if (t.nbytes > 0) {
            std::memcpy(v.data(), file.data() + t.dataBegin, t.nbytes);
        }
        payloads.emplace(t.name, std::move(v));
    }
    return RuneFile{std::move(man), std::move(payloads)};
}

RuneFile RuneFile::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("cannot open '" + path + "'");
    }
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    try {
        return parse(std::span<const std::uint8_t>(buf.data(), buf.size()));
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string{e.what()} + " [file '" + path + "']");
    }
}

std::vector<std::uint8_t> RuneFile::serialize() const {
    json m;
    m["format"]           = std::string{kRuneFormatName};
    m["version"]          = _manifest.version;
    m["name"]             = _manifest.name;
    m["base_fingerprint"] = _manifest.baseFingerprint;
    m["provenance"]       = json::parse(_manifest.provenanceJson);
    json ops = json::array();
    for (const auto& op : _manifest.ops) {
        ops.push_back(opToJson(op));
    }
    m["ops"] = std::move(ops);

    json header = json::object();
    std::size_t offset = 0;
    for (const auto& [name, values] : _payloads) {
        const std::size_t nbytes = values.size() * sizeof(float);
        header[name] = {
            {"dtype", "F32"},
            {"shape", json::array({values.size()})},
            {"data_offsets", json::array({offset, offset + nbytes})},
        };
        offset += nbytes;
    }
    header["__metadata__"] = {{std::string{kRuneMetadataKey}, m.dump()}};

    std::string headerText = header.dump();
    // Pad the JSON header with spaces so the data region starts 8-byte aligned.
    while ((sizeof(std::uint64_t) + headerText.size()) % 8 != 0) {
        headerText.push_back(' ');
    }

    std::vector<std::uint8_t> out(sizeof(std::uint64_t));
    const std::uint64_t headerLen = headerText.size();
    std::memcpy(out.data(), &headerLen, sizeof(headerLen));
    out.insert(out.end(), headerText.begin(), headerText.end());
    for (const auto& [name, values] : _payloads) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(values.data());
        out.insert(out.end(), p, p + values.size() * sizeof(float));
    }
    return out;
}

std::span<const float> RuneFile::payload(std::string_view name) const {
    const auto it = _payloads.find(std::string{name});
    if (it == _payloads.end()) {
        fail("payload '" + std::string{name} + "' not present");
    }
    return {it->second.data(), it->second.size()};
}

} // namespace mimirmind::core::rune

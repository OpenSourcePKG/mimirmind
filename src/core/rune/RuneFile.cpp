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
    if (opKindIsOverlay(op.kind)) {
        // Engine overlay: no weight is bound, the base fingerprint is.
        if (j.contains("tensor_hash")) {
            op.tensorHash = requireString(j, "tensor_hash", ctx);
        }
        if ((opKindIsSteer(op.kind) || op.kind == RuneOpKind::LowRank) && j.contains("alpha")) {
            if (!j["alpha"].is_number()) {
                fail(ctx + ": field 'alpha' is not a number");
            }
            op.alpha = j["alpha"].get<double>();
        }
        if (op.kind == RuneOpKind::SteerIf) {
            const std::string mode = requireString(j, "mode", ctx);
            const auto m = steerModeFromName(mode);
            if (!m) {
                fail(ctx + ": mode must be 'add' or 'project', got '" + mode + "'");
            }
            op.steerMode = *m;
            op.condition = requireString(j, "condition", ctx);
            const std::uint64_t cb = requireUint(j, "condition_block", ctx);
            if (cb > 0xFFFFFFFFULL) {
                fail(ctx + ": condition_block out of range");
            }
            op.conditionBlock = static_cast<std::uint32_t>(cb);
            const json& th = requireField(j, "threshold", ctx);
            if (!th.is_number()) {
                fail(ctx + ": field 'threshold' is not a number");
            }
            op.threshold = th.get<double>();
        }
        if (op.kind == RuneOpKind::LowRank) {
            op.payloadB = requireString(j, "payload_b", ctx);
            const std::uint64_t r = requireUint(j, "rank", ctx);
            if (r == 0 || r > 0xFFFFFFFFULL) {
                fail(ctx + ": rank must be in [1, 2^32)");
            }
            op.rank = static_cast<std::uint32_t>(r);
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
    if (!opKindIsOverlay(op.kind) || !op.tensorHash.empty()) {
        j["tensor_hash"] = op.tensorHash;
    }
    if (opKindIsSteer(op.kind) || op.kind == RuneOpKind::LowRank) {
        j["alpha"] = op.alpha;
    }
    if (op.kind == RuneOpKind::LowRank) {
        j["payload_b"] = op.payloadB;
        j["rank"]      = op.rank;
    }
    if (op.kind == RuneOpKind::SteerIf) {
        j["mode"]            = std::string{steerModeName(op.steerMode)};
        j["condition"]       = op.condition;
        j["condition_block"] = op.conditionBlock;
        j["threshold"]       = op.threshold;
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
        const bool hashOptional = opKindIsOverlay(op.kind) && op.tensorHash.empty();
        if (!hashOptional && !wellFormedHash(op.tensorHash)) {
            fail(ctx + ": tensor_hash '" + op.tensorHash +
                 "' is not of the form fnv1a64:<16 hex>");
        }
        if (opKindIsSteer(op.kind)) {
            const auto blk = steerBlockOf(op.target);
            if (!blk) {
                fail(ctx + ": steer target must be 'blk.<i>', got '" + op.target + "'");
            }
            if (!std::isfinite(op.alpha)) {
                fail(ctx + ": alpha is not finite");
            }
            const bool project = op.kind == RuneOpKind::SteerProject ||
                                 (op.kind == RuneOpKind::SteerIf &&
                                  op.steerMode == RuneSteerMode::Project);
            // A projection with alpha > 1 reflects the component (installs the
            // opposite behaviour) instead of removing it; alpha <= 0 is a no-op
            // or an amplification. Both are refused (7.12 guardrail).
            if (project && !(op.alpha > 0.0 && op.alpha <= 1.0)) {
                fail(ctx + ": projection alpha must be in (0, 1], got " +
                     std::to_string(op.alpha));
            }
            if (op.kind == RuneOpKind::SteerIf) {
                if (op.conditionBlock > *blk) {
                    fail(ctx + ": condition_block " + std::to_string(op.conditionBlock) +
                         " is after the steered block " + std::to_string(*blk) +
                         " (the condition must be known when the steer runs)");
                }
                if (!std::isfinite(op.threshold) || op.threshold <= -1.0 ||
                    op.threshold >= 1.0) {
                    fail(ctx + ": threshold must be a cosine in (-1, 1)");
                }
                const auto ct = _payloads.find(op.condition);
                if (op.condition.empty() || ct == _payloads.end() || ct->second.empty()) {
                    fail(ctx + ": condition payload '" + op.condition +
                         "' not present in the container");
                }
            }
        }
        if (op.kind == RuneOpKind::Mask && !expertMaskBlockOf(op.target)) {
            fail(ctx + ": mask target must be a router 'blk.<i>.ffn_gate_inp.weight', got '" +
                 op.target + "'");
        }
        if (op.kind == RuneOpKind::LowRank) {
            if (!std::isfinite(op.alpha)) {
                fail(ctx + ": alpha is not finite");
            }
            const auto a = _payloads.find(op.payload);
            const auto b = _payloads.find(op.payloadB);
            if (a == _payloads.end() || b == _payloads.end() || a->second.empty() ||
                b->second.empty()) {
                fail(ctx + ": lowrank needs non-empty payloads '" + op.payload + "' (A) and '" +
                     op.payloadB + "' (B)");
            }
            if (op.rank == 0 || a->second.size() % op.rank != 0 ||
                b->second.size() % op.rank != 0) {
                fail(ctx + ": payload sizes are not multiples of rank " + std::to_string(op.rank));
            }
            continue;
        }
        const bool needsPayload = op.kind != RuneOpKind::WriteQuantBlocks;
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

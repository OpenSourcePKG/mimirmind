// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/InspectTools.hpp"

#include "core/gguf/GgufTypes.hpp"
#include "core/rune/Futhark.hpp"
#include "core/rune/RuneFile.hpp"
#include "core/rune/RuneFormat.hpp"

#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace mimirmind::hugin {

namespace {

using nlohmann::json;
namespace gg = core::gguf;
namespace rn = core::rune;

json schema(json properties, json required = json::array()) {
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}};
}

bool patchable(gg::GgmlType t) noexcept {
    return t == gg::GgmlType::F32 || t == gg::GgmlType::F16 || t == gg::GgmlType::BF16;
}

json tensorJson(const gg::GgufTensor& t) {
    return {{"name", t.name},
            {"type", std::string{gg::typeInfo(t.type).name}},
            {"dims", t.dimensions},
            {"rows", ModelView::rowsOf(t)},
            {"cols", ModelView::colsOf(t)},
            {"bytes", t.nbytes},
            {"rune_patchable", patchable(t.type)}};
}

std::vector<std::string> stringList(const json& args, const char* key) {
    std::vector<std::string> out;
    if (!args.contains(key) || !args[key].is_array() || args[key].empty()) {
        throw std::runtime_error(std::string{"'"} + key + "' must be a non-empty array of strings");
    }
    for (const auto& v : args[key]) {
        out.push_back(v.get<std::string>());
    }
    return out;
}

} // namespace

InspectTools::InspectTools(const std::string& initialModel, std::string tokenizerGguf) {
    if (!initialModel.empty()) {
        open(initialModel, std::move(tokenizerGguf));
    }
}

void InspectTools::open(const std::string& path, std::string tokenizerGguf) {
    if (std::filesystem::is_directory(path)) {
        // NVFP4 checkpoint: the weights only exist after the engine's device
        // materialisation, so only the forward tools can use it.
        _model.reset();
        _checkpointDir = path;
    } else {
        _model = std::make_unique<ModelView>(path);
        _checkpointDir.clear();
    }
    _tokenizerGguf = std::move(tokenizerGguf);
}

const ModelView& InspectTools::model() const {
    if (!_model) {
        if (!_checkpointDir.empty()) {
            throw std::runtime_error("'" + _checkpointDir + "' is an NVFP4 checkpoint — the weight "
                                     "inspection tools are GGUF-only; the forward tools work on it");
        }
        throw std::runtime_error("no model loaded — call open_model first");
    }
    return *_model;
}

void InspectTools::registerOn(McpServer& server) {
    server.addTool(
        "open_model",
        "Open a model: a GGUF file (weight inspection via read-only mmap + forward tools) or an "
        "NVFP4 checkpoint directory (forward tools only; needs a CUDA build/GPU). "
        "'tokenizer_gguf' supplies the tokenizer for NVFP4 checkpoints that need one.",
        schema({{"path", {{"type", "string"}, {"description", "absolute path to a .gguf file or checkpoint dir"}}},
                {"tokenizer_gguf", {{"type", "string"}}}},
               {"path"}),
        [this](const json& a) {
            open(a.at("path").get<std::string>(), a.value("tokenizer_gguf", std::string{}));
            if (!_model) {
                return json{{"path", _checkpointDir}, {"format", "nvfp4"},
                            {"weight_inspection", false}};
            }
            return json{{"path", _model->path()}, {"format", "gguf"},
                        {"base_fingerprint", _model->fingerprint()}};
        });

    server.addTool(
        "model_info",
        "Summary of the loaded model: architecture, base fingerprint (what runes bind to), "
        "tensor count, bytes per tensor type.",
        schema(json::object()),
        [this](const json&) {
            const ModelView& m = model();
            std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> byType;
            for (const auto& t : m.reader().tensors()) {
                auto& e = byType[std::string{gg::typeInfo(t.type).name}];
                ++e.first;
                e.second += t.nbytes;
            }
            json types = json::object();
            for (const auto& [k, v] : byType) {
                types[k] = {{"tensors", v.first}, {"bytes", v.second}};
            }
            return json{{"path", m.path()},
                        {"format", "gguf"},
                        {"architecture", m.architecture()},
                        {"base_fingerprint", m.fingerprint()},
                        {"tensor_count", m.reader().tensorCount()},
                        {"tensor_bytes", m.reader().totalTensorBytes()},
                        {"types", types}};
        });

    server.addTool(
        "list_tensors",
        "List tensors (name, type, dims, rows/cols, bytes, whether a rune can patch it). "
        "Optional substring filter and limit.",
        schema({{"filter", {{"type", "string"}}}, {"limit", {{"type", "integer"}}}}),
        [this](const json& a) {
            const std::string filter = a.value("filter", std::string{});
            const std::size_t limit  = a.value("limit", std::size_t{200});
            json list = json::array();
            std::size_t matched = 0;
            for (const auto& t : model().reader().tensors()) {
                if (!filter.empty() && t.name.find(filter) == std::string::npos) {
                    continue;
                }
                ++matched;
                if (list.size() < limit) {
                    list.push_back(tensorJson(t));
                }
            }
            return json{{"matched", matched}, {"tensors", list}};
        });

    server.addTool(
        "tensor_stats",
        "Statistics of one tensor (or a row range): mean, std, min, max, abs-max, L2, "
        "NaN/Inf counts, per-row L2 norm min/mean/max. Dequantises on the host.",
        schema({{"name", {{"type", "string"}}},
                {"row_begin", {{"type", "integer"}}},
                {"row_end", {{"type", "integer"}}}},
               {"name"}),
        [this](const json& a) {
            const ModelView& m = model();
            const auto& t = m.require(a.at("name").get<std::string>());
            std::optional<std::uint64_t> rb, re;
            if (a.contains("row_begin")) rb = a["row_begin"].get<std::uint64_t>();
            if (a.contains("row_end")) re = a["row_end"].get<std::uint64_t>();
            const TensorStats s = m.stats(t, rb, re);
            return json{{"tensor", tensorJson(t)},
                        {"rows", s.rows}, {"elements", s.elements},
                        {"mean", s.mean}, {"std", s.stddev}, {"min", s.min}, {"max", s.max},
                        {"abs_max", s.absMax}, {"l2", s.l2},
                        {"nan", s.nanCount}, {"inf", s.infCount},
                        {"row_norm", {{"min", s.rowNormMin}, {"mean", s.rowNormMean},
                                      {"max", s.rowNormMax}}}};
        });

    server.addTool(
        "tensor_hash",
        "Content hash (fnv1a64) of a tensor as loaded — the tensor_hash a rune op must carry.",
        schema({{"name", {{"type", "string"}}}}, {"name"}),
        [this](const json& a) {
            const ModelView& m = model();
            const auto& t = m.require(a.at("name").get<std::string>());
            const auto b = m.bytes(t);
            return json{{"name", t.name}, {"tensor_hash", rn::tensorContentHash(b.data(), b.size())}};
        });

    server.addTool(
        "rune_inspect", "Decode a rune file's manifest (name, base, provenance, ops).",
        schema({{"path", {{"type", "string"}}}}, {"path"}),
        [](const json& a) {
            const rn::RuneFile r = rn::RuneFile::load(a.at("path").get<std::string>());
            const rn::RuneManifest& m = r.manifest();
            json ops = json::array();
            for (const auto& op : m.ops) {
                json o{{"op", std::string{rn::opKindName(op.kind)}},
                       {"target", op.target},
                       {"tensor_hash", op.tensorHash},
                       {"payload", op.payload},
                       {"applicable_in_this_build", rn::opKindApplicable(op.kind)}};
                if (!op.payload.empty()) {
                    o["payload_values"] = r.payload(op.payload).size();
                }
                if (op.kind == rn::RuneOpKind::WriteSlice) {
                    o["row_begin"] = op.rowBegin;
                    o["row_end"]   = op.rowEnd;
                } else if (op.kind == rn::RuneOpKind::Scale) {
                    o["axis"] = op.axis == rn::RuneScaleAxis::Rows ? "rows" : "cols";
                }
                ops.push_back(std::move(o));
            }
            return json{{"name", m.name},
                        {"version", m.version},
                        {"base_fingerprint", m.baseFingerprint},
                        {"provenance", json::parse(m.provenanceJson)},
                        {"ops", ops}};
        });

    server.addTool(
        "rune_check",
        "Dry-run rune files (in stack order) against the loaded model with the engine's own "
        "Futhark checks: base fingerprint, op applicability, target type/shape, per-op tensor "
        "hash chain, payload sizes. Nothing is written. Returns per-rune touched tensors with "
        "hash before/after, or the exact refusal the engine would log.",
        schema({{"paths", {{"type", "array"}, {"items", {{"type", "string"}}}}}}, {"paths"}),
        [this](const json& a) {
            const ModelView& m = model();
            const std::vector<std::string> paths = stringList(a, "paths");
            const gg::WeightsMap weights = m.readOnlyWeights();
            rn::FutharkDeviceIo io;
            io.readback = [](void* dst, const void* src, std::size_t n) { std::memcpy(dst, src, n); };
            io.upload   = [](void*, const void*, std::size_t) {};   // dry run: never write
            rn::Futhark futhark{std::move(io)};
            const auto res = futhark.applyFiles(paths, m.fingerprint(), weights);
            if (!res) {
                return json{{"ok", false}, {"error", res.error()},
                            {"base_fingerprint", m.fingerprint()}};
            }
            json runes = json::array();
            for (const auto& rep : res->reports) {
                json touched = json::array();
                for (const auto& t : rep.touched) {
                    touched.push_back({{"tensor", t.name}, {"hash_before", t.hashBefore},
                                       {"hash_after", t.hashAfter}});
                }
                runes.push_back({{"name", rep.runeName}, {"ops", rep.opsApplied},
                                 {"touched", touched}});
            }
            return json{{"ok", true}, {"base_fingerprint", m.fingerprint()}, {"runes", runes}};
        });
}

} // namespace mimirmind::hugin

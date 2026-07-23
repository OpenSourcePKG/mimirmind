// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/HfQuantConfig.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>

namespace mimirmind::core::modelopt {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("hf_quant_config: " + msg);
}

/// Minimal glob match supporting `*` (matches any run of characters,
/// including none). No `?` or char classes — the ModelOpt exclude patterns
/// only use `*` (e.g. `mtp*`, `mtp.layers.0*`).
bool globMatch(std::string_view pat, std::string_view text) noexcept {
    std::size_t p = 0, t = 0;
    std::size_t star = std::string_view::npos;
    std::size_t tStar = 0;
    while (t < text.size()) {
        if (p < pat.size() && pat[p] == '*') {
            star  = p++;      // remember star position, matching zero chars first
            tStar = t;
        } else if (p < pat.size() && pat[p] == text[t]) {
            ++p;
            ++t;
        } else if (star != std::string_view::npos) {
            p = star + 1;     // backtrack: let the star swallow one more char
            t = ++tStar;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*') {
        ++p;
    }
    return p == pat.size();
}

} // namespace

HfQuantConfig HfQuantConfig::parse(std::string_view jsonText) {
    nlohmann::json top =
        nlohmann::json::parse(jsonText.begin(), jsonText.end(), nullptr,
                              /*allow_exceptions=*/false);
    if (top.is_discarded()) {
        fail("not valid JSON");
    }
    if (!top.is_object()) {
        fail("top level is not an object");
    }
    if (!top.contains("quantization") || !top.at("quantization").is_object()) {
        fail("missing or non-object 'quantization'");
    }
    const auto& q = top.at("quantization");

    HfQuantConfig cfg;

    if (q.contains("quant_algo") && q.at("quant_algo").is_string()) {
        cfg._topLevelAlgo = q.at("quant_algo").get<std::string>();
    }
    if (q.contains("kv_cache_quant_algo") && q.at("kv_cache_quant_algo").is_string()) {
        cfg._kvCacheAlgo = q.at("kv_cache_quant_algo").get<std::string>();
    }

    if (q.contains("exclude_modules")) {
        const auto& ex = q.at("exclude_modules");
        if (!ex.is_array()) {
            fail("'exclude_modules' is not an array");
        }
        for (const auto& pat : ex) {
            if (!pat.is_string()) {
                fail("'exclude_modules' entry is not a string");
            }
            cfg._exclude.push_back(pat.get<std::string>());
        }
    }

    if (q.contains("quantized_layers")) {
        const auto& ql = q.at("quantized_layers");
        if (!ql.is_object()) {
            fail("'quantized_layers' is not an object");
        }
        for (const auto& [module, spec] : ql.items()) {
            if (!spec.is_object() || !spec.contains("quant_algo")
                || !spec.at("quant_algo").is_string()) {
                fail("quantized_layers['" + module + "'] lacks a string quant_algo");
            }
            const std::string algo = spec.at("quant_algo").get<std::string>();
            const auto scheme = schemeFromQuantAlgo(algo);
            if (!scheme.has_value()) {
                fail("quantized_layers['" + module + "'] has unsupported quant_algo '"
                     + algo + "'");
            }
            ModuleQuant mq;
            mq.scheme = *scheme;
            if (spec.contains("group_size")) {
                const auto& gs = spec.at("group_size");
                if (!gs.is_number_unsigned()) {
                    fail("quantized_layers['" + module + "'] group_size is not an unsigned integer");
                }
                mq.groupSize = static_cast<std::uint16_t>(gs.get<std::uint64_t>());
            }
            cfg._modules.emplace(module, mq);
        }
    }

    // Uniform fallback: a non-MIXED top-level scheme with no per-module map.
    if (cfg._modules.empty() && !cfg._topLevelAlgo.empty()) {
        if (const auto s = schemeFromQuantAlgo(cfg._topLevelAlgo)) {
            cfg._uniform.scheme = *s;
            if (q.contains("group_size") && q.at("group_size").is_number_unsigned()) {
                cfg._uniform.groupSize =
                    static_cast<std::uint16_t>(q.at("group_size").get<std::uint64_t>());
            }
            cfg._hasUniform = true;
        }
    }

    return cfg;
}

bool HfQuantConfig::isExcluded(std::string_view name) const noexcept {
    for (const std::string& pat : _exclude) {
        if (globMatch(pat, name)) {
            return true;
        }
    }
    return false;
}

std::optional<ModuleQuant> HfQuantConfig::resolve(std::string_view name) const {
    if (isExcluded(name)) {
        return std::nullopt;
    }

    if (!_modules.empty()) {
        // Longest boundary-prefix match: trim trailing `.segment`s until a
        // module key matches exactly. Keys carry no trailing dot, so an
        // exact find respects `.` boundaries.
        std::string key(name);
        while (true) {
            const auto it = _modules.find(key);
            if (it != _modules.end()) {
                return it->second;
            }
            const auto pos = key.rfind('.');
            if (pos == std::string::npos) {
                break;
            }
            key.resize(pos);
        }
        return std::nullopt;
    }

    if (_hasUniform) {
        return _uniform;
    }
    return std::nullopt;
}

std::optional<ModelOptQuantScheme>
HfQuantConfig::schemeForTensor(std::string_view name) const {
    if (const auto mq = resolve(name)) {
        return mq->scheme;
    }
    return std::nullopt;
}

} // namespace mimirmind::core::modelopt
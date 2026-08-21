// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/CompressedTensorsConfig.hpp"

#include <nlohmann/json.hpp>

#include <regex>
#include <string_view>

namespace mimirmind::core::modelopt {

namespace {

// A target/ignore entry matches a module path either as a `re:`-prefixed
// regex (llm-compressor convention) or as a literal exact string.
bool entryMatches(const std::string& entry, const std::string& modulePath) {
    constexpr std::string_view kRe{"re:"};
    if (entry.size() > kRe.size()
        && std::string_view{entry}.substr(0, kRe.size()) == kRe) {
        try {
            const std::regex re(entry.substr(kRe.size()));
            return std::regex_search(modulePath, re);
        } catch (const std::regex_error&) {
            return false;  // a malformed pattern never matches
        }
    }
    return entry == modulePath;
}

// Strip a trailing ".weight" so matching is on the module path, which is what
// the config_groups regex targets (`re:.*self_attn\.(q|k|v|o)_proj$`) key on.
std::string modulePathOf(const std::string& hfWeightName) {
    constexpr std::string_view kW{".weight"};
    if (hfWeightName.size() > kW.size()
        && std::string_view{hfWeightName}.substr(hfWeightName.size() - kW.size())
               == kW) {
        return hfWeightName.substr(0, hfWeightName.size() - kW.size());
    }
    return hfWeightName;
}

} // namespace

CompressedTensorsConfig
CompressedTensorsConfig::parse(const std::string& configJson) {
    CompressedTensorsConfig out;
    nlohmann::json top =
        nlohmann::json::parse(configJson, nullptr, /*allow_exceptions=*/false);
    if (top.is_discarded() || !top.is_object()) {
        return out;  // invalid
    }
    const auto qcIt = top.find("quantization_config");
    if (qcIt == top.end() || !qcIt->is_object()) {
        return out;  // no compressed-tensors config
    }
    const auto& qc = *qcIt;
    // Only claim this path for compressed-tensors checkpoints.
    if (qc.value("quant_method", std::string{}) != "compressed-tensors") {
        return out;
    }

    const auto cgIt = qc.find("config_groups");
    if (cgIt != qc.end() && cgIt->is_object()) {
        // nlohmann's object is key-ordered (group_0 before group_1), so a
        // layer-specific FP8 override in group_0 wins over the broad NVFP4 rule.
        for (const auto& [name, grp] : cgIt->items()) {
            if (!grp.is_object()) continue;
            Group g;
            const auto tIt = grp.find("targets");
            if (tIt != grp.end() && tIt->is_array()) {
                for (const auto& t : *tIt) {
                    if (t.is_string()) g.targets.push_back(t.get<std::string>());
                }
            }
            const auto wIt = grp.find("weights");
            if (wIt != grp.end() && wIt->is_object()) {
                g.numBits = wIt->value("num_bits", 0);
            }
            if (!g.targets.empty() && g.numBits > 0) {
                out._groups.push_back(std::move(g));
            }
        }
    }

    const auto igIt = qc.find("ignore");
    if (igIt != qc.end() && igIt->is_array()) {
        for (const auto& e : *igIt) {
            if (e.is_string()) out._ignore.push_back(e.get<std::string>());
        }
    }

    out._valid = !out._groups.empty();
    return out;
}

CompressedTensorsConfig::Scheme
CompressedTensorsConfig::schemeForTensor(const std::string& hfWeightName) const {
    const std::string mod = modulePathOf(hfWeightName);
    for (const auto& ig : _ignore) {
        if (entryMatches(ig, mod)) return Scheme::Bf16;
    }
    for (const auto& g : _groups) {
        for (const auto& t : g.targets) {
            if (entryMatches(t, mod)) {
                return g.numBits == 4 ? Scheme::Nvfp4 : Scheme::Fp8;
            }
        }
    }
    return Scheme::Bf16;
}

} // namespace mimirmind::core::modelopt

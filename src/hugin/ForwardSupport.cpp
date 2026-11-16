// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardSupport.hpp"

#include "hugin/LogitMath.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mimirmind::hugin::forward_support {

using nlohmann::json;

json schema(json properties, json required) {
    return {{"type", "object"}, {"properties", std::move(properties)}, {"required", std::move(required)}};
}

json textProps(json extra) {
    json p{{"text", {{"type", "string"}, {"description", "raw text (no chat template is applied)"}}},
           {"tokens", {{"type", "array"}, {"items", {{"type", "integer"}}},
                       {"description", "token ids; used instead of text when given"}}},
           {"add_bos", {{"type", "boolean"}, {"description", "prepend BOS when tokenizing text (default true)"}}}};
    for (auto it = extra.begin(); it != extra.end(); ++it) {
        p[it.key()] = it.value();
    }
    return p;
}

std::vector<std::int32_t> inputTokens(const Engine& e, const json& a) {
    std::vector<std::int32_t> ids;
    if (a.contains("tokens")) {
        ids = a["tokens"].get<std::vector<std::int32_t>>();
    } else if (a.contains("text")) {
        ids = e.tokenizer().encode(a["text"].get<std::string>(), a.value("add_bos", true));
    } else {
        throw std::runtime_error("pass 'text' or 'tokens'");
    }
    if (ids.empty()) {
        throw std::runtime_error("input tokenizes to zero tokens");
    }
    return ids;
}

std::string piece(const Engine& e, std::int32_t id) {
    return std::string{e.tokenizer().tokenText(id)};
}

/// One stateless forward: empty cache in, empty cache out.
std::vector<std::vector<float>> forward(Engine& e, std::span<const std::int32_t> ids,
                                        Engine::ForwardProbe* probe) {
    e.resetCache();
    auto logits = e.forwardVerify(ids, probe);
    e.resetCache();
    return logits;
}

std::vector<double> logProbs(const Engine& e, std::vector<float> logits) {
    LogitMath::softcap(logits, e.finalLogitSoftcap());
    return LogitMath::logSoftmax(logits);
}

json topJson(const Engine& e, const std::vector<double>& logp, std::size_t k) {
    json out = json::array();
    for (const std::int32_t id : LogitMath::topK(logp, k)) {
        out.push_back({{"id", id}, {"piece", piece(e, id)},
                       {"prob", std::exp(logp[static_cast<std::size_t>(id)])}});
    }
    return out;
}

/// Requested blocks (default: all), validated against the model.
std::vector<std::uint32_t> blockList(const Engine& e, const json& a) {
    const std::uint32_t n = e.config().blockCount;
    std::vector<std::uint32_t> out;
    if (a.contains("blocks")) {
        for (const auto& v : a["blocks"]) {
            const auto b = v.get<std::int64_t>();
            if (b < 0 || b >= static_cast<std::int64_t>(n)) {
                throw std::runtime_error("block " + std::to_string(b) + " out of range [0," +
                                         std::to_string(n) + ")");
            }
            out.push_back(static_cast<std::uint32_t>(b));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    } else {
        for (std::uint32_t b = 0; b < n; ++b) {
            out.push_back(b);
        }
    }
    return out;
}

std::size_t positionArg(const json& a, std::size_t n) {
    if (!a.contains("position")) {
        return n - 1;
    }
    const auto p = a["position"].get<std::int64_t>();
    const auto q = p < 0 ? static_cast<std::int64_t>(n) + p : p;   // -1 = last
    if (q < 0 || q >= static_cast<std::int64_t>(n)) {
        throw std::runtime_error("position " + std::to_string(p) + " out of range for " +
                                 std::to_string(n) + " tokens");
    }
    return static_cast<std::size_t>(q);
}

std::vector<std::int32_t> textTokens(const Engine& e, const std::string& text, bool addBos) {
    auto ids = e.tokenizer().encode(text, addBos);
    if (ids.empty()) {
        throw std::runtime_error("input tokenizes to zero tokens");
    }
    return ids;
}

} // namespace mimirmind::hugin::forward_support

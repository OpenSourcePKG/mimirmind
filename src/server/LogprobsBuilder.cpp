// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/LogprobsBuilder.hpp"

#include "model/Tokenizer.hpp"

#include <string>

namespace mimirmind::server {

nlohmann::json LogprobsBuilder::entryJson(std::int32_t tokId,
                                          const runtime::TokenLogprobs& lp,
                                          const model::Tokenizer& tok) {
    const auto bytesOf = [](const std::string& s) {
        nlohmann::json b = nlohmann::json::array();
        for (unsigned char c : s) { b.push_back(static_cast<int>(c)); }
        return b;
    };
    const auto decode1 = [&tok](std::int32_t id) {
        return tok.decode(std::vector<std::int32_t>{id}, /*skipSpecial=*/false);
    };
    const std::string piece = decode1(tokId);
    nlohmann::json top = nlohmann::json::array();
    for (const auto& tp : lp.top) {
        const std::string p = decode1(tp.token);
        top.push_back({{"token", p},
                       {"logprob", tp.logprob},
                       {"bytes", bytesOf(p)}});
    }
    return nlohmann::json{{"token", piece},
                          {"logprob", lp.logprob},
                          {"bytes", bytesOf(piece)},
                          {"top_logprobs", std::move(top)}};
}

nlohmann::json LogprobsBuilder::buildJson(
        const std::vector<std::int32_t>&                  generated,
        const std::vector<runtime::TokenLogprobs>&        lp,
        const model::Tokenizer&                           tok) {
    nlohmann::json content = nlohmann::json::array();
    for (std::size_t i = 0; i < generated.size() && i < lp.size(); ++i) {
        if (!lp[i].captured) { continue; }
        content.push_back(entryJson(generated[i], lp[i], tok));
    }
    return nlohmann::json{{"content", std::move(content)}};
}

} // namespace mimirmind::server

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/ModelProvider.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <vector>

namespace mimirmind::server {

/**
 * Shape the additive `models` block of GET /v1/system/memory from a snapshot
 * of resident-model memory (RequestDispatcher::residentModelsMemory()).
 *
 * Pure — it touches no engine, device, or lock — so it is unit-testable
 * without a GPU. `poolMode` selects the reported `mode` string ("pool" vs
 * "eager"); `poolCapacity` is the pool's K (1 in eager mode). An empty
 * `resident` yields `{available:false}` with a mode-specific reason (e.g. a
 * cold pool with nothing materialized).
 *
 * Contract (a downstream consumer parses these exact keys):
 *   { "available": true, "mode": "eager"|"pool", "capacity": <K>,
 *     "resident": [ { "id", "title", "default",
 *                     "role": "chat"|"embedding"|"rerank",
 *                     "in_generative_pool": <bool>,
 *                     "weight_bytes",
 *                     "kv": { "serving_active": <bool>,
 *                             "resident_bytes"?, "num_blocks"? } } ] }
 * `mode`/`capacity` describe the GENERATIVE (chat) pool only; encoder engines
 * (embedding/rerank) are always present in `resident[]` with
 * `in_generative_pool:false`. The `kv` object is ALWAYS present so clients
 * render every model consistently: `serving_active:false` (no
 * resident_bytes/num_blocks) for models with no paged KV cache — which is
 * every encoder, and any chat model on the single-session path.
 */
[[nodiscard]] inline nlohmann::json
buildModelsMemoryJson(const std::vector<ResidentModelMemory>& resident,
                      bool poolMode, std::size_t poolCapacity) {
    using nlohmann::json;

    if (resident.empty()) {
        return json{
            {"available", false},
            {"reason", poolMode ? "pool mode: no model currently materialized"
                                : "no resident model engines"}};
    }

    json arr = json::array();
    for (const auto& r : resident) {
        json entry{
            {"id",                 r.id},
            {"title",              r.title},
            {"default",            r.isDefault},
            {"role",               r.role},
            {"in_generative_pool", r.inGenerativePool},
            {"weight_bytes",       r.weightBytes},
        };
        // `kv` is ALWAYS present (self-describing). Models with an active
        // paged serving cache carry the byte/block detail; encoders and any
        // single-session model report `serving_active:false` with no detail
        // (they hold no paged KV cache) so a client never has to special-case
        // a missing field.
        if (r.servingActive) {
            entry["kv"] = json{
                {"serving_active", true},
                {"resident_bytes", r.kvResidentBytes},
                {"num_blocks",     r.kvNumBlocks},
            };
        } else {
            entry["kv"] = json{{"serving_active", false}};
        }
        arr.push_back(std::move(entry));
    }

    return json{
        {"available", true},
        {"mode",      poolMode ? "pool" : "eager"},
        {"capacity",  poolCapacity},
        {"resident",  std::move(arr)},
    };
}

} // namespace mimirmind::server

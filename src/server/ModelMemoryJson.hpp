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
 *                     "weight_bytes",
 *                     "kv": { "serving_active", "resident_bytes", "num_blocks" }? } ] }
 * The `kv` object is omitted when the model has no active serving cache.
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
            {"id",           r.id},
            {"title",        r.title},
            {"default",      r.isDefault},
            {"weight_bytes", r.weightBytes},
        };
        // Omit `kv` entirely when the engine has no active serving cache
        // (single-session generate() path, or a not-yet-serving model).
        if (r.servingActive) {
            entry["kv"] = json{
                {"serving_active", true},
                {"resident_bytes", r.kvResidentBytes},
                {"num_blocks",     r.kvNumBlocks},
            };
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

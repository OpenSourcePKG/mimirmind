// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/RequestTracker.hpp"

namespace mimirmind::server {

using nlohmann::json;

void RequestTracker::begin(std::string id, std::size_t promptTokens,
                             std::size_t maxNew, bool streaming) {
    std::lock_guard<std::mutex> lk{_mutex};
    CurrentRequest r;
    r.request_id     = id;
    r.started_at     = std::chrono::steady_clock::now();
    r.prompt_tokens  = promptTokens;
    r.max_new_tokens = maxNew;
    r.streaming      = streaming;
    _requests[std::move(id)] = std::move(r);
}

void RequestTracker::end(const std::string& id) noexcept {
    std::lock_guard<std::mutex> lk{_mutex};
    _requests.erase(id);
}

void RequestTracker::updatePrefillProgress(const std::string& id,
                                            std::size_t blocksDone,
                                            std::size_t blocksTotal,
                                            double      elapsedMs) {
    std::lock_guard<std::mutex> lk{_mutex};
    auto it = _requests.find(id);
    if (it == _requests.end()) return;
    it->second.prefill_blocks_done  = blocksDone;
    it->second.prefill_blocks_total = blocksTotal;
    it->second.prefill_elapsed_ms   = elapsedMs;
}

void RequestTracker::markPrefillDone(const std::string& id) {
    std::lock_guard<std::mutex> lk{_mutex};
    auto it = _requests.find(id);
    if (it == _requests.end()) return;
    it->second.prefill_done = true;
}

void RequestTracker::incrementDecodeTokens(const std::string& id) {
    std::lock_guard<std::mutex> lk{_mutex};
    auto it = _requests.find(id);
    if (it == _requests.end()) return;
    ++it->second.decode_tokens_emitted;
}

json RequestTracker::requestJson(const CurrentRequest& r) {
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - r.started_at).count();
    return json{
        {"request_id",      r.request_id},
        {"streaming",       r.streaming},
        {"elapsed_ms",      elapsedMs},
        {"prompt_tokens",   r.prompt_tokens},
        {"prefill", {
            {"blocks_done",  r.prefill_blocks_done},
            {"blocks_total", r.prefill_blocks_total},
            {"elapsed_ms",   r.prefill_elapsed_ms},
            {"done",         r.prefill_done},
        }},
        {"decode", {
            {"tokens_emitted", r.decode_tokens_emitted},
            {"max_new_tokens", r.max_new_tokens},
        }},
    };
}

json RequestTracker::buildStatusBlock() const {
    std::lock_guard<std::mutex> lk{_mutex};
    if (_requests.empty()) {
        return json{{"active", false}};
    }

    // Deterministic ordering by admission time so the single-tenant
    // mirror (and the array) are stable across polls rather than at the
    // mercy of unordered_map bucket order.
    const CurrentRequest* oldest = nullptr;
    json requests = json::array();
    for (const auto& [id, r] : _requests) {
        requests.push_back(requestJson(r));
        if (oldest == nullptr || r.started_at < oldest->started_at) {
            oldest = &r;
        }
    }

    // Mirror the oldest active request's fields at the top level so the
    // pre-multi-tenant `current_request.request_id` contract keeps
    // working; `requests[]` carries the full concurrent set.
    json out = requestJson(*oldest);
    out["active"]   = true;
    out["count"]    = _requests.size();
    out["requests"] = std::move(requests);
    return out;
}

} // namespace mimirmind::server
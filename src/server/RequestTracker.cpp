// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/RequestTracker.hpp"

namespace mimirmind::server {

using nlohmann::json;

void RequestTracker::begin(std::string id, std::size_t promptTokens,
                             std::size_t maxNew, bool streaming) {
    std::lock_guard<std::mutex> lk{_mutex};
    CurrentRequest r;
    r.request_id     = std::move(id);
    r.started_at     = std::chrono::steady_clock::now();
    r.prompt_tokens  = promptTokens;
    r.max_new_tokens = maxNew;
    r.streaming      = streaming;
    _current         = std::move(r);
}

void RequestTracker::end() noexcept {
    std::lock_guard<std::mutex> lk{_mutex};
    _current.reset();
}

void RequestTracker::updatePrefillProgress(std::size_t blocksDone,
                                            std::size_t blocksTotal,
                                            double      elapsedMs) {
    std::lock_guard<std::mutex> lk{_mutex};
    if (!_current.has_value()) return;
    _current->prefill_blocks_done  = blocksDone;
    _current->prefill_blocks_total = blocksTotal;
    _current->prefill_elapsed_ms   = elapsedMs;
}

void RequestTracker::markPrefillDone() {
    std::lock_guard<std::mutex> lk{_mutex};
    if (!_current.has_value()) return;
    _current->prefill_done = true;
}

void RequestTracker::incrementDecodeTokens() {
    std::lock_guard<std::mutex> lk{_mutex};
    if (!_current.has_value()) return;
    ++_current->decode_tokens_emitted;
}

json RequestTracker::buildStatusBlock() const {
    std::lock_guard<std::mutex> lk{_mutex};
    if (!_current.has_value()) {
        return json{{"active", false}};
    }
    const auto& r = *_current;
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - r.started_at).count();
    return json{
        {"active",          true},
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

} // namespace mimirmind::server
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/RequestScheduler.hpp"

#include "runtime/serving/PagedKvBlockAllocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace mimirmind::runtime::serving {

RequestScheduler::RequestScheduler(std::int32_t           tokenBudget,
                                   std::size_t            maxActiveRequests,
                                   PagedKvBlockAllocator* allocator)
    : _prefillScheduler(tokenBudget)
    , _maxActiveRequests(maxActiveRequests)
    , _allocator(allocator)
{
    if (maxActiveRequests == 0) {
        throw std::invalid_argument{
            "RequestScheduler: maxActiveRequests must be > 0"};
    }
    // tokenBudget validation is delegated to ChunkedPrefillScheduler's
    // ctor — same exception type propagates through.
}

std::uint64_t RequestScheduler::admit(std::int32_t promptLength,
                                      std::int32_t maxTokens) noexcept
{
    if (promptLength <= 0 || maxTokens <= 0) return 0;

    RequestStateData r{};
    r.request_id     = _nextRequestId++;
    r.state          = RequestState::Waiting;
    r.prompt_length  = promptLength;
    r.tokens_pending = promptLength;
    r.tokens_decoded = 0;
    r.max_tokens     = maxTokens;
    r.admit_seq      = _nextAdmitSeq++;
    if (_allocator != nullptr) {
        r.kv_sequence.emplace(*_allocator);
    }
    _requests.push_back(std::move(r));
    return _requests.back().request_id;
}

void RequestScheduler::reenqueuePreempted() noexcept {
    for (auto& r : _requests) {
        if (r.state == RequestState::Preempted) {
            r.state          = RequestState::Waiting;
            r.tokens_pending = r.prompt_length;
            r.tokens_decoded = 0;
            // admit_seq stays — preserves LIFO ordering so a
            // just-preempted request doesn't leapfrog fresh admissions.
            // KV blocks were released when the request went to
            // Preempted (preemptOne calls seq.reset()); no further
            // work needed here.
        }
    }
}

void RequestScheduler::promoteWaitingToActive() noexcept {
    std::size_t active = numActive();
    if (active >= _maxActiveRequests) return;

    // Waiting queue is FIFO by admission order — walk _requests in
    // storage order (matches admission order because we push_back on
    // admit and never reorder).
    for (auto& r : _requests) {
        if (active >= _maxActiveRequests) break;
        if (r.state != RequestState::Waiting) continue;
        // Waiting → Prefilling (if prompt has tokens) or directly to
        // Decoding (if prompt was 0 — defensive; admit() rejects that
        // but a re-enqueued preempted request could theoretically have
        // been mutated).
        r.state = (r.tokens_pending > 0)
                    ? RequestState::Prefilling
                    : RequestState::Decoding;
        ++active;
    }
}

std::vector<RequestSlice> RequestScheduler::snapshotActiveSlices() const {
    std::vector<RequestSlice> out;
    out.reserve(_requests.size());
    for (const auto& r : _requests) {
        if (r.state != RequestState::Prefilling &&
            r.state != RequestState::Decoding) continue;
        RequestSlice s{};
        s.request_id     = r.request_id;
        s.tokens_pending = r.tokens_pending;
        s.tokens_decoded = r.tokens_decoded;
        s.is_decoding    = (r.state == RequestState::Decoding);
        out.push_back(s);
    }
    return out;
}

BatchSchedule RequestScheduler::tick() noexcept {
    reenqueuePreempted();
    promoteWaitingToActive();
    const auto slices = snapshotActiveSlices();
    return _prefillScheduler.schedule(slices);
}

void RequestScheduler::commitProgress(
    const BatchSchedule&           executed,
    std::span<const std::uint8_t>  reachedEos) noexcept
{
    // Prefill assignments — advance tokens_pending, promote to
    // Decoding when fully prefilled.
    for (const auto& p : executed.prefills) {
        auto* r = findMut(p.request_id);
        if (r == nullptr) continue;
        if (r->state != RequestState::Prefilling) continue;
        r->tokens_pending -= p.chunk_size;
        if (r->tokens_pending <= 0) {
            r->tokens_pending = 0;
            r->state = RequestState::Decoding;
        }
    }

    // Decode assignments — advance tokens_decoded, promote to
    // Completed on EOS or max_tokens hit. Guard against caller-bug
    // size mismatch (documented in the header): treat missing EOS
    // flags as false.
    const bool eosSizesMatch = (reachedEos.size() == executed.decodes.size());
    for (std::size_t i = 0; i < executed.decodes.size(); ++i) {
        const auto& d = executed.decodes[i];
        auto* r = findMut(d.request_id);
        if (r == nullptr) continue;
        if (r->state != RequestState::Decoding) continue;
        r->tokens_decoded += 1;
        const bool eos = eosSizesMatch ? (reachedEos[i] != 0) : false;
        if (eos || r->tokens_decoded >= r->max_tokens) {
            r->state = RequestState::Completed;
        }
    }
}

std::uint64_t RequestScheduler::preemptOne() noexcept {
    // LIFO: pick the active request with the highest admit_seq. That
    // one has invested the least KV so RECOMPUTE is cheapest.
    RequestStateData* victim = nullptr;
    for (auto& r : _requests) {
        if (r.state != RequestState::Prefilling &&
            r.state != RequestState::Decoding) continue;
        if (victim == nullptr || r.admit_seq > victim->admit_seq) {
            victim = &r;
        }
    }
    if (victim == nullptr) return 0;
    victim->state = RequestState::Preempted;
    // Release KV blocks eagerly so they're available to whoever gets
    // promoted next iteration. If the request is re-enqueued via
    // reenqueuePreempted → promoteWaitingToActive, the caller's
    // Phase-D pass will re-feed the prompt tokens through
    // feedPrefillTokens which will allocate fresh blocks.
    if (victim->kv_sequence.has_value()) {
        victim->kv_sequence->reset();
    }
    return victim->request_id;
}

std::vector<std::uint64_t> RequestScheduler::drainCompleted() noexcept {
    std::vector<std::uint64_t> drained;
    drained.reserve(_requests.size());
    // Two-pass: collect ids first (preserves admission order), then
    // erase-remove in one shot. The erased entries' `kv_sequence`
    // destructors release their blocks back to the pool.
    for (const auto& r : _requests) {
        if (r.state == RequestState::Completed) {
            drained.push_back(r.request_id);
        }
    }
    _requests.erase(
        std::remove_if(_requests.begin(), _requests.end(),
                       [](const RequestStateData& r) {
                           return r.state == RequestState::Completed;
                       }),
        _requests.end());
    return drained;
}

bool RequestScheduler::feedPrefillTokens(
    std::uint64_t                  id,
    std::span<const std::int32_t>  tokens) noexcept
{
    auto* r = findMut(id);
    if (r == nullptr) return false;
    if (!r->kv_sequence.has_value()) return false;
    for (const auto tok : tokens) {
        if (!r->kv_sequence->appendToken(tok)) {
            // Pool exhausted mid-append. Partial state stays as-is —
            // caller should preempt to recover (documented on the
            // header).
            return false;
        }
    }
    return true;
}

bool RequestScheduler::feedDecodeToken(std::uint64_t id,
                                       std::int32_t  token) noexcept
{
    auto* r = findMut(id);
    if (r == nullptr) return false;
    if (!r->kv_sequence.has_value()) return false;
    return r->kv_sequence->appendToken(token);
}

std::size_t RequestScheduler::numWaiting() const noexcept {
    std::size_t n = 0;
    for (const auto& r : _requests) {
        if (r.state == RequestState::Waiting) ++n;
    }
    return n;
}

std::size_t RequestScheduler::numActive() const noexcept {
    std::size_t n = 0;
    for (const auto& r : _requests) {
        if (r.state == RequestState::Prefilling ||
            r.state == RequestState::Decoding) ++n;
    }
    return n;
}

std::size_t RequestScheduler::numCompleted() const noexcept {
    std::size_t n = 0;
    for (const auto& r : _requests) {
        if (r.state == RequestState::Completed) ++n;
    }
    return n;
}

std::size_t RequestScheduler::numPreempted() const noexcept {
    std::size_t n = 0;
    for (const auto& r : _requests) {
        if (r.state == RequestState::Preempted) ++n;
    }
    return n;
}

const RequestStateData* RequestScheduler::find(std::uint64_t id) const noexcept {
    for (const auto& r : _requests) {
        if (r.request_id == id) return &r;
    }
    return nullptr;
}

RequestStateData* RequestScheduler::findMut(std::uint64_t id) noexcept {
    for (auto& r : _requests) {
        if (r.request_id == id) return &r;
    }
    return nullptr;
}

} // namespace mimirmind::runtime::serving
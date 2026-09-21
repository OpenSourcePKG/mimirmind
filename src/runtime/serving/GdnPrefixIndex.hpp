// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Roadmap 5.28.1.2.b — cross-slot GDN prefix sharing: the host-side keyed prefix
// index. Maps a block-aligned token prefix -> an opaque payloadId (the device
// SSM+conv+KV checkpoint image, OWNED BY THE CALLER — the index only tracks
// metadata). Pure host, dependency-free (STL only) so it unit-tests on a bare
// host like the Inc-0 gate, and header-only so it compiles into the serving TU.
//
// Responsibilities (the correctness-critical logic):
//   * key a prefix by an FNV-1a hash of its token ids, block-aligned;
//   * lookup returns the LARGEST cached block-aligned prefix that EXACTLY equals
//     prompt[0,pos) — an exact token compare guards against hash collisions
//     (a hash-only match must NEVER be served: wrong-prefix restore = silent
//     cross-request contamination);
//   * refcount pin/unpin so an entry in use by a live slot is never evicted;
//   * bounded LRU eviction among unpinned entries, returning the evicted
//     payloadIds so the caller frees their device images.
//
// It does NOT own or touch device memory — that plumbing (copy-OUT/in of the
// SSM/conv/KV images) lives in the ServingSession integration layer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace mimirmind::runtime {

class GdnPrefixIndex {
public:
    using PayloadId = std::uint64_t;
    static constexpr PayloadId kNone = 0;   // reserved: "no payload"

    /// `blockSize` = checkpoint granularity in tokens (512, matching Inc-3);
    /// `maxEntries` = LRU cap (0 = unbounded). Prefixes are keyed only at
    /// multiples of blockSize.
    explicit GdnPrefixIndex(std::size_t blockSize = 512, std::size_t maxEntries = 0)
        : _blockSize(blockSize ? blockSize : 512), _maxEntries(maxEntries) {}

    struct Hit {
        bool        found{false};
        std::size_t pos{0};          // block-aligned matched prefix length
        PayloadId   payload{kNone};
    };

    /// Consumer. Returns the largest block-aligned prefix pos (<= promptLen) whose
    /// cached tokens EXACTLY equal prompt[0,pos). Bumps that entry's LRU. Misses
    /// (found=false) when no cached prefix matches — the caller then cold-prefills.
    Hit lookup(const std::int32_t* prompt, std::size_t promptLen) {
        if (prompt == nullptr || promptLen < _blockSize) {
            return {};
        }
        const std::size_t maxPos = (promptLen / _blockSize) * _blockSize;
        for (std::size_t pos = maxPos; pos >= _blockSize; pos -= _blockSize) {
            const std::uint64_t h = hashTokens(prompt, pos);
            const auto it = _byHash.find(h);
            if (it == _byHash.end()) {
                continue;
            }
            for (const PayloadId pid : it->second) {
                Entry& e = _entries.at(pid);
                if (e.tokens.size() == pos &&
                    std::memcmp(e.tokens.data(), prompt,
                                pos * sizeof(std::int32_t)) == 0) {
                    e.lastUsed = ++_tick;               // exact match — safe
                    return {true, pos, pid};
                }
            }
        }
        return {};
    }

    /// Producer. Cache tokens[0,pos) -> payload (pos MUST be a multiple of
    /// blockSize; payload MUST NOT be kNone). If the exact prefix is already
    /// cached, the existing entry is kept and `payload` is appended to `evicted`
    /// (the caller frees the redundant image). New entries start unpinned
    /// (refcount 0) — acquire() right after to pin before the LRU can reclaim.
    /// LRU victims (unpinned) are appended to `evicted`.
    void insert(const std::int32_t* tokens, std::size_t pos, PayloadId payload,
                std::vector<PayloadId>& evicted) {
        if (tokens == nullptr || pos == 0 || (pos % _blockSize) != 0 ||
            payload == kNone || _entries.count(payload) != 0) {
            return;                                     // invalid / id reuse
        }
        const std::uint64_t h = hashTokens(tokens, pos);
        auto& bucket = _byHash[h];
        for (const PayloadId pid : bucket) {
            Entry& e = _entries.at(pid);
            if (e.tokens.size() == pos &&
                std::memcmp(e.tokens.data(), tokens,
                            pos * sizeof(std::int32_t)) == 0) {
                e.lastUsed = ++_tick;                   // dup prefix — keep old
                evicted.push_back(payload);
                return;
            }
        }
        Entry e;
        e.tokens.assign(tokens, tokens + pos);
        e.payload  = payload;
        e.refcount = 0;
        e.lastUsed = ++_tick;
        _entries.emplace(payload, std::move(e));
        bucket.push_back(payload);
        _payloadHash.emplace(payload, h);
        evictIfNeeded(evicted);
    }

    /// Pin/unpin an entry so a live slot's prefix survives eviction. No-op on an
    /// unknown payload (already evicted).
    void acquire(PayloadId payload) {
        const auto it = _entries.find(payload);
        if (it != _entries.end()) {
            ++it->second.refcount;
        }
    }
    void release(PayloadId payload) {
        const auto it = _entries.find(payload);
        if (it != _entries.end() && it->second.refcount > 0) {
            --it->second.refcount;
        }
    }

    [[nodiscard]] std::size_t size() const { return _entries.size(); }
    [[nodiscard]] std::size_t blockSize() const { return _blockSize; }
    [[nodiscard]] std::uint64_t refcount(PayloadId payload) const {
        const auto it = _entries.find(payload);
        return it == _entries.end() ? 0 : it->second.refcount;
    }

private:
    struct Entry {
        std::vector<std::int32_t> tokens;   // [0,pos), for the exact-match guard
        PayloadId                 payload{kNone};
        std::uint64_t             refcount{0};
        std::uint64_t             lastUsed{0};
    };

    static std::uint64_t hashTokens(const std::int32_t* t, std::size_t n) {
        std::uint64_t h = 1469598103934665603ULL;      // FNV-1a 64 offset basis
        for (std::size_t i = 0; i < n; ++i) {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t[i]));
            h *= 1099511628211ULL;                      // FNV prime
        }
        return h;
    }

    void eraseEntry(PayloadId pid) {
        const auto hit = _payloadHash.find(pid);
        if (hit != _payloadHash.end()) {
            auto bit = _byHash.find(hit->second);
            if (bit != _byHash.end()) {
                auto& v = bit->second;
                for (std::size_t i = 0; i < v.size(); ++i) {
                    if (v[i] == pid) { v[i] = v.back(); v.pop_back(); break; }
                }
                if (v.empty()) _byHash.erase(bit);
            }
            _payloadHash.erase(hit);
        }
        _entries.erase(pid);
    }

    void evictIfNeeded(std::vector<PayloadId>& evicted) {
        while (_maxEntries > 0 && _entries.size() > _maxEntries) {
            PayloadId     victim = kNone;
            std::uint64_t best   = std::numeric_limits<std::uint64_t>::max();
            for (const auto& [pid, e] : _entries) {
                if (e.refcount == 0 && e.lastUsed < best) {
                    best = e.lastUsed;
                    victim = pid;
                }
            }
            if (victim == kNone) {
                break;                                  // all pinned — cannot evict
            }
            eraseEntry(victim);
            evicted.push_back(victim);
        }
    }

    std::size_t   _blockSize;
    std::size_t   _maxEntries;
    std::uint64_t _tick{0};
    std::unordered_map<PayloadId, Entry>                     _entries;
    std::unordered_map<std::uint64_t, std::vector<PayloadId>> _byHash;
    std::unordered_map<PayloadId, std::uint64_t>             _payloadHash;
};

} // namespace mimirmind::runtime

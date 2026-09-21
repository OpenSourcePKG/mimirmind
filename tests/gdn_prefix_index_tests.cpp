// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Roadmap 5.28.1.2.b — cross-slot GDN prefix sharing. Correctness gate for the
// host-side keyed prefix index (GdnPrefixIndex): largest-block-aligned-prefix
// match, the exact-token guard against hash collisions (wrong-prefix restore =
// contamination), refcount pinning, and bounded LRU eviction. Pure CPU, no GPU:
//   cmake --build build --target gdn_prefix_index_tests &&
//   build/gdn_prefix_index_tests

#include "TestFramework.hpp"

#include "runtime/serving/GdnPrefixIndex.hpp"

#include <cstdint>
#include <vector>

namespace {

using ::mimirmind::runtime::GdnPrefixIndex;

// Deterministic token vector: tokens[i] = base + i (distinct -> distinct hashes).
std::vector<std::int32_t> seq(std::int32_t base, std::size_t n) {
    std::vector<std::int32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + static_cast<std::int32_t>(i);
    }
    return v;
}

constexpr std::size_t B = 512;

TEST(gdn_prefix_index_hit_at_block_boundary) {
    GdnPrefixIndex idx(B, /*maxEntries=*/0);
    const auto prefix = seq(1000, B);          // cache tokens[0,512)
    std::vector<GdnPrefixIndex::PayloadId> ev;
    idx.insert(prefix.data(), B, /*payload=*/7, ev);
    EXPECT_TRUE(ev.empty());
    EXPECT_EQ(idx.size(), std::size_t{1});

    // A prompt that continues that prefix hits at pos=512.
    auto prompt = seq(1000, B + 40);
    const auto h = idx.lookup(prompt.data(), prompt.size());
    EXPECT_TRUE(h.found);
    EXPECT_EQ(h.pos, B);
    EXPECT_EQ(h.payload, GdnPrefixIndex::PayloadId{7});
}

TEST(gdn_prefix_index_largest_prefix_wins) {
    GdnPrefixIndex idx(B, 0);
    const auto p512  = seq(2000, B);
    const auto p1024 = seq(2000, 2 * B);       // p512 is a prefix of p1024
    std::vector<GdnPrefixIndex::PayloadId> ev;
    idx.insert(p512.data(),  B,     10, ev);
    idx.insert(p1024.data(), 2 * B, 20, ev);
    EXPECT_EQ(idx.size(), std::size_t{2});

    auto prompt = seq(2000, 2 * B + 8);        // matches both; largest = 1024
    const auto h = idx.lookup(prompt.data(), prompt.size());
    EXPECT_TRUE(h.found);
    EXPECT_EQ(h.pos, std::size_t{2 * B});
    EXPECT_EQ(h.payload, GdnPrefixIndex::PayloadId{20});
}

TEST(gdn_prefix_index_wrong_prefix_rejected) {
    GdnPrefixIndex idx(B, 0);
    const auto a = seq(3000, B);
    std::vector<GdnPrefixIndex::PayloadId> ev;
    idx.insert(a.data(), B, 30, ev);

    // A different prefix of the SAME length must NOT match (exact-token guard).
    auto other = seq(9000, B + 5);
    EXPECT_TRUE(!idx.lookup(other.data(), other.size()).found);

    // A prompt shorter than one block never matches.
    auto tiny = seq(3000, B - 1);
    EXPECT_TRUE(!idx.lookup(tiny.data(), tiny.size()).found);

    // Same length, one token differs at the end -> miss.
    auto nearly = seq(3000, B);
    nearly[B - 1] = 424242;
    EXPECT_TRUE(!idx.lookup(nearly.data(), nearly.size()).found);
}

TEST(gdn_prefix_index_lru_evicts_oldest_unpinned) {
    GdnPrefixIndex idx(B, /*maxEntries=*/2);
    std::vector<GdnPrefixIndex::PayloadId> ev;
    const auto a = seq(100, B), b = seq(200, B), c = seq(300, B);
    idx.insert(a.data(), B, 1, ev);   // oldest
    idx.insert(b.data(), B, 2, ev);
    EXPECT_TRUE(ev.empty());
    idx.insert(c.data(), B, 3, ev);   // over cap -> evict LRU (payload 1)
    EXPECT_EQ(ev.size(), std::size_t{1});
    EXPECT_EQ(ev[0], GdnPrefixIndex::PayloadId{1});
    EXPECT_EQ(idx.size(), std::size_t{2});
    // Evicted prefix no longer hits; the survivors do.
    auto pa = seq(100, B + 1), pc = seq(300, B + 1);
    EXPECT_TRUE(!idx.lookup(pa.data(), pa.size()).found);
    EXPECT_TRUE(idx.lookup(pc.data(), pc.size()).found);
}

TEST(gdn_prefix_index_pinned_survives_eviction) {
    GdnPrefixIndex idx(B, /*maxEntries=*/1);
    std::vector<GdnPrefixIndex::PayloadId> ev;
    const auto a = seq(500, B), b = seq(600, B);
    idx.insert(a.data(), B, 1, ev);
    idx.acquire(1);                    // pin payload 1
    idx.insert(b.data(), B, 2, ev);    // over cap, but 1 is pinned -> evict 2? no:
    // insert adds 2 then evicts LRU-unpinned; 1 is pinned so 2 (newer, unpinned)
    // is the only evictable -> 2 is evicted, pinned 1 stays.
    EXPECT_EQ(ev.size(), std::size_t{1});
    EXPECT_EQ(ev[0], GdnPrefixIndex::PayloadId{2});
    EXPECT_EQ(idx.refcount(1), std::uint64_t{1});
    auto pa = seq(500, B + 1);
    EXPECT_TRUE(idx.lookup(pa.data(), pa.size()).found);   // pinned prefix alive

    // Release then insert again -> now 1 is evictable.
    idx.release(1);
    const auto c = seq(700, B);
    idx.insert(c.data(), B, 3, ev);
    EXPECT_EQ(idx.size(), std::size_t{1});                 // capped at 1
}

TEST(gdn_prefix_index_duplicate_prefix_keeps_old_payload) {
    GdnPrefixIndex idx(B, 0);
    std::vector<GdnPrefixIndex::PayloadId> ev;
    const auto a = seq(800, B);
    idx.insert(a.data(), B, 1, ev);
    idx.insert(a.data(), B, 2, ev);    // same prefix, new payload -> keep old (1)
    EXPECT_EQ(idx.size(), std::size_t{1});
    EXPECT_EQ(ev.size(), std::size_t{1});
    EXPECT_EQ(ev[0], GdnPrefixIndex::PayloadId{2});   // caller frees the redundant image
    auto pa = seq(800, B + 1);
    const auto h = idx.lookup(pa.data(), pa.size());
    EXPECT_TRUE(h.found);
    EXPECT_EQ(h.payload, GdnPrefixIndex::PayloadId{1});
}

} // namespace

int main() {
    return mm::test::run();
}

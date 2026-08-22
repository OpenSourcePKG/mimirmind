// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Pure-CPU tests for the M-Munin.3 worker-side model cache
// (runtime::serving::AttachedModelPool). No GPU: a fake payload with
// live/created counters stands in for the materialized model bundle, so the
// switch / evict / pin / serialize logic is exercised directly.

#include "TestFramework.hpp"

#include "runtime/serving/AttachedModelPool.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using ::mimirmind::runtime::serving::AttachedModelPool;
using namespace std::chrono_literals;

namespace {

struct Stats {
    std::atomic<int> live{0};
    std::atomic<int> created{0};
};

struct FakeModel {
    std::string id;
    Stats*      st;
    FakeModel(std::string i, Stats* s) : id{std::move(i)}, st{s} {
        st->live.fetch_add(1);
        st->created.fetch_add(1);
    }
    ~FakeModel() { st->live.fetch_sub(1); }
};

// Factory that (optionally) sleeps to widen the materialization window, so the
// serialize/blocking behaviour is observable.
AttachedModelPool<FakeModel>::Factory mkFactory(Stats& st, std::chrono::milliseconds delay) {
    return [&st, delay](const std::string& id) -> std::unique_ptr<FakeModel> {
        if (delay.count() > 0) {
            std::this_thread::sleep_for(delay);
        }
        return std::make_unique<FakeModel>(id, &st);
    };
}

} // namespace

TEST(pool_hitAndSwitch_evictsLru_k1) {
    Stats st;
    AttachedModelPool<FakeModel> pool{1, {"a", "b"}, mkFactory(st, 0ms)};

    {
        auto ha = pool.acquire("a");
        EXPECT_TRUE(static_cast<bool>(ha));
        EXPECT_EQ(ha->id, "a");
        EXPECT_TRUE(pool.isResident("a"));
        EXPECT_EQ(pool.residentCount(), std::size_t{1});
        EXPECT_EQ(st.created.load(), 1);
        EXPECT_EQ(st.live.load(), 1);
    } // release a

    {
        auto hb = pool.acquire("b");       // K=1 -> evicts a, materializes b
        EXPECT_EQ(hb->id, "b");
        EXPECT_TRUE(pool.isResident("b"));
        EXPECT_TRUE(!pool.isResident("a"));
        EXPECT_EQ(pool.residentCount(), std::size_t{1});
        EXPECT_EQ(st.created.load(), 2);   // a + b materialized
        EXPECT_EQ(st.live.load(), 1);      // a destroyed on evict, only b live
    }
}

TEST(pool_sharedHit_noRematerialize) {
    Stats st;
    AttachedModelPool<FakeModel> pool{1, {"a"}, mkFactory(st, 0ms)};
    auto h1 = pool.acquire("a");
    auto h2 = pool.acquire("a");           // hit — same slot
    EXPECT_EQ(h1.get(), h2.get());
    EXPECT_EQ(st.created.load(), 1);
    EXPECT_EQ(pool.residentCount(), std::size_t{1});
}

TEST(pool_unknownModel_throws) {
    Stats st;
    AttachedModelPool<FakeModel> pool{1, {"a"}, mkFactory(st, 0ms)};
    bool threw = false;
    try {
        auto h = pool.acquire("nope");
        (void)h;
    } catch (const std::exception& x) {
        threw = std::string{x.what()}.find("unknown model") != std::string::npos;
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(st.created.load(), 0);
}

TEST(pool_capacity2_keepsBothWarm) {
    Stats st;
    AttachedModelPool<FakeModel> pool{2, {"a", "b", "c"}, mkFactory(st, 0ms)};
    { auto ha = pool.acquire("a"); }
    { auto hb = pool.acquire("b"); }
    EXPECT_EQ(pool.residentCount(), std::size_t{2});
    { auto ha = pool.acquire("a"); EXPECT_EQ(ha->id, "a"); }  // still warm, no re-materialize
    EXPECT_EQ(st.created.load(), 2);
    // A third distinct model evicts the LRU (a was touched last -> b is LRU).
    { auto hc = pool.acquire("c"); }
    EXPECT_EQ(pool.residentCount(), std::size_t{2});
    EXPECT_TRUE(pool.isResident("c"));
    EXPECT_TRUE(!pool.isResident("b"));
    EXPECT_TRUE(pool.isResident("a"));
}

TEST(pool_pinBlocksEviction_thenSwitches) {
    Stats st;
    AttachedModelPool<FakeModel> pool{1, {"a", "b"}, mkFactory(st, 0ms)};

    auto ha = pool.acquire("a");           // pin a
    EXPECT_TRUE(pool.isResident("a"));

    std::atomic<bool> bDone{false};
    std::thread t([&] {
        auto hb = pool.acquire("b");       // blocks: a is pinned, K=1
        bDone.store(true);
    });

    std::this_thread::sleep_for(80ms);
    EXPECT_TRUE(!bDone.load());            // still blocked
    EXPECT_TRUE(!pool.isResident("b"));

    ha.reset();                            // release a -> switch proceeds
    t.join();
    EXPECT_TRUE(bDone.load());
    EXPECT_TRUE(pool.isResident("b"));
    EXPECT_TRUE(!pool.isResident("a"));
    EXPECT_EQ(st.created.load(), 2);
    EXPECT_EQ(st.live.load(), 1);
}

TEST(pool_concurrentSameModel_materializesOnce) {
    Stats st;
    AttachedModelPool<FakeModel> pool{1, {"a"}, mkFactory(st, 60ms)};

    std::atomic<int>         ok{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) {
        ts.emplace_back([&] {
            auto h = pool.acquire("a");
            if (h && h->id == "a") ok.fetch_add(1);
            std::this_thread::sleep_for(20ms);
        });
    }
    for (auto& th : ts) th.join();

    EXPECT_EQ(ok.load(), 4);
    EXPECT_EQ(st.created.load(), 1);       // one materialization, shared by all
}

TEST(pool_factoryFailure_throwsAndRecovers) {
    Stats st;
    AttachedModelPool<FakeModel> pool{
        1, {"a", "bad"},
        [&st](const std::string& id) -> std::unique_ptr<FakeModel> {
            if (id == "bad") return nullptr;             // signal failure
            return std::make_unique<FakeModel>(id, &st);
        }};

    bool threw = false;
    try {
        auto h = pool.acquire("bad");
        (void)h;
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT_TRUE(threw);

    // The pool recovered (switch token cleared): a normal acquire still works.
    auto h = pool.acquire("a");
    EXPECT_TRUE(pool.isResident("a"));
    EXPECT_EQ(st.created.load(), 1);
}

int main() {
    return mm::test::run();
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Pure-CPU tests for the M-Munin.3 worker-side model cache
// (runtime::serving::AttachedModelPool). No GPU: a fake payload with
// live/created counters stands in for the materialized model bundle, so the
// switch / evict / pin / serialize logic is exercised directly.

#include "TestFramework.hpp"

#include "runtime/serving/AttachedModelPool.hpp"
#include "server/ModelMemoryJson.hpp"
#include "server/ModelProvider.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
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

// --------------------------------------------------------------------------
// buildModelsMemoryJson — the additive `models` block of /v1/system/memory.
// Pure data->JSON shaping, so it needs no engine/GPU: we feed
// ResidentModelMemory snapshots directly.
// --------------------------------------------------------------------------

using ::mimirmind::server::ResidentModelMemory;
using ::mimirmind::server::buildModelsMemoryJson;

TEST(modelsJson_eager_singleDefault_withKv) {
    std::vector<ResidentModelMemory> resident = {
        ResidentModelMemory{"primary", "Qwen3.6-35B-A3B NVFP4", /*isDefault=*/true,
                            /*weightBytes=*/27693715968ULL, /*servingActive=*/true,
                            /*kvResidentBytes=*/5905580032ULL, /*kvNumBlocks=*/32768ULL}};

    const auto j = buildModelsMemoryJson(resident, /*poolMode=*/false, /*poolCapacity=*/1);

    EXPECT_TRUE(j.at("available").get<bool>());
    EXPECT_TRUE(j.at("mode").get<std::string>() == "eager");
    EXPECT_EQ(j.at("capacity").get<std::size_t>(), std::size_t{1});
    EXPECT_EQ(j.at("resident").size(), std::size_t{1});

    const auto& e0 = j.at("resident").at(0);
    EXPECT_TRUE(e0.at("id").get<std::string>() == "primary");
    EXPECT_TRUE(e0.at("title").get<std::string>() == "Qwen3.6-35B-A3B NVFP4");
    EXPECT_TRUE(e0.at("default").get<bool>());
    EXPECT_EQ(e0.at("weight_bytes").get<std::size_t>(), std::size_t{27693715968ULL});
    EXPECT_TRUE(e0.contains("kv"));
    EXPECT_TRUE(e0.at("kv").at("serving_active").get<bool>());
    EXPECT_EQ(e0.at("kv").at("resident_bytes").get<std::size_t>(), std::size_t{5905580032ULL});
    EXPECT_EQ(e0.at("kv").at("num_blocks").get<std::size_t>(), std::size_t{32768ULL});
}

TEST(modelsJson_eager_multiModel_oneDefault_kvOmittedWhenIdle) {
    std::vector<ResidentModelMemory> resident = {
        ResidentModelMemory{"primary", "Primary", true,  1000, true,  200, 8},
        ResidentModelMemory{"embed",   "BGE-M3",  false, 300,  false, 0,   0},
        ResidentModelMemory{"rerank",  "Rerank",  false, 250,  false, 0,   0}};

    const auto j = buildModelsMemoryJson(resident, /*poolMode=*/false, /*poolCapacity=*/1);

    EXPECT_TRUE(j.at("available").get<bool>());
    EXPECT_EQ(j.at("resident").size(), std::size_t{3});

    int defaults = 0;
    std::size_t weightSum = 0;
    for (const auto& e : j.at("resident")) {
        if (e.at("default").get<bool>()) {
            ++defaults;
        }
        weightSum += e.at("weight_bytes").get<std::size_t>();
    }
    EXPECT_EQ(defaults, 1);                       // exactly one default
    EXPECT_EQ(weightSum, std::size_t{1550});      // 1000 + 300 + 250

    // Idle (non-serving) models omit the kv object entirely.
    EXPECT_TRUE(j.at("resident").at(0).contains("kv"));    // primary serves
    EXPECT_TRUE(!j.at("resident").at(1).contains("kv"));   // embed idle
    EXPECT_TRUE(!j.at("resident").at(2).contains("kv"));   // rerank idle
}

TEST(modelsJson_pool_reportsModeAndCapacity) {
    std::vector<ResidentModelMemory> resident = {
        ResidentModelMemory{"qwen4exp", "Qwen3.8-Flash-Next", true, 77000, true, 6000, 4096}};

    const auto j = buildModelsMemoryJson(resident, /*poolMode=*/true, /*poolCapacity=*/4);

    EXPECT_TRUE(j.at("available").get<bool>());
    EXPECT_TRUE(j.at("mode").get<std::string>() == "pool");
    EXPECT_EQ(j.at("capacity").get<std::size_t>(), std::size_t{4});
    EXPECT_EQ(j.at("resident").size(), std::size_t{1});
}

TEST(modelsJson_empty_reportsUnavailableWithReason) {
    const std::vector<ResidentModelMemory> none;

    const auto jPool = buildModelsMemoryJson(none, /*poolMode=*/true, /*poolCapacity=*/2);
    EXPECT_TRUE(!jPool.at("available").get<bool>());
    EXPECT_TRUE(jPool.contains("reason"));
    EXPECT_TRUE(!jPool.contains("resident"));

    const auto jEager = buildModelsMemoryJson(none, /*poolMode=*/false, /*poolCapacity=*/1);
    EXPECT_TRUE(!jEager.at("available").get<bool>());
    EXPECT_TRUE(jEager.contains("reason"));
}

// --------------------------------------------------------------------------
// AttachedModelPool::snapshotResident — read-only slot enumeration that must
// not disturb LRU order. Uses the FakeModel payload above (no engine/GPU).
// --------------------------------------------------------------------------

TEST(pool_snapshotResident_listsResident_withoutPerturbingLru) {
    Stats st;
    AttachedModelPool<FakeModel> pool{2, {"a", "b", "c"}, mkFactory(st, 0ms)};

    // Materialize a then b (a is now the LRU-oldest, b the newest).
    pool.acquire("a").reset();
    pool.acquire("b").reset();
    EXPECT_EQ(pool.residentCount(), std::size_t{2});

    // Snapshot must see both resident ids and leave residentCount untouched.
    std::vector<std::string> seen;
    pool.snapshotResident([&](const std::string& id, const FakeModel&) {
        seen.push_back(id);
    });
    EXPECT_EQ(seen.size(), std::size_t{2});
    bool hasA = false, hasB = false;
    for (const auto& id : seen) {
        hasA = hasA || (id == "a");
        hasB = hasB || (id == "b");
    }
    EXPECT_TRUE(hasA);
    EXPECT_TRUE(hasB);
    EXPECT_EQ(pool.residentCount(), std::size_t{2});

    // The snapshot did NOT touch LRU: acquiring c (pool full, k=2) must still
    // evict the untouched LRU-oldest 'a', not 'b'. If snapshotResident had
    // bumped lastUsed, 'a' would look fresh and 'b' would be evicted instead.
    pool.acquire("c").reset();
    EXPECT_TRUE(!pool.isResident("a"));   // a was the LRU victim
    EXPECT_TRUE(pool.isResident("b"));
    EXPECT_TRUE(pool.isResident("c"));
}

int main() {
    return mm::test::run();
}

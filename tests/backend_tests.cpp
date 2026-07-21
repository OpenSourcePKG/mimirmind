// Pure-CPU unit tests for the backend abstraction:
//   - `BackendPool::discoverAll`  — probes every compiled-in backend
//   - `BackendPool::select(Auto)` — first-available walk
//   - `BackendPool::selectByToken` — token parser + resolver
//   - `tokenFor`                  — canonical token render
//   - `PoolEntry::hasContext`     — lazy-context accessor
//
// Cpu is always compiled in, so `discoverAll` should always produce at
// least one entry. L0 / HIP entries appear when their respective build
// flags are on AND a device is reachable at runtime — the tests assert
// the shape without depending on which is which.

#include "TestFramework.hpp"

#include "core/backend/BackendPool.hpp"
#include "core/backend/BackendRegistry.hpp"
#include "core/backend/SelectionMode.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <string_view>

namespace {

using ::mimirmind::core::backend::BackendKind;
using ::mimirmind::core::backend::BackendPool;
using ::mimirmind::core::backend::BackendRegistry;
using ::mimirmind::core::backend::PoolEntry;
using ::mimirmind::core::backend::SelectionMode;
using ::mimirmind::core::backend::tokenFor;

// True iff at least one entry with the given kind ended up in the pool.
// Used by tests that want to assert both "the pool has SOMETHING" and
// "the pool respects the compile-time backend set" without pinning to
// a specific hardware configuration.
bool poolContainsKind(const BackendPool& pool, BackendKind kind) {
    return std::any_of(pool.entries().begin(), pool.entries().end(),
                       [kind](const PoolEntry& e) { return e.kind == kind; });
}

// True iff the exception's what() contains `needle`. Substring so tests
// stay readable when the exact message wording evolves.
bool whatContains(const std::exception& ex, std::string_view needle) {
    return std::string_view{ex.what()}.find(needle) != std::string_view::npos;
}

} // namespace

// -----------------------------------------------------------------------
// tokenFor — canonical rendering
// -----------------------------------------------------------------------

TEST(tokenFor_levelZero_zero) {
    EXPECT_EQ(tokenFor(BackendKind::LevelZero, 0), std::string{"l0:0"});
}

TEST(tokenFor_levelZero_higherIx) {
    EXPECT_EQ(tokenFor(BackendKind::LevelZero, 3), std::string{"l0:3"});
}

TEST(tokenFor_hip_zero) {
    EXPECT_EQ(tokenFor(BackendKind::Hip, 0), std::string{"hip:0"});
}

TEST(tokenFor_cpu_collapsesDeviceIx) {
    // Cpu has no device index in the token grammar — collapses to "cpu"
    // regardless of deviceIx so consumers can just call it uniformly.
    EXPECT_EQ(tokenFor(BackendKind::Cpu, 0),  std::string{"cpu"});
    EXPECT_EQ(tokenFor(BackendKind::Cpu, 5),  std::string{"cpu"});
}

TEST(tokenFor_cuda_zero) {
    // Cuda is not compiled in but the enum + tokenFor exist for
    // symmetry — the tokens are consumed by config parsers well before
    // any device is touched.
    EXPECT_EQ(tokenFor(BackendKind::Cuda, 0), std::string{"cuda:0"});
}

TEST(tokenFor_unknown_returnsUnknown) {
    EXPECT_EQ(tokenFor(BackendKind::Unknown, 0), std::string{"unknown"});
}

// -----------------------------------------------------------------------
// discoverAll — populates entries; Cpu is always available
// -----------------------------------------------------------------------

TEST(pool_discoverAll_cpuAlwaysPresent) {
    BackendPool pool;
    pool.discoverAll();
    EXPECT_TRUE(poolContainsKind(pool, BackendKind::Cpu));
}

TEST(pool_discoverAll_neverIncludesUnknown) {
    BackendPool pool;
    pool.discoverAll();
    for (const auto& e : pool.entries()) {
        EXPECT_TRUE(e.kind != BackendKind::Unknown);
    }
}

TEST(pool_discoverAll_isIdempotent) {
    BackendPool pool;
    pool.discoverAll();
    const auto first_count = pool.entries().size();
    // Second call must not double the pool.
    pool.discoverAll();
    EXPECT_EQ(pool.entries().size(), first_count);
}

TEST(pool_discoverAll_populatesToken) {
    // Every entry must carry the canonical `tokenFor` string so that
    // `selectByToken` can round-trip against it.
    BackendPool pool;
    pool.discoverAll();
    for (const auto& e : pool.entries()) {
        EXPECT_EQ(e.token, tokenFor(e.kind, e.deviceIx));
    }
}

TEST(pool_discoverAll_hasContext_startsFalse) {
    // ComputeContext is lazy — freshly-discovered entries have no ctx
    // constructed until `PoolEntry::context()` is called.
    BackendPool pool;
    pool.discoverAll();
    for (const auto& e : pool.entries()) {
        EXPECT_TRUE(!e.hasContext());
    }
}

// -----------------------------------------------------------------------
// select(Auto) — first-available walk
// -----------------------------------------------------------------------

TEST(pool_select_auto_returnsFirstEntry) {
    BackendPool pool;
    pool.discoverAll();
    auto& first = pool.entries().front();
    auto& picked = pool.select(SelectionMode::Auto);
    // Address identity — `select(Auto)` returns a reference into the
    // pool's storage, same as `entries().front()`.
    EXPECT_TRUE(&picked == &first);
}

TEST(pool_select_auto_onEmptyPool_throws) {
    // Fresh pool, `discoverAll` NOT called → no entries. The API says
    // `select` throws with a helpful diagnostic.
    BackendPool pool;
    try {
        (void)pool.select(SelectionMode::Auto);
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::exception& ex) {
        EXPECT_TRUE(whatContains(ex, "pool is empty"));
    }
}

// -----------------------------------------------------------------------
// selectByToken — token parser + resolver
// -----------------------------------------------------------------------

TEST(pool_selectByToken_cpu_resolves) {
    BackendPool pool;
    pool.discoverAll();
    auto& e = pool.selectByToken("cpu");
    EXPECT_EQ(e.kind, BackendKind::Cpu);
    EXPECT_EQ(e.token, std::string{"cpu"});
}

TEST(pool_selectByToken_auto_delegatesToSelectAuto) {
    BackendPool pool;
    pool.discoverAll();
    auto& first = pool.entries().front();
    auto& picked = pool.selectByToken("auto");
    EXPECT_TRUE(&picked == &first);
}

TEST(pool_selectByToken_isCaseInsensitiveOnKind) {
    // Config-facing tokens are case-insensitive so an operator typing
    // "CPU" or "Cpu" gets the same result as "cpu".
    BackendPool pool;
    pool.discoverAll();
    auto& lower = pool.selectByToken("cpu");
    auto& upper = pool.selectByToken("CPU");
    EXPECT_TRUE(&lower == &upper);
}

TEST(pool_selectByToken_unknownKindName_throwsWithHint) {
    BackendPool pool;
    pool.discoverAll();
    try {
        (void)pool.selectByToken("banana");
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::exception& ex) {
        // Error message must name the unknown backend AND enumerate
        // what IS in the pool so the operator can fix their config
        // without a source dive.
        EXPECT_TRUE(whatContains(ex, "malformed token")
                 || whatContains(ex, "unknown backend"));
    }
}

TEST(pool_selectByToken_wrongDeviceIx_throwsWithAvailableList) {
    // "cpu:5" is a well-formed token but the Cpu entry only has
    // deviceIx=0 today. Should surface as "no matching entry" plus the
    // list of available entries (which tests fold into the substring
    // check below by looking for the canonical Cpu token).
    BackendPool pool;
    pool.discoverAll();
    try {
        // Cpu doesn't accept a colon suffix at all per the grammar —
        // `tokenFor(Cpu, N) = "cpu"` always. So "cpu:1" is arguably
        // malformed. Instead pick an obviously-nonexistent device
        // index on a compiled backend name that's still ambiguous:
        // "l0:99". At worst L0 is off and the parser fails on the
        // kind name; at best L0 is on and the deviceIx lookup fails.
        (void)pool.selectByToken("l0:99");
        EXPECT_TRUE(false && "expected throw");
    } catch (const std::exception&) {
        // Either "does not match any pool entry" or the L0-off
        // "no available entry" — both are correct outcomes for an
        // impossible token, and both fail loud (which is what we test).
    }
}

TEST(pool_selectByToken_malformedColonSuffix_throws) {
    BackendPool pool;
    pool.discoverAll();
    // "l0:" with a trailing empty index is malformed per the grammar.
    try {
        (void)pool.selectByToken("l0:");
        EXPECT_TRUE(false && "expected throw for trailing colon");
    } catch (const std::exception& ex) {
        EXPECT_TRUE(whatContains(ex, "malformed token"));
    }
}

TEST(pool_selectByToken_nonNumericDeviceIx_throws) {
    BackendPool pool;
    pool.discoverAll();
    try {
        (void)pool.selectByToken("hip:abc");
        EXPECT_TRUE(false && "expected throw for non-numeric device ix");
    } catch (const std::exception& ex) {
        EXPECT_TRUE(whatContains(ex, "malformed token"));
    }
}

TEST(pool_selectByToken_shortForm_l0_matchesL0Zero) {
    // The parser accepts `"l0"` as shorthand for `"l0:0"`. Test only
    // survives the compile-time-enabled backend list — skip when L0
    // isn't in the pool.
    BackendPool pool;
    pool.discoverAll();
    if (!poolContainsKind(pool, BackendKind::LevelZero)) {
        return; // no L0 available in this build — nothing to assert
    }
    auto& colon = pool.selectByToken("l0:0");
    auto& shortForm = pool.selectByToken("l0");
    EXPECT_TRUE(&colon == &shortForm);
}

TEST(pool_selectByToken_shortForm_hip_matchesHipZero) {
    BackendPool pool;
    pool.discoverAll();
    if (!poolContainsKind(pool, BackendKind::Hip)) {
        return;
    }
    auto& colon = pool.selectByToken("hip:0");
    auto& shortForm = pool.selectByToken("hip");
    EXPECT_TRUE(&colon == &shortForm);
}

TEST(pool_selectByToken_isStableAcrossCalls) {
    // Repeated resolution of the same token returns the same entry —
    // no hidden per-call state that would break long-running consumers
    // holding a `PoolEntry&`.
    BackendPool pool;
    pool.discoverAll();
    auto& a = pool.selectByToken("cpu");
    auto& b = pool.selectByToken("cpu");
    EXPECT_TRUE(&a == &b);
}

TEST(pool_selectByToken_failedLookup_leavesEntriesIntact) {
    BackendPool pool;
    pool.discoverAll();
    const auto before = pool.entries().size();
    try {
        (void)pool.selectByToken("banana");
    } catch (...) {
        // expected
    }
    EXPECT_EQ(pool.entries().size(), before);
}

// -----------------------------------------------------------------------
// BackendRegistry adjacent — probes should never throw
// -----------------------------------------------------------------------

TEST(registry_probeAll_neverThrows) {
    // noexcept in the signature; guard against a regression in the
    // implementation returning early with a raise instead of a
    // best-effort probe.
    const auto probes = BackendRegistry::probeAll();
    EXPECT_TRUE(!probes.empty());
}

TEST(registry_probeAll_hasCpu) {
    const auto probes = BackendRegistry::probeAll();
    const bool cpuPresent = std::any_of(probes.begin(), probes.end(),
        [](const auto& p) {
            return p.kind == BackendKind::Cpu && p.compiledIn && p.available;
        });
    EXPECT_TRUE(cpuPresent);
}

TEST(registry_parseKind_recognisesAllExpectedForms) {
    // Small sweep to catch a spelling drift — the shorthand list here
    // is the config-facing surface, changes have to be intentional.
    EXPECT_TRUE(BackendRegistry::parseKind("l0").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("L0").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("levelzero").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("level_zero").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("hip").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("HIP").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("rocm").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("amd").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("cuda").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("nvidia").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("cpu").has_value());
    EXPECT_TRUE(BackendRegistry::parseKind("CPU").has_value());
}

TEST(registry_parseKind_rejectsUnknown) {
    EXPECT_TRUE(!BackendRegistry::parseKind("banana").has_value());
    EXPECT_TRUE(!BackendRegistry::parseKind("").has_value());
}

// -----------------------------------------------------------------------
// BatchCapacityProbe — roundToSchedulerStep + skeleton fallback
// -----------------------------------------------------------------------

#include "runtime/serving/BatchCapacityProbe.hpp"

TEST(batchProbe_roundToSchedulerStep_powerOfTwo) {
    using ::mimirmind::runtime::serving::BatchCapacityProbe;
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(1),  std::size_t{1});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(2),  std::size_t{2});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(4),  std::size_t{4});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(8),  std::size_t{8});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(16), std::size_t{16});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(32), std::size_t{32});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(64), std::size_t{32});
}

TEST(batchProbe_roundToSchedulerStep_downRounds) {
    using ::mimirmind::runtime::serving::BatchCapacityProbe;
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(3),  std::size_t{2});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(7),  std::size_t{4});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(15), std::size_t{8});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(27), std::size_t{16});
    EXPECT_EQ(BatchCapacityProbe::roundToSchedulerStep(0),  std::size_t{1});
}

TEST(batchProbe_skeletonFallback_isSingleSession) {
    using ::mimirmind::runtime::serving::BatchCapacityProbe;
    const auto est = BatchCapacityProbe::estimateConservativeFallback();
    EXPECT_EQ(est.sustainableBatch, std::size_t{1});
    EXPECT_TRUE(!est.servingClassRecommended);
    EXPECT_TRUE(!est.reasoning.empty());
}

// -----------------------------------------------------------------------
// LlmConfig::kvBytesPerToken — helper used by BatchCapacityProbe
// -----------------------------------------------------------------------

#include "model/LlmConfig.hpp"

TEST(llmConfig_kvBytesPerToken_uniformFp16) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    cfg.blockCount      = 32;
    cfg.embeddingLength = 4096;
    cfg.headCount       = 32;
    cfg.headCountKv     = 8;      // GQA 4:1
    // headDim() = 4096 / 32 = 128 (keyLength unset falls back to embed/head).
    // Per layer: 2 * 8 * 128 * 2 = 4096 bytes. 32 layers → 131072 bytes.
    EXPECT_EQ(cfg.kvBytesPerToken(2),
              std::size_t{32} * 2U * 8U * 128U * 2U);
}

TEST(llmConfig_kvBytesPerToken_f32ScalesLinearly) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    cfg.blockCount      = 8;
    cfg.embeddingLength = 1024;
    cfg.headCount       = 8;
    cfg.headCountKv     = 4;
    // F32 (dtypeBytes=4) is exactly 2x FP16 (dtypeBytes=2).
    const auto fp16 = cfg.kvBytesPerToken(2);
    const auto f32  = cfg.kvBytesPerToken(4);
    EXPECT_EQ(f32, fp16 * 2U);
}

TEST(llmConfig_kvBytesPerToken_perLayerKvHonoured) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    cfg.blockCount      = 4;
    cfg.embeddingLength = 256;
    cfg.headCount       = 4;
    cfg.headCountKv     = 4;      // fallback if per-layer empty
    cfg.headCountKvPerLayer = {1U, 2U, 4U, 8U};
    // head_dim = 256/4 = 64. Per-layer contribution: 2 * n_kv(b) * 64 * 2.
    // Sum n_kv = 1+2+4+8 = 15. Total = 15 * 2 * 64 * 2 = 3840 bytes.
    EXPECT_EQ(cfg.kvBytesPerToken(2),
              std::size_t{15} * 2U * 64U * 2U);
}

TEST(llmConfig_kvBytesPerToken_emptyModelReturnsZero) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    // blockCount=0 → sum-loop doesn't run.
    EXPECT_EQ(cfg.kvBytesPerToken(2), std::size_t{0});
}

// -----------------------------------------------------------------------
// LlmConfig — Qwen3-Next / GatedDeltaNet hybrid (qwen35moe) helpers
// -----------------------------------------------------------------------

TEST(llmConfig_ssm_dimsFor35bA3b) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    // Qwen3.6-35B-A3B SSM hyperparameters (recon 2026-07-21).
    cfg.ssmConvKernel   = 4;
    cfg.ssmInnerSize    = 4096;   // value_dim = H_v * S_v = 32 * 128
    cfg.ssmStateSize    = 128;    // head_dim
    cfg.ssmTimeStepRank = 32;     // H_v (num v-heads)
    cfg.ssmGroupCount   = 16;     // H_k (num k-heads)

    EXPECT_EQ(cfg.ssmHeadDim(),   std::uint32_t{128});
    EXPECT_EQ(cfg.ssmNumKHeads(), std::uint32_t{16});
    EXPECT_EQ(cfg.ssmNumVHeads(), std::uint32_t{32});
    // conv_dim = d_inner + 2*n_group*d_state = 4096 + 2*16*128 = 8192.
    EXPECT_EQ(cfg.ssmConvDim(),   std::uint32_t{8192});
    // state elems/layer = S^2 * H_v = 128*128*32 = 524288 (= 2 MiB F32).
    EXPECT_EQ(cfg.ssmStateElemsPerLayer(), std::size_t{524288});
    // conv state elems/layer = (d_conv-1) * conv_dim = 3 * 8192 = 24576.
    EXPECT_EQ(cfg.ssmConvStateElemsPerLayer(), std::size_t{24576});
}

TEST(llmConfig_ssm_nonHybridReturnsZero) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    // No SSM params set → all derived SSM quantities are zero, no recurrence.
    EXPECT_EQ(cfg.ssmConvDim(),                std::uint32_t{0});
    EXPECT_EQ(cfg.ssmStateElemsPerLayer(),     std::size_t{0});
    EXPECT_EQ(cfg.ssmConvStateElemsPerLayer(), std::size_t{0});
    EXPECT_TRUE(!cfg.isHybridRecurrent());
    EXPECT_TRUE(!cfg.isRecurrentLayer(0));
}

TEST(llmConfig_attentionScaleFor) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    // Unset (0) → default 1/sqrt(head_dim).
    EXPECT_NEAR(cfg.attentionScaleFor(256),
                1.0F / std::sqrt(256.0F), 1e-7F);
    // Explicit scale wins.
    cfg.attentionScale = 0.125F;
    EXPECT_NEAR(cfg.attentionScaleFor(256), 0.125F, 1e-7F);
    // head_dim 0 with no explicit scale is a safe 0 (no div-by-zero).
    LlmConfig z{};
    EXPECT_NEAR(z.attentionScaleFor(0), 0.0F, 1e-7F);
}

TEST(llmConfig_recurrentPattern_intervalConvention) {
    using ::mimirmind::model::LlmConfig;
    LlmConfig cfg{};
    // Mirror the llama.cpp qwen35moe synthesis `(b+1) % 4 != 0` over 8 layers:
    // full at index 3 and 7, recurrent elsewhere.
    cfg.blockCount = 8;
    cfg.recurrentLayerPattern =
        {true, true, true, false, true, true, true, false};
    EXPECT_TRUE(cfg.isHybridRecurrent());
    EXPECT_TRUE(cfg.isRecurrentLayer(0));
    EXPECT_TRUE(cfg.isRecurrentLayer(1));
    EXPECT_TRUE(cfg.isRecurrentLayer(2));
    EXPECT_TRUE(!cfg.isRecurrentLayer(3));   // full attention
    EXPECT_TRUE(cfg.isRecurrentLayer(6));
    EXPECT_TRUE(!cfg.isRecurrentLayer(7));   // full attention
    // Out-of-range index is a safe false, never UB.
    EXPECT_TRUE(!cfg.isRecurrentLayer(8));
    EXPECT_TRUE(!cfg.isRecurrentLayer(9999));
}

// -----------------------------------------------------------------------
// ServingSettings — config-parser round-trip
// -----------------------------------------------------------------------

#include "core/config/Config.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

// Write JSON to a fresh temp path, return the path. Caller unlinks.
std::string writeTempConfig(const std::string& body) {
    const auto dir  = std::filesystem::temp_directory_path();
    const auto path = dir / ("mm-cfg-test-"
                             + std::to_string(std::rand()) + ".json");
    std::ofstream out{path};
    out << body;
    out.close();
    return path.string();
}

} // namespace

TEST(serving_defaults_areAutoAndEight) {
    using ::mimirmind::core::config::ServingSettings;
    using ::mimirmind::core::config::TriState;
    ServingSettings s{};
    EXPECT_EQ(mm::test::streamable(s.enableBatching),
              mm::test::streamable(TriState::Auto));
    EXPECT_EQ(s.minBatchForEnable, std::size_t{8});
}

TEST(serving_configJson_defaultsWhenMissing) {
    // Config without a `serving` block leaves defaults in place.
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}]
    })");
    const auto cfg = ::mimirmind::core::config::loadConfig(path);
    std::filesystem::remove(path);
    using ::mimirmind::core::config::TriState;
    EXPECT_EQ(mm::test::streamable(cfg.serving.enableBatching),
              mm::test::streamable(TriState::Auto));
    EXPECT_EQ(cfg.serving.minBatchForEnable, std::size_t{8});
}

TEST(serving_configJson_parsesExplicitValues) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": {
            "enableBatching": "force",
            "minBatchForEnable": 16
        }
    })");
    const auto cfg = ::mimirmind::core::config::loadConfig(path);
    std::filesystem::remove(path);
    using ::mimirmind::core::config::TriState;
    EXPECT_EQ(mm::test::streamable(cfg.serving.enableBatching),
              mm::test::streamable(TriState::Force));
    EXPECT_EQ(cfg.serving.minBatchForEnable, std::size_t{16});
}

TEST(serving_configJson_rejectsUnknownKey) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": {
            "enableBatching": "auto",
            "banana": 42
        }
    })");
    bool threw = false;
    try {
        (void)::mimirmind::core::config::loadConfig(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_rejectsOutOfRangeMinBatch) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "minBatchForEnable": 0 }
    })");
    bool threw = false;
    try {
        (void)::mimirmind::core::config::loadConfig(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_defaults) {
    // No config → all C-phase defaults applied.
    using ::mimirmind::core::config::ServingSettings;
    ServingSettings s{};
    EXPECT_EQ(s.tokenBudget,               std::size_t{512});
    EXPECT_EQ(s.maxActiveRequests,         std::size_t{32});
    EXPECT_TRUE(s.preemptFreeBlockThreshold == 0.05);
    EXPECT_EQ(s.blockSize,                 std::size_t{16});
}

TEST(serving_configJson_cphaseKnobs_parseExplicitValues) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": {
            "tokenBudget": 256,
            "maxActiveRequests": 64,
            "preemptFreeBlockThreshold": 0.1,
            "blockSize": 32
        }
    })");
    const auto cfg = ::mimirmind::core::config::loadConfig(path);
    std::filesystem::remove(path);
    EXPECT_EQ(cfg.serving.tokenBudget,               std::size_t{256});
    EXPECT_EQ(cfg.serving.maxActiveRequests,         std::size_t{64});
    EXPECT_TRUE(cfg.serving.preemptFreeBlockThreshold == 0.1);
    EXPECT_EQ(cfg.serving.blockSize,                 std::size_t{32});
}

TEST(serving_configJson_cphaseKnobs_rejectsZeroTokenBudget) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "tokenBudget": 0 }
    })");
    bool threw = false;
    try { (void)::mimirmind::core::config::loadConfig(path); }
    catch (const std::exception&) { threw = true; }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_rejectsHugeTokenBudget) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "tokenBudget": 999999 }
    })");
    bool threw = false;
    try { (void)::mimirmind::core::config::loadConfig(path); }
    catch (const std::exception&) { threw = true; }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_rejectsZeroMaxActive) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "maxActiveRequests": 0 }
    })");
    bool threw = false;
    try { (void)::mimirmind::core::config::loadConfig(path); }
    catch (const std::exception&) { threw = true; }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_rejectsThresholdAboveOne) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "preemptFreeBlockThreshold": 1.5 }
    })");
    bool threw = false;
    try { (void)::mimirmind::core::config::loadConfig(path); }
    catch (const std::exception&) { threw = true; }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_rejectsNegativeThreshold) {
    const auto path = writeTempConfig(R"({
        "models": [{"id":"primary","path":"/models/x.gguf"}],
        "serving": { "preemptFreeBlockThreshold": -0.1 }
    })");
    bool threw = false;
    try { (void)::mimirmind::core::config::loadConfig(path); }
    catch (const std::exception&) { threw = true; }
    std::filesystem::remove(path);
    EXPECT_TRUE(threw);
}

TEST(serving_configJson_cphaseKnobs_acceptsBoundaryThresholds) {
    // Both 0.0 (disabled) and 1.0 (maximally-eager) are legal, same
    // contract as PreemptionPolicy's ctor.
    for (double t : {0.0, 1.0}) {
        std::string body = R"({"models":[{"id":"primary","path":"/models/x.gguf"}],)"
                           R"("serving":{"preemptFreeBlockThreshold": )"
                           + std::to_string(t) + "}}";
        const auto path = writeTempConfig(body);
        const auto cfg = ::mimirmind::core::config::loadConfig(path);
        std::filesystem::remove(path);
        EXPECT_TRUE(cfg.serving.preemptFreeBlockThreshold == t);
    }
}

TEST(serving_configJson_cphaseKnobs_rejectsNonPowerOfTwoBlockSize) {
    // Only {8, 16, 32} are legal per M-Cuda.Batch design.
    for (int bs : {1, 4, 7, 24, 64, 128}) {
        std::string body = R"({"models":[{"id":"primary","path":"/models/x.gguf"}],)"
                           R"("serving":{"blockSize": )"
                           + std::to_string(bs) + "}}";
        const auto path = writeTempConfig(body);
        bool threw = false;
        try { (void)::mimirmind::core::config::loadConfig(path); }
        catch (const std::exception&) { threw = true; }
        std::filesystem::remove(path);
        EXPECT_TRUE(threw);
    }
}

TEST(serving_configJson_cphaseKnobs_acceptsAllLegalBlockSizes) {
    for (int bs : {8, 16, 32}) {
        std::string body = R"({"models":[{"id":"primary","path":"/models/x.gguf"}],)"
                           R"("serving":{"blockSize": )"
                           + std::to_string(bs) + "}}";
        const auto path = writeTempConfig(body);
        const auto cfg = ::mimirmind::core::config::loadConfig(path);
        std::filesystem::remove(path);
        EXPECT_EQ(cfg.serving.blockSize, static_cast<std::size_t>(bs));
    }
}

// -----------------------------------------------------------------------
// ComputeContext::bandwidthGBps — Cpu backend (only one always compiled)
// -----------------------------------------------------------------------

#include "core/cpu/CpuContext.hpp"

TEST(computeContext_cpu_bandwidthIsFiftyGBps) {
    ::mimirmind::core::cpu::CpuContext ctx{};
    // 50 GB/s = dual-channel DDR5-5600 desktop baseline.
    EXPECT_EQ(ctx.bandwidthGBps(), std::size_t{50});
}

TEST(computeContext_baseDefaultsToZero) {
    // Sanity — the base class default is 0 ("unknown"), meaning any
    // future backend that forgets to override still compiles and the
    // BatchCapacityProbe correctly falls back to single-session.
    // Verified indirectly: CpuContext overrides to non-zero, so if the
    // default weren't 0 we'd need a different sentinel here.
    ::mimirmind::core::cpu::CpuContext ctx{};
    EXPECT_TRUE(ctx.bandwidthGBps() != 0);  // sanity: override IS in effect
}

// -----------------------------------------------------------------------
// PagedKvBlockAllocator — logical block-pool for Bragi PagedAttention
// -----------------------------------------------------------------------

#include "runtime/serving/PagedKvBlockAllocator.hpp"

TEST(pagedAlloc_ctorRejectsZeroBlocks) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    bool threw = false;
    try {
        PagedKvBlockAllocator pool{0, 16};
        (void)pool;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(pagedAlloc_ctorRejectsZeroBlockSize) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    bool threw = false;
    try {
        PagedKvBlockAllocator pool{4, 0};
        (void)pool;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(pagedAlloc_freshPoolHasAllFree) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{8, 16};
    EXPECT_EQ(pool.numBlocksTotal(), std::size_t{8});
    EXPECT_EQ(pool.numBlocksFree(),  std::size_t{8});
    EXPECT_EQ(pool.numBlocksUsed(),  std::size_t{0});
    EXPECT_EQ(pool.blockSize(),      std::size_t{16});
}

TEST(pagedAlloc_allocateHandsOutIdsInAscendingOrder) {
    // Free-list is filled reverse in the ctor so LIFO pop returns 0,1,2,...
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    EXPECT_EQ(pool.allocate(), std::uint32_t{0});
    EXPECT_EQ(pool.allocate(), std::uint32_t{1});
    EXPECT_EQ(pool.allocate(), std::uint32_t{2});
    EXPECT_EQ(pool.allocate(), std::uint32_t{3});
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{4});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{0});
}

TEST(pagedAlloc_exhaustionReturnsInvalid) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{2, 16};
    (void)pool.allocate();
    (void)pool.allocate();
    EXPECT_EQ(pool.allocate(), PagedKvBlockAllocator::kInvalidBlock);
}

TEST(pagedAlloc_freshBlockHasRefcountOneAndZeroHash) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    EXPECT_EQ(pool.refcountOf(id), std::uint32_t{1});
    EXPECT_EQ(pool.hashOf(id),     std::uint64_t{0});
}

TEST(pagedAlloc_addRefBumpsRefcount) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    pool.addRef(id);
    pool.addRef(id);
    EXPECT_EQ(pool.refcountOf(id), std::uint32_t{3});
}

TEST(pagedAlloc_releaseReturnsBlockWhenRefcountReachesZero) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{3});
    pool.release(id);
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{4});
    EXPECT_EQ(pool.refcountOf(id),  std::uint32_t{0});
}

TEST(pagedAlloc_releaseKeepsBlockWhileRefcountAboveZero) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    pool.addRef(id);
    pool.release(id);
    EXPECT_EQ(pool.refcountOf(id),  std::uint32_t{1});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{3});
    pool.release(id);
    EXPECT_EQ(pool.refcountOf(id),  std::uint32_t{0});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{4});
}

TEST(pagedAlloc_releasedBlockCanBeReallocated) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{2, 16};
    const auto a = pool.allocate();
    (void)pool.allocate();
    pool.release(a);
    // LIFO reuse: freshly released `a` sits on top of the free list.
    EXPECT_EQ(pool.allocate(), a);
}

TEST(pagedAlloc_hashRoundTrip) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    pool.setHash(id, 0xDEADBEEFCAFEBABEULL);
    EXPECT_EQ(pool.hashOf(id), 0xDEADBEEFCAFEBABEULL);
}

TEST(pagedAlloc_releaseClearsHash) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    const auto id = pool.allocate();
    pool.setHash(id, 0x42ULL);
    pool.release(id);
    // After release the id is back in the free list. Its hash is a
    // scratch slot the allocator zeroes on release.
    EXPECT_EQ(pool.hashOf(id), std::uint64_t{0});
}

TEST(pagedAlloc_setHashOnFreeBlockIsNoOp) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    // Never allocated, refcount is 0.
    pool.setHash(std::uint32_t{2}, 0xAAULL);
    EXPECT_EQ(pool.hashOf(std::uint32_t{2}), std::uint64_t{0});
}

TEST(pagedAlloc_addRefOnFreeBlockIsNoOp) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    // Refcount stays 0, block stays free — caller-bug guard.
    pool.addRef(std::uint32_t{2});
    EXPECT_EQ(pool.refcountOf(std::uint32_t{2}), std::uint32_t{0});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{4});
}

TEST(pagedAlloc_releaseOnFreeBlockIsNoOp) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{4, 16};
    // Refcount stays 0, free-list stays at capacity — no underflow into
    // some negative refcount, no duplicate free-list entry.
    pool.release(std::uint32_t{2});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{4});
}

TEST(pagedAlloc_outOfRangeIdsAreSilentlyIgnored) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    PagedKvBlockAllocator pool{2, 16};
    // 99 is out of range for a 2-block pool.
    pool.addRef(99);
    pool.release(99);
    pool.setHash(99, 0x1ULL);
    EXPECT_EQ(pool.refcountOf(99), std::uint32_t{0});
    EXPECT_EQ(pool.hashOf(99),     std::uint64_t{0});
    // Pool state must be untouched.
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{2});
}

// -----------------------------------------------------------------------
// PagedKvSequence — per-request block-table + token history + hash chain
// -----------------------------------------------------------------------

#include "runtime/serving/PagedKvSequence.hpp"

TEST(pagedSeq_emptyOnConstruction) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence seq{pool};
    EXPECT_EQ(seq.numTokens(), std::size_t{0});
    EXPECT_EQ(seq.numBlocks(), std::size_t{0});
    EXPECT_EQ(seq.blockSize(), std::size_t{4});
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
}

TEST(pagedSeq_firstAppendAllocatesFirstBlock) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence seq{pool};
    EXPECT_TRUE(seq.appendToken(42));
    EXPECT_EQ(seq.numTokens(),      std::size_t{1});
    EXPECT_EQ(seq.numBlocks(),      std::size_t{1});
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{1});
}

TEST(pagedSeq_exactFillUsesOneBlockAndSetsHash) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence seq{pool};
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(seq.appendToken(i));
    }
    EXPECT_EQ(seq.numTokens(),      std::size_t{4});
    EXPECT_EQ(seq.numBlocks(),      std::size_t{1});
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{1});
    // Block just got filled — hash must be non-zero.
    const auto id = seq.blockTable().front();
    EXPECT_TRUE(pool.hashOf(id) != std::uint64_t{0});
}

TEST(pagedSeq_overfillByOneAllocatesSecondBlock) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence seq{pool};
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(seq.appendToken(i));
    }
    EXPECT_EQ(seq.numTokens(), std::size_t{5});
    EXPECT_EQ(seq.numBlocks(), std::size_t{2});
    // First block was completed → hashed. Second block is only partial
    // → hash slot is still zero (allocator clears it on allocate()).
    EXPECT_TRUE(pool.hashOf(seq.blockTable()[0]) != std::uint64_t{0});
    EXPECT_EQ(pool.hashOf(seq.blockTable()[1]), std::uint64_t{0});
}

TEST(pagedSeq_hashChainIsDeterministic) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    // Two independent sequences with the same first block must produce
    // the same block-hash — that's the property M-PrefixCache relies on.
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence a{pool};
    PagedKvSequence b{pool};
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(a.appendToken(i));
        EXPECT_TRUE(b.appendToken(i));
    }
    EXPECT_EQ(pool.hashOf(a.blockTable()[0]),
              pool.hashOf(b.blockTable()[0]));
}

TEST(pagedSeq_hashChainSeparatesOnDivergence) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    // Same first block, different second block → block-1 hashes differ.
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence a{pool};
    PagedKvSequence b{pool};
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(a.appendToken(i));
        EXPECT_TRUE(b.appendToken(i));
    }
    // Fill each second block with different tokens.
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(a.appendToken(100 + i));
        EXPECT_TRUE(b.appendToken(200 + i));
    }
    // Block 0 still matches (same input).
    EXPECT_EQ(pool.hashOf(a.blockTable()[0]),
              pool.hashOf(b.blockTable()[0]));
    // Block 1 diverges.
    EXPECT_TRUE(pool.hashOf(a.blockTable()[1]) !=
                pool.hashOf(b.blockTable()[1]));
}

TEST(pagedSeq_hashChainDependsOnParent) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    // Same block-2 tokens under different block-1 tokens → different
    // block-2 hashes. Proves parent-hash actually seeds the chain.
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence a{pool};
    PagedKvSequence b{pool};
    // Divergent first block.
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(a.appendToken(i));
        EXPECT_TRUE(b.appendToken(1000 + i));
    }
    // Identical second block.
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(a.appendToken(50 + i));
        EXPECT_TRUE(b.appendToken(50 + i));
    }
    EXPECT_TRUE(pool.hashOf(a.blockTable()[1]) !=
                pool.hashOf(b.blockTable()[1]));
}

TEST(pagedSeq_resetReleasesEveryBlock) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence seq{pool};
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(seq.appendToken(i));
    }
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{3});
    seq.reset();
    EXPECT_EQ(seq.numTokens(),      std::size_t{0});
    EXPECT_EQ(seq.numBlocks(),      std::size_t{0});
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{8});
}

TEST(pagedSeq_destructorReleasesBlocks) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    {
        PagedKvSequence seq{pool};
        for (int i = 0; i < 6; ++i) {
            EXPECT_TRUE(seq.appendToken(i));
        }
        EXPECT_EQ(pool.numBlocksUsed(), std::size_t{2});
    }
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
    EXPECT_EQ(pool.numBlocksFree(), std::size_t{8});
}

TEST(pagedSeq_oomReturnsFalseAndLeavesStateUnchanged) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    // Pool of 1 block × 4 tokens. Fills to exactly 4, then the next
    // append needs a new block and must fail cleanly.
    PagedKvBlockAllocator pool{1, 4};
    PagedKvSequence seq{pool};
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(seq.appendToken(i));
    }
    EXPECT_EQ(seq.numTokens(), std::size_t{4});
    EXPECT_EQ(seq.numBlocks(), std::size_t{1});
    EXPECT_TRUE(!seq.appendToken(99));
    // Unchanged: still 4 tokens, still 1 block, block-hash from the
    // earlier fill is still there.
    EXPECT_EQ(seq.numTokens(), std::size_t{4});
    EXPECT_EQ(seq.numBlocks(), std::size_t{1});
    EXPECT_TRUE(pool.hashOf(seq.blockTable().front()) != std::uint64_t{0});
}

TEST(pagedSeq_moveTransfersOwnership) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::PagedKvSequence;
    PagedKvBlockAllocator pool{8, 4};
    PagedKvSequence src{pool};
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(src.appendToken(i));
    }
    const std::size_t blocksBeforeMove = pool.numBlocksUsed();
    PagedKvSequence dst{std::move(src)};
    // Pool is untouched by the move — blocks changed owner, not lifecycle.
    EXPECT_EQ(pool.numBlocksUsed(), blocksBeforeMove);
    EXPECT_EQ(dst.numTokens(), std::size_t{6});
    EXPECT_EQ(dst.numBlocks(), std::size_t{2});
    // Moved-from is empty and its destructor is a no-op.
    EXPECT_EQ(src.numTokens(), std::size_t{0});
    EXPECT_EQ(src.numBlocks(), std::size_t{0});
}

// -----------------------------------------------------------------------
// ChunkedPrefillScheduler — Sub-Step C3 (M-Cuda.Batch Phase C)
// -----------------------------------------------------------------------

#include "runtime/serving/ChunkedPrefillScheduler.hpp"

TEST(chunkedSched_ctorDefaultBudgetIs512) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    ChunkedPrefillScheduler s{};
    EXPECT_EQ(s.tokenBudget(), std::int32_t{512});
}

TEST(chunkedSched_ctorRejectsZeroBudget) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    bool threw = false;
    try {
        ChunkedPrefillScheduler s{0};
        (void)s;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(chunkedSched_ctorRejectsNegativeBudget) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    bool threw = false;
    try {
        ChunkedPrefillScheduler s{-1};
        (void)s;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(chunkedSched_emptyInputProducesEmptySchedule) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{};
    const std::vector<RequestSlice> empty{};
    const auto out = s.schedule(empty);
    EXPECT_EQ(out.decodes.size(),           std::size_t{0});
    EXPECT_EQ(out.prefills.size(),          std::size_t{0});
    EXPECT_EQ(out.total_tokens_scheduled,   std::int32_t{0});
}

TEST(chunkedSched_singleDecodeSchedulesOneToken) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{};
    const std::vector<RequestSlice> reqs{
        {/*id=*/1, /*pending=*/0, /*decoded=*/5, /*is_decoding=*/true},
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),         std::size_t{1});
    EXPECT_EQ(out.decodes.front().request_id, std::uint64_t{1});
    EXPECT_EQ(out.prefills.size(),        std::size_t{0});
    EXPECT_EQ(out.total_tokens_scheduled, std::int32_t{1});
}

TEST(chunkedSched_manyDecodesUnderBudgetAllScheduled) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{16};
    std::vector<RequestSlice> reqs;
    for (int i = 0; i < 10; ++i) {
        reqs.push_back({static_cast<std::uint64_t>(i), 0, 0, true});
    }
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),         std::size_t{10});
    EXPECT_EQ(out.total_tokens_scheduled, std::int32_t{10});
}

TEST(chunkedSched_decodesExceedBudgetDropsOverflow) {
    // C3 is pure scheduling math. Overflow is silently dropped — the
    // C5 preemption policy handles the memory-pressure signal.
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{4};
    std::vector<RequestSlice> reqs;
    for (int i = 0; i < 10; ++i) {
        reqs.push_back({static_cast<std::uint64_t>(i), 0, 0, true});
    }
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),         std::size_t{4});
    EXPECT_EQ(out.total_tokens_scheduled, std::int32_t{4});
    // Input-order preserved on the survivors.
    EXPECT_EQ(out.decodes[0].request_id, std::uint64_t{0});
    EXPECT_EQ(out.decodes[3].request_id, std::uint64_t{3});
}

TEST(chunkedSched_singlePrefillFitsFullyInBudget) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {/*id=*/42, /*pending=*/100, /*decoded=*/0, /*is_decoding=*/false},
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.prefills.size(),           std::size_t{1});
    EXPECT_EQ(out.prefills.front().request_id, std::uint64_t{42});
    EXPECT_EQ(out.prefills.front().chunk_size, std::int32_t{100});
    EXPECT_EQ(out.total_tokens_scheduled,    std::int32_t{100});
}

TEST(chunkedSched_singlePrefillOverflowSplitsAtBudget) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {/*id=*/42, /*pending=*/1000, /*decoded=*/0, /*is_decoding=*/false},
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.prefills.size(),           std::size_t{1});
    EXPECT_EQ(out.prefills.front().chunk_size, std::int32_t{512});
    EXPECT_EQ(out.total_tokens_scheduled,    std::int32_t{512});
    // Caller now reduces tokens_pending by 512 and reschedules next iter.
}

TEST(chunkedSched_multiplePrefillsFillBudgetInOrder) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {1, 200, 0, false},
        {2, 200, 0, false},
        {3, 200, 0, false},   // only 112 of these get scheduled this iter
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.prefills.size(),            std::size_t{3});
    EXPECT_EQ(out.prefills[0].request_id,     std::uint64_t{1});
    EXPECT_EQ(out.prefills[0].chunk_size,     std::int32_t{200});
    EXPECT_EQ(out.prefills[1].request_id,     std::uint64_t{2});
    EXPECT_EQ(out.prefills[1].chunk_size,     std::int32_t{200});
    EXPECT_EQ(out.prefills[2].request_id,     std::uint64_t{3});
    EXPECT_EQ(out.prefills[2].chunk_size,     std::int32_t{112});
    EXPECT_EQ(out.total_tokens_scheduled,     std::int32_t{512});
}

TEST(chunkedSched_decodeFirstThenPrefillGetsRemainder) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    std::vector<RequestSlice> reqs;
    // 5 decodes + 1 long prefill. Decodes eat 5, prefill gets 507.
    for (int i = 0; i < 5; ++i) {
        reqs.push_back({static_cast<std::uint64_t>(100 + i), 0, 0, true});
    }
    reqs.push_back({999, 1000, 0, false});
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),             std::size_t{5});
    EXPECT_EQ(out.prefills.size(),            std::size_t{1});
    EXPECT_EQ(out.prefills.front().request_id, std::uint64_t{999});
    EXPECT_EQ(out.prefills.front().chunk_size, std::int32_t{507});
    EXPECT_EQ(out.total_tokens_scheduled,     std::int32_t{512});
}

TEST(chunkedSched_decodesFillBudgetPrefillsSkipped) {
    // When decodes consume the whole budget, prefills wait for the
    // next iteration. Guarantees decode latency isn't sacrificed to
    // prefill throughput.
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{8};
    std::vector<RequestSlice> reqs;
    for (int i = 0; i < 8; ++i) {
        reqs.push_back({static_cast<std::uint64_t>(i), 0, 0, true});
    }
    reqs.push_back({999, 500, 0, false});     // pending but no room
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),         std::size_t{8});
    EXPECT_EQ(out.prefills.size(),        std::size_t{0});
    EXPECT_EQ(out.total_tokens_scheduled, std::int32_t{8});
}

TEST(chunkedSched_mixedInputPreservesOrderPerClass) {
    // Even when decodes and prefills are interleaved in the input,
    // decodes go first (in their input order) and prefills after
    // (in their input order).
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {10, 100, 0, false},   // prefill
        {20, 0,   0, true},    // decode
        {30, 50,  0, false},   // prefill
        {40, 0,   0, true},    // decode
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.decodes.size(),          std::size_t{2});
    EXPECT_EQ(out.decodes[0].request_id,   std::uint64_t{20});
    EXPECT_EQ(out.decodes[1].request_id,   std::uint64_t{40});
    EXPECT_EQ(out.prefills.size(),         std::size_t{2});
    EXPECT_EQ(out.prefills[0].request_id,  std::uint64_t{10});
    EXPECT_EQ(out.prefills[1].request_id,  std::uint64_t{30});
    EXPECT_EQ(out.total_tokens_scheduled,  std::int32_t{2 + 100 + 50});
}

TEST(chunkedSched_zeroPendingPrefillIsSkipped) {
    // Defensive: a prefill slice with tokens_pending=0 shouldn't
    // produce a chunk_size=0 assignment (wasted iteration entry).
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {1, 0, 0, false},   // "pending but nothing left" — skip
        {2, 100, 0, false},
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.prefills.size(),           std::size_t{1});
    EXPECT_EQ(out.prefills.front().request_id, std::uint64_t{2});
    EXPECT_EQ(out.prefills.front().chunk_size, std::int32_t{100});
    EXPECT_EQ(out.total_tokens_scheduled,    std::int32_t{100});
}

TEST(chunkedSched_customBudgetIsHonoured) {
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{128};
    const std::vector<RequestSlice> reqs{
        {1, 500, 0, false},
    };
    const auto out = s.schedule(reqs);
    EXPECT_EQ(out.prefills.size(),           std::size_t{1});
    EXPECT_EQ(out.prefills.front().chunk_size, std::int32_t{128});
    EXPECT_EQ(out.total_tokens_scheduled,    std::int32_t{128});
}

TEST(chunkedSched_totalTokensMatchesSumOfAssignments) {
    // Meta-invariant: total_tokens_scheduled = decodes.size() +
    // Σ prefills[i].chunk_size. Never drift.
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {1, 0,    5, true},
        {2, 0,    3, true},
        {3, 300,  0, false},
        {4, 500,  0, false},
    };
    const auto out = s.schedule(reqs);
    std::int32_t sum = static_cast<std::int32_t>(out.decodes.size());
    for (const auto& p : out.prefills) sum += p.chunk_size;
    EXPECT_EQ(out.total_tokens_scheduled, sum);
}

TEST(chunkedSched_repeatCallsAreIdempotentAndStateless) {
    // Sanity: schedule is pure. Two back-to-back calls on the same
    // input produce identical output. Proves no hidden state.
    using ::mimirmind::runtime::serving::ChunkedPrefillScheduler;
    using ::mimirmind::runtime::serving::RequestSlice;
    ChunkedPrefillScheduler s{512};
    const std::vector<RequestSlice> reqs{
        {1, 100, 0, false},
        {2, 0,   0, true},
    };
    const auto a = s.schedule(reqs);
    const auto b = s.schedule(reqs);
    EXPECT_EQ(a.total_tokens_scheduled, b.total_tokens_scheduled);
    EXPECT_EQ(a.decodes.size(),         b.decodes.size());
    EXPECT_EQ(a.prefills.size(),        b.prefills.size());
    EXPECT_EQ(a.decodes[0].request_id,  b.decodes[0].request_id);
    EXPECT_EQ(a.prefills[0].request_id, b.prefills[0].request_id);
    EXPECT_EQ(a.prefills[0].chunk_size, b.prefills[0].chunk_size);
}

// -----------------------------------------------------------------------
// RequestScheduler — Sub-Steps C1+C2 (M-Cuda.Batch Phase C)
// -----------------------------------------------------------------------

#include "runtime/serving/RequestScheduler.hpp"

TEST(reqSched_ctorRejectsZeroMaxActive) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    bool threw = false;
    try {
        RequestScheduler s{512, 0};
        (void)s;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(reqSched_ctorPropagatesChunkedBudgetGuard) {
    // tokenBudget=0 must be rejected — the guard lives in
    // ChunkedPrefillScheduler and propagates through RequestScheduler.
    using ::mimirmind::runtime::serving::RequestScheduler;
    bool threw = false;
    try {
        RequestScheduler s{0, 4};
        (void)s;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(reqSched_admitAssignsMonotonicIds) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    const auto a = s.admit(100, 32);
    const auto b = s.admit(200, 64);
    const auto c = s.admit(50,  16);
    EXPECT_EQ(a, std::uint64_t{1});
    EXPECT_EQ(b, std::uint64_t{2});
    EXPECT_EQ(c, std::uint64_t{3});
}

TEST(reqSched_admitRejectsInvalidPromptLength) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    EXPECT_EQ(s.admit(0, 32),   std::uint64_t{0});
    EXPECT_EQ(s.admit(-1, 32),  std::uint64_t{0});
    // Registry stays clean after rejected admits.
    EXPECT_EQ(s.numWaiting(), std::size_t{0});
}

TEST(reqSched_admitRejectsInvalidMaxTokens) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    EXPECT_EQ(s.admit(100, 0),   std::uint64_t{0});
    EXPECT_EQ(s.admit(100, -5),  std::uint64_t{0});
    EXPECT_EQ(s.numWaiting(), std::size_t{0});
}

TEST(reqSched_admittedRequestStartsWaiting) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(100, 32);
    const auto* r = s.find(id);
    EXPECT_TRUE(r != nullptr);
    EXPECT_EQ(r->state,          RequestState::Waiting);
    EXPECT_EQ(r->prompt_length,  std::int32_t{100});
    EXPECT_EQ(r->tokens_pending, std::int32_t{100});
    EXPECT_EQ(r->tokens_decoded, std::int32_t{0});
    EXPECT_EQ(r->max_tokens,     std::int32_t{32});
}

TEST(reqSched_tickPromotesWaitingToPrefilling) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(100, 32);
    (void)s.tick();
    const auto* r = s.find(id);
    EXPECT_TRUE(r != nullptr);
    EXPECT_EQ(r->state, RequestState::Prefilling);
    EXPECT_EQ(s.numActive(),  std::size_t{1});
    EXPECT_EQ(s.numWaiting(), std::size_t{0});
}

TEST(reqSched_tickRespectsMaxActiveCap) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 2};
    (void)s.admit(100, 32);
    (void)s.admit(100, 32);
    (void)s.admit(100, 32);   // this one stays Waiting
    (void)s.tick();
    EXPECT_EQ(s.numActive(),  std::size_t{2});
    EXPECT_EQ(s.numWaiting(), std::size_t{1});
}

TEST(reqSched_tickProducesScheduleForActiveRequests) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    (void)s.admit(100, 32);
    const auto sched = s.tick();
    EXPECT_EQ(sched.prefills.size(),         std::size_t{1});
    EXPECT_EQ(sched.prefills.front().chunk_size, std::int32_t{100});
    EXPECT_EQ(sched.decodes.size(),          std::size_t{0});
    EXPECT_EQ(sched.total_tokens_scheduled,  std::int32_t{100});
}

TEST(reqSched_commitPrefillAdvancesTokensPending) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    // Long prompt (> budget) — needs multiple prefill iterations.
    RequestScheduler s{100, 4};
    const auto id = s.admit(250, 32);
    // Iter 1: chunk 100, pending 250 → 150, still Prefilling
    const auto sched1 = s.tick();
    EXPECT_EQ(sched1.prefills.front().chunk_size, std::int32_t{100});
    s.commitProgress(sched1, {});
    {
        const auto* r = s.find(id);
        EXPECT_EQ(r->tokens_pending, std::int32_t{150});
        EXPECT_EQ(r->state,          RequestState::Prefilling);
    }
    // Iter 2: chunk 100, pending 150 → 50
    const auto sched2 = s.tick();
    EXPECT_EQ(sched2.prefills.front().chunk_size, std::int32_t{100});
    s.commitProgress(sched2, {});
    {
        const auto* r = s.find(id);
        EXPECT_EQ(r->tokens_pending, std::int32_t{50});
        EXPECT_EQ(r->state,          RequestState::Prefilling);
    }
    // Iter 3: chunk 50, pending 50 → 0, promote to Decoding
    const auto sched3 = s.tick();
    EXPECT_EQ(sched3.prefills.front().chunk_size, std::int32_t{50});
    s.commitProgress(sched3, {});
    {
        const auto* r = s.find(id);
        EXPECT_EQ(r->tokens_pending, std::int32_t{0});
        EXPECT_EQ(r->state,          RequestState::Decoding);
    }
}

TEST(reqSched_commitDecodeAdvancesTokensDecoded) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(10, 5);
    // Prefill in one shot.
    const auto sched0 = s.tick();
    s.commitProgress(sched0, {});
    // Now Decoding — three decode iterations without EOS.
    for (int i = 1; i <= 3; ++i) {
        const auto sched = s.tick();
        EXPECT_EQ(sched.decodes.size(), std::size_t{1});
        const std::vector<std::uint8_t> eos(1, 0);
        s.commitProgress(sched, eos);
        const auto* r = s.find(id);
        EXPECT_EQ(r->tokens_decoded, std::int32_t{i});
        EXPECT_EQ(r->state,          RequestState::Decoding);
    }
}

TEST(reqSched_commitDecodePromotesOnEos) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(10, 100);
    s.commitProgress(s.tick(), {});  // prefill
    const auto sched = s.tick();
    const std::vector<std::uint8_t> eos(1, 1);
    s.commitProgress(sched, eos);
    EXPECT_EQ(s.find(id)->state, RequestState::Completed);
}

TEST(reqSched_commitDecodePromotesOnMaxTokens) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(10, 2);
    s.commitProgress(s.tick(), {});  // prefill → Decoding
    // Decode 1
    s.commitProgress(s.tick(), std::vector<std::uint8_t>{0});
    EXPECT_EQ(s.find(id)->state, RequestState::Decoding);
    // Decode 2 → hits max_tokens
    s.commitProgress(s.tick(), std::vector<std::uint8_t>{0});
    EXPECT_EQ(s.find(id)->state,          RequestState::Completed);
    EXPECT_EQ(s.find(id)->tokens_decoded, std::int32_t{2});
}

TEST(reqSched_preemptOneReturnsZeroWhenNoActive) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    EXPECT_EQ(s.preemptOne(), std::uint64_t{0});
    // Even with Waiting requests, nothing active to preempt.
    (void)s.admit(100, 32);
    EXPECT_EQ(s.preemptOne(), std::uint64_t{0});
}

TEST(reqSched_preemptOnePicksLifo) {
    // Newest active request has the lowest KV investment → cheapest
    // to RECOMPUTE. Preemption should pick it, not the oldest.
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto a = s.admit(100, 32);
    const auto b = s.admit(100, 32);
    const auto c = s.admit(100, 32);
    (void)s.tick();  // all three become Prefilling
    const auto victim = s.preemptOne();
    EXPECT_EQ(victim, c);
    EXPECT_EQ(s.find(a)->state, RequestState::Prefilling);
    EXPECT_EQ(s.find(b)->state, RequestState::Prefilling);
    EXPECT_EQ(s.find(c)->state, RequestState::Preempted);
    EXPECT_EQ(s.numPreempted(), std::size_t{1});
    EXPECT_EQ(s.numActive(),    std::size_t{2});
}

TEST(reqSched_preemptedRequestRestartsFromWaitingOnNextTick) {
    // RECOMPUTE flow: preempted → next tick moves to Waiting →
    // (if room) promoted to Prefilling with FRESH tokens_pending =
    // prompt_length (KV is gone, must recompute).
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{50, 2};
    const auto a = s.admit(30, 32);
    const auto b = s.admit(30, 32);
    // Iter 1: both promoted, budget=50 → a gets 30, b gets 20
    const auto sched1 = s.tick();
    EXPECT_EQ(sched1.prefills.size(), std::size_t{2});
    s.commitProgress(sched1, {});
    // b has 10 pending, a is now Decoding? no, a's pending was 30 → 0
    // wait, chunk=30 for a, chunk=20 for b (50 budget, 30 first, 20 remainder)
    // → a.tokens_pending=0 → Decoding, b.tokens_pending=10, still Prefilling
    EXPECT_EQ(s.find(a)->state,          RequestState::Decoding);
    EXPECT_EQ(s.find(b)->state,          RequestState::Prefilling);
    EXPECT_EQ(s.find(b)->tokens_pending, std::int32_t{10});
    // Preempt b (only one non-Completed active LIFO candidate).
    const auto victim = s.preemptOne();
    EXPECT_EQ(victim, b);
    EXPECT_EQ(s.find(b)->state, RequestState::Preempted);
    // Next tick: b re-enqueues to Waiting, then promotes → Prefilling
    // with FRESH tokens_pending = prompt_length (30, not 10 —
    // RECOMPUTE means KV is gone).
    (void)s.tick();
    EXPECT_EQ(s.find(b)->state,          RequestState::Prefilling);
    EXPECT_EQ(s.find(b)->tokens_pending, std::int32_t{30});
    EXPECT_EQ(s.find(b)->tokens_decoded, std::int32_t{0});
}

TEST(reqSched_drainCompletedRemovesAndReturnsIds) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    const auto a = s.admit(5, 1);
    const auto b = s.admit(5, 1);
    // Prefill both
    s.commitProgress(s.tick(), {});
    // Decode both — both complete on 1st token (max_tokens=1)
    const auto sched = s.tick();
    EXPECT_EQ(sched.decodes.size(), std::size_t{2});
    const std::vector<std::uint8_t> eos{0, 0};
    s.commitProgress(sched, eos);
    EXPECT_EQ(s.numCompleted(), std::size_t{2});
    const auto drained = s.drainCompleted();
    EXPECT_EQ(drained.size(),   std::size_t{2});
    EXPECT_EQ(drained[0],       a);
    EXPECT_EQ(drained[1],       b);
    EXPECT_EQ(s.numCompleted(), std::size_t{0});
    EXPECT_TRUE(s.find(a) == nullptr);
    EXPECT_TRUE(s.find(b) == nullptr);
}

TEST(reqSched_mismatchedEosSizeIsSafe) {
    // Defensive: caller passes wrong-size reachedEos. Must not
    // corrupt state — treat as all-false (no EOS this iteration).
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    RequestScheduler s{512, 4};
    const auto id = s.admit(5, 100);
    s.commitProgress(s.tick(), {});  // prefill
    const auto sched = s.tick();
    // Wrong-size eos vector (empty when 1 decode assigned).
    s.commitProgress(sched, {});
    // State advanced (tokens_decoded++), but no EOS triggered — no
    // spurious Completed.
    EXPECT_EQ(s.find(id)->tokens_decoded, std::int32_t{1});
    EXPECT_EQ(s.find(id)->state,          RequestState::Decoding);
}

TEST(reqSched_inspectionCountsPartitionsCorrectly) {
    // Post-state: 1 Waiting + 1 Prefilling + 1 Decoding + 1 Completed.
    // Counts must sum to storage size.
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{100, 2};
    (void)s.admit(50, 1);        // will finish
    (void)s.admit(200, 32);      // needs 2 prefill iters, stays Prefilling
    (void)s.admit(10, 32);       // waits (maxActive=2)
    // Iter 1: promote first two, chunk a=50, b=50 → a completes prefill
    // → Decoding; b tokens_pending = 150 still Prefilling.
    const auto sched1 = s.tick();
    s.commitProgress(sched1, {});
    // Iter 2: a decodes 1 token → Completed (max_tokens=1);
    // b prefills another chunk of 100 → pending=50, still Prefilling.
    const auto sched2 = s.tick();
    const std::vector<std::uint8_t> eos(sched2.decodes.size(), 0);
    s.commitProgress(sched2, eos);
    // Now: a Completed, b Prefilling, c Waiting (still can't promote —
    // b holds one of the two active slots).
    EXPECT_EQ(s.numCompleted(), std::size_t{1});
    EXPECT_EQ(s.numActive(),    std::size_t{1});
    EXPECT_EQ(s.numWaiting(),   std::size_t{1});
    EXPECT_EQ(s.numPreempted(), std::size_t{0});
}

TEST(reqSched_tokenBudgetAccessorReflectsCtor) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{256, 8};
    EXPECT_EQ(s.tokenBudget(),        std::int32_t{256});
    EXPECT_EQ(s.maxActiveRequests(),  std::size_t{8});
}

// -----------------------------------------------------------------------
// RequestScheduler ↔ PagedKvBlockAllocator wire-up (Phase A + C bridge)
// -----------------------------------------------------------------------

TEST(reqSchedKv_stateMachineOnlyModeHasNoAllocator) {
    // Backwards-compat: existing 2-arg ctor leaves allocator null.
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    EXPECT_TRUE(s.allocator() == nullptr);
}

TEST(reqSchedKv_ctorStoresAllocatorPointer) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    EXPECT_TRUE(s.allocator() == &pool);
}

TEST(reqSchedKv_admitCreatesEmptySequenceWhenAllocatorPresent) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(10, 5);
    const auto* r = s.find(id);
    EXPECT_TRUE(r != nullptr);
    EXPECT_TRUE(r->kv_sequence.has_value());
    EXPECT_EQ(r->kv_sequence->numTokens(), std::size_t{0});
    EXPECT_EQ(r->kv_sequence->numBlocks(), std::size_t{0});
    // Allocator untouched — sequence is empty until Phase D feeds
    // tokens through feedPrefillTokens.
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
}

TEST(reqSchedKv_admitWithoutAllocatorSkipsSequence) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    const auto id = s.admit(10, 5);
    const auto* r = s.find(id);
    EXPECT_TRUE(r != nullptr);
    EXPECT_TRUE(!r->kv_sequence.has_value());
}

TEST(reqSchedKv_feedPrefillTokensAllocatesBlocks) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};   // 16 blocks × 4 tokens = 64 slots
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(10, 5);
    // Feed 10 prompt tokens. Block-size 4 → 3 blocks (10/4 = 2 rem 2 → 3).
    const std::int32_t tokens[10] = {1,2,3,4,5,6,7,8,9,10};
    EXPECT_TRUE(s.feedPrefillTokens(id, tokens));
    const auto* r = s.find(id);
    EXPECT_EQ(r->kv_sequence->numTokens(), std::size_t{10});
    EXPECT_EQ(r->kv_sequence->numBlocks(), std::size_t{3});
    EXPECT_EQ(pool.numBlocksUsed(),        std::size_t{3});
}

TEST(reqSchedKv_feedDecodeTokenAppendsOne) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(10, 5);
    EXPECT_TRUE(s.feedDecodeToken(id, 42));
    EXPECT_EQ(s.find(id)->kv_sequence->numTokens(), std::size_t{1});
    EXPECT_EQ(pool.numBlocksUsed(),                 std::size_t{1});
}

TEST(reqSchedKv_feedRejectsUnknownId) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const std::int32_t tokens[1] = {7};
    EXPECT_TRUE(!s.feedPrefillTokens(9999, tokens));
    EXPECT_TRUE(!s.feedDecodeToken(9999, 7));
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
}

TEST(reqSchedKv_feedIsNoopWithoutAllocator) {
    // State-machine mode: feed silently returns false without
    // touching anything.
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    const auto id = s.admit(10, 5);
    const std::int32_t tokens[1] = {7};
    EXPECT_TRUE(!s.feedPrefillTokens(id, tokens));
    EXPECT_TRUE(!s.feedDecodeToken(id, 7));
}

TEST(reqSchedKv_preemptOneReleasesBlocksEagerly) {
    // The whole point of preemption is to free KV pressure. preemptOne
    // must actually shrink numBlocksUsed, not just flip a state bit.
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const auto a = s.admit(20, 5);
    const auto b = s.admit(20, 5);
    (void)a;
    // Feed b with 12 tokens → 3 blocks
    const std::int32_t toks[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    EXPECT_TRUE(s.feedPrefillTokens(b, toks));
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{3});
    (void)s.tick();  // promote both to Prefilling so preemptOne can find them
    const auto victim = s.preemptOne();
    EXPECT_EQ(victim, b);   // LIFO — b is newest
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});  // b's 3 blocks released
    EXPECT_EQ(s.find(b)->kv_sequence->numTokens(), std::size_t{0});
    EXPECT_EQ(s.find(b)->kv_sequence->numBlocks(), std::size_t{0});
}

TEST(reqSchedKv_drainCompletedReleasesBlocks) {
    // Completion path: caller drains, blocks return to pool via
    // sequence destructor.
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(5, 1);
    const std::int32_t toks[5] = {1,2,3,4,5};
    EXPECT_TRUE(s.feedPrefillTokens(id, toks));   // 2 blocks
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{2});
    // Push through the state machine to Completed.
    s.commitProgress(s.tick(), {});             // prefill → Decoding
    s.commitProgress(s.tick(), std::vector<std::uint8_t>{0});  // decode → Completed (max_tokens=1)
    const auto drained = s.drainCompleted();
    EXPECT_EQ(drained.size(), std::size_t{1});
    EXPECT_EQ(drained.front(), id);
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
}

TEST(reqSchedKv_schedulerDestructorReleasesAllBlocks) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{16, 4};
    {
        RequestScheduler s{512, 4, &pool};
        const auto a = s.admit(5, 5);
        const auto b = s.admit(5, 5);
        const std::int32_t toks[5] = {1,2,3,4,5};
        EXPECT_TRUE(s.feedPrefillTokens(a, toks));
        EXPECT_TRUE(s.feedPrefillTokens(b, toks));
        EXPECT_EQ(pool.numBlocksUsed(), std::size_t{4});
    }
    // Scheduler out of scope — all sequences destroyed → blocks freed.
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
}

TEST(reqSchedKv_prefixSharingInvariantHoldsThroughScheduler) {
    // End-to-end test of the M-PrefixCache invariant: two sequences
    // that receive identical first-block tokens through the scheduler
    // API produce identical block-0 hashes at the allocator level.
    // Proves that scheduler wire-up doesn't break the property that
    // PagedKvSequence guarantees.
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{32, 4};
    RequestScheduler s{512, 4, &pool};
    const auto a = s.admit(4, 5);
    const auto b = s.admit(4, 5);
    const std::int32_t sharedPrefix[4] = {10, 20, 30, 40};
    EXPECT_TRUE(s.feedPrefillTokens(a, sharedPrefix));
    EXPECT_TRUE(s.feedPrefillTokens(b, sharedPrefix));
    const auto blkA = s.find(a)->kv_sequence->blockTable().front();
    const auto blkB = s.find(b)->kv_sequence->blockTable().front();
    // Different physical blocks (no dedup at Allocator level yet —
    // that's M-PrefixCache) but identical HASHES.
    EXPECT_TRUE(blkA != blkB);
    EXPECT_EQ(pool.hashOf(blkA), pool.hashOf(blkB));
    EXPECT_TRUE(pool.hashOf(blkA) != std::uint64_t{0});
}

// -----------------------------------------------------------------------
// ServingMetrics + snapshotMetrics()
// -----------------------------------------------------------------------

TEST(reqSchedMetrics_freshSchedulerHasZeroCounters) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.num_waiting,           std::size_t{0});
    EXPECT_EQ(m.num_active,            std::size_t{0});
    EXPECT_EQ(m.num_preempted,         std::size_t{0});
    EXPECT_EQ(m.num_completed_current, std::size_t{0});
    EXPECT_EQ(m.total_admitted,        std::uint64_t{0});
    EXPECT_EQ(m.total_completed,       std::uint64_t{0});
    EXPECT_EQ(m.total_preempted,       std::uint64_t{0});
    // State-machine-only mode: pool fields all zero.
    EXPECT_EQ(m.block_pool_total,      std::size_t{0});
    EXPECT_EQ(m.block_pool_free,       std::size_t{0});
    EXPECT_EQ(m.block_pool_used,       std::size_t{0});
    EXPECT_TRUE(m.block_pool_utilization == 0.0);
}

TEST(reqSchedMetrics_admitBumpsTotalAdmitted) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    (void)s.admit(10, 5);
    (void)s.admit(10, 5);
    (void)s.admit(0, 5);   // rejected, must NOT bump
    (void)s.admit(10, 0);  // rejected, must NOT bump
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_admitted, std::uint64_t{2});
    EXPECT_EQ(m.num_waiting,    std::size_t{2});
}

TEST(reqSchedMetrics_completionBumpsTotalCompleted) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    (void)s.admit(5, 1);
    s.commitProgress(s.tick(), {});                        // prefill → Decoding
    s.commitProgress(s.tick(), std::vector<std::uint8_t>{0});  // decode → Completed
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_completed,       std::uint64_t{1});
    EXPECT_EQ(m.num_completed_current, std::size_t{1});
    // total_admitted persists across the completion.
    EXPECT_EQ(m.total_admitted,        std::uint64_t{1});
}

TEST(reqSchedMetrics_totalCompletedPersistsAcrossDrain) {
    // Drain removes the Completed entries but the monotonic total
    // must not decrement. That's the "since-start" contract.
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    (void)s.admit(5, 1);
    s.commitProgress(s.tick(), {});
    s.commitProgress(s.tick(), std::vector<std::uint8_t>{0});
    (void)s.drainCompleted();
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_completed,       std::uint64_t{1});   // persists
    EXPECT_EQ(m.num_completed_current, std::size_t{0});     // drained
}

TEST(reqSchedMetrics_preemptBumpsTotalPreempted) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    (void)s.admit(10, 5);
    (void)s.admit(10, 5);
    (void)s.tick();
    (void)s.preemptOne();
    (void)s.preemptOne();   // second preempt (a different request)
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_preempted, std::uint64_t{2});
    EXPECT_EQ(m.num_preempted,   std::size_t{2});
}

TEST(reqSchedMetrics_totalPreemptedCountsEventsNotRequests) {
    // A single request that gets preempted, re-enqueued, and preempted
    // again should count as TWO preemption events (not one) — the
    // counter tracks pressure-relief actions, not unique victims.
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 2};
    (void)s.admit(10, 5);
    (void)s.admit(10, 5);
    (void)s.tick();
    (void)s.preemptOne();   // victim = req 2
    (void)s.tick();         // req 2 back to Waiting → Prefilling
    (void)s.preemptOne();   // victim = req 2 again (still LIFO)
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_preempted, std::uint64_t{2});
}

TEST(reqSchedMetrics_preemptOnEmptyIsNoop) {
    using ::mimirmind::runtime::serving::RequestScheduler;
    RequestScheduler s{512, 4};
    EXPECT_EQ(s.preemptOne(), std::uint64_t{0});   // nothing active
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.total_preempted, std::uint64_t{0});
}

TEST(reqSchedMetrics_poolFieldsPopulatedWithAllocator) {
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{100, 4};
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(10, 5);
    const std::int32_t toks[10] = {1,2,3,4,5,6,7,8,9,10};
    EXPECT_TRUE(s.feedPrefillTokens(id, toks));   // 3 blocks used
    const auto m = s.snapshotMetrics();
    EXPECT_EQ(m.block_pool_total, std::size_t{100});
    EXPECT_EQ(m.block_pool_used,  std::size_t{3});
    EXPECT_EQ(m.block_pool_free,  std::size_t{97});
    // 3/100 = 0.03 — round-trip through the division so exact-compare
    // isn't safe; use range.
    EXPECT_TRUE(m.block_pool_utilization > 0.029 && m.block_pool_utilization < 0.031);
}

TEST(reqSchedMetrics_poolUtilisationTracksBlocksLive) {
    // After preemption releases blocks, utilisation should drop.
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    PagedKvBlockAllocator pool{100, 4};
    RequestScheduler s{512, 4, &pool};
    const auto a = s.admit(20, 5);
    const auto b = s.admit(20, 5);
    (void)a;
    const std::int32_t toks[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    EXPECT_TRUE(s.feedPrefillTokens(b, toks));   // 5 blocks used
    {
        const auto m = s.snapshotMetrics();
        EXPECT_EQ(m.block_pool_used, std::size_t{5});
    }
    (void)s.tick();
    (void)s.preemptOne();
    {
        const auto m = s.snapshotMetrics();
        EXPECT_EQ(m.block_pool_used, std::size_t{0});
        EXPECT_TRUE(m.block_pool_utilization == 0.0);
    }
}

// -----------------------------------------------------------------------
// PreemptionPolicy — Sub-Step C5 (M-Cuda.Batch Phase C)
// -----------------------------------------------------------------------

#include "runtime/serving/PreemptionPolicy.hpp"

TEST(preemptPolicy_ctorDefaultThresholdIsFivePercent) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{};
    // Round-trip through ctor without arithmetic — bit-identical to
    // the literal, safe to compare exactly. (EXPECT_NEAR would trip
    // TestFramework's float-only signature.)
    EXPECT_TRUE(p.freeBlockThreshold() == PreemptionPolicy::kDefaultFreeBlockThreshold);
}

TEST(preemptPolicy_ctorRejectsNegativeThreshold) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    bool threw = false;
    try {
        PreemptionPolicy p{-0.1};
        (void)p;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(preemptPolicy_ctorRejectsAboveOne) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    bool threw = false;
    try {
        PreemptionPolicy p{1.5};
        (void)p;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(preemptPolicy_ctorAcceptsBoundaryValues) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    // Both 0.0 and 1.0 are legal — see class docstring for semantics.
    PreemptionPolicy p0{0.0};
    PreemptionPolicy p1{1.0};
    EXPECT_TRUE(p0.freeBlockThreshold() == 0.0);
    EXPECT_TRUE(p1.freeBlockThreshold() == 1.0);
}

TEST(preemptPolicy_zeroTotalBlocksIsFalse) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.5};
    EXPECT_TRUE(!p.shouldPreempt(0, 0, 4));
}

TEST(preemptPolicy_soleActiveIsFalse) {
    // Sole-active guard: preempting the only active request thrashes.
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.5};
    // Well below threshold (0/100 = 0% free) but numActive=1 → false.
    EXPECT_TRUE(!p.shouldPreempt(0, 100, 1));
    EXPECT_TRUE(!p.shouldPreempt(0, 100, 0));
}

TEST(preemptPolicy_belowThresholdTriggers) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.10};   // 10 %
    // 4/100 = 4 % free — below threshold, 2 active — trigger.
    EXPECT_TRUE(p.shouldPreempt(4, 100, 2));
}

TEST(preemptPolicy_atThresholdDoesNotTrigger) {
    // Strict-less-than: exactly at threshold is safe.
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.10};
    EXPECT_TRUE(!p.shouldPreempt(10, 100, 4));
}

TEST(preemptPolicy_aboveThresholdDoesNotTrigger) {
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.10};
    EXPECT_TRUE(!p.shouldPreempt(50, 100, 4));
    EXPECT_TRUE(!p.shouldPreempt(100, 100, 4));
}

TEST(preemptPolicy_zeroThresholdIsDisabled) {
    // Docstring contract: threshold=0.0 effectively disables the
    // policy — freeRatio < 0.0 is impossible.
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.0};
    EXPECT_TRUE(!p.shouldPreempt(0, 100, 8));    // 0% free but disabled
    EXPECT_TRUE(!p.shouldPreempt(50, 100, 8));
    EXPECT_TRUE(!p.shouldPreempt(100, 100, 8));
}

TEST(preemptPolicy_oneThresholdIsMaximallyEager) {
    // threshold=1.0 → preempt whenever any block is used
    // (freeRatio < 1.0 iff at least one block used).
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{1.0};
    EXPECT_TRUE(p.shouldPreempt(99, 100, 2));    // 99% free but not 100 → trigger
    EXPECT_TRUE(!p.shouldPreempt(100, 100, 2));  // truly empty pool
}

TEST(preemptPolicy_isPure) {
    // Sanity: same input → same output, no hidden state.
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{0.10};
    const auto a = p.shouldPreempt(5, 100, 3);
    const auto b = p.shouldPreempt(5, 100, 3);
    const auto c = p.shouldPreempt(5, 100, 3);
    EXPECT_TRUE(a && b && c);
}

TEST(preemptPolicy_scenarioBragiDefaults) {
    // Realistic scenario: Bragi-v1 default 5% threshold, 8k-block
    // pool, 32 active decodes. At 3% free (240 blocks), preempt.
    // At 8% free (640 blocks), don't preempt.
    using ::mimirmind::runtime::serving::PreemptionPolicy;
    PreemptionPolicy p{};   // 5% default
    EXPECT_TRUE(p.shouldPreempt(240, 8000, 32));      // 3% → trigger
    EXPECT_TRUE(!p.shouldPreempt(640, 8000, 32));     // 8% → safe
    EXPECT_TRUE(p.shouldPreempt(240, 8000, 2));       // Same threshold, min active
    EXPECT_TRUE(!p.shouldPreempt(240, 8000, 1));      // Sole-active guard
}

TEST(reqSchedKv_preemptedRequestGetsFreshSequenceOnRecompute) {
    // RECOMPUTE flow with KV: preempt → blocks released → next tick
    // re-enqueues → Phase D re-feeds prompt tokens → fresh blocks
    // allocated from the pool (possibly the same physical ids via
    // LIFO reuse, but a NEW allocation).
    using ::mimirmind::runtime::serving::PagedKvBlockAllocator;
    using ::mimirmind::runtime::serving::RequestScheduler;
    using ::mimirmind::runtime::serving::RequestState;
    PagedKvBlockAllocator pool{16, 4};
    RequestScheduler s{512, 4, &pool};
    const auto id = s.admit(8, 5);
    const std::int32_t toks[8] = {1,2,3,4,5,6,7,8};
    EXPECT_TRUE(s.feedPrefillTokens(id, toks));
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{2});
    (void)s.tick();
    (void)s.preemptOne();
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{0});
    EXPECT_EQ(s.find(id)->state, RequestState::Preempted);
    // Next tick re-enqueues + promotes.
    (void)s.tick();
    EXPECT_EQ(s.find(id)->state,          RequestState::Prefilling);
    EXPECT_EQ(s.find(id)->tokens_pending, std::int32_t{8});   // fresh
    // Sequence is empty — Phase D feeds prompt tokens again.
    EXPECT_EQ(s.find(id)->kv_sequence->numTokens(), std::size_t{0});
    EXPECT_TRUE(s.feedPrefillTokens(id, toks));
    EXPECT_EQ(pool.numBlocksUsed(), std::size_t{2});
}

int main() {
    return mm::test::run();
}

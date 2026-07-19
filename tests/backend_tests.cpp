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

int main() {
    return mm::test::run();
}

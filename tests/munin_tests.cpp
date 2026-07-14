// Pure-CPU tests for the M-Munin wire-op layer. Covers the JSON
// envelope serde for request / healthz / error frames. The socket-
// integration layer (AttachSession + SocketServer) is exercised in
// the docker-based smoke tests that come with Schritt 8 — nothing
// meaningful to unit-test here without a live L0 device and a peer
// worker.
//
// Run via: cmake --build build --target munin_tests && build/munin_tests

#include "TestFramework.hpp"

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/WireOps.hpp"
#include "core/os/GovernorLock.hpp"
#include "munin/ChunkAllocator.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using ::mimirmind::core::gguf::GgmlType;
using ::mimirmind::core::gguf::GgufTensor;
using ::mimirmind::core::gguf::WeightsMap;
using ::mimirmind::core::ipc::HealthzResponse;
using ::mimirmind::core::ipc::ModelSummaryWire;
using ::mimirmind::core::ipc::RequestEnvelope;
using ::mimirmind::core::ipc::makeErrorJson;
using ::mimirmind::core::ipc::parseErrorJson;
using ::mimirmind::core::os::GovernorLock;
using ::mimirmind::munin::ChunkAllocator;

namespace {

// Build a valid path under $TMPDIR (or /tmp) unique to the test process.
std::string tmpFilePath(const std::string& tag) {
    const char* d = std::getenv("TMPDIR");
    std::string root = (d != nullptr && d[0] != '\0') ? d : "/tmp";
    return root + "/mimirmind-test-" + tag + "-" +
           std::to_string(::getpid()) + ".lock";
}

} // namespace

// ---- RequestEnvelope --------------------------------------------------------

TEST(requestEnvelope_healthz_parses) {
    const auto r = RequestEnvelope::fromJson(R"({"op":"healthz"})");
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->op, std::string{"healthz"});
    EXPECT_EQ(r->modelId, std::string{""});
}

TEST(requestEnvelope_attach_withModelId_parses) {
    const auto r = RequestEnvelope::fromJson(
        R"({"op":"attach","modelId":"gemma-4-e4b"})");
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->op, std::string{"attach"});
    EXPECT_EQ(r->modelId, std::string{"gemma-4-e4b"});
}

TEST(requestEnvelope_missingOp_rejected) {
    const auto r = RequestEnvelope::fromJson(R"({"modelId":"x"})");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_modelIdWrongType_rejected) {
    const auto r = RequestEnvelope::fromJson(
        R"({"op":"attach","modelId":42})");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_malformedJson_rejected) {
    const auto r = RequestEnvelope::fromJson(R"({"op":)");
    EXPECT_TRUE(!r);
}

TEST(requestEnvelope_rootNotObject_rejected) {
    const auto r = RequestEnvelope::fromJson(R"("healthz")");
    EXPECT_TRUE(!r);
}

// ---- HealthzResponse round-trip --------------------------------------------

TEST(healthzResponse_roundtrip_emptyModels) {
    HealthzResponse r{};
    r.pid           = 4711;
    r.governorOwner = "munin";

    const std::string s = r.toJson();
    const auto parsed = HealthzResponse::fromJson(s);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->protocolVersion, 1U);
    EXPECT_EQ(parsed->status, std::string{"ok"});
    EXPECT_EQ(parsed->governorOwner, std::string{"munin"});
    EXPECT_EQ(parsed->pid, 4711U);
    EXPECT_EQ(parsed->models.size(), 0U);
}

TEST(healthzResponse_roundtrip_twoModels) {
    HealthzResponse r{};
    r.pid = 123;
    {
        ModelSummaryWire m{};
        m.id          = "gemma-4-e4b";
        m.fingerprint = "720.25169342464.deadbeefcafebabe";
        m.totalBytes  = 25169342464ULL;
        m.tensorCount = 720;
        r.models.push_back(std::move(m));
    }
    {
        ModelSummaryWire m{};
        m.id          = "qwen2-7b";
        m.fingerprint = "512.7000000000.1234567890abcdef";
        m.totalBytes  = 7000000000ULL;
        m.tensorCount = 512;
        r.models.push_back(std::move(m));
    }

    const auto parsed = HealthzResponse::fromJson(r.toJson());
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->models.size(), 2U);
    EXPECT_EQ(parsed->models[0].id, std::string{"gemma-4-e4b"});
    EXPECT_EQ(parsed->models[0].tensorCount, 720U);
    EXPECT_EQ(parsed->models[0].totalBytes, 25169342464ULL);
    EXPECT_EQ(parsed->models[1].id, std::string{"qwen2-7b"});
    EXPECT_EQ(parsed->models[1].fingerprint,
              std::string{"512.7000000000.1234567890abcdef"});
}

TEST(healthzResponse_missingField_rejected) {
    // Drop `pid` — fromJson must refuse rather than default.
    const auto parsed = HealthzResponse::fromJson(
        R"({"protocol_version":1,"status":"ok","governor_owner":"munin","models":[]})");
    EXPECT_TRUE(!parsed);
}

// ---- Error envelope --------------------------------------------------------

TEST(errorEnvelope_roundtrip) {
    const std::string body = makeErrorJson("no model with id 'x' loaded");
    const auto parsed = parseErrorJson(body);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(*parsed, std::string{"no model with id 'x' loaded"});
}

TEST(errorEnvelope_notAnError_rejected) {
    const auto parsed = parseErrorJson(R"({"status":"ok"})");
    EXPECT_TRUE(!parsed);
}

TEST(errorEnvelope_wrongType_rejected) {
    const auto parsed = parseErrorJson(R"({"error":42})");
    EXPECT_TRUE(!parsed);
}

// ---- WeightsMap attached-mode ---------------------------------------------

TEST(weightsMap_fromAttached_findWorks) {
    std::vector<GgufTensor> ts;
    {
        GgufTensor t{};
        t.name       = "token_embd.weight";
        t.type       = GgmlType::Q4_K;
        t.dimensions = {2560, 262144};
        t.nbytes     = 128;
        t.usmPtr     = reinterpret_cast<void*>(0xdeadbeefULL);
        ts.push_back(std::move(t));
    }
    {
        GgufTensor t{};
        t.name       = "blk.0.attn_q.weight";
        t.type       = GgmlType::Q6_K;
        t.dimensions = {2560, 2560};
        t.nbytes     = 64;
        t.usmPtr     = reinterpret_cast<void*>(0xcafebabeULL);
        ts.push_back(std::move(t));
    }

    WeightsMap m{std::move(ts)};
    EXPECT_TRUE(m.isAttached());
    EXPECT_EQ(m.size(), 2U);

    const auto* q = m.find("blk.0.attn_q.weight");
    EXPECT_TRUE(q != nullptr);
    EXPECT_EQ(q->name, std::string{"blk.0.attn_q.weight"});
    EXPECT_TRUE(q->usmPtr == reinterpret_cast<void*>(0xcafebabeULL));

    const auto* missing = m.find("token_embd.output");
    EXPECT_TRUE(missing == nullptr);

    // findBlock helper — same result as explicit name lookup.
    const auto* qViaBlock = m.findBlock(0, "attn_q.weight");
    EXPECT_TRUE(qViaBlock == q);
}

TEST(weightsMap_fromAttached_requireThrowsForMissing) {
    // Attach one dummy tensor so the map's isAttached() (which reports
    // "we own tensor entries") is true, then require() something that
    // doesn't exist.
    std::vector<GgufTensor> ts;
    GgufTensor t{};
    t.name       = "irrelevant";
    t.type       = GgmlType::F32;
    t.dimensions = {1};
    ts.push_back(std::move(t));
    WeightsMap m{std::move(ts)};
    EXPECT_TRUE(m.isAttached());

    bool threw = false;
    try {
        (void)m.require("nope");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

// ---- WeightsMap::fromAttachedChunked --------------------------------------
//
// Pure-CPU coverage of the chunk-attached construction path. No L0
// device needed: we fabricate two byte-buffers as fake chunk bases and
// verify each tensor's usmPtr is computed as `base + chunkOffset`.

TEST(weightsMap_fromAttachedChunked_computesTensorPointers) {
    using ::mimirmind::core::ipc::ChunkDesc;
    using ::mimirmind::core::ipc::ManifestEntry;
    using ::mimirmind::core::ipc::TensorManifest;

    // Fake two chunks as heap buffers. Only pointer arithmetic matters,
    // never dereferenced by fromAttachedChunked.
    std::vector<std::byte> chunk0(4096);
    std::vector<std::byte> chunk1(2048);
    std::vector<void*> chunkBases{
        static_cast<void*>(chunk0.data()),
        static_cast<void*>(chunk1.data()),
    };

    TensorManifest m{};
    m.modelId          = "id";
    m.modelFingerprint = "fp";
    m.chunks.push_back({.chunkIndex = 0, .bytes = 4096});
    m.chunks.push_back({.chunkIndex = 1, .bytes = 2048});

    m.tensors.push_back({
        .name        = "blk.0.attn_q.weight",
        .type        = GgmlType::Q4_K,
        .dims        = {2560, 4096},
        .bytes       = 128,
        .chunkIndex  = 0,
        .chunkOffset = 64,
    });
    m.tensors.push_back({
        .name        = "token_embd.weight",
        .type        = GgmlType::Q6_K,
        .dims        = {2560, 262144},
        .bytes       = 256,
        .chunkIndex  = 1,
        .chunkOffset = 128,
    });

    auto wm = WeightsMap::fromAttachedChunked(
        m, std::span<void* const>{chunkBases});
    EXPECT_TRUE(wm.isAttached());
    EXPECT_EQ(wm.size(), 2U);

    const auto* q = wm.find("blk.0.attn_q.weight");
    EXPECT_TRUE(q != nullptr);
    EXPECT_TRUE(q->usmPtr == static_cast<void*>(chunk0.data() + 64));
    EXPECT_EQ(q->chunkIndex,  0U);
    EXPECT_EQ(q->chunkOffset, std::uint64_t{64});
    // nelements should be recomputed from dims.
    EXPECT_EQ(q->nelements, std::uint64_t{2560} * 4096);

    const auto* e = wm.find("token_embd.weight");
    EXPECT_TRUE(e != nullptr);
    EXPECT_TRUE(e->usmPtr == static_cast<void*>(chunk1.data() + 128));
    EXPECT_EQ(e->chunkIndex,  1U);
    EXPECT_EQ(e->chunkOffset, std::uint64_t{128});
}

TEST(weightsMap_fromAttachedChunked_outOfRangeChunkIndex_throws) {
    using ::mimirmind::core::ipc::ManifestEntry;
    using ::mimirmind::core::ipc::TensorManifest;

    std::vector<std::byte> chunk0(1024);
    std::vector<void*> chunkBases{static_cast<void*>(chunk0.data())};

    TensorManifest m{};
    m.modelId          = "id";
    m.modelFingerprint = "fp";
    m.chunks.push_back({.chunkIndex = 0, .bytes = 1024});
    // Tensor points at chunkIndex=2 but chunkBases has only one entry.
    // Manifest parser normally refuses this — we still guard defensively
    // because a hand-built manifest can bypass fromJson.
    m.tensors.push_back({
        .name        = "orphan",
        .type        = GgmlType::F32,
        .dims        = {16},
        .bytes       = 64,
        .chunkIndex  = 2,
        .chunkOffset = 0,
    });

    bool threw = false;
    try {
        (void)WeightsMap::fromAttachedChunked(
            m, std::span<void* const>{chunkBases});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(weightsMap_fromAttachedChunked_nullChunkBase_throws) {
    using ::mimirmind::core::ipc::TensorManifest;

    std::vector<void*> chunkBases{nullptr};

    TensorManifest m{};
    m.modelId          = "id";
    m.modelFingerprint = "fp";
    m.chunks.push_back({.chunkIndex = 0, .bytes = 1024});
    m.tensors.push_back({
        .name        = "t",
        .type        = GgmlType::F32,
        .dims        = {8},
        .bytes       = 32,
        .chunkIndex  = 0,
        .chunkOffset = 0,
    });

    bool threw = false;
    try {
        (void)WeightsMap::fromAttachedChunked(
            m, std::span<void* const>{chunkBases});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(weightsMap_fromAttachedChunked_emptyManifest_yieldsEmptyMap) {
    using ::mimirmind::core::ipc::TensorManifest;

    std::vector<void*> chunkBases{};
    TensorManifest m{};
    m.modelId          = "id";
    m.modelFingerprint = "fp";

    auto wm = WeightsMap::fromAttachedChunked(
        m, std::span<void* const>{chunkBases});
    EXPECT_EQ(wm.size(), 0U);
    // isAttached() looks at owned vector — for an empty manifest it
    // reports false, matching the standalone-mode convention that
    // "isAttached" implies "we hold at least one tensor". Not a
    // load-bearing behavior; just documenting current semantics.
    EXPECT_TRUE(!wm.isAttached());
}

// ---- GovernorLock ----------------------------------------------------------

TEST(governorLock_acquire_thenRelease) {
    const std::string p = tmpFilePath("acquire");
    (void)::unlink(p.c_str());  // clean from any previous test crash

    auto lk = GovernorLock::tryAcquire(p);
    EXPECT_TRUE(static_cast<bool>(lk));
    EXPECT_TRUE(lk->held());
    EXPECT_EQ(std::string{lk->path()}, p);

    // Second attempt from same process must fail — flock is per-open-fd,
    // and the first lock is held by our own `lk` object.
    auto lk2 = GovernorLock::tryAcquire(p);
    EXPECT_TRUE(!lk2);

    // Release the first, then a fresh acquire should succeed.
    lk->release();
    EXPECT_TRUE(!lk->held());

    auto lk3 = GovernorLock::tryAcquire(p);
    EXPECT_TRUE(static_cast<bool>(lk3));
    lk3->release();
    (void)::unlink(p.c_str());
}

TEST(governorLock_moveTransfersOwnership) {
    const std::string p = tmpFilePath("move");
    (void)::unlink(p.c_str());

    auto lk = GovernorLock::tryAcquire(p);
    EXPECT_TRUE(static_cast<bool>(lk));

    GovernorLock moved{std::move(*lk)};
    EXPECT_TRUE(moved.held());
    // After move-from, source is empty; concurrent acquire on the same
    // path should still fail because `moved` holds the flock.
    auto lk2 = GovernorLock::tryAcquire(p);
    EXPECT_TRUE(!lk2);

    moved.release();
    (void)::unlink(p.c_str());
}

// ---- ChunkAllocator::layoutInsideChunk --------------------------------------
//
// Pure-CPU coverage of the bump-math extracted from ChunkAllocator. The full
// alloc cycle (zeMemAllocHost + destructor) needs a live L0 device and lives
// in the docker smoke test (see M-Munin.1a ADR Schritt 10).

TEST(chunkLayout_emptyChunk_fitsAtOffsetZero) {
    const auto p = ChunkAllocator::layoutInsideChunk(0, 128, 64, 1024);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{0});
}

TEST(chunkLayout_alignedCursor_placesRequestAtCursor) {
    // currentUsed already a 64-B multiple → offset equals currentUsed.
    const auto p = ChunkAllocator::layoutInsideChunk(256, 128, 64, 1024);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{256});
}

TEST(chunkLayout_unalignedCursor_padsUpToAlignment) {
    // 100 is not a 64-B multiple → round up to 128.
    const auto p = ChunkAllocator::layoutInsideChunk(100, 32, 64, 1024);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{128});
}

TEST(chunkLayout_fitsExactlyAtEnd_noNewChunk) {
    // chunkBytes=1024, currentUsed=896, align=64, bytes=128 → offset=896,
    // 896+128=1024 which is exactly the chunk boundary → still fits.
    const auto p = ChunkAllocator::layoutInsideChunk(896, 128, 64, 1024);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{896});
}

TEST(chunkLayout_wouldOverflow_requestsNewChunk) {
    // chunkBytes=1024, currentUsed=1000, align=64 → alignUp(1000,64)=1024,
    // 1024+32=1056 > 1024 → needsNewChunk.
    const auto p = ChunkAllocator::layoutInsideChunk(1000, 32, 64, 1024);
    EXPECT_TRUE(p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{0});
}

TEST(chunkLayout_alignmentPadPushesPastEnd_requestsNewChunk) {
    // The alignment pad alone kicks us past the chunk boundary, even
    // though the payload would technically fit in the raw tail bytes.
    // chunkBytes=200, currentUsed=140, align=64 → alignUp=192, 192+16=208
    // > 200 → new chunk.
    const auto p = ChunkAllocator::layoutInsideChunk(140, 16, 64, 200);
    EXPECT_TRUE(p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{0});
}

TEST(chunkLayout_alignZero_clampedToOne) {
    // align=0 is treated as align=1 — no padding is added, request lands
    // at the raw cursor. Keeps the caller from tripping on a zero-align
    // programming error while still producing a well-defined placement.
    const auto p = ChunkAllocator::layoutInsideChunk(37, 11, 0, 1024);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{37});
}

TEST(chunkLayout_largeChunkBytes_doesNotOverflowMath) {
    // 1 GiB chunk with GGUF-typical tensor size (~600 MiB) placed near
    // the start of a fresh chunk. Sanity check that the uint64 arithmetic
    // does not lose precision.
    constexpr std::uint64_t oneGiB = 1ULL << 30;
    constexpr std::size_t   bytes  = 605ULL * 1024ULL * 1024ULL; // ~605 MiB
    const auto p = ChunkAllocator::layoutInsideChunk(0, bytes, 64, oneGiB);
    EXPECT_TRUE(!p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{0});
}

TEST(chunkLayout_secondLargeTensor_triggersNewChunk) {
    // Two ~605 MiB tensors do not both fit in a 1 GiB chunk — the second
    // must land in a fresh chunk.
    constexpr std::uint64_t oneGiB = 1ULL << 30;
    constexpr std::size_t   bytes  = 605ULL * 1024ULL * 1024ULL;
    const auto p = ChunkAllocator::layoutInsideChunk(bytes, bytes, 64, oneGiB);
    EXPECT_TRUE(p.needsNewChunk);
    EXPECT_EQ(p.offset, std::uint64_t{0});
}

int main() {
    return mm::test::run();
}
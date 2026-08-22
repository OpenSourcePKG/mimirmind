#include "TestFramework.hpp"

#include "core/gguf/GgufTypes.hpp"
#include "core/ipc/IpcTransport.hpp"
#include "core/ipc/ShmIpcExporter.hpp"
#include "core/ipc/ShmIpcImporter.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/TensorFingerprint.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/ipc/ShmIpcExporter.hpp"
#include "core/ipc/ShmMuninClient.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/UnixSocketFrame.hpp"
#include "munin/ShmChunkAllocator.hpp"
#include "munin/ShmLoadedModel.hpp"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <expected>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using ::mimirmind::core::gguf::GgmlType;
using ::mimirmind::core::ipc::Frame;
using ::mimirmind::core::ipc::IpcHandle;
using ::mimirmind::core::ipc::ManifestEntry;
using ::mimirmind::core::ipc::ShmIpcExporter;
using ::mimirmind::core::ipc::ShmIpcImporter;
using ::mimirmind::core::ipc::TensorManifest;
using ::mimirmind::core::ipc::UnixSocketFrame;
using ::mimirmind::core::gguf::WeightsMap;
using ::mimirmind::core::ipc::ShmMuninClient;
using ::mimirmind::munin::ShmChunkAllocator;
using ::mimirmind::munin::ShmLoadedModel;

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

namespace {

// RAII socketpair. Cleans up whichever end wasn't consumed.
struct SocketPair {
    int a{-1};
    int b{-1};

    SocketPair() {
        int fds[2] = {-1, -1};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0) {
            std::abort();
        }
        a = fds[0];
        b = fds[1];
    }
    ~SocketPair() {
        if (a >= 0) ::close(a);
        if (b >= 0) ::close(b);
    }
    SocketPair(const SocketPair&)            = delete;
    SocketPair& operator=(const SocketPair&) = delete;
};

std::vector<std::byte> makeBytes(std::string_view s) {
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

std::string bytesToString(std::span<const std::byte> b) {
    return std::string{reinterpret_cast<const char*>(b.data()), b.size()};
}

} // namespace

// ---- UnixSocketFrame --------------------------------------------------------

TEST(unixSocketFrame_emptyPayload_noFds) {
    SocketPair sp;
    const auto s = UnixSocketFrame::send(sp.a, {});
    EXPECT_TRUE(static_cast<bool>(s));

    auto r = UnixSocketFrame::recv(sp.b);
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->payload.size(), 0U);
    EXPECT_EQ(r->fds.size(),     0U);
}

TEST(unixSocketFrame_shortPayload_noFds) {
    SocketPair sp;
    const std::string msg{"hello munin"};
    const auto bytes = makeBytes(msg);
    EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, bytes)));

    auto r = UnixSocketFrame::recv(sp.b);
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r->payload.size(), msg.size());
    EXPECT_EQ(bytesToString(r->payload), msg);
    EXPECT_EQ(r->fds.size(),     0U);
}

TEST(unixSocketFrame_largePayload_noFds) {
    SocketPair sp;
    // 512 KiB payload — exercises the partial-write / partial-read loop
    // in writeExactWithFds / readExact. Larger than SO_SNDBUF default
    // (~200 KiB on Linux) so send/recv must actually interleave — hence
    // the reader thread.
    std::vector<std::byte> big(512 * 1024);
    for (std::size_t i = 0; i < big.size(); ++i) {
        big[i] = static_cast<std::byte>(i & 0xff);
    }

    std::expected<Frame, std::string> recvResult{Frame{}};
    std::thread reader([&] {
        recvResult = UnixSocketFrame::recv(sp.b, /*maxPayloadBytes=*/2 * 1024 * 1024);
    });

    const auto s = UnixSocketFrame::send(sp.a, big);
    reader.join();

    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_TRUE(static_cast<bool>(recvResult));
    EXPECT_EQ(recvResult->payload.size(), big.size());
    EXPECT_TRUE(std::equal(recvResult->payload.begin(), recvResult->payload.end(),
                           big.begin()));
}

TEST(unixSocketFrame_oneFdViaScmRights) {
    // Move a real pipe fd across the socket, verify the receiver sees a
    // functional descriptor that speaks to the sender's write end.
    int pipefd[2] = {-1, -1};
    EXPECT_EQ(::pipe(pipefd), 0);
    const int senderReadEnd  = pipefd[0];
    const int senderWriteEnd = pipefd[1];

    SocketPair sp;
    const std::string msg{"here comes fd"};
    const auto bytes = makeBytes(msg);
    const int fdsOut[1] = {senderReadEnd};
    EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, bytes, fdsOut)));

    auto r = UnixSocketFrame::recv(sp.b);
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(bytesToString(r->payload), msg);
    EXPECT_EQ(r->fds.size(), 1U);

    // Prove the received fd is a live pipe: write on sender end, read
    // on received fd.
    const char* payload = "PING";
    EXPECT_EQ(::write(senderWriteEnd, payload, 4), 4);
    char buf[5]{};
    EXPECT_EQ(::read(r->fds[0], buf, 4), 4);
    EXPECT_EQ(std::string_view{buf}, std::string_view{"PING"});

    // Cleanup: the sender-side read fd is now dup'd on both sides.
    ::close(senderReadEnd);
    ::close(senderWriteEnd);
    ::close(r->fds[0]);
}

TEST(unixSocketFrame_tooManyFds_rejected) {
    SocketPair sp;
    std::vector<int> tooMany(UnixSocketFrame::kMaxFdsPerFrame + 1, -1);
    const auto s = UnixSocketFrame::send(sp.a, {}, tooMany);
    EXPECT_TRUE(!static_cast<bool>(s));
    // Reject reason should mention SCM_MAX_FD or similar.
    EXPECT_TRUE(s.error().find("SCM_MAX_FD") != std::string::npos);
}

TEST(unixSocketFrame_peerClosed_returnsError) {
    SocketPair sp;
    ::close(sp.a);
    sp.a = -1;
    auto r = UnixSocketFrame::recv(sp.b);
    EXPECT_TRUE(!static_cast<bool>(r));
    EXPECT_TRUE(r.error().find("peer closed") != std::string::npos);
}

TEST(unixSocketFrame_recvMaxSize_rejects) {
    SocketPair sp;
    std::vector<std::byte> payload(64);
    EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, payload)));
    // Refuse anything > 32 bytes — should trip on our 64-byte send.
    auto r = UnixSocketFrame::recv(sp.b, /*maxPayloadBytes=*/32);
    EXPECT_TRUE(!static_cast<bool>(r));
    EXPECT_TRUE(r.error().find("frame length") != std::string::npos);
}

// ---- TensorManifest ---------------------------------------------------------

TEST(tensorManifest_roundTripEmpty) {
    TensorManifest m{};
    m.modelId          = "test-model";
    m.modelFingerprint = "sha256:cafe";

    const std::string j = m.toJson();
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->protocolVersion,  TensorManifest::kCurrentProtocolVersion);
    EXPECT_EQ(parsed->modelId,          "test-model");
    EXPECT_EQ(parsed->modelFingerprint, "sha256:cafe");
    EXPECT_EQ(parsed->chunks.size(),    0U);
    EXPECT_EQ(parsed->tensors.size(),   0U);
}

TEST(tensorManifest_roundTripPopulated) {
    TensorManifest m{};
    m.modelId          = "google_gemma-4-E4B-it-Q4_K_M";
    m.modelFingerprint = "header-sum:deadbeef";

    // Two chunks: one full 1 GiB packing several early tensors, one
    // partial holding the vocab tail. Wire format carries used-bytes,
    // not raw chunk capacity.
    m.chunks.push_back({.chunkIndex = 0, .bytes = 1073741824ULL});
    m.chunks.push_back({.chunkIndex = 1, .bytes = 560760832ULL});

    m.tensors.push_back({
        .name        = "blk.0.attn_q.weight",
        .type        = GgmlType::Q4_K,
        .dims        = {2560, 4096},
        .bytes       = 5898240,
        .chunkIndex  = 0,
        .chunkOffset = 0,
    });
    m.tensors.push_back({
        .name        = "blk.0.attn_k.weight",
        .type        = GgmlType::Q5_K,
        .dims        = {2560, 1024},
        .bytes       = 1802240,
        .chunkIndex  = 0,
        .chunkOffset = 5898240,
    });
    m.tensors.push_back({
        .name        = "token_embd.weight",
        .type        = GgmlType::Q6_K,
        .dims        = {2560, 262144},
        .bytes       = 550502400,
        .chunkIndex  = 1,
        .chunkOffset = 0,
    });

    const std::string j = m.toJson();
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->chunks.size(),  2U);
    EXPECT_EQ(parsed->tensors.size(), 3U);

    for (std::size_t i = 0; i < parsed->chunks.size(); ++i) {
        EXPECT_EQ(parsed->chunks[i].chunkIndex, m.chunks[i].chunkIndex);
        EXPECT_EQ(parsed->chunks[i].bytes,      m.chunks[i].bytes);
    }
    for (std::size_t i = 0; i < parsed->tensors.size(); ++i) {
        const auto& a = m.tensors[i];
        const auto& b = parsed->tensors[i];
        EXPECT_EQ(a.name,        b.name);
        EXPECT_EQ(a.type,        b.type);
        EXPECT_EQ(a.dims.size(), b.dims.size());
        for (std::size_t k = 0; k < a.dims.size(); ++k) {
            EXPECT_EQ(a.dims[k], b.dims[k]);
        }
        EXPECT_EQ(a.bytes,       b.bytes);
        EXPECT_EQ(a.chunkIndex,  b.chunkIndex);
        EXPECT_EQ(a.chunkOffset, b.chunkOffset);
    }
}

TEST(tensorManifest_tensorReferencesUnknownChunk_rejects) {
    // Tensor points at chunk_index=2 but the manifest only declares
    // one chunk. Consistency check should refuse at parse time so the
    // worker never dereferences an out-of-range chunkBases[] entry.
    std::string j = R"({"protocol_version":2,"model_id":"x","model_fingerprint":"y",)"
                    R"("chunks":[{"chunk_index":0,"bytes":1024}],)"
                    R"("tensors":[{"name":"t","type_id":0,"dims":[1],"bytes":4,)"
                    R"("chunk_index":2,"chunk_offset":0}]})";
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("chunk_index=2") != std::string::npos);
}

TEST(tensorManifest_versionMismatch_rejects) {
    // Craft a JSON with a bumped protocol_version, verify parser refuses.
    // v1 is the retired legacy — receiver must refuse even though the
    // rest of the payload looks superficially valid.
    std::string j = R"({"protocol_version":1,"model_id":"x","model_fingerprint":"y","chunks":[],"tensors":[]})";
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("protocol_version") != std::string::npos);
}

TEST(tensorManifest_malformed_rejects) {
    auto parsed = TensorManifest::fromJson(std::string_view{"not json at all }{"});
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("parse error") != std::string::npos);
}

TEST(tensorManifest_missingRequiredField_rejects) {
    // Missing model_id.
    std::string j = R"({"protocol_version":2,"model_fingerprint":"y","chunks":[],"tensors":[]})";
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("model_id") != std::string::npos);
}

TEST(tensorManifest_missingChunksArray_rejects) {
    // v2 requires the chunks field even when empty. Older senders that
    // omit it entirely should be refused rather than silently attaching
    // with zero chunks and mysterious runtime failures downstream.
    std::string j = R"({"protocol_version":2,"model_id":"x","model_fingerprint":"y","tensors":[]})";
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("chunks") != std::string::npos);
}

TEST(tensorManifest_wireFormatIsCompactJson) {
    // Regression: keep the wire format tight so a 720-tensor manifest
    // stays under 100 KB. Empty manifest with two short strings should
    // fit in well under 200 bytes.
    TensorManifest m{};
    m.modelId          = "id";
    m.modelFingerprint = "fp";
    const std::string j = m.toJson();
    EXPECT_TRUE(j.size() < 200U);
    // No pretty-printing.
    EXPECT_TRUE(j.find('\n') == std::string::npos);
}

// ---- Framing + manifest end-to-end ------------------------------------------

TEST(ipc_manifestOverSocketpair_roundTrip) {
    SocketPair sp;

    // Sender: build manifest, ship it as a JSON frame.
    TensorManifest sent{};
    sent.modelId          = "e4b-q4k";
    sent.modelFingerprint = "hs:1234";
    sent.chunks.push_back({.chunkIndex = 0, .bytes = 10240});
    sent.tensors.push_back({
        .name        = "output_norm.weight",
        .type        = GgmlType::F32,
        .dims        = {2560},
        .bytes       = 10240,
        .chunkIndex  = 0,
        .chunkOffset = 0,
    });
    const std::string j = sent.toJson();
    const auto payload = makeBytes(j);
    EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, payload)));

    // Receiver: read frame, parse manifest.
    auto rframe = UnixSocketFrame::recv(sp.b);
    EXPECT_TRUE(static_cast<bool>(rframe));

    auto rparsed = TensorManifest::fromJson(bytesToString(rframe->payload));
    EXPECT_TRUE(static_cast<bool>(rparsed));
    EXPECT_EQ(rparsed->modelId,           sent.modelId);
    EXPECT_EQ(rparsed->modelFingerprint,  sent.modelFingerprint);
    EXPECT_EQ(rparsed->tensors.size(),    1U);
    EXPECT_EQ(rparsed->tensors[0].name,   "output_norm.weight");
    EXPECT_EQ(rparsed->tensors[0].type,   GgmlType::F32);
}

// ---- ShmIpcExporter / ShmIpcImporter ----------------------------------------
//
// The chosen M-Munin.CUDA transport (ADR 2026-08-14, step 1b). These tests
// exercise the pure-POSIX seam end-to-end: a memfd stands in for a Munin
// model chunk, the exporter turns it into an IpcHandle, the handle crosses a
// socketpair via the REAL SCM_RIGHTS wire (UnixSocketFrame), and the importer
// mmaps the received memfd — exactly the AttachSession -> MuninClient flow for
// the shm backend, minus the CUDA kernel deref (proven separately in
// tools/cuda-ipc-testrig --kind shm).

namespace {

constexpr std::uint32_t kShmOwnerPat    = 0xA5A5A5A5u;
constexpr std::uint32_t kShmAttacherPat = 0x5A5A5A5Au;

// Create a memfd of `len` bytes (syscall wrapper — no glibc memfd_create
// dependency). Returns -1 on failure.
int makeMemfd(std::size_t len) {
    const long fd = ::syscall(SYS_memfd_create, "munin-test-shm", MFD_CLOEXEC);
    if (fd < 0) return -1;
    if (::ftruncate(static_cast<int>(fd), static_cast<off_t>(len)) != 0) {
        ::close(static_cast<int>(fd));
        return -1;
    }
    return static_cast<int>(fd);
}

std::uint64_t decodeMapLen(const std::array<std::byte, 64>& bytes) {
    std::uint64_t v = 0;
    std::memcpy(&v, bytes.data(), sizeof(v));
    return v;
}

} // namespace

TEST(shmIpc_exportEncodesLengthAndBorrowsFd) {
    constexpr std::size_t kLen = 3 * 4096;  // 3 pages
    const int memfd = makeMemfd(kLen);
    EXPECT_TRUE(memfd >= 0);

    const int memfds[1]              = {memfd};
    const std::uint64_t mapBytes[1]  = {kLen};
    ShmIpcExporter exp{std::span<const int>{memfds, 1},
                       std::span<const std::uint64_t>{mapBytes, 1}};

    auto h = exp.exportChunk(0, /*base=*/nullptr, /*usedBytes=*/kLen);
    EXPECT_TRUE(static_cast<bool>(h));
    EXPECT_EQ(h->fd, memfd);                 // borrowed, not dup'd
    EXPECT_EQ(decodeMapLen(h->bytes), static_cast<std::uint64_t>(kLen));

    // Out-of-range index is a clean error, not a crash.
    auto bad = exp.exportChunk(1, nullptr, kLen);
    EXPECT_TRUE(!static_cast<bool>(bad));
    EXPECT_TRUE(bad.error().find("out of range") != std::string::npos);

    ::close(memfd);
}

TEST(shmIpc_roundTripOverScmRights_bothDirectionsCoherent) {
    constexpr std::size_t kLen   = 4 * 4096;      // 16 KiB
    constexpr std::size_t kWords = kLen / 4;
    const int memfd = makeMemfd(kLen);
    EXPECT_TRUE(memfd >= 0);

    // Server side: map the memfd and fill it with the owner pattern, as
    // Munin's shm allocator would populate a resident chunk.
    void* ownerMap = ::mmap(nullptr, kLen, PROT_READ | PROT_WRITE,
                            MAP_SHARED, memfd, 0);
    EXPECT_TRUE(ownerMap != MAP_FAILED);
    auto* ownerWords = static_cast<std::uint32_t*>(ownerMap);
    for (std::size_t i = 0; i < kWords; ++i) ownerWords[i] = kShmOwnerPat;

    // Export -> ship the handle over the real SCM_RIGHTS wire.
    const int memfds[1]             = {memfd};
    const std::uint64_t mapBytes[1] = {kLen};
    ShmIpcExporter exp{std::span<const int>{memfds, 1},
                       std::span<const std::uint64_t>{mapBytes, 1}};
    auto h = exp.exportChunk(0, nullptr, kLen);
    EXPECT_TRUE(static_cast<bool>(h));

    SocketPair sp;
    const auto payload =
        std::span<const std::byte>{h->bytes.data(), h->bytes.size()};
    const int fdsOut[1] = {h->fd};
    EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, payload, fdsOut)));

    auto rframe = UnixSocketFrame::recv(sp.b, /*maxPayloadBytes=*/128);
    EXPECT_TRUE(static_cast<bool>(rframe));
    EXPECT_EQ(rframe->payload.size(), 64U);
    EXPECT_EQ(rframe->fds.size(),     1U);

    // Worker side: import the received (dup'd) memfd.
    ShmIpcImporter imp;
    std::span<const std::byte, 64> handleBytes{rframe->payload.data(), 64};
    auto p = imp.importChunk(handleBytes, rframe->fds[0]);
    EXPECT_TRUE(static_cast<bool>(p));
    auto* workerWords = static_cast<std::uint32_t*>(*p);

    // Owner data is visible to the worker's mapping...
    bool ownerSeen = true;
    for (std::size_t i = 0; i < kWords; ++i)
        if (workerWords[i] != kShmOwnerPat) { ownerSeen = false; break; }
    EXPECT_TRUE(ownerSeen);

    // ...and the worker's writes are coherent back on the owner mapping
    // (this is what the CUDA kernel does through the same pointer).
    for (std::size_t i = 0; i < 1024; ++i) workerWords[i] = kShmAttacherPat;
    bool writesVisible = true;
    for (std::size_t i = 0; i < 1024; ++i)
        if (ownerWords[i] != kShmAttacherPat) { writesVisible = false; break; }
    EXPECT_TRUE(writesVisible);

    // closeChunk releases the worker mapping; a second close is a no-op.
    imp.closeChunk(*p);
    imp.closeChunk(*p);

    ::munmap(ownerMap, kLen);
    ::close(memfd);
}

TEST(shmIpc_importRejectsBadFdAndZeroLength) {
    ShmIpcImporter imp;

    std::array<std::byte, 64> okBytes{};
    const std::uint64_t len = 4096;
    std::memcpy(okBytes.data(), &len, sizeof(len));
    auto negFd = imp.importChunk(std::span<const std::byte, 64>{okBytes.data(), 64},
                                 /*receivedFd=*/-1);
    EXPECT_TRUE(!static_cast<bool>(negFd));
    EXPECT_TRUE(negFd.error().find("negative") != std::string::npos);

    // Zero-length handle is refused even with a valid fd.
    const int memfd = makeMemfd(4096);
    EXPECT_TRUE(memfd >= 0);
    std::array<std::byte, 64> zeroBytes{};  // mapLen == 0
    auto zeroLen = imp.importChunk(std::span<const std::byte, 64>{zeroBytes.data(), 64},
                                   memfd);
    EXPECT_TRUE(!static_cast<bool>(zeroLen));
    EXPECT_TRUE(zeroLen.error().find("zero-length") != std::string::npos);
    ::close(memfd);
}

// ---- ShmChunkAllocator ------------------------------------------------------
//
// The server-side memfd backing store for the shm transport (step 2). The
// bump/layout math mirrors the L0 ChunkAllocator (already covered in
// munin_tests); here we lock the mirror and, more importantly, prove the
// full server->worker data path: pack tensors into memfd chunks, export each
// chunk via ShmIpcExporter over the real SCM_RIGHTS wire, import on the
// worker side, and resolve every tensor's pointer via
// chunkBase[chunkIndex] + chunkOffset.

TEST(shmChunkAlloc_layoutMirrorsChunkAllocator) {
    // Fits inside the current chunk.
    auto a = ShmChunkAllocator::layoutInsideChunk(/*used=*/100, /*bytes=*/64,
                                                  /*align=*/64, /*chunkBytes=*/8192);
    EXPECT_TRUE(!a.needsNewChunk);
    EXPECT_EQ(a.offset, 128U);  // alignUp(100,64)=128

    // Does not fit -> fresh chunk.
    auto b = ShmChunkAllocator::layoutInsideChunk(/*used=*/8100, /*bytes=*/128,
                                                  /*align=*/64, /*chunkBytes=*/8192);
    EXPECT_TRUE(b.needsNewChunk);
    EXPECT_EQ(b.offset, 0U);
}

TEST(shmChunkAlloc_rejectsOversizedAndZero) {
    ShmChunkAllocator alloc{8192};
    bool threwOversize = false;
    try {
        (void)alloc.allocate(9000);  // > chunkBytes
    } catch (const std::exception&) {
        threwOversize = true;
    }
    EXPECT_TRUE(threwOversize);

    bool threwZero = false;
    try {
        (void)alloc.allocate(0);
    } catch (const std::exception&) {
        threwZero = true;
    }
    EXPECT_TRUE(threwZero);
}

TEST(shmChunkAlloc_endToEnd_attachResolvesTensors) {
    // 8 KiB chunks force the third tensor into a second chunk, exercising
    // multi-chunk export/import and per-tensor offset resolution.
    ShmChunkAllocator alloc{8192};

    struct Tensor {
        std::size_t                 bytes;
        std::uint8_t                pattern;
        ShmChunkAllocator::Allocation alloc{};
    };
    Tensor tensors[3] = {
        {3000, 0x11, {}},
        {3000, 0x22, {}},
        {3000, 0x33, {}},
    };

    for (auto& t : tensors) {
        t.alloc = alloc.allocate(t.bytes);           // default 64 B align
        std::memset(t.alloc.ptr, t.pattern, t.bytes); // "load" the tensor
    }

    // Expect 2 chunks: t0@chunk0 off0, t1@chunk0 off3008, t2@chunk1 off0.
    EXPECT_EQ(alloc.chunkCount(), 2U);
    EXPECT_EQ(tensors[0].alloc.chunkIndex, 0U);
    EXPECT_EQ(tensors[0].alloc.chunkOffset, 0U);
    EXPECT_EQ(tensors[1].alloc.chunkIndex, 0U);
    EXPECT_EQ(tensors[1].alloc.chunkOffset, 3008U);   // alignUp(3000,64)
    EXPECT_EQ(tensors[2].alloc.chunkIndex, 1U);
    EXPECT_EQ(tensors[2].alloc.chunkOffset, 0U);
    EXPECT_EQ(alloc.chunkUsedBytes(0), 6008U);
    EXPECT_EQ(alloc.chunkUsedBytes(1), 3000U);
    EXPECT_EQ(alloc.bytesUsed(), 9000U);

    // Server: export each chunk; ship over the SCM_RIGHTS wire. Worker:
    // import and record the per-chunk base pointer in this address space.
    const std::vector<int>           memfds   = alloc.chunkMemfds();
    const std::vector<std::uint64_t> mapBytes = alloc.chunkMapBytes();
    ShmIpcExporter exp{std::span<const int>{memfds},
                       std::span<const std::uint64_t>{mapBytes}};
    ShmIpcImporter imp;

    std::vector<void*> workerBases(alloc.chunkCount(), nullptr);
    for (std::uint32_t i = 0; i < alloc.chunkCount(); ++i) {
        auto h = exp.exportChunk(i, alloc.chunkBase(i), alloc.chunkUsedBytes(i));
        EXPECT_TRUE(static_cast<bool>(h));

        SocketPair sp;
        const auto payload =
            std::span<const std::byte>{h->bytes.data(), h->bytes.size()};
        const int fdsOut[1] = {h->fd};
        EXPECT_TRUE(static_cast<bool>(UnixSocketFrame::send(sp.a, payload, fdsOut)));

        auto rframe = UnixSocketFrame::recv(sp.b, /*maxPayloadBytes=*/128);
        EXPECT_TRUE(static_cast<bool>(rframe));
        EXPECT_EQ(rframe->fds.size(), 1U);

        std::span<const std::byte, 64> hb{rframe->payload.data(), 64};
        auto p = imp.importChunk(hb, rframe->fds[0]);
        EXPECT_TRUE(static_cast<bool>(p));
        workerBases[i] = *p;
    }

    // Worker resolves each tensor via chunkBase[chunkIndex] + chunkOffset
    // and sees exactly the bytes the server wrote.
    for (const auto& t : tensors) {
        const auto* base = static_cast<const std::uint8_t*>(workerBases[t.alloc.chunkIndex]);
        const std::uint8_t* tp = base + t.alloc.chunkOffset;
        bool ok = true;
        for (std::size_t i = 0; i < t.bytes; ++i)
            if (tp[i] != t.pattern) { ok = false; break; }
        EXPECT_TRUE(ok);
    }

    for (void* b : workerBases) imp.closeChunk(b);
}

// ---- GgufReader::loadTensorsIntoShmChunks + ShmLoadedModel ------------------
//
// The shm ModelStore load path (step 2-tail): parse a GGUF, copy each
// tensor's raw payload into memfd chunks, build the wire manifest. We
// synthesize a minimal GGUF v3 in a temp file (no fixture on disk) so the
// whole load path is exercised end-to-end, then check chunk placement, the
// manifest, and that the raw bytes actually landed in the shm chunks.

namespace {

// Minimal GGUF v3 builder. Three 1-D F32 tensors, distinct byte patterns,
// packed contiguously in the data section. Returns the file bytes.
std::vector<std::uint8_t> makeMinimalGguf() {
    std::vector<std::uint8_t> b;
    auto putU32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    };
    auto putU64 = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    };
    auto putStr = [&](std::string_view s) {
        putU64(s.size());
        for (char ch : s) b.push_back(static_cast<std::uint8_t>(ch));
    };

    constexpr std::uint32_t kMagic     = 0x46554747u; // 'GGUF'
    constexpr std::uint32_t kAlignment = 32;
    constexpr std::uint64_t kElems     = 2000;        // F32 -> 8000 bytes/tensor
    constexpr std::uint64_t kTBytes    = kElems * 4;

    putU32(kMagic);
    putU32(3);          // version
    putU64(3);          // tensor count
    putU64(1);          // metadata count

    // metadata[0]: general.alignment : UInt32 = 32
    putStr("general.alignment");
    putU32(4);          // GgufValueType::UInt32
    putU32(kAlignment);

    // tensor index: name, ndim=1, dim, type=F32(0), fileOffset
    const char* names[3] = {"a", "bb", "ccc"};
    for (int i = 0; i < 3; ++i) {
        putStr(names[i]);
        putU32(1);                                   // ndim
        putU64(kElems);                              // dim[0]
        putU32(0);                                   // GgmlType::F32
        putU64(static_cast<std::uint64_t>(i) * kTBytes); // fileOffset in data section
    }

    // pad to alignment, then the data section (3 patterns).
    while ((b.size() % kAlignment) != 0) b.push_back(0);
    const std::uint8_t pat[3] = {0x11, 0x22, 0x33};
    for (int i = 0; i < 3; ++i) {
        for (std::uint64_t k = 0; k < kTBytes; ++k) b.push_back(pat[i]);
    }
    return b;
}

std::string writeTempGguf(const std::vector<std::uint8_t>& bytes) {
    std::string path = "/tmp/munin_shm_test_" +
                       std::to_string(::getpid()) + ".gguf";
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    os.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
    os.close();
    return path;
}

} // namespace

TEST(shmModel_loadGgufIntoShmChunks_buildsManifestAndData) {
    const std::string path = writeTempGguf(makeMinimalGguf());

    ShmLoadedModel lm{};
    lm.id     = "test-shm-model";
    lm.chunks = std::make_unique<ShmChunkAllocator>(/*chunkBytes=*/8192);
    lm.reader = std::make_unique<::mimirmind::core::gguf::GgufReader>();
    lm.reader->open(path);
    EXPECT_EQ(lm.reader->tensorCount(), 3U);

    lm.reader->loadTensorsIntoShmChunks(*lm.chunks);
    lm.totalBytes  = lm.reader->totalTensorBytes();
    lm.fingerprint = ::mimirmind::core::gguf::tensorFingerprint(*lm.reader);

    // 8000-byte tensors do not share an 8192-byte chunk -> one chunk each.
    EXPECT_EQ(lm.chunks->chunkCount(), 3U);
    EXPECT_EQ(lm.chunks->chunkUsedBytes(0), 8000U);
    EXPECT_EQ(lm.chunks->chunkUsedBytes(1), 8000U);
    EXPECT_EQ(lm.chunks->chunkUsedBytes(2), 8000U);
    EXPECT_EQ(lm.totalBytes, 24000U);

    // Manifest: 3 chunks + 3 tensors, each tensor alone at offset 0 in its
    // own chunk.
    const auto m = lm.buildManifest();
    EXPECT_EQ(m.modelId,        "test-shm-model");
    EXPECT_EQ(m.chunks.size(),  3U);
    EXPECT_EQ(m.tensors.size(), 3U);
    for (std::uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(m.chunks[i].bytes,        8000U);
        EXPECT_EQ(m.tensors[i].chunkIndex,  i);
        EXPECT_EQ(m.tensors[i].chunkOffset, 0U);
        EXPECT_EQ(m.tensors[i].bytes,       8000U);
    }

    // The raw GGUF payload actually landed in the shm chunks: each chunk
    // holds its tensor's distinct pattern.
    const std::uint8_t expect[3] = {0x11, 0x22, 0x33};
    for (std::uint32_t i = 0; i < 3; ++i) {
        const auto* base = static_cast<const std::uint8_t*>(lm.chunks->chunkBase(i));
        bool ok = base != nullptr;
        for (std::uint64_t k = 0; ok && k < 8000; ++k)
            if (base[k] != expect[i]) { ok = false; }
        EXPECT_TRUE(ok);
    }

    ::unlink(path.c_str());
}

// ---- Full shm attach: server <-> ShmMuninClient -> WeightsMap ---------------
//
// The whole M-Munin.CUDA data path in one process, over a real socket: an
// in-test "Munin" holds a model in memfd chunks (ShmLoadedModel) and speaks
// the attach wire (manifest + per-chunk HANDLE frames via ShmIpcExporter);
// ShmMuninClient attaches, imports each chunk (mmap), and the result feeds
// the backend-neutral WeightsMap::fromAttachedChunked. We then verify every
// tensor resolves — through the worker's own mappings — to the exact bytes
// the server loaded. Only the final CUDA-kernel deref of these pointers is
// left for on-box (proven separately by tools/cuda-ipc-testrig --kind shm).

TEST(shmAttach_endToEnd_clientResolvesWeightsMap) {
    const std::string path = writeTempGguf(makeMinimalGguf());

    // Server-held model in memfd chunks.
    ShmLoadedModel lm{};
    lm.id     = "test-shm-model";
    lm.chunks = std::make_unique<ShmChunkAllocator>(/*chunkBytes=*/8192);
    lm.reader = std::make_unique<::mimirmind::core::gguf::GgufReader>();
    lm.reader->open(path);
    lm.reader->loadTensorsIntoShmChunks(*lm.chunks);
    lm.fingerprint = ::mimirmind::core::gguf::tensorFingerprint(*lm.reader);
    ::unlink(path.c_str());

    SocketPair sp;

    // In-test Munin server on sp.a: read attach request, send manifest, then
    // one HANDLE frame per chunk (payload + SCM_RIGHTS memfd).
    std::thread server([&] {
        auto req = UnixSocketFrame::recv(sp.a, /*maxPayloadBytes=*/64 * 1024);
        if (!req) return;

        const std::string manifestJson = lm.buildManifest().toJson();
        if (!UnixSocketFrame::send(
                sp.a, std::span<const std::byte>{
                          reinterpret_cast<const std::byte*>(manifestJson.data()),
                          manifestJson.size()})) {
            return;
        }

        const std::vector<int>           memfds   = lm.chunks->chunkMemfds();
        const std::vector<std::uint64_t> mapBytes = lm.chunks->chunkMapBytes();
        ::mimirmind::core::ipc::ShmIpcExporter exp{
            std::span<const int>{memfds}, std::span<const std::uint64_t>{mapBytes}};
        for (std::uint32_t i = 0; i < lm.chunks->chunkCount(); ++i) {
            auto h = exp.exportChunk(i, lm.chunks->chunkBase(i),
                                     lm.chunks->chunkUsedBytes(i));
            if (!h) return;
            const int fds[1] = {h->fd};
            (void)UnixSocketFrame::send(
                sp.a,
                std::span<const std::byte>{h->bytes.data(), h->bytes.size()},
                std::span<const int>{fds, 1});
        }
    });

    // Worker on sp.b: attach + import.
    ShmMuninClient client;
    auto res = client.attachOnConnectedFd(sp.b, "test-shm-model");
    server.join();

    EXPECT_TRUE(static_cast<bool>(res));
    if (res) {
        sp.b = -1;  // client owns the session fd now; don't double-close.

        EXPECT_EQ(res->manifest.modelId,   "test-shm-model");
        EXPECT_EQ(res->manifest.tensors.size(), 3U);
        EXPECT_EQ(res->chunkBases.size(),  3U);

        // Backend-neutral resolver: manifest + imported bases -> tensors.
        WeightsMap wm = WeightsMap::fromAttachedChunked(
            res->manifest, std::span<void* const>{res->chunkBases});
        EXPECT_EQ(wm.size(), 3U);

        // Each tensor's pointer (through the WORKER's mapping) holds the
        // exact bytes the server loaded from the GGUF.
        const char*        names[3]  = {"a", "bb", "ccc"};
        const std::uint8_t expect[3] = {0x11, 0x22, 0x33};
        for (int i = 0; i < 3; ++i) {
            const auto* t = wm.find(names[i]);
            EXPECT_TRUE(t != nullptr);
            if (t != nullptr) {
                const auto* p = static_cast<const std::uint8_t*>(t->usmPtr);
                bool ok = p != nullptr;
                for (std::uint64_t k = 0; ok && k < 8000; ++k)
                    if (p[k] != expect[i]) { ok = false; }
                EXPECT_TRUE(ok);
            }
        }
    }
}

int main() {
    return mm::test::run();
}
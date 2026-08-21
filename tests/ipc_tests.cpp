#include "TestFramework.hpp"

#include "core/gguf/GgufTypes.hpp"
#include "core/ipc/IpcTransport.hpp"
#include "core/ipc/ShmIpcExporter.hpp"
#include "core/ipc/ShmIpcImporter.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/UnixSocketFrame.hpp"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
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

int main() {
    return mm::test::run();
}
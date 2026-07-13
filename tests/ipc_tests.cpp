#include "TestFramework.hpp"

#include "core/gguf/GgufTypes.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/ipc/UnixSocketFrame.hpp"

#include <sys/socket.h>
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
using ::mimirmind::core::ipc::ManifestEntry;
using ::mimirmind::core::ipc::TensorManifest;
using ::mimirmind::core::ipc::UnixSocketFrame;

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
    EXPECT_EQ(parsed->tensors.size(),   0U);
}

TEST(tensorManifest_roundTripPopulated) {
    TensorManifest m{};
    m.modelId          = "google_gemma-4-E4B-it-Q4_K_M";
    m.modelFingerprint = "header-sum:deadbeef";

    m.tensors.push_back({
        .name        = "blk.0.attn_q.weight",
        .type        = GgmlType::Q4_K,
        .dims        = {2560, 4096},
        .bytes       = 5898240,
        .handleIndex = 0,
    });
    m.tensors.push_back({
        .name        = "blk.0.attn_k.weight",
        .type        = GgmlType::Q5_K,
        .dims        = {2560, 1024},
        .bytes       = 1802240,
        .handleIndex = 1,
    });
    m.tensors.push_back({
        .name        = "token_embd.weight",
        .type        = GgmlType::Q6_K,
        .dims        = {2560, 262144},
        .bytes       = 550502400,
        .handleIndex = 2,
    });

    const std::string j = m.toJson();
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(static_cast<bool>(parsed));
    EXPECT_EQ(parsed->tensors.size(), 3U);

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
        EXPECT_EQ(a.handleIndex, b.handleIndex);
    }
}

TEST(tensorManifest_versionMismatch_rejects) {
    // Craft a JSON with a bumped protocol_version, verify parser refuses.
    std::string j = R"({"protocol_version":999,"model_id":"x","model_fingerprint":"y","tensors":[]})";
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
    std::string j = R"({"protocol_version":1,"model_fingerprint":"y","tensors":[]})";
    auto parsed = TensorManifest::fromJson(j);
    EXPECT_TRUE(!static_cast<bool>(parsed));
    EXPECT_TRUE(parsed.error().find("model_id") != std::string::npos);
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
    sent.tensors.push_back({
        .name = "output_norm.weight", .type = GgmlType::F32,
        .dims = {2560}, .bytes = 10240, .handleIndex = 0,
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

int main() {
    return mm::test::run();
}
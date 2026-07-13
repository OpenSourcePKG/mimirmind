#include "core/ipc/MuninClient.hpp"

#include "core/ipc/IpcImporter.hpp"
#include "core/ipc/UnixSocketFrame.hpp"
#include "core/log/Log.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace mimirmind::core::ipc {

namespace {

std::string errnoTag(const char* where, int e) {
    std::ostringstream os;
    os << where << ": " << std::strerror(e) << " (errno=" << e << ")";
    return os.str();
}

std::span<const std::byte> asBytes(const std::string& s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

std::string_view asStringView(std::span<const std::byte> b) {
    return {reinterpret_cast<const char*>(b.data()), b.size()};
}

// Open + connect an AF_UNIX SOCK_STREAM socket to `path`. Returns fd on
// success, error string otherwise. On error the fd is closed for the
// caller — this helper owns the failure path.
std::expected<int, std::string> connectUnix(std::string_view path) noexcept {
    if (path.size() >= sizeof(::sockaddr_un{}.sun_path)) {
        return std::unexpected(std::string{
            "MuninClient: socket path exceeds sun_path capacity"});
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return std::unexpected(errnoTag("socket(AF_UNIX)", errno));
    }
    ::sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.data(), path.size());
    if (::connect(fd, reinterpret_cast<::sockaddr*>(&addr), sizeof(addr)) < 0) {
        const int e = errno;
        ::close(fd);
        std::ostringstream os;
        os << "MuninClient: connect(" << path << ") failed: "
           << std::strerror(e) << " (errno=" << e << ")";
        return std::unexpected(os.str());
    }
    return fd;
}

// Send a request envelope and read exactly one response frame. Response
// is expected to be a JSON payload with no fds. Used for healthz — the
// attach flow diverges after the first frame.
std::expected<std::string, std::string>
oneRequest(int fd, const std::string& requestJson) noexcept {
    if (auto s = UnixSocketFrame::send(fd, asBytes(requestJson)); !s) {
        return std::unexpected(s.error());
    }
    auto rsp = UnixSocketFrame::recv(fd);
    if (!rsp) {
        return std::unexpected(rsp.error());
    }
    if (!rsp->fds.empty()) {
        // Server sent fds on a healthz-style response — protocol misuse.
        for (int f : rsp->fds) {
            ::close(f);
        }
        return std::unexpected(std::string{
            "MuninClient: response carried unexpected SCM_RIGHTS fds"});
    }
    return std::string{asStringView(rsp->payload)};
}

} // namespace

MuninClient::MuninClient(::mimirmind::core::l0::L0Context& l0)
    : _l0{l0} {}

MuninClient::~MuninClient() {
    detach();
}

void MuninClient::detach() noexcept {
    if (_sessionFd >= 0) {
        ::close(_sessionFd);
        _sessionFd = -1;
    }
}

std::expected<HealthzResponse, std::string>
MuninClient::healthz(std::string_view socketPath) noexcept {
    auto fd = connectUnix(socketPath);
    if (!fd) {
        return std::unexpected(fd.error());
    }
    const int sock = *fd;

    RequestEnvelope req{};
    req.op = std::string{op::kHealthz};
    // Envelope has no toJson — build it inline to avoid growing the
    // shared header with a client-only helper. Two fields; hand-rolled
    // is smaller than pulling in nlohmann here.
    const std::string body = R"({"op":"healthz"})";

    auto rsp = oneRequest(sock, body);
    ::close(sock);
    if (!rsp) {
        return std::unexpected(rsp.error());
    }

    // Try healthz-response first; on failure, check for an error
    // envelope so we can bubble a clean error message up.
    auto parsed = HealthzResponse::fromJson(*rsp);
    if (parsed) {
        return *parsed;
    }
    if (auto errBody = parseErrorJson(*rsp); errBody) {
        return std::unexpected(std::string{"Munin healthz error: "} + *errBody);
    }
    return std::unexpected(parsed.error());
}

std::expected<MuninClient::AttachResult, std::string>
MuninClient::attach(std::string_view socketPath,
                    std::string_view modelId) noexcept {
    if (_sessionFd >= 0) {
        return std::unexpected(std::string{
            "MuninClient: attach called on already-attached client"});
    }
    if (modelId.empty()) {
        return std::unexpected(std::string{
            "MuninClient: attach requires a non-empty modelId"});
    }

    auto fd = connectUnix(socketPath);
    if (!fd) {
        return std::unexpected(fd.error());
    }
    const int sock = *fd;

    // Hand-rolled request JSON — tiny, avoids pulling nlohmann into
    // the hot code path here. Keys match RequestEnvelope::fromJson.
    std::string body;
    body.append(R"({"op":"attach","modelId":")");
    // modelId is operator-provided config; assume no JSON-hostile chars.
    // A future guard could escape backslash/quote — but if the operator
    // put those in a model id, other things break first.
    body.append(modelId).append(R"("})");

    if (auto s = UnixSocketFrame::send(sock, asBytes(body)); !s) {
        ::close(sock);
        return std::unexpected(s.error());
    }

    // First response frame = manifest JSON (or error envelope).
    auto first = UnixSocketFrame::recv(sock);
    if (!first) {
        ::close(sock);
        return std::unexpected(first.error());
    }
    if (!first->fds.empty()) {
        for (int f : first->fds) ::close(f);
        ::close(sock);
        return std::unexpected(std::string{
            "MuninClient: manifest frame unexpectedly carried fds"});
    }

    // Peek for error envelope before trying manifest parse.
    if (auto errBody = parseErrorJson(asStringView(first->payload)); errBody) {
        ::close(sock);
        return std::unexpected(std::string{"Munin attach error: "} + *errBody);
    }

    auto manifest = TensorManifest::fromJson(asStringView(first->payload));
    if (!manifest) {
        ::close(sock);
        return std::unexpected(manifest.error());
    }

    // Now N HANDLE frames, one per tensor. Order matches
    // manifest.tensors[i].handleIndex — server iterates in the same
    // order, so `i` and `handleIndex` coincide.
    AttachResult out{};
    out.manifest = std::move(*manifest);
    out.tensors.reserve(out.manifest.tensors.size());

    for (std::size_t i = 0; i < out.manifest.tensors.size(); ++i) {
        auto frame = UnixSocketFrame::recv(sock, /*maxPayloadBytes=*/128);
        if (!frame) {
            ::close(sock);
            return std::unexpected(frame.error());
        }
        if (frame->payload.size() != 64) {
            for (int f : frame->fds) ::close(f);
            ::close(sock);
            std::ostringstream os;
            os << "MuninClient: handle frame[" << i << "] has "
               << frame->payload.size() << " payload bytes, expected 64";
            return std::unexpected(os.str());
        }
        if (frame->fds.size() != 1) {
            for (int f : frame->fds) ::close(f);
            ::close(sock);
            std::ostringstream os;
            os << "MuninClient: handle frame[" << i << "] has "
               << frame->fds.size() << " SCM_RIGHTS fds, expected 1";
            return std::unexpected(os.str());
        }

        std::span<const std::byte, 64> handleBytes{
            frame->payload.data(), 64};
        auto ptr = IpcImporter::importOne(_l0, handleBytes, frame->fds[0]);
        if (!ptr) {
            // IpcImporter takes ownership on success only — on failure
            // the caller (us) must close the fd.
            ::close(frame->fds[0]);
            ::close(sock);
            std::ostringstream os;
            os << "MuninClient: import handle[" << i << "] ('"
               << out.manifest.tensors[i].name << "') failed: " << ptr.error();
            return std::unexpected(os.str());
        }

        ::mimirmind::core::gguf::GgufTensor t{};
        t.name       = out.manifest.tensors[i].name;
        t.type       = out.manifest.tensors[i].type;
        t.dimensions = out.manifest.tensors[i].dims;
        // Recompute nelements from dims for the downstream code that
        // reads it; Munin only ships bytes on the wire, but standalone
        // code paths rely on `nelements` being populated.
        std::uint64_t nel = 1;
        for (auto d : t.dimensions) nel *= d;
        t.nelements  = nel;
        t.nbytes     = static_cast<std::size_t>(out.manifest.tensors[i].bytes);
        t.fileOffset = 0;   // not meaningful in attached mode
        t.usmPtr     = *ptr;
        out.tensors.push_back(std::move(t));
    }

    _sessionFd = sock;
    MM_LOG_INFO("munin-client",
                "attached to model '{}' fingerprint='{}' tensors={} "
                "over socket '{}'",
                out.manifest.modelId, out.manifest.modelFingerprint,
                out.tensors.size(), socketPath);
    return out;
}

} // namespace mimirmind::core::ipc
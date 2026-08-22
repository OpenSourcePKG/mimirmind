// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/ShmMuninClient.hpp"

#include "core/ipc/UnixSocketFrame.hpp"
#include "core/log/Log.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <span>
#include <sstream>
#include <string>
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

std::expected<int, std::string> connectUnix(std::string_view path) noexcept {
    if (path.size() >= sizeof(::sockaddr_un{}.sun_path)) {
        return std::unexpected(std::string{
            "ShmMuninClient: socket path exceeds sun_path capacity"});
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
        os << "ShmMuninClient: connect(" << path << ") failed: "
           << std::strerror(e) << " (errno=" << e << ")";
        return std::unexpected(os.str());
    }
    return fd;
}

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
        for (int f : rsp->fds) ::close(f);
        return std::unexpected(std::string{
            "ShmMuninClient: response carried unexpected SCM_RIGHTS fds"});
    }
    return std::string{asStringView(rsp->payload)};
}

} // namespace

ShmMuninClient::~ShmMuninClient() {
    detach();
}

void ShmMuninClient::detach() noexcept {
    if (_sessionFd >= 0) {
        ::close(_sessionFd);
        _sessionFd = -1;
    }
}

std::expected<HealthzResponse, std::string>
ShmMuninClient::healthz(std::string_view socketPath) noexcept {
    auto fd = connectUnix(socketPath);
    if (!fd) {
        return std::unexpected(fd.error());
    }
    const int sock = *fd;

    auto rsp = oneRequest(sock, R"({"op":"healthz"})");
    ::close(sock);
    if (!rsp) {
        return std::unexpected(rsp.error());
    }

    auto parsed = HealthzResponse::fromJson(*rsp);
    if (parsed) {
        return *parsed;
    }
    if (auto errBody = parseErrorJson(*rsp); errBody) {
        return std::unexpected(std::string{"Munin healthz error: "} + *errBody);
    }
    return std::unexpected(parsed.error());
}

std::expected<ShmMuninClient::AttachResult, std::string>
ShmMuninClient::attach(std::string_view socketPath,
                       std::string_view modelId) noexcept {
    if (_sessionFd >= 0) {
        return std::unexpected(std::string{
            "ShmMuninClient: attach called on already-attached client"});
    }
    auto fd = connectUnix(socketPath);
    if (!fd) {
        return std::unexpected(fd.error());
    }
    auto res = attachOnConnectedFd(*fd, modelId);
    if (!res) {
        ::close(*fd);  // attachOnConnectedFd did not take ownership on failure
        return std::unexpected(res.error());
    }
    return res;
}

std::expected<ShmMuninClient::AttachResult, std::string>
ShmMuninClient::attachOnConnectedFd(int fd, std::string_view modelId) noexcept {
    if (_sessionFd >= 0) {
        return std::unexpected(std::string{
            "ShmMuninClient: attach called on already-attached client"});
    }
    if (modelId.empty()) {
        return std::unexpected(std::string{
            "ShmMuninClient: attach requires a non-empty modelId"});
    }

    std::string body;
    body.append(R"({"op":"attach","modelId":")").append(modelId).append(R"("})");
    if (auto s = UnixSocketFrame::send(fd, asBytes(body)); !s) {
        return std::unexpected(s.error());
    }

    // First response frame = manifest JSON (or error envelope), no fds.
    auto first = UnixSocketFrame::recv(fd);
    if (!first) {
        return std::unexpected(first.error());
    }
    if (!first->fds.empty()) {
        for (int f : first->fds) ::close(f);
        return std::unexpected(std::string{
            "ShmMuninClient: manifest frame unexpectedly carried fds"});
    }
    if (auto errBody = parseErrorJson(asStringView(first->payload)); errBody) {
        return std::unexpected(std::string{"Munin attach error: "} + *errBody);
    }
    auto manifest = TensorManifest::fromJson(asStringView(first->payload));
    if (!manifest) {
        return std::unexpected(manifest.error());
    }

    AttachResult out{};
    out.manifest = std::move(*manifest);
    const std::size_t nChunks = out.manifest.chunks.size();
    out.chunkBases.reserve(nChunks);

    // Roll back every mapping opened so far, for any mid-stream failure.
    auto rollback = [&](std::string msg) {
        for (void* b : out.chunkBases) _importer.closeChunk(b);
        return std::unexpected<std::string>(std::move(msg));
    };

    // One HANDLE frame per chunk: 64-byte payload + one SCM_RIGHTS memfd.
    for (std::size_t i = 0; i < nChunks; ++i) {
        auto frame = UnixSocketFrame::recv(fd, /*maxPayloadBytes=*/128);
        if (!frame) {
            return rollback(frame.error());
        }
        if (frame->payload.size() != 64 || frame->fds.size() != 1) {
            for (int f : frame->fds) ::close(f);
            std::ostringstream os;
            os << "ShmMuninClient: chunk-handle frame[" << i << "] has "
               << frame->payload.size() << " payload bytes and "
               << frame->fds.size() << " fd(s), expected 64 + 1";
            return rollback(os.str());
        }

        std::span<const std::byte, 64> hb{frame->payload.data(), 64};
        auto p = _importer.importChunk(hb, frame->fds[0]);
        if (!p) {
            ::close(frame->fds[0]);  // importChunk did not consume it on failure
            std::ostringstream os;
            os << "ShmMuninClient: import of chunk[" << i << "] failed: "
               << p.error();
            return rollback(os.str());
        }
        out.chunkBases.push_back(*p);
    }

    _sessionFd = fd;  // take ownership of the session socket on success
    MM_LOG_INFO("shm-munin-client",
                "attached to model '{}' fingerprint='{}' tensors={} chunks={}",
                out.manifest.modelId, out.manifest.modelFingerprint,
                out.manifest.tensors.size(), nChunks);
    return out;
}

} // namespace mimirmind::core::ipc

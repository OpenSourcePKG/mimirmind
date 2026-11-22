// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/ipc/WeightArenaExporter.hpp"

#include "core/ipc/UnixSocketFrame.hpp"
#include "core/log/Log.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <span>

namespace mimirmind::core::ipc {

namespace {

std::expected<sockaddr_un, std::string> unixAddr(const std::string& path) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof(addr.sun_path)) {
        return std::unexpected("weight arena socket path empty or too long: '" + path + "'");
    }
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    return addr;
}

std::span<const std::byte> asBytes(const std::string& s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

} // namespace

WeightArenaExporter::~WeightArenaExporter() {
    stop();
}

std::expected<void, std::string>
WeightArenaExporter::start(const std::string& socketPath, int arenaFd,
                           WeightArenaManifest manifest) {
    if (_thread.joinable()) {
        return std::unexpected("weight arena exporter already running");
    }
    auto addr = unixAddr(socketPath);
    if (!addr) {
        return std::unexpected(addr.error());
    }
    // A read-only open file description of the memfd: an attacher's
    // mmap(PROT_WRITE, MAP_SHARED) on it fails with EACCES.
    const std::string procPath = "/proc/self/fd/" + std::to_string(arenaFd);
    _roFd = ::open(procPath.c_str(), O_RDONLY | O_CLOEXEC);
    if (_roFd < 0) {
        return std::unexpected("weight arena exporter: reopen read-only failed: " +
                               std::string{std::strerror(errno)});
    }
    _listenFd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (_listenFd < 0) {
        stop();
        return std::unexpected("weight arena exporter: socket: " +
                               std::string{std::strerror(errno)});
    }
    ::unlink(socketPath.c_str());
    if (::bind(_listenFd, reinterpret_cast<const sockaddr*>(&*addr), sizeof(*addr)) != 0 ||
        ::listen(_listenFd, 4) != 0) {
        const std::string e = std::strerror(errno);
        stop();
        return std::unexpected("weight arena exporter: bind/listen '" + socketPath +
                               "': " + e);
    }
    ::chmod(socketPath.c_str(), 0660);
    _path         = socketPath;
    _manifestJson = manifest.toJson();
    _stop.store(false);
    _thread = std::thread([this] { run(); });
    return {};
}

void WeightArenaExporter::stop() noexcept {
    _stop.store(true);
    if (_thread.joinable()) {
        _thread.join();
    }
    if (_listenFd >= 0) {
        ::close(_listenFd);
        _listenFd = -1;
    }
    if (!_path.empty()) {
        ::unlink(_path.c_str());
        _path.clear();
    }
    if (_roFd >= 0) {
        ::close(_roFd);
        _roFd = -1;
    }
}

void WeightArenaExporter::run() noexcept {
    while (!_stop.load()) {
        pollfd p{_listenFd, POLLIN, 0};
        const int rc = ::poll(&p, 1, 200);
        if (rc <= 0 || (p.revents & POLLIN) == 0) {
            continue;
        }
        const int c = ::accept4(_listenFd, nullptr, nullptr, SOCK_CLOEXEC);
        if (c < 0) {
            continue;
        }
        // Bound the handshake so a silent client cannot wedge the thread.
        timeval tv{5, 0};
        ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        auto req = UnixSocketFrame::recv(c, 64 * 1024);
        if (!req) {
            MM_LOG_WARN("arena", "weight arena: bad attach request: {}", req.error());
            ::close(c);
            continue;
        }
        const int fds[1] = {_roFd};
        auto sent = UnixSocketFrame::send(c, asBytes(_manifestJson), std::span<const int>(fds, 1));
        if (!sent) {
            MM_LOG_WARN("arena", "weight arena: reply failed: {}", sent.error());
        } else {
            const auto n = _attaches.fetch_add(1) + 1;
            MM_LOG_INFO("arena", "weight arena: attacher #{} served on '{}' ({} request bytes)",
                        n, _path, req->payload.size());
        }
        for (int fd : req->fds) {
            ::close(fd);
        }
        ::close(c);
    }
}

std::expected<WeightArenaHandle, std::string>
fetchWeightArena(const std::string& socketPath, const std::string& clientName) noexcept {
    auto addr = unixAddr(socketPath);
    if (!addr) {
        return std::unexpected(addr.error());
    }
    const int s = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) {
        return std::unexpected("weight arena attach: socket: " +
                               std::string{std::strerror(errno)});
    }
    if (::connect(s, reinterpret_cast<const sockaddr*>(&*addr), sizeof(*addr)) != 0) {
        const std::string e = std::strerror(errno);
        ::close(s);
        return std::unexpected("weight arena attach: connect '" + socketPath + "': " + e +
                               " (is the engine running with sharedWeights.socket set?)");
    }
    const std::string req = std::string{R"({"op":"attach","client":")"} + clientName + "\"}";
    auto sent = UnixSocketFrame::send(s, asBytes(req));
    if (!sent) {
        ::close(s);
        return std::unexpected("weight arena attach: send: " + sent.error());
    }
    auto reply = UnixSocketFrame::recv(s);
    ::close(s);
    if (!reply) {
        return std::unexpected("weight arena attach: recv: " + reply.error());
    }
    if (reply->fds.size() != 1) {
        for (int fd : reply->fds) ::close(fd);
        return std::unexpected("weight arena attach: expected 1 fd, got " +
                               std::to_string(reply->fds.size()));
    }
    const std::string json(reinterpret_cast<const char*>(reply->payload.data()),
                           reply->payload.size());
    auto m = WeightArenaManifest::fromJson(json);
    if (!m) {
        ::close(reply->fds[0]);
        return std::unexpected(m.error());
    }
    return WeightArenaHandle{std::move(*m), reply->fds[0]};
}

} // namespace mimirmind::core::ipc

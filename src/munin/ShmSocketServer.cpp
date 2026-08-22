// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "munin/ShmSocketServer.hpp"

#include "core/log/Log.hpp"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace mimirmind::munin {

namespace {

// Stale-socket detection: file exists AND is a socket AND connect() fails
// with ECONNREFUSED (no listener). Everything else -> do not unlink.
bool isStaleSocket(const std::string& path) noexcept {
    struct ::stat st{};
    if (::stat(path.c_str(), &st) < 0) return false;
    if (!S_ISSOCK(st.st_mode)) return false;
    const int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe < 0) return false;
    struct ::sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(probe);
        return false;
    }
    std::memcpy(addr.sun_path, path.data(), path.size());
    const int r = ::connect(
        probe, reinterpret_cast<struct ::sockaddr*>(&addr), sizeof(addr));
    const int e = errno;
    ::close(probe);
    if (r == 0) return false;  // live listener — do NOT touch
    return e == ECONNREFUSED;
}

pid_t peerPidOf(int fd) noexcept {
    struct ::ucred cred{};
    ::socklen_t sz = sizeof(cred);
    if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &sz) < 0) return 0;
    return cred.pid;
}

} // namespace

ShmSocketServer::ShmSocketServer(const ShmModelStore& store,
                                 std::string          socketPath)
    : _store{store}, _socketPath{std::move(socketPath)} {}

ShmSocketServer::~ShmSocketServer() {
    stop();
}

void ShmSocketServer::closeListenFd() noexcept {
    if (_listenFd >= 0) {
        ::close(_listenFd);
        _listenFd = -1;
    }
}

void ShmSocketServer::serve(int shutdownEventFd) {
    if (_socketPath.empty()) {
        throw std::runtime_error{"ShmSocketServer: socket path is empty"};
    }
    if (_socketPath.size() >= sizeof(::sockaddr_un{}.sun_path)) {
        throw std::runtime_error{
            "ShmSocketServer: socket path exceeds sun_path capacity ("
            + std::to_string(sizeof(::sockaddr_un{}.sun_path) - 1) + " bytes)"};
    }

    if (isStaleSocket(_socketPath)) {
        MM_LOG_INFO("munin",
                    "ShmSocketServer: unlinking stale socket at {}", _socketPath);
        if (::unlink(_socketPath.c_str()) < 0) {
            throw std::runtime_error{
                "ShmSocketServer: unlink stale socket failed: "
                + std::string{std::strerror(errno)}};
        }
    }

    _listenFd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (_listenFd < 0) {
        throw std::runtime_error{
            "ShmSocketServer: socket(AF_UNIX) failed: "
            + std::string{std::strerror(errno)}};
    }

    ::sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, _socketPath.data(), _socketPath.size());

    if (::bind(_listenFd, reinterpret_cast<::sockaddr*>(&addr), sizeof(addr)) < 0) {
        const int e = errno;
        closeListenFd();
        throw std::runtime_error{
            "ShmSocketServer: bind(" + _socketPath + ") failed: "
            + std::string{std::strerror(e)}};
    }

    if (::chmod(_socketPath.c_str(), 0660) < 0) {
        MM_LOG_WARN("munin",
                    "ShmSocketServer: chmod 0660 on socket failed (errno={}) — "
                    "continuing", errno);
    }

    if (::listen(_listenFd, /*backlog=*/16) < 0) {
        const int e = errno;
        closeListenFd();
        ::unlink(_socketPath.c_str());
        throw std::runtime_error{
            "ShmSocketServer: listen failed: " + std::string{std::strerror(e)}};
    }

    MM_LOG_INFO("munin", "ShmSocketServer: listening on {}", _socketPath);

    while (!_stopRequested.load()) {
        pollfd pfds[2]{};
        pfds[0].fd     = _listenFd;
        pfds[0].events = POLLIN;
        int nfds = 1;
        if (shutdownEventFd >= 0) {
            pfds[1].fd     = shutdownEventFd;
            pfds[1].events = POLLIN;
            nfds = 2;
        }

        const int pr = ::poll(pfds, static_cast<::nfds_t>(nfds), /*timeout_ms=*/-1);
        if (pr < 0) {
            if (errno == EINTR) continue;
            MM_LOG_ERROR("munin",
                         "ShmSocketServer: poll failed: {} (errno={})",
                         std::strerror(errno), errno);
            break;
        }

        if (shutdownEventFd >= 0
            && (pfds[1].revents & (POLLIN | POLLHUP)) != 0) {
            MM_LOG_INFO("munin",
                        "ShmSocketServer: shutdown event received, exiting accept loop");
            break;
        }

        if ((pfds[0].revents & POLLIN) == 0) continue;

        const int connFd = ::accept4(_listenFd, nullptr, nullptr, SOCK_CLOEXEC);
        if (connFd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (errno == EBADF || errno == EINVAL) break;  // listen fd closed on stop()
            MM_LOG_WARN("munin",
                        "ShmSocketServer: accept4 failed: {} (errno={})",
                        std::strerror(errno), errno);
            continue;
        }

        const std::uint32_t sid = _nextSessionId.fetch_add(1);
        const pid_t         pid = peerPidOf(connFd);

        auto slot     = std::make_unique<SessionSlot>();
        slot->session = std::make_unique<ShmAttachSession>(connFd, pid, sid, _store);
        slot->connFd  = connFd;

        ShmAttachSession* raw = slot->session.get();
        slot->thread = std::thread([raw]() noexcept { raw->run(); });

        {
            std::lock_guard<std::mutex> lk{_sessionsMx};
            _sessions.push_back(std::move(slot));
        }
    }

    // Shutdown: wind down every live session, then join.
    {
        std::lock_guard<std::mutex> lk{_sessionsMx};
        for (auto& s : _sessions) {
            if (s->session) s->session->requestStop();
            if (s->connFd >= 0) ::shutdown(s->connFd, SHUT_RDWR);
        }
    }

    std::vector<std::unique_ptr<SessionSlot>> local;
    {
        std::lock_guard<std::mutex> lk{_sessionsMx};
        local.swap(_sessions);
    }
    for (auto& s : local) {
        if (s->thread.joinable()) s->thread.join();
    }
    local.clear();

    closeListenFd();
    ::unlink(_socketPath.c_str());
    MM_LOG_INFO("munin", "ShmSocketServer: exited cleanly, socket {} unlinked",
                _socketPath);
}

void ShmSocketServer::stop() noexcept {
    if (_stopRequested.exchange(true)) return;
    if (_listenFd >= 0) ::shutdown(_listenFd, SHUT_RDWR);
    std::lock_guard<std::mutex> lk{_sessionsMx};
    for (auto& s : _sessions) {
        if (s->session) s->session->requestStop();
        if (s->connFd >= 0) ::shutdown(s->connFd, SHUT_RDWR);
    }
}

std::vector<ShmSocketServer::SessionInfo> ShmSocketServer::sessions() const {
    std::vector<SessionInfo> out;
    std::lock_guard<std::mutex> lk{_sessionsMx};
    out.reserve(_sessions.size());
    for (const auto& s : _sessions) {
        SessionInfo i{};
        i.sessionId = s->session ? s->session->sessionId() : 0;
        i.peerPid   = s->session ? s->session->peerPid()   : 0;
        i.modelId   = s->session ? s->session->attachedModelId() : "";
        out.push_back(std::move(i));
    }
    return out;
}

} // namespace mimirmind::munin

#include "munin/SocketServer.hpp"

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
#include <string_view>
#include <utility>

namespace mimirmind::munin {

namespace {

// Try to detect a stale socket file at `path`. Returns true when the
// file exists AND is a socket AND connect() to it fails with
// ECONNREFUSED (i.e. no listener). All other file states (regular
// file, live socket, EACCES, ...) return false and the caller refuses
// to unlink.
bool isStaleSocket(const std::string& path) noexcept {
    struct ::stat st{};
    if (::stat(path.c_str(), &st) < 0) {
        return false;
    }
    if (!S_ISSOCK(st.st_mode)) {
        return false;
    }
    const int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (probe < 0) {
        return false;
    }
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
    if (r == 0) {
        return false;  // live listener — do NOT touch
    }
    return e == ECONNREFUSED;
}

// Peer's PID from SO_PEERCRED. 0 on error — used only for logging and
// bookkeeping, never for authorisation.
pid_t peerPidOf(int fd) noexcept {
    struct ::ucred cred{};
    ::socklen_t sz = sizeof(cred);
    if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &sz) < 0) {
        return 0;
    }
    return cred.pid;
}

} // namespace

SocketServer::SocketServer(const ModelStore&                  store,
                           ::mimirmind::core::l0::L0Context&  l0,
                           std::string                        socketPath)
    : _store{store}, _l0{l0}, _socketPath{std::move(socketPath)} {}

SocketServer::~SocketServer() {
    stop();
}

void SocketServer::closeListenFd() noexcept {
    if (_listenFd >= 0) {
        ::close(_listenFd);
        _listenFd = -1;
    }
}

void SocketServer::serve(int shutdownEventFd) {
    // ---- bind ---------------------------------------------------------
    if (_socketPath.empty()) {
        throw std::runtime_error{"SocketServer: socket path is empty"};
    }
    if (_socketPath.size() >= sizeof(::sockaddr_un{}.sun_path)) {
        throw std::runtime_error{
            "SocketServer: socket path exceeds sun_path capacity ("
            + std::to_string(sizeof(::sockaddr_un{}.sun_path) - 1)
            + " bytes)"};
    }

    // Reap any stale socket left over from a previous unclean shutdown.
    if (isStaleSocket(_socketPath)) {
        MM_LOG_INFO("munin",
                    "SocketServer: unlinking stale socket at {}",
                    _socketPath);
        if (::unlink(_socketPath.c_str()) < 0) {
            const int e = errno;
            throw std::runtime_error{
                "SocketServer: unlink stale socket failed: "
                + std::string{std::strerror(e)}};
        }
    }

    _listenFd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (_listenFd < 0) {
        const int e = errno;
        throw std::runtime_error{
            "SocketServer: socket(AF_UNIX) failed: "
            + std::string{std::strerror(e)}};
    }

    ::sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, _socketPath.data(), _socketPath.size());

    if (::bind(_listenFd, reinterpret_cast<::sockaddr*>(&addr), sizeof(addr)) < 0) {
        const int e = errno;
        closeListenFd();
        throw std::runtime_error{
            "SocketServer: bind(" + _socketPath + ") failed: "
            + std::string{std::strerror(e)}};
    }

    // Attached workers connect from the same host; there is no filesystem
    // permission model above uid/gid. 0660 with the daemon's primary gid
    // is the sensible default — non-root workers in the same group can
    // attach without exposing the socket to the world.
    if (::chmod(_socketPath.c_str(), 0660) < 0) {
        MM_LOG_WARN("munin",
                    "SocketServer: chmod 0660 on socket failed (errno={}) — "
                    "continuing", errno);
    }

    if (::listen(_listenFd, /*backlog=*/16) < 0) {
        const int e = errno;
        closeListenFd();
        ::unlink(_socketPath.c_str());
        throw std::runtime_error{
            "SocketServer: listen failed: "
            + std::string{std::strerror(e)}};
    }

    MM_LOG_INFO("munin",
                "SocketServer: listening on {}", _socketPath);

    // ---- accept loop --------------------------------------------------
    // poll on the listen fd + optional shutdown fd. accept() itself
    // is level-triggered so we do not need edge-tricks.
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
                         "SocketServer: poll failed: {} (errno={})",
                         std::strerror(errno), errno);
            break;
        }

        if (shutdownEventFd >= 0
            && (pfds[1].revents & (POLLIN | POLLHUP)) != 0) {
            MM_LOG_INFO("munin",
                        "SocketServer: shutdown event received, exiting accept loop");
            break;
        }

        if ((pfds[0].revents & POLLIN) == 0) {
            continue;
        }

        const int connFd = ::accept4(_listenFd, nullptr, nullptr, SOCK_CLOEXEC);
        if (connFd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            if (errno == EBADF || errno == EINVAL) {
                // listen fd was closed underneath us — expected on stop().
                break;
            }
            MM_LOG_WARN("munin",
                        "SocketServer: accept4 failed: {} (errno={})",
                        std::strerror(errno), errno);
            continue;
        }

        // Bookkeep the session before we hand its fd to a thread.
        const std::uint32_t sid = _nextSessionId.fetch_add(1);
        const pid_t         pid = peerPidOf(connFd);

        auto slot          = std::make_unique<SessionSlot>();
        slot->session      = std::make_unique<AttachSession>(
            connFd, pid, sid, _store, _l0);
        slot->connFd       = connFd;

        AttachSession* raw = slot->session.get();
        slot->thread       = std::thread([raw]() noexcept {
            raw->run();
        });

        {
            std::lock_guard<std::mutex> lk{_sessionsMx};
            _sessions.push_back(std::move(slot));
        }
    }

    // ---- shutdown: tell every live session to wind down, then join ---
    {
        std::lock_guard<std::mutex> lk{_sessionsMx};
        for (auto& s : _sessions) {
            if (s->session) {
                s->session->requestStop();
            }
            // Shutting down the read side wakes any blocking recv on
            // the session's thread with an EOF-equivalent return, which
            // is how the session exits its wait-for-peer-close loop.
            if (s->connFd >= 0) {
                ::shutdown(s->connFd, SHUT_RDWR);
            }
        }
    }

    // Join outside the lock: the session threads may briefly acquire it
    // if we ever grow such a call; today they don't, but we defensive
    // -join without holding the lock.
    std::vector<std::unique_ptr<SessionSlot>> local;
    {
        std::lock_guard<std::mutex> lk{_sessionsMx};
        local.swap(_sessions);
    }
    for (auto& s : local) {
        if (s->thread.joinable()) {
            s->thread.join();
        }
    }
    local.clear();

    closeListenFd();
    ::unlink(_socketPath.c_str());
    MM_LOG_INFO("munin", "SocketServer: exited cleanly, socket {} unlinked",
                _socketPath);
}

void SocketServer::stop() noexcept {
    if (_stopRequested.exchange(true)) {
        return;
    }
    // Break accept() in the serve thread by shutting the listen fd.
    if (_listenFd >= 0) {
        ::shutdown(_listenFd, SHUT_RDWR);
    }
    // Wake any active session's recv.
    std::lock_guard<std::mutex> lk{_sessionsMx};
    for (auto& s : _sessions) {
        if (s->session) {
            s->session->requestStop();
        }
        if (s->connFd >= 0) {
            ::shutdown(s->connFd, SHUT_RDWR);
        }
    }
}

void SocketServer::reapFinishedSessions() noexcept {
    // Walk _sessions and drop any slot whose thread has already run to
    // completion. Cheap because we only touch it after accept(), and
    // Munin's steady-state has O(few) sessions.
    std::vector<std::unique_ptr<SessionSlot>> reaped;
    {
        std::lock_guard<std::mutex> lk{_sessionsMx};
        auto it = _sessions.begin();
        while (it != _sessions.end()) {
            auto& s = *it;
            // A joinable thread is either running or joinable-because-
            // finished. Peek by non-blocking join via native_handle
            // would be racy; instead we approximate: sessions that have
            // seen their session's fd already closed by the peer will
            // typically have their thread near-exit. A cheap heuristic
            // is not available without pthread_tryjoin_np which is
            // non-portable — for MVP we do a simple pass and rely on
            // stop() at shutdown for the final sweep. Leaving this
            // function a no-op for now to stay honest.
            (void)s;
            ++it;
        }
        (void)reaped;
    }
}

std::vector<SocketServer::SessionInfo> SocketServer::sessions() const {
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
#include "core/ipc/UnixSocketFrame.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sstream>

namespace mimirmind::core::ipc {

namespace {

std::string errnoTag(const char* where, int e) {
    std::ostringstream os;
    os << where << ": " << std::strerror(e) << " (errno=" << e << ")";
    return os.str();
}

// Read exactly `n` bytes into `dst`, appending any SCM_RIGHTS fds
// received en route into `fds`. Handles partial reads, EINTR retries,
// and peer-closed detection.
std::expected<void, std::string>
readExact(int sock, std::byte* dst, std::size_t n, std::vector<int>& fds) noexcept {
    while (n > 0) {
        // Room for up to kMaxFdsPerFrame fds in one recvmsg. In practice
        // Munin sends one fd per handle frame, so this is generous.
        char cbuf[CMSG_SPACE(sizeof(int) * UnixSocketFrame::kMaxFdsPerFrame)];
        iovec  iov{};
        iov.iov_base = dst;
        iov.iov_len  = n;
        msghdr msg{};
        msg.msg_iov        = &iov;
        msg.msg_iovlen     = 1;
        msg.msg_control    = cbuf;
        msg.msg_controllen = sizeof(cbuf);

        const ssize_t r = ::recvmsg(sock, &msg, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return std::unexpected(errnoTag("recvmsg", errno));
        }
        if (r == 0) {
            return std::unexpected(std::string{"recvmsg: peer closed connection"});
        }

        // Harvest fds regardless of which recvmsg they arrived on.
        for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
            if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) {
                continue;
            }
            const std::size_t dataLen = c->cmsg_len - CMSG_LEN(0);
            const std::size_t nFds    = dataLen / sizeof(int);
            for (std::size_t i = 0; i < nFds; ++i) {
                int fd = -1;
                std::memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof(int));
                fds.push_back(fd);
            }
        }

        dst += r;
        n   -= static_cast<std::size_t>(r);
    }
    return {};
}

// Write exactly `n` bytes from `src`. First sendmsg attaches SCM_RIGHTS
// with the provided fds; subsequent sends (if partial) are plain writes.
std::expected<void, std::string>
writeExactWithFds(int sock,
                  const std::byte*     src,
                  std::size_t          n,
                  std::span<const int> fds) noexcept {
    bool controlAttached = false;
    while (n > 0) {
        iovec iov{};
        iov.iov_base = const_cast<std::byte*>(src);
        iov.iov_len  = n;
        msghdr msg{};
        msg.msg_iov        = &iov;
        msg.msg_iovlen     = 1;

        // Attach control only on the first sendmsg; SCM_RIGHTS travels
        // with the first byte of the frame and only needs to fire once.
        char cbuf[CMSG_SPACE(sizeof(int) * UnixSocketFrame::kMaxFdsPerFrame)];
        if (!controlAttached && !fds.empty()) {
            std::memset(cbuf, 0, sizeof(cbuf));
            msg.msg_control    = cbuf;
            msg.msg_controllen = CMSG_SPACE(sizeof(int) * fds.size());
            cmsghdr* c = CMSG_FIRSTHDR(&msg);
            c->cmsg_level = SOL_SOCKET;
            c->cmsg_type  = SCM_RIGHTS;
            c->cmsg_len   = CMSG_LEN(sizeof(int) * fds.size());
            std::memcpy(CMSG_DATA(c), fds.data(), sizeof(int) * fds.size());
        }

        const ssize_t r = ::sendmsg(sock, &msg, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return std::unexpected(errnoTag("sendmsg", errno));
        }
        if (r == 0) {
            return std::unexpected(std::string{"sendmsg returned 0"});
        }
        controlAttached = true;
        src += r;
        n   -= static_cast<std::size_t>(r);
    }
    return {};
}

} // namespace

std::expected<void, std::string>
UnixSocketFrame::send(int sock,
                      std::span<const std::byte> payload,
                      std::span<const int>       fds) noexcept {
    if (fds.size() > kMaxFdsPerFrame) {
        std::ostringstream os;
        os << "send: fds.size()=" << fds.size()
           << " exceeds SCM_MAX_FD=" << kMaxFdsPerFrame;
        return std::unexpected(os.str());
    }

    const std::uint32_t lenLE = static_cast<std::uint32_t>(payload.size());
    std::byte prefix[4];
    std::memcpy(prefix, &lenLE, 4);

    // Prefix carries the control message. Payload streams without control.
    if (auto r = writeExactWithFds(sock, prefix, 4, fds); !r) {
        return std::unexpected(r.error());
    }
    if (!payload.empty()) {
        if (auto r = writeExactWithFds(sock, payload.data(), payload.size(), {});
            !r) {
            return std::unexpected(r.error());
        }
    }
    return {};
}

std::expected<Frame, std::string>
UnixSocketFrame::recv(int sock, std::size_t maxPayloadBytes) noexcept {
    Frame f{};

    std::byte prefix[4];
    if (auto r = readExact(sock, prefix, 4, f.fds); !r) {
        return std::unexpected(r.error());
    }
    std::uint32_t lenLE = 0;
    std::memcpy(&lenLE, prefix, 4);

    if (lenLE > maxPayloadBytes) {
        std::ostringstream os;
        os << "recv: frame length " << lenLE
           << " exceeds max " << maxPayloadBytes;
        return std::unexpected(os.str());
    }

    if (lenLE > 0) {
        f.payload.resize(lenLE);
        if (auto r = readExact(sock, f.payload.data(), lenLE, f.fds); !r) {
            return std::unexpected(r.error());
        }
    }
    return f;
}

} // namespace mimirmind::core::ipc
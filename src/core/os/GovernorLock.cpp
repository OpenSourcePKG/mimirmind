#include "core/os/GovernorLock.hpp"

#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace mimirmind::core::os {

namespace {

// Best-effort holder identification. Reads `/proc/locks` and looks for
// a FLOCK entry that matches the inode of `path`. Returns a formatted
// hint string on success (empty when nothing usable was found). Never
// throws.
std::string identifyHolder(const std::string& path) noexcept {
    struct ::stat st{};
    if (::stat(path.c_str(), &st) < 0) {
        return {};
    }
    std::ifstream f{"/proc/locks"};
    if (!f) {
        return {};
    }
    std::string line;
    while (std::getline(f, line)) {
        // Sample: "3: FLOCK  ADVISORY  WRITE 12345 fd:01:2621452 0 EOF"
        // Fields: id: kind mode WRITE|READ pid maj:min:inode start end
        // We split on whitespace and pick the pid and the maj:min:inode.
        std::istringstream is{line};
        std::string tok;
        int col = 0;
        std::string kind, pidStr, mmi;
        while (is >> tok) {
            switch (col) {
                case 1: kind   = tok; break;
                case 4: pidStr = tok; break;
                case 5: mmi    = tok; break;
                default: break;
            }
            ++col;
        }
        if (kind != "FLOCK") continue;
        const auto colon2 = mmi.rfind(':');
        if (colon2 == std::string::npos) continue;
        const std::string inodeStr = mmi.substr(colon2 + 1);
        try {
            const auto inode = std::stoull(inodeStr);
            if (inode == static_cast<unsigned long long>(st.st_ino)) {
                std::ostringstream os;
                os << " (held by pid " << pidStr << ")";
                return os.str();
            }
        } catch (...) {
            // continue
        }
    }
    return {};
}

// mkdir -p for the parent directory of the lock path. The 0755 mode
// matches what the Munin daemon uses for /var/run/munin/. Returns true
// on success or "already exists"; false otherwise.
bool ensureParentDir(const std::string& path) noexcept {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    std::string dir = path.substr(0, slash);
    std::string acc;
    acc.reserve(dir.size());
    for (std::size_t i = 0; i < dir.size(); ++i) {
        acc.push_back(dir[i]);
        if (dir[i] != '/' && i + 1 != dir.size()) continue;
        if (acc == "/") continue;
        if (::mkdir(acc.c_str(), 0755) < 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

} // namespace

GovernorLock::GovernorLock(GovernorLock&& other) noexcept
    : _fd{other._fd}, _path{std::move(other._path)} {
    other._fd = -1;
}

GovernorLock& GovernorLock::operator=(GovernorLock&& other) noexcept {
    if (this != &other) {
        release();
        _fd       = other._fd;
        _path     = std::move(other._path);
        other._fd = -1;
    }
    return *this;
}

GovernorLock::~GovernorLock() {
    release();
}

void GovernorLock::release() noexcept {
    if (_fd >= 0) {
        // Closing the fd releases the flock; explicit LOCK_UN is
        // redundant on Linux but harmless.
        ::flock(_fd, LOCK_UN);
        ::close(_fd);
        _fd = -1;
    }
}

std::expected<GovernorLock, std::string>
GovernorLock::tryAcquire(std::string_view path) noexcept {
    std::string p{path};
    if (!ensureParentDir(p)) {
        std::ostringstream os;
        os << "GovernorLock: cannot create parent directory for '" << p
           << "': " << std::strerror(errno);
        return std::unexpected(os.str());
    }
    const int fd = ::open(p.c_str(),
                          O_RDWR | O_CREAT | O_CLOEXEC,
                          0664);
    if (fd < 0) {
        std::ostringstream os;
        os << "GovernorLock: open('" << p << "') failed: "
           << std::strerror(errno) << " (errno=" << errno << ")";
        return std::unexpected(os.str());
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) < 0) {
        const int e = errno;
        ::close(fd);
        std::ostringstream os;
        os << "GovernorLock: flock('" << p << "') failed: "
           << std::strerror(e) << " (errno=" << e << ")";
        os << identifyHolder(p);
        return std::unexpected(os.str());
    }

    // Best-effort record: write our PID into the file so an operator
    // reading it sees who has it. The lock semantics do not depend on
    // this — flock is per-fd — but the file content is a useful hint.
    const std::string pidLine = std::to_string(::getpid()) + "\n";
    if (::ftruncate(fd, 0) == 0) {
        // Return value intentionally ignored — the pid write is a hint
        // for humans reading the lock file, not a correctness signal.
        const ::ssize_t w = ::write(fd, pidLine.data(), pidLine.size());
        (void)w;
    }

    GovernorLock lk{};
    lk._fd   = fd;
    lk._path = std::move(p);
    return lk;
}

} // namespace mimirmind::core::os
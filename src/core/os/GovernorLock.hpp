#pragma once

#include <expected>
#include <string>
#include <string_view>

namespace mimirmind::core::os {

/**
 * RAII exclusive `flock(2)` on a fixed lock file, used to serialise
 * governor ownership between mimirmind (standalone or attached) and
 * Munin. Exactly one process at a time may hold the lock; the second
 * attempt fails fast with `EWOULDBLOCK` and a message pointing at the
 * conflicting PID (read from `/proc/locks` on a best-effort basis).
 *
 * Semantics — governor ownership:
 *   - Standalone `mimirmind serve` acquires the lock at startup.
 *   - Munin acquires the lock at startup.
 *   - `mimirmind serve --attach ...` does NOT try to acquire it. It
 *     reads Munin's `healthz.governor_owner` field and skips its own
 *     governor init when that says `"munin"`.
 *
 * The lock file itself carries no state; it is a coordination point.
 * `flock` is per-open-file-description on Linux, so we hold the fd
 * open for the lock's lifetime.
 */
class GovernorLock {
public:
    /**
     * Default lock file path. Prod runtime dir; the Docker compose
     * files for Munin + attached-worker mount `/var/run/mimirmind/`
     * from a shared tmpfs so both containers see the same inode.
     */
    static constexpr std::string_view kDefaultPath =
        "/var/run/mimirmind/governor.lock";

    /**
     * Try to acquire the lock. Returns a live `GovernorLock` on
     * success or an error string that includes the conflicting
     * holder's PID when possible.
     *
     * The file is created with mode 0664 if missing. The lock is
     * released when the returned object is destroyed.
     */
    [[nodiscard]] static std::expected<GovernorLock, std::string>
    tryAcquire(std::string_view path = kDefaultPath) noexcept;

    ~GovernorLock();

    GovernorLock(const GovernorLock&)            = delete;
    GovernorLock& operator=(const GovernorLock&) = delete;
    GovernorLock(GovernorLock&& other) noexcept;
    GovernorLock& operator=(GovernorLock&& other) noexcept;

    [[nodiscard]] std::string_view path() const noexcept { return _path; }
    [[nodiscard]] bool             held() const noexcept { return _fd >= 0; }

    /**
     * Release the lock explicitly. Idempotent. The destructor calls
     * this — you only need it when you want to hand ownership over
     * before the object goes out of scope.
     */
    void release() noexcept;

private:
    GovernorLock() = default;

    int         _fd{-1};
    std::string _path{};
};

} // namespace mimirmind::core::os
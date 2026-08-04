// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>
#include <utility>

namespace mimirmind::core::security {

/**
 * Thread-local tenant label for the request currently being serviced.
 *
 * cpp-httplib gives each request start-to-finish to a single worker
 * thread, so a `thread_local` slot set in the pre-routing auth handler is
 * readable — without races — anywhere further down the same handler chain
 * (e.g. the chat-completion handler that eventually reaches the batcher's
 * admission control). `ScopedTenant` is an RAII guard: it stashes the
 * previous value on construction and restores it on destruction, so
 * nested handling on a reused pool thread never inherits a stale label.
 *
 * The per-tenant admission accounting itself is a follow-up (Bragi) — for
 * now this only carries the label so that wiring can read it later without
 * touching the auth hook again.
 */
class ScopedTenant {
public:
    explicit ScopedTenant(std::string tenantId, bool isAdmin = false)
        : _previous{std::move(current())}, _previousAdmin{currentAdmin()} {
        current()      = std::move(tenantId);
        currentAdmin() = isAdmin;
    }

    ~ScopedTenant() {
        current()      = std::move(_previous);
        currentAdmin() = _previousAdmin;
    }

    ScopedTenant(const ScopedTenant&)            = delete;
    ScopedTenant& operator=(const ScopedTenant&) = delete;
    ScopedTenant(ScopedTenant&&)                 = delete;
    ScopedTenant& operator=(ScopedTenant&&)      = delete;

    /// Tenant label for the current thread's in-flight request, or the
    /// empty string when auth is off / no guard is active.
    [[nodiscard]] static const std::string& active() { return current(); }

    /// Whether the current thread's request was authenticated with an
    /// admin-scoped key. False when auth is off or the key is a normal
    /// tenant key — the operator-only routes gate on this.
    [[nodiscard]] static bool activeIsAdmin() { return currentAdmin(); }

    /// Directly set the current thread's tenant label + admin flag. Used by
    /// the auth pre-routing hook, whose lambda returns before the route
    /// handler runs, so an RAII guard would be destroyed too early — the
    /// label has to outlive the hook and be picked up by the subsequent
    /// handler on the same pool thread. Cleared (empty tenant, non-admin)
    /// for open / unauthenticated paths so a reused thread never inherits a
    /// previous tenant's identity or privilege.
    static void set(std::string tenantId, bool isAdmin = false) {
        current()      = std::move(tenantId);
        currentAdmin() = isAdmin;
    }

private:
    static std::string& current() {
        static thread_local std::string tenant{};
        return tenant;
    }
    static bool& currentAdmin() {
        static thread_local bool admin{false};
        return admin;
    }

    std::string _previous;
    bool        _previousAdmin;
};

} // namespace mimirmind::core::security

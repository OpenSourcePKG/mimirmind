// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/backend/BackendPool.hpp"

#include "core/backend/BackendRegistry.hpp"
#include "core/log/Log.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::core::backend {

namespace {

std::string toLower(std::string_view s) {
    std::string out{s};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return out;
}

// Parse either "kind" or "kind:index". Returns the tail after the
// colon (as size_t) or 0 when the colon is absent. Returns nullopt on
// malformed input.
struct TokenParts {
    std::string_view kindName;
    std::size_t      deviceIx{0};
};

std::optional<TokenParts> splitToken(std::string_view lower) {
    const auto colon = lower.find(':');
    if (colon == std::string_view::npos) {
        return TokenParts{lower, 0U};
    }
    const auto kindName = lower.substr(0, colon);
    const auto ixText   = lower.substr(colon + 1);
    if (ixText.empty()) {
        return std::nullopt;
    }
    std::size_t ix = 0;
    const auto* first = ixText.data();
    const auto* last  = first + ixText.size();
    const auto res = std::from_chars(first, last, ix);
    if (res.ec != std::errc{} || res.ptr != last) {
        return std::nullopt;
    }
    return TokenParts{kindName, ix};
}

} // namespace

// ---- PoolEntry ------------------------------------------------------------

ComputeContext& PoolEntry::context() {
    if (_ctx == nullptr) {
        // Ignore deviceIx for now — createContext takes only a kind.
        // Multi-device-per-kind is a follow-up; single-device backends
        // still bind to their first device today.
        _ctx = BackendRegistry::createContext(kind);
    }
    return *_ctx;
}

// ---- BackendPool ----------------------------------------------------------

void BackendPool::discoverAll() {
    _entries.clear();
    const auto probes = BackendRegistry::probeAll();
    for (const auto& p : probes) {
        if (!p.compiledIn || !p.available) {
            continue;
        }
        PoolEntry entry;
        entry.kind     = p.kind;
        entry.deviceIx = 0;
        entry.name     = BackendRegistry::name(p.kind);
        entry.token    = tokenFor(p.kind, entry.deviceIx);
        entry.detail   = p.detail;
        _entries.push_back(std::move(entry));
    }
    MM_LOG_INFO("backend-pool",
                "discoverAll: {} entries", _entries.size());
    for (const auto& e : _entries) {
        MM_LOG_INFO("backend-pool",
                    "  {} — {} ({})", e.token, e.name, e.detail);
    }
}

PoolEntry& BackendPool::select(SelectionMode mode) {
    if (_entries.empty()) {
        throw std::runtime_error(
            "BackendPool::select: pool is empty — call discoverAll() "
            "first, and ensure at least one backend is compiled-in "
            "and available");
    }
    switch (mode) {
        case SelectionMode::Auto:
            // Entries are ordered by BackendKind enum order (LevelZero,
            // Hip, Cuda, Cpu) inside discoverAll — same priority as
            // BackendRegistry::autoSelect. GPU entries win over Cpu
            // by construction.
            return _entries.front();
    }
    throw std::runtime_error("BackendPool::select: unreachable mode");
}

PoolEntry& BackendPool::selectByToken(std::string_view token) {
    const auto lower = toLower(token);
    if (lower == "auto") {
        return select(SelectionMode::Auto);
    }
    const auto parts = splitToken(lower);
    if (!parts) {
        std::ostringstream os;
        os << "BackendPool::selectByToken: malformed token '" << token
           << "' — expected one of auto | cpu | l0[:N] | hip[:N] | cuda[:N]";
        throw std::runtime_error(os.str());
    }
    const auto kindOpt = BackendRegistry::parseKind(parts->kindName);
    if (!kindOpt) {
        std::ostringstream os;
        os << "BackendPool::selectByToken: unknown backend name '"
           << parts->kindName << "' in token '" << token << "'";
        throw std::runtime_error(os.str());
    }
    for (auto& e : _entries) {
        if (e.kind == *kindOpt && e.deviceIx == parts->deviceIx) {
            return e;
        }
    }
    // Compose a helpful "here's what IS in the pool" message so the
    // operator sees why the token failed to resolve.
    std::ostringstream os;
    os << "BackendPool::selectByToken: token '" << token
       << "' does not match any pool entry. Available: [";
    bool first = true;
    for (const auto& e : _entries) {
        if (!first) os << ", ";
        os << e.token;
        first = false;
    }
    os << "]";
    throw std::runtime_error(os.str());
}

// ---- tokenFor helper ------------------------------------------------------

std::string tokenFor(BackendKind kind, std::size_t deviceIx) noexcept {
    switch (kind) {
        case BackendKind::LevelZero:
            return "l0:" + std::to_string(deviceIx);
        case BackendKind::Hip:
            return "hip:" + std::to_string(deviceIx);
        case BackendKind::Cuda:
            return "cuda:" + std::to_string(deviceIx);
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Unknown:
            return "unknown";
    }
    return "unknown";
}

} // namespace mimirmind::core::backend

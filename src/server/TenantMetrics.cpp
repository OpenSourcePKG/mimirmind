// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/TenantMetrics.hpp"

#include "runtime/serving/ContinuousBatcher.hpp"
#include "core/log/Log.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

namespace {

std::int64_t nowUnix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/// Escape a tenant label for an OpenMetrics label value (backslash, double
/// quote, newline) per the exposition-format spec.
std::string escapeLabel(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            default:   out.push_back(c);
        }
    }
    return out;
}

} // namespace

TenantMetrics::TenantMetrics(std::string persistPath)
    : _persistPath{std::move(persistPath)} {
    if (_persistPath.empty()) {
        return;   // in-memory only
    }
    load();
    _running = true;
    _flusher = std::thread(&TenantMetrics::flushLoop, this);
    MM_LOG_INFO("metrics",
                "TenantMetrics: per-tenant accounting persisted to {} "
                "({} tenant row(s) restored)",
                _persistPath, _byTenant.size());
}

TenantMetrics::~TenantMetrics() {
    if (_persistPath.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _running = false;
    }
    _cv.notify_all();
    if (_flusher.joinable()) {
        _flusher.join();
    }
    // Final synchronous flush so shutdown never loses the last window.
    std::lock_guard<std::mutex> lk(_mtx);
    if (_dirty) {
        flushLocked();
    }
}

TenantMetrics::TenantStat& TenantMetrics::touchLocked(const std::string& tenant) {
    TenantStat& s = _byTenant[tenant];
    s.last_seen   = nowUnix();
    _dirty        = true;
    return s;
}

void TenantMetrics::recordSuccess(const std::string& tenant,
                                  std::uint64_t promptTokens,
                                  std::uint64_t completionTokens,
                                  double energyJoules, double prefillMs,
                                  double decodeMs) {
    std::lock_guard<std::mutex> lk(_mtx);
    TenantStat& s = touchLocked(tenant);
    s.requests_ok        += 1;
    s.prompt_tokens      += promptTokens;
    s.completion_tokens  += completionTokens;
    if (energyJoules > 0.0) s.energy_joules += energyJoules;
    s.prefill_ms         += prefillMs;
    s.decode_ms          += decodeMs;
    _cv.notify_all();
}

void TenantMetrics::recordQuotaRejected(const std::string& tenant) {
    std::lock_guard<std::mutex> lk(_mtx);
    touchLocked(tenant).rejected_quota += 1;
    _cv.notify_all();
}

void TenantMetrics::recordOverloadRejected(const std::string& tenant) {
    std::lock_guard<std::mutex> lk(_mtx);
    touchLocked(tenant).rejected_overload += 1;
    _cv.notify_all();
}

void TenantMetrics::recordError(const std::string& tenant) {
    std::lock_guard<std::mutex> lk(_mtx);
    touchLocked(tenant).errors += 1;
    _cv.notify_all();
}

nlohmann::json TenantMetrics::snapshotJson(
        const runtime::serving::ContinuousBatcher* batcher) const {
    std::lock_guard<std::mutex> lk(_mtx);
    json tenants = json::array();
    for (const auto& [id, s] : _byTenant) {
        const std::uint64_t total = s.requests_ok + s.rejected_quota +
                                    s.rejected_overload + s.errors;
        json row = {
            {"tenant",              id},
            {"requests_total",      total},
            {"requests_ok",         s.requests_ok},
            {"rejected_quota",      s.rejected_quota},
            {"rejected_overload",   s.rejected_overload},
            {"errors",              s.errors},
            {"prompt_tokens",       s.prompt_tokens},
            {"completion_tokens",   s.completion_tokens},
            {"total_tokens",        s.prompt_tokens + s.completion_tokens},
            {"energy_joules",       s.energy_joules},
            {"prefill_ms_total",    s.prefill_ms},
            {"decode_ms_total",     s.decode_ms},
            {"last_seen",           s.last_seen},
        };
        if (batcher != nullptr) {
            row["active_now"] = batcher->inflightForTenant(id);
        }
        tenants.push_back(std::move(row));
    }
    return json{{"object", "list"}, {"tenants", std::move(tenants)}};
}

std::string TenantMetrics::snapshotOpenMetrics(
        const runtime::serving::ContinuousBatcher* batcher) const {
    // Each metric family: one HELP + TYPE header, then one line per tenant.
    struct Family {
        const char*                                        name;
        const char*                                        type;
        const char*                                        help;
        std::function<std::string(const TenantStat&,
                                  const std::string&)>      value;
    };

    std::lock_guard<std::mutex> lk(_mtx);

    auto u = [](std::uint64_t v) { return std::to_string(v); };
    auto d = [](double v) { std::ostringstream o; o << v; return o.str(); };

    const std::vector<Family> fams = {
        {"mimirmind_tenant_requests_total", "counter",
         "Total chat requests accepted per tenant (all outcomes).",
         [&](const TenantStat& s, const std::string&) {
             return u(s.requests_ok + s.rejected_quota + s.rejected_overload +
                      s.errors);
         }},
        {"mimirmind_tenant_requests_ok_total", "counter",
         "Successfully completed (200) chat requests per tenant.",
         [&](const TenantStat& s, const std::string&) { return u(s.requests_ok); }},
        {"mimirmind_tenant_rejected_quota_total", "counter",
         "Requests shed by the per-tenant admission quota (429).",
         [&](const TenantStat& s, const std::string&) { return u(s.rejected_quota); }},
        {"mimirmind_tenant_rejected_overload_total", "counter",
         "Requests shed by whole-server overload / thermal (503).",
         [&](const TenantStat& s, const std::string&) { return u(s.rejected_overload); }},
        {"mimirmind_tenant_errors_total", "counter",
         "Requests that failed with a server error (500).",
         [&](const TenantStat& s, const std::string&) { return u(s.errors); }},
        {"mimirmind_tenant_prompt_tokens_total", "counter",
         "Prompt (input) tokens processed per tenant.",
         [&](const TenantStat& s, const std::string&) { return u(s.prompt_tokens); }},
        {"mimirmind_tenant_completion_tokens_total", "counter",
         "Completion (output) tokens generated per tenant.",
         [&](const TenantStat& s, const std::string&) { return u(s.completion_tokens); }},
        {"mimirmind_tenant_energy_joules_total", "counter",
         "Package energy (J) attributed to a tenant's completed requests.",
         [&](const TenantStat& s, const std::string&) { return d(s.energy_joules); }},
        {"mimirmind_tenant_prefill_ms_total", "counter",
         "Cumulative prefill wall-time (ms) per tenant.",
         [&](const TenantStat& s, const std::string&) { return d(s.prefill_ms); }},
        {"mimirmind_tenant_decode_ms_total", "counter",
         "Cumulative decode wall-time (ms) per tenant.",
         [&](const TenantStat& s, const std::string&) { return d(s.decode_ms); }},
        {"mimirmind_tenant_active_requests", "gauge",
         "In-flight (running + queued) requests per tenant right now.",
         [&](const TenantStat&, const std::string& id) {
             return u(batcher != nullptr ? batcher->inflightForTenant(id) : 0);
         }},
        {"mimirmind_tenant_last_seen_seconds", "gauge",
         "Unix time of the tenant's most recent request.",
         [&](const TenantStat& s, const std::string&) {
             return std::to_string(s.last_seen);
         }},
    };

    std::ostringstream out;
    for (const Family& f : fams) {
        out << "# HELP " << f.name << ' ' << f.help << '\n';
        out << "# TYPE " << f.name << ' ' << f.type << '\n';
        for (const auto& [id, s] : _byTenant) {
            out << f.name << "{tenant=\"" << escapeLabel(id) << "\"} "
                << f.value(s, id) << '\n';
        }
    }
    return out.str();
}

// ---- persistence ----------------------------------------------------------

void TenantMetrics::load() {
    std::ifstream in{_persistPath};
    if (!in) {
        return;   // first boot / no file yet — not an error
    }
    json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        MM_LOG_WARN("metrics",
                    "TenantMetrics: ignoring unreadable store {} ({})",
                    _persistPath, e.what());
        return;
    }
    if (!j.is_object() || !j.contains("tenants") || !j["tenants"].is_object()) {
        return;
    }
    for (const auto& [id, v] : j["tenants"].items()) {
        TenantStat s;
        s.requests_ok       = v.value("requests_ok",       std::uint64_t{0});
        s.rejected_quota    = v.value("rejected_quota",    std::uint64_t{0});
        s.rejected_overload = v.value("rejected_overload", std::uint64_t{0});
        s.errors            = v.value("errors",            std::uint64_t{0});
        s.prompt_tokens     = v.value("prompt_tokens",     std::uint64_t{0});
        s.completion_tokens = v.value("completion_tokens", std::uint64_t{0});
        s.energy_joules     = v.value("energy_joules",     0.0);
        s.prefill_ms        = v.value("prefill_ms",        0.0);
        s.decode_ms         = v.value("decode_ms",         0.0);
        s.last_seen         = v.value("last_seen",         std::int64_t{0});
        _byTenant.emplace(id, s);
    }
}

void TenantMetrics::flushLocked() const {
    json tenants = json::object();
    for (const auto& [id, s] : _byTenant) {
        tenants[id] = {
            {"requests_ok",       s.requests_ok},
            {"rejected_quota",    s.rejected_quota},
            {"rejected_overload", s.rejected_overload},
            {"errors",            s.errors},
            {"prompt_tokens",     s.prompt_tokens},
            {"completion_tokens", s.completion_tokens},
            {"energy_joules",     s.energy_joules},
            {"prefill_ms",        s.prefill_ms},
            {"decode_ms",         s.decode_ms},
            {"last_seen",         s.last_seen},
        };
    }
    const json j = {{"version", 1}, {"tenants", std::move(tenants)}};

    // Atomic: write a sibling temp file then rename over the target so a
    // reader (or a crash) never sees a half-written store.
    const std::string tmp = _persistPath + ".tmp";
    {
        std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
        if (!out) {
            MM_LOG_WARN("metrics",
                        "TenantMetrics: cannot write {} — metrics not persisted",
                        tmp);
            return;
        }
        out << j.dump(2);
        if (!out) {
            MM_LOG_WARN("metrics", "TenantMetrics: write to {} failed", tmp);
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, _persistPath, ec);
    if (ec) {
        MM_LOG_WARN("metrics",
                    "TenantMetrics: rename {} -> {} failed ({})",
                    tmp, _persistPath, ec.message());
        std::remove(tmp.c_str());
        return;
    }
    _dirty = false;
}

void TenantMetrics::flush() const {
    if (_persistPath.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(_mtx);
    if (_dirty) {
        flushLocked();
    }
}

void TenantMetrics::flushLoop() {
    constexpr auto kInterval = std::chrono::seconds(15);
    std::unique_lock<std::mutex> lk(_mtx);
    while (_running) {
        _cv.wait_for(lk, kInterval, [this] { return !_running; });
        if (_dirty) {
            flushLocked();
        }
    }
}

} // namespace mimirmind::server

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

// 8.20.1 — Prometheus /metrics registry (Bragi observability).
//
// A dependency-free, header-only, process-global metrics registry that emits
// the Prometheus text exposition format. Design note:
// [[Design — Prometheus /metrics Endpoint (Bragi Observability)]].
//
// Why header-only + singleton: the producers (ContinuousBatcher / slot
// lifecycle in runtime/serving, request completion in src/server) and the
// scrape handler (src/server/ApiServer) all reference ONE registry without a
// new link target — C++17 inline semantics give a single instance across TUs.
//
// Cost discipline: counters/histograms are updated at REQUEST / slot-lifecycle
// granularity (once per finished request, never per token), so the single
// coarse mutex is negligible and never touches the decode hot path. Gauges are
// PULL-based providers evaluated at scrape time — the live slot/KV state is
// owned by the batcher/allocator; we read it on /metrics, we do not mirror it.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::runtime::serving {

class MetricsRegistry {
public:
    // Ordered label set. Callers pass {{"model","qwen3.6"}} etc.; order is
    // normalised (sorted by key) so the same logical series maps to one entry.
    using Labels = std::vector<std::pair<std::string, std::string>>;

    static MetricsRegistry& instance() {
        static MetricsRegistry reg;
        return reg;
    }

    // --- one-time family declarations (help text + type). Idempotent. ---
    void declareCounter(const std::string& name, std::string help) {
        std::lock_guard<std::mutex> lk(_mu);
        auto& f = _counters[name];
        if (f.help.empty()) f.help = std::move(help);
    }
    void declareHistogram(const std::string& name, std::string help,
                          std::vector<double> buckets) {
        std::lock_guard<std::mutex> lk(_mu);
        auto& f = _histograms[name];
        if (f.help.empty()) {
            f.help    = std::move(help);
            f.buckets = std::move(buckets);   // strictly ascending upper bounds
        }
    }
    // Pull-gauge: `provider` is evaluated at scrape time (single unlabeled value).
    void registerGauge(std::string name, std::string help,
                       std::function<double()> provider) {
        std::lock_guard<std::mutex> lk(_mu);
        _gauges.push_back({std::move(name), std::move(help), std::move(provider)});
    }

    // --- updates (request / lifecycle granularity) ---
    void incCounter(const std::string& name, double v, Labels labels = {}) {
        if (v == 0.0) return;
        normalise(labels);
        std::lock_guard<std::mutex> lk(_mu);
        _counters[name].series[key(labels)].value += v;
        _counters[name].series[key(labels)].labels = labels;
    }
    void observe(const std::string& name, double v, Labels labels = {}) {
        normalise(labels);
        std::lock_guard<std::mutex> lk(_mu);
        auto  fit = _histograms.find(name);
        if (fit == _histograms.end()) return;   // must be declared (has buckets)
        auto& h   = fit->second;
        auto& s   = h.series[key(labels)];
        if (s.bucketCounts.empty()) {
            s.bucketCounts.assign(h.buckets.size(), 0);
            s.labels = labels;
        }
        // Per-bucket (non-cumulative) tally: bump only the smallest bucket whose
        // upper bound contains v; exposition() cumulates. Values above the last
        // bound fall into no bucket and are caught by the +Inf bucket (== count).
        for (std::size_t i = 0; i < h.buckets.size(); ++i) {
            if (v <= h.buckets[i]) { ++s.bucketCounts[i]; break; }
        }
        s.sum += v;
        ++s.count;
    }

    // --- Prometheus text exposition ---
    std::string exposition() const {
        std::lock_guard<std::mutex> lk(_mu);
        std::ostringstream os;
        os.setf(std::ios::fixed);
        // counters
        for (const auto& [name, fam] : _counters) {
            if (!fam.help.empty()) os << "# HELP " << name << ' ' << fam.help << '\n';
            os << "# TYPE " << name << " counter\n";
            for (const auto& [k, s] : fam.series) {
                (void)k;
                os << name << labelStr(s.labels) << ' ' << fmt(s.value) << '\n';
            }
        }
        // gauges (pull)
        for (const auto& g : _gauges) {
            if (!g.help.empty()) os << "# HELP " << g.name << ' ' << g.help << '\n';
            os << "# TYPE " << g.name << " gauge\n";
            double v = 0.0;
            if (g.provider) { try { v = g.provider(); } catch (...) { v = 0.0; } }
            os << g.name << ' ' << fmt(v) << '\n';
        }
        // histograms (cumulative buckets + _sum + _count, + le="+Inf")
        for (const auto& [name, h] : _histograms) {
            if (!h.help.empty()) os << "# HELP " << name << ' ' << h.help << '\n';
            os << "# TYPE " << name << " histogram\n";
            for (const auto& [k, s] : h.series) {
                (void)k;
                std::uint64_t cum = 0;
                for (std::size_t i = 0; i < h.buckets.size(); ++i) {
                    cum += s.bucketCounts[i];
                    os << name << "_bucket" << labelStr(withLe(s.labels, num(h.buckets[i])))
                       << ' ' << cum << '\n';
                }
                os << name << "_bucket" << labelStr(withLe(s.labels, "+Inf"))
                   << ' ' << s.count << '\n';
                os << name << "_sum"   << labelStr(s.labels) << ' ' << fmt(s.sum)   << '\n';
                os << name << "_count" << labelStr(s.labels) << ' ' << s.count << '\n';
            }
        }
        return os.str();
    }

    // Standard latency buckets (seconds) for TTFT / e2e.
    static std::vector<double> latencyBuckets() {
        return {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0};
    }

private:
    MetricsRegistry() = default;

    struct CounterSeries { Labels labels; double value{0.0}; };
    struct CounterFamily { std::string help; std::map<std::string, CounterSeries> series; };
    struct HistSeries   { Labels labels; std::vector<std::uint64_t> bucketCounts; double sum{0.0}; std::uint64_t count{0}; };
    struct HistFamily   { std::string help; std::vector<double> buckets; std::map<std::string, HistSeries> series; };
    struct GaugeEntry   { std::string name; std::string help; std::function<double()> provider; };

    static void normalise(Labels& l) {
        std::sort(l.begin(), l.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    static std::string key(const Labels& l) {
        std::string k;
        for (const auto& [n, v] : l) { k += n; k += '\x1f'; k += v; k += '\x1e'; }
        return k;
    }
    static std::string labelStr(const Labels& l) {
        if (l.empty()) return {};
        std::string s = "{";
        for (std::size_t i = 0; i < l.size(); ++i) {
            if (i) s += ',';
            s += l[i].first; s += "=\""; s += escape(l[i].second); s += '"';
        }
        s += '}';
        return s;
    }
    static Labels withLe(Labels l, std::string le) {
        l.push_back({"le", std::move(le)});
        return l;   // le goes last; order is fine for exposition
    }
    static std::string escape(const std::string& v) {
        std::string o;
        for (char c : v) {
            if (c == '\\' || c == '"') { o += '\\'; o += c; }
            else if (c == '\n') { o += "\\n"; }
            else o += c;
        }
        return o;
    }
    static std::string fmt(double v) {
        std::ostringstream o; o << v; return o.str();
    }
    static std::string num(double v) {
        std::ostringstream o; o << v; return o.str();
    }

    mutable std::mutex                     _mu;
    std::map<std::string, CounterFamily>   _counters;
    std::map<std::string, HistFamily>      _histograms;
    std::vector<GaugeEntry>                _gauges;
};

} // namespace mimirmind::runtime::serving

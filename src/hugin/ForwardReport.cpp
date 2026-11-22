// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ForwardReport.hpp"

#include "hugin/LogitMath.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mimirmind::hugin {

using nlohmann::json;

namespace {

/// Top-k positions of `w` restricted to the attended keys [lo, hi): a key the
/// sliding window masks has weight 0 by construction and is not an attention
/// edge, so it must never be listed (also not when k exceeds the window).
std::vector<std::size_t> topInRange(const std::vector<double>& w, std::size_t lo, std::size_t hi,
                                    std::size_t k) {
    std::vector<std::size_t> out;
    if (lo >= hi) {
        return out;
    }
    const std::span<const double> view(w.data() + lo, hi - lo);
    for (const auto j : LogitMath::topK(view, k)) {
        out.push_back(lo + static_cast<std::size_t>(j));
    }
    return out;
}

} // namespace

json ForwardReport::attention(const AttentionStages& s, std::span<const std::int32_t> ids,
                              const Piece& piece, std::uint32_t block, const AttentionOptions& o) {
    if (s.q == nullptr || s.k == nullptr || s.v == nullptr || s.ctx == nullptr) {
        throw std::runtime_error("attention stages missing");
    }
    const Stage& Q = *s.q;
    const Stage& K = *s.k;
    const Stage& V = *s.v;
    const Stage& C = *s.ctx;
    // Per-layer geometry from the captured shapes (GQA width and head dim may
    // differ per layer, e.g. kolibri1 SWA vs full layers).
    const std::size_t H = s.headCount;
    if (H == 0 || C.cols % H != 0) {
        throw std::runtime_error("attn_ctx width is not a multiple of the head count");
    }
    const std::size_t hd  = C.cols / H;
    const std::size_t Hkv = hd > 0 ? K.cols / hd : 0;
    if (Q.cols != H * hd || Hkv == 0 || K.cols != Hkv * hd || V.cols != Hkv * hd ||
        H % Hkv != 0 || Q.rows != ids.size() || K.rows < ids.size() || V.rows < ids.size()) {
        throw std::runtime_error("unexpected stage shapes for attention_pattern");
    }
    const std::size_t pos = o.position;
    if (pos >= ids.size()) {
        throw std::runtime_error("position out of range");
    }
    std::size_t spanLo = 0, spanHi = 0;
    if (o.span) {
        spanLo = o.span->first;
        spanHi = o.span->second;
        if (spanHi <= spanLo || spanHi > ids.size()) {
            throw std::runtime_error("'span' must be [start, end) with 0 <= start < end <= tokens");
        }
    }
    std::vector<std::size_t> heads = o.heads;
    if (heads.empty()) {
        for (std::size_t h = 0; h < H; ++h) heads.push_back(h);
    }
    for (const auto h : heads) {
        if (h >= H) throw std::runtime_error("head " + std::to_string(h) + " out of range");
    }

    const std::size_t n = pos + 1;   // causal: keys 0..pos
    // Sliding window as the kernel applies it: keys [n - window, n).
    const std::size_t kMin = (s.window > 0 && n > s.window) ? n - s.window : 0;
    double maxErr = 0, maxRef = 0;
    std::vector<double> avg(n, 0.0);
    json out = json::array();
    std::vector<std::pair<double, std::size_t>> bySpan;
    for (const auto h : heads) {
        const std::size_t kvh = h / (H / Hkv);
        const float* q = Q.data.data() + pos * Q.cols + h * hd;
        std::vector<double> w(n, 0.0);
        double mx = -1e300;
        for (std::size_t j = kMin; j < n; ++j) {
            const float* kr = K.data.data() + j * K.cols + kvh * hd;
            double dot = 0;
            for (std::size_t d = 0; d < hd; ++d) dot += static_cast<double>(q[d]) * static_cast<double>(kr[d]);
            w[j] = dot * s.scale;
            mx = std::max(mx, w[j]);
        }
        double z = 0;
        for (std::size_t j = kMin; j < n; ++j) { w[j] = std::exp(w[j] - mx); z += w[j]; }
        double ent = 0, spanW = 0;
        for (std::size_t j = kMin; j < n; ++j) {
            w[j] /= z;
            avg[j] += w[j] / static_cast<double>(heads.size());
            if (w[j] > 0) ent -= w[j] * std::log(w[j]);
            if (o.span && j >= spanLo && j < spanHi) spanW += w[j];
        }
        for (std::size_t d = 0; d < hd; ++d) {
            double ctx = 0;
            for (std::size_t j = 0; j < n; ++j) ctx += w[j] * static_cast<double>(V.data[j * V.cols + kvh * hd + d]);
            const double ref = C.data[pos * C.cols + h * hd + d];
            maxErr = std::max(maxErr, std::fabs(ctx - ref));
            maxRef = std::max(maxRef, std::fabs(ref));
        }
        json top = json::array();
        for (const auto j : topInRange(w, kMin, n, o.topK)) {
            top.push_back({{"pos", j}, {"token", piece(ids[j])}, {"weight", w[j]}});
        }
        json e{{"head", h}, {"kv_head", kvh}, {"entropy", ent}, {"bos_weight", w[0]},
               {"self_weight", w[pos]}, {"top", top}};
        if (o.span) {
            e["span_weight"] = spanW;
            bySpan.emplace_back(spanW, h);
        }
        if (o.full) e["weights"] = w;
        out.push_back(std::move(e));
    }
    json avgTop = json::array();
    for (const auto j : topInRange(avg, kMin, n, o.topK)) {
        avgTop.push_back({{"pos", j}, {"token", piece(ids[j])}, {"weight", avg[j]}});
    }
    json res{{"block", block}, {"position", pos}, {"query_token", piece(ids[pos])},
             {"heads", H}, {"kv_heads", Hkv}, {"head_dim", hd}, {"scale", s.scale},
             {"window", s.window},
             {"consistency", {{"max_abs_err", maxErr}, {"max_abs_ref", maxRef}}},
             {"head_average_top", avgTop}, {"per_head", out}};
    if (o.span) {
        std::sort(bySpan.begin(), bySpan.end(), std::greater<>{});
        json ranked = json::array();
        for (const auto& [wgt, h] : bySpan) ranked.push_back({{"head", h}, {"span_weight", wgt}});
        res["span"] = {spanLo, spanHi};
        res["heads_by_span_weight"] = ranked;
    }
    return res;
}

json ForwardReport::residual(std::uint32_t block, std::span<const float> x,
                             std::optional<std::span<const float>> prev, std::size_t topDims,
                             bool values) {
    const std::size_t d = x.size();
    // Statistics over finite values; non-finite ones are counted (a NaN/Inf
    // residual is itself a finding worth reporting).
    double sum = 0, sq = 0, amax = 0;
    std::size_t nonFinite = 0;
    std::vector<double> mag(d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        const double v = x[i];
        if (!std::isfinite(v)) {
            ++nonFinite;
            continue;
        }
        sum += v;
        sq += v * v;
        mag[i] = std::fabs(v);
        amax   = std::max(amax, mag[i]);
    }
    const double finiteN = static_cast<double>(std::max<std::size_t>(1, d - nonFinite));
    const double mean = sum / finiteN;
    json dims = json::array();
    for (const auto i : LogitMath::topK(mag, topDims)) {
        dims.push_back({{"dim", i}, {"value", x[static_cast<std::size_t>(i)]}});
    }
    json o{{"block", block}, {"l2", std::sqrt(sq)}, {"mean", mean},
           {"std", std::sqrt(std::max(0.0, sq / finiteN - mean * mean))},
           {"abs_max", amax}, {"non_finite", nonFinite}, {"top_dims", dims}};
    if (prev) {
        if (prev->size() != d) {
            throw std::runtime_error("residual: previous block has a different width");
        }
        double dsq = 0, dot = 0, psq = 0;
        for (std::size_t i = 0; i < d; ++i) {
            const double xv = x[i];
            const double pv = (*prev)[i];
            dsq += (xv - pv) * (xv - pv);
            dot += xv * pv;
            psq += pv * pv;
        }
        o["update_l2"]      = std::sqrt(dsq);
        o["cos_prev_block"] = (sq > 0 && psq > 0) ? dot / std::sqrt(sq * psq) : 0.0;
    }
    if (values) {
        o["values"] = std::vector<float>(x.begin(), x.end());
    }
    return o;
}

json ForwardReport::expertBlock(std::uint32_t block, const Stage& idx, const Stage* weight,
                                std::size_t experts, std::size_t topN,
                                std::optional<std::size_t> position,
                                std::vector<double>& totalCount) {
    const std::size_t T = idx.rows, K = idx.cols;
    if (totalCount.size() < experts) {
        totalCount.resize(experts, 0.0);
    }
    std::vector<double> count(experts, 0.0), wsum(experts, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            const auto ex = static_cast<std::size_t>(idx.data[t * K + k]);
            if (ex >= experts) {
                throw std::runtime_error("expert id out of range in block " + std::to_string(block));
            }
            count[ex] += 1.0;
            totalCount[ex] += 1.0;
            if (weight != nullptr) {
                wsum[ex] += static_cast<double>(weight->data[t * K + k]);
            }
        }
    }
    std::size_t distinct = 0;
    double ent = 0;
    const double picks = static_cast<double>(T * K);
    for (const double c : count) {
        if (c > 0) {
            ++distinct;
            ent -= (c / picks) * std::log(c / picks);
        }
    }
    json byCount = json::array(), byWeight = json::array();
    for (const auto ex : LogitMath::topK(count, topN)) {
        if (count[static_cast<std::size_t>(ex)] > 0)
            byCount.push_back({{"expert", ex}, {"tokens", count[static_cast<std::size_t>(ex)]}});
    }
    for (const auto ex : LogitMath::topK(wsum, topN)) {
        if (wsum[static_cast<std::size_t>(ex)] > 0)
            byWeight.push_back({{"expert", ex}, {"weight", wsum[static_cast<std::size_t>(ex)]}});
    }
    json b{{"block", block}, {"tokens", T}, {"k", K}, {"distinct_experts", distinct},
           {"load_entropy", ent},
           {"max_entropy", std::log(static_cast<double>(std::max<std::size_t>(1, std::min<std::size_t>(experts, T * K))))},
           {"top_by_count", byCount}, {"top_by_weight", byWeight}};
    if (position) {
        if (*position >= T) {
            throw std::runtime_error("position out of range for the routing tap");
        }
        json picksAt = json::array();
        for (std::size_t k = 0; k < K; ++k) {
            json p{{"expert", static_cast<std::int64_t>(idx.data[*position * K + k])}};
            if (weight != nullptr) p["weight"] = weight->data[*position * K + k];
            picksAt.push_back(std::move(p));
        }
        b["picks_at_position"] = picksAt;
    }
    return b;
}

json ForwardReport::expertSummary(const std::vector<double>& totalCount, std::size_t topN) {
    json overall = json::array();
    for (const auto ex : LogitMath::topK(totalCount, topN)) {
        overall.push_back({{"expert", ex}, {"tokens", totalCount[static_cast<std::size_t>(ex)]}});
    }
    std::size_t neverUsed = 0;
    for (const double c : totalCount) neverUsed += c == 0 ? 1U : 0U;
    return json{{"never_used_experts", neverUsed}, {"top_overall", overall}};
}

} // namespace mimirmind::hugin

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/DenseModel.hpp"

#include "hugin/train/CrossEntropyGrad.hpp"
#include "hugin/train/KlDivGrad.hpp"
#include "hugin/train/LinearGrad.hpp"
#include "hugin/train/RmsNormGrad.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin::train {

DenseModel::DenseModel(std::vector<DenseBlock> blocks, Head head)
    : _blocks(std::move(blocks)), _head(std::move(head)) {
    if (_blocks.empty()) throw std::runtime_error("DenseModel: no blocks");
    _d = _blocks.front().config().dModel;
    if (_head.vocab == 0 || _head.embed.size() != _head.vocab * _d) {
        throw std::runtime_error("DenseModel: embedding must be [vocab, dModel]");
    }
    if (_head.finalNorm.size() != _d) throw std::runtime_error("DenseModel: final norm must be [dModel]");
    if (!_head.lmHead.empty() && _head.lmHead.size() != _head.vocab * _d) {
        throw std::runtime_error("DenseModel: lm_head must be [vocab, dModel]");
    }
}

void DenseModel::embed(std::span<const std::int32_t> ids, float* x) const {
    for (std::size_t t = 0; t < ids.size(); ++t) {
        if (ids[t] < 0 || static_cast<std::size_t>(ids[t]) >= _head.vocab) {
            throw std::runtime_error("DenseModel: token id " + std::to_string(ids[t]) + " out of range");
        }
        const float* row = _head.embed.data() + static_cast<std::size_t>(ids[t]) * _d;
        std::copy(row, row + _d, x + t * _d);
    }
}

std::vector<float> DenseModel::logits(std::span<const std::int32_t> ids,
                                      std::span<const std::size_t> rows) const {
    const std::size_t T = ids.size();
    std::vector<float> x(T * _d), y(T * _d);
    embed(ids, x.data());
    for (const auto& b : _blocks) {
        b.forward(x.data(), T, y.data());
        x.swap(y);
    }
    std::vector<std::size_t> sel(rows.begin(), rows.end());
    if (sel.empty()) {
        for (std::size_t t = 0; t < T; ++t) sel.push_back(t);
    }
    std::vector<float> xr(sel.size() * _d), n(sel.size() * _d), out(sel.size() * _head.vocab);
    for (std::size_t i = 0; i < sel.size(); ++i) {
        std::copy(x.begin() + static_cast<std::ptrdiff_t>(sel[i] * _d),
                  x.begin() + static_cast<std::ptrdiff_t>((sel[i] + 1) * _d), xr.begin() + static_cast<std::ptrdiff_t>(i * _d));
    }
    RmsNormGrad::forward(xr.data(), sel.size(), _d, _head.finalNorm.data(), _head.eps, n.data());
    LinearGrad::forward(n.data(), sel.size(), _d, lmHead().data(), _head.vocab, nullptr, out.data());
    return out;
}

double DenseModel::lossAndGrad(std::span<const std::int32_t> ids, std::span<const std::int32_t> targets) {
    return lossAndGrad(ids, targets, nullptr, 0.0).ce;
}

DenseModel::Anchor DenseModel::anchor(std::span<const std::int32_t> ids, std::span<const std::size_t> rows,
                                      std::size_t k) const {
    if (rows.empty()) throw std::runtime_error("DenseModel: anchor needs rows");
    Anchor a;
    a.rows.assign(rows.begin(), rows.end());
    a.k = k;
    a.ids.resize(rows.size() * k);
    a.p.resize(rows.size() * k);
    a.tail.resize(rows.size());
    const auto lg = logits(ids, rows);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        KlDivGrad::topK(lg.data() + i * _head.vocab, _head.vocab, k, a.ids.data() + i * k, a.p.data() + i * k,
                        a.tail[i]);
    }
    return a;
}

DenseModel::Loss DenseModel::lossAndGrad(std::span<const std::int32_t> ids, std::span<const std::int32_t> targets,
                                         const Anchor* anchor, double klWeight) {
    const std::size_t T = ids.size();
    if (targets.size() != T) throw std::runtime_error("DenseModel: targets must match ids");
    // Union of CE rows and anchor rows; tgt = -1 on anchor-only rows.
    std::vector<std::int32_t> rowTarget(T, -1);
    std::vector<char> used(T, 0);
    Loss out;
    for (std::size_t t = 0; t < T; ++t) {
        if (targets[t] >= 0) {
            used[t] = 1;
            rowTarget[t] = targets[t];
            ++out.ceRows;
        }
    }
    const bool withKl = anchor != nullptr && klWeight != 0.0 && !anchor->rows.empty();
    if (withKl) {
        if (anchor->ids.size() != anchor->rows.size() * anchor->k || anchor->p.size() != anchor->ids.size() ||
            anchor->tail.size() != anchor->rows.size()) {
            throw std::runtime_error("DenseModel: malformed anchor");
        }
        for (const auto r : anchor->rows) {
            if (r >= T) throw std::runtime_error("DenseModel: anchor row out of range");
            used[r] = 1;
        }
        out.klRows = anchor->rows.size();
    }
    std::vector<std::size_t> rows;
    std::vector<std::int32_t> tgt;
    std::vector<std::size_t> slot(T, 0);   // sequence row -> union index
    for (std::size_t t = 0; t < T; ++t) {
        if (used[t] != 0) {
            slot[t] = rows.size();
            rows.push_back(t);
            tgt.push_back(rowTarget[t]);
        }
    }
    if (rows.empty()) return out;

    // Lowest block with an adapter: backward stops there.
    std::size_t lowest = _blocks.size();
    for (std::size_t b = 0; b < _blocks.size() && lowest == _blocks.size(); ++b) {
        for (const auto p : {DenseBlock::Proj::Q, DenseBlock::Proj::K, DenseBlock::Proj::V, DenseBlock::Proj::O,
                             DenseBlock::Proj::Gate, DenseBlock::Proj::Up, DenseBlock::Proj::Down}) {
            if (_blocks[b].lora(p) != nullptr) {
                lowest = b;
                break;
            }
        }
    }

    // Forward, keeping the block inputs from `lowest` on.
    std::vector<std::vector<float>> inputs(_blocks.size());
    std::vector<float> x(T * _d), y(T * _d);
    embed(ids, x.data());
    for (std::size_t b = 0; b < _blocks.size(); ++b) {
        if (b >= lowest) inputs[b] = x;
        _blocks[b].forward(x.data(), T, y.data());
        x.swap(y);
    }

    // Head on the used rows only.
    const std::size_t R = rows.size(), V = _head.vocab;
    std::vector<float> xr(R * _d), n(R * _d), lg(R * V), dLg(R * V);
    for (std::size_t i = 0; i < R; ++i) {
        std::copy(x.begin() + static_cast<std::ptrdiff_t>(rows[i] * _d),
                  x.begin() + static_cast<std::ptrdiff_t>((rows[i] + 1) * _d), xr.begin() + static_cast<std::ptrdiff_t>(i * _d));
    }
    RmsNormGrad::forward(xr.data(), R, _d, _head.finalNorm.data(), _head.eps, n.data());
    LinearGrad::forward(n.data(), R, _d, lmHead().data(), V, nullptr, lg.data());
    out.ce = CrossEntropyGrad::lossAndGrad(lg.data(), tgt.data(), R, V, dLg.data());
    if (withKl) {
        const auto  K = anchor->k;
        const float w = static_cast<float>(klWeight / static_cast<double>(anchor->rows.size()));
        double sum = 0;
        for (std::size_t i = 0; i < anchor->rows.size(); ++i) {
            const std::size_t u = slot[anchor->rows[i]];
            sum += KlDivGrad::rowLossAndGrad(lg.data() + u * V, V, K, anchor->ids.data() + i * K,
                                             anchor->p.data() + i * K, anchor->tail[i], w, dLg.data() + u * V);
        }
        out.kl = sum / static_cast<double>(anchor->rows.size());
    }
    if (lowest == _blocks.size()) return out;   // no adapters: loss only

    std::vector<float> dN(R * _d), dXr(R * _d);
    LinearGrad::backward(dLg.data(), n.data(), R, _d, lmHead().data(), V, nullptr, dN.data(), nullptr, nullptr);
    RmsNormGrad::backward(dN.data(), xr.data(), R, _d, _head.finalNorm.data(), _head.eps, dXr.data(), nullptr);
    std::vector<float> dx(T * _d, 0.0F), dPrev(T * _d);
    for (std::size_t i = 0; i < R; ++i) {
        std::copy(dXr.begin() + static_cast<std::ptrdiff_t>(i * _d), dXr.begin() + static_cast<std::ptrdiff_t>((i + 1) * _d),
                  dx.begin() + static_cast<std::ptrdiff_t>(rows[i] * _d));
    }
    for (std::size_t b = _blocks.size(); b-- > lowest;) {
        _blocks[b].backward(inputs[b].data(), T, dx.data(), dPrev.data());
        dx.swap(dPrev);
    }
    return out;
}

void DenseModel::zeroGrad() {
    for (auto& b : _blocks) b.zeroGrad();
}

} // namespace mimirmind::hugin::train

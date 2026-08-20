// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/SnacDecoder.hpp"

#include "compute/quant/Float16.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mimirmind::runtime::audio {

namespace {

// A 1-D signal as [channels, time], row-major: x[c * T + t].
struct Signal {
    std::vector<float> data;
    std::size_t        c{0};
    std::size_t        t{0};
    float&       at(std::size_t ch, std::size_t ti)       { return data[ch * t + ti]; }
    float        at(std::size_t ch, std::size_t ti) const { return data[ch * t + ti]; }
};

// Conv1d, PyTorch semantics. weight [Cout, Cin/groups, K] row-major, bias
// [Cout] (may be empty). Returns [Cout, Tout].
Signal conv1d(const Signal& x, const std::vector<float>& w,
              const std::vector<float>& bias, std::size_t cout, std::size_t k,
              std::size_t stride, std::size_t pad, std::size_t dil,
              std::size_t groups) {
    const std::size_t cinG  = x.c / groups;
    const std::size_t coutG = cout / groups;
    const long        T     = static_cast<long>(x.t);
    const long        eff   = static_cast<long>(dil * (k - 1));
    const std::size_t tout  =
        static_cast<std::size_t>((static_cast<long>(x.t) + 2L * static_cast<long>(pad)
                                  - eff - 1L) / static_cast<long>(stride) + 1L);
    Signal out{std::vector<float>(cout * tout, 0.0F), cout, tout};

    for (std::size_t co = 0; co < cout; ++co) {
        const std::size_t g   = co / coutG;
        const float       b   = bias.empty() ? 0.0F : bias[co];
        for (std::size_t ot = 0; ot < tout; ++ot) {
            float acc = b;
            const long base = static_cast<long>(ot * stride) - static_cast<long>(pad);
            for (std::size_t cl = 0; cl < cinG; ++cl) {
                const std::size_t ci = g * cinG + cl;
                const float* wp = &w[(co * cinG + cl) * k];
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const long it = base + static_cast<long>(kk * dil);
                    if (it >= 0 && it < T) {
                        acc += x.at(ci, static_cast<std::size_t>(it)) * wp[kk];
                    }
                }
            }
            out.at(co, ot) = acc;
        }
    }
    return out;
}

// ConvTranspose1d, PyTorch semantics, groups = 1. weight [Cin, Cout, K]
// row-major, bias [Cout]. Returns [Cout, Tout].
Signal convTranspose1d(const Signal& x, const std::vector<float>& w,
                       const std::vector<float>& bias, std::size_t cout,
                       std::size_t k, std::size_t stride, std::size_t pad,
                       std::size_t outPad) {
    const long tout = static_cast<long>((x.t - 1) * stride) - 2L * static_cast<long>(pad)
                      + static_cast<long>(k - 1) + static_cast<long>(outPad) + 1L;
    const std::size_t toutU = static_cast<std::size_t>(tout);
    Signal out{std::vector<float>(cout * toutU, 0.0F), cout, toutU};

    for (std::size_t ci = 0; ci < x.c; ++ci) {
        for (std::size_t ti = 0; ti < x.t; ++ti) {
            const float xv = x.at(ci, ti);
            if (xv == 0.0F) continue;
            const long base = static_cast<long>(ti * stride) - static_cast<long>(pad);
            for (std::size_t co = 0; co < cout; ++co) {
                const float* wp = &w[(ci * cout + co) * k];
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const long ot = base + static_cast<long>(kk);
                    if (ot >= 0 && ot < tout) {
                        out.at(co, static_cast<std::size_t>(ot)) += xv * wp[kk];
                    }
                }
            }
        }
    }
    if (!bias.empty()) {
        for (std::size_t co = 0; co < cout; ++co) {
            for (std::size_t ot = 0; ot < toutU; ++ot) out.at(co, ot) += bias[co];
        }
    }
    return out;
}

// Snake activation in place: x += 1/(alpha+1e-9) * sin(alpha*x)^2, per channel.
void snake(Signal& x, const std::vector<float>& alpha) {
    for (std::size_t c = 0; c < x.c; ++c) {
        const float a   = alpha[c];
        const float inv = 1.0F / (a + 1e-9F);
        for (std::size_t ti = 0; ti < x.t; ++ti) {
            const float s = std::sin(a * x.at(c, ti));
            x.at(c, ti) += inv * s * s;
        }
    }
}

} // namespace

std::vector<float>
SnacDecoder::conv1dTest(const std::vector<float>& x, std::size_t cin,
                        std::size_t T, const std::vector<float>& w,
                        const std::vector<float>& bias, std::size_t cout,
                        std::size_t k, std::size_t stride, std::size_t pad,
                        std::size_t dil, std::size_t groups, std::size_t& tout) {
    Signal xs{x, cin, T};
    Signal r = conv1d(xs, w, bias, cout, k, stride, pad, dil, groups);
    tout = r.t;
    return r.data;
}

std::vector<float>
SnacDecoder::convTranspose1dTest(const std::vector<float>& x, std::size_t cin,
                                 std::size_t T, const std::vector<float>& w,
                                 const std::vector<float>& bias,
                                 std::size_t cout, std::size_t k,
                                 std::size_t stride, std::size_t pad,
                                 std::size_t outPad, std::size_t& tout) {
    Signal xs{x, cin, T};
    Signal r = convTranspose1d(xs, w, bias, cout, k, stride, pad, outPad);
    tout = r.t;
    return r.data;
}

std::vector<float>
SnacDecoder::snakeTest(std::vector<float> x, std::size_t c, std::size_t T,
                       const std::vector<float>& alpha) {
    Signal xs{std::move(x), c, T};
    snake(xs, alpha);
    return xs.data;
}

const SnacDecoder::HostTensor& SnacDecoder::t(const std::string& name) const {
    const auto it = _w.find(name);
    if (it == _w.end()) {
        throw std::runtime_error("SnacDecoder: missing tensor '" + name + "'");
    }
    return it->second;
}

void SnacDecoder::load(std::string_view path, bool noise) {
    _noise = noise;
    _w.clear();

    core::safetensors::SafetensorsModel sm;
    sm.open(path);

    for (const auto* meta : sm.tensors()) {
        HostTensor ht;
        ht.shape.assign(meta->shape.begin(), meta->shape.end());
        ht.data.resize(static_cast<std::size_t>(meta->nelements));
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(meta->name);
        switch (meta->dtype) {
        case core::safetensors::SafetensorsDtype::F32:
            std::memcpy(ht.data.data(), bytes.data(),
                        ht.data.size() * sizeof(float));
            break;
        case core::safetensors::SafetensorsDtype::F16:
            compute::quant::Float16::instance().dequantToF32(
                bytes.data(), ht.data.size(), ht.data.data());
            break;
        default:
            throw std::runtime_error("SnacDecoder: tensor '" + meta->name +
                                     "' has unsupported dtype (expected F32/F16)");
        }
        _w.emplace(meta->name, std::move(ht));
    }

    // Presence check on the load-bearing tensors so a bad checkpoint fails at
    // load, not mid-decode.
    (void)t("dec.in.dw.weight");
    (void)t("dec.in.pw.weight");
    (void)t("dec.out.conv.weight");
    for (int i = 0; i < 3; ++i) {
        (void)t("q." + std::to_string(i) + ".codebook");
        (void)t("q." + std::to_string(i) + ".out_proj.weight");
    }
    _loaded = true;
}

std::vector<float>
SnacDecoder::decode(const std::vector<std::vector<std::int32_t>>& codes) const {
    if (!_loaded) throw std::runtime_error("SnacDecoder: not loaded");
    if (codes.size() != 3) {
        throw std::runtime_error("SnacDecoder: expected 3 code levels, got " +
                                 std::to_string(codes.size()));
    }
    // Common (fine) frame count T; each level i has T/stride[i] entries.
    const std::size_t T = codes[2].size();
    if (T == 0) return {};
    for (int i = 0; i < 3; ++i) {
        const std::size_t s = static_cast<std::size_t>(kStrides[i]);
        if (codes[static_cast<std::size_t>(i)].size() * s != T) {
            throw std::runtime_error(
                "SnacDecoder: code level " + std::to_string(i) +
                " length inconsistent with vq_stride");
        }
    }

    // ---- quantizer.from_codes -> z_q [768, T] ----
    Signal z{std::vector<float>(static_cast<std::size_t>(kLatentDim) * T, 0.0F),
             static_cast<std::size_t>(kLatentDim), T};
    for (int i = 0; i < 3; ++i) {
        const auto&       code = codes[static_cast<std::size_t>(i)];
        const std::size_t s    = static_cast<std::size_t>(kStrides[i]);
        const std::size_t Ti   = code.size();
        const HostTensor& cb    = t("q." + std::to_string(i) + ".codebook");        // [4096,8]
        const HostTensor& opW   = t("q." + std::to_string(i) + ".out_proj.weight"); // [768,8,1]
        const HostTensor& opB   = t("q." + std::to_string(i) + ".out_proj.bias");   // [768]
        const std::size_t D     = static_cast<std::size_t>(kCodebookDim);

        for (std::size_t ti = 0; ti < Ti; ++ti) {
            const std::int32_t idx = code[ti];
            if (idx < 0 || static_cast<std::size_t>(idx) * D + D > cb.data.size()) {
                throw std::runtime_error("SnacDecoder: code index out of range");
            }
            const float* emb = &cb.data[static_cast<std::size_t>(idx) * D];
            // out_proj (k1): o[768] = bias + W[768,8] . emb[8]
            for (std::size_t o = 0; o < static_cast<std::size_t>(kLatentDim); ++o) {
                float acc = opB.data[o];
                const float* wr = &opW.data[o * D];
                for (std::size_t d = 0; d < D; ++d) acc += wr[d] * emb[d];
                // repeat_interleave by stride: write to [ti*s, ti*s + s).
                for (std::size_t j = 0; j < s; ++j) z.at(o, ti * s + j) += acc;
            }
        }
    }

    // ---- Decoder ----
    // in: depthwise conv (768->768,k7,g768) then pointwise (768->1024,k1)
    Signal x = conv1d(z, t("dec.in.dw.weight").data, t("dec.in.dw.bias").data,
                      768, 7, 1, 3, 1, 768);
    x = conv1d(x, t("dec.in.pw.weight").data, t("dec.in.pw.bias").data,
               1024, 1, 1, 0, 1, 1);

    for (int b = 0; b < 4; ++b) {
        const std::string p    = "dec.block." + std::to_string(b) + ".";
        const std::size_t s    = static_cast<std::size_t>(kDecRates[b]);
        const std::size_t outD = 1024u >> (b + 1);   // input dim is x.c (== 1024>>b)

        snake(x, t(p + "snake.alpha").data);
        x = convTranspose1d(x, t(p + "up.weight").data, t(p + "up.bias").data,
                            outD, 2 * s, s, static_cast<std::size_t>((s + 1) / 2),
                            s % 2);
        // NoiseBlock: x += randn * Conv1d(x). Deterministic build keeps noise
        // off (identity); enabled only when _noise is set (stochastic).
        // (Left as identity here; a seeded generator is a later quality step.)

        for (int r = 0; r < 3; ++r) {
            const std::size_t dil = (r == 0) ? 1u : (r == 1) ? 3u : 9u;
            const std::string rp  = p + "res." + std::to_string(r) + ".";
            Signal y = x;                         // residual copy
            snake(y, t(rp + "snake1.alpha").data);
            y = conv1d(y, t(rp + "dw.weight").data, t(rp + "dw.bias").data,
                       outD, 7, 1, (7u - 1u) * dil / 2u, dil, outD);
            snake(y, t(rp + "snake2.alpha").data);
            y = conv1d(y, t(rp + "pw.weight").data, t(rp + "pw.bias").data,
                       outD, 1, 1, 0, 1, 1);
            for (std::size_t n = 0; n < x.data.size(); ++n) x.data[n] += y.data[n];
        }
    }

    snake(x, t("dec.out.snake.alpha").data);
    x = conv1d(x, t("dec.out.conv.weight").data, t("dec.out.conv.bias").data,
               1, 7, 1, 3, 1, 1);

    // tanh -> mono waveform.
    std::vector<float> pcm(x.t);
    for (std::size_t ti = 0; ti < x.t; ++ti) pcm[ti] = std::tanh(x.at(0, ti));
    return pcm;
}

} // namespace mimirmind::runtime::audio

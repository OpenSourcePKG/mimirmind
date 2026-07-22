// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/GatedDeltaNet.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mimirmind::compute {

void gatedDeltaNetRecurrent(const float* q,
                            const float* k,
                            const float* v,
                            const float* gLog,
                            const float* beta,
                            float*       state,
                            float*       out,
                            std::size_t  T,
                            std::size_t  H,
                            std::size_t  S) {
    const float qScale = 1.0F / std::sqrt(static_cast<float>(S));

    // Per-token scratch reused across heads: the predicted value s^T k and
    // the gated error d, both length S.
    std::vector<float> sk(S);
    std::vector<float> d(S);
    std::vector<float> qs(S);

    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            const float* qt = q + (t * H + h) * S;
            const float* kt = k + (t * H + h) * S;
            const float* vt = v + (t * H + h) * S;
            float*       s  = state + h * S * S;   // s[i*S + j]
            float*       ot = out + (t * H + h) * S;

            const float gDecay = std::exp(gLog[t * H + h]);
            const float b       = beta[t * H + h];

            for (std::size_t i = 0; i < S; ++i) {
                qs[i] = qt[i] * qScale;
            }

            // Decay the state, then predict sk[j] = sum_i s[i,j] * k[i]
            // from the decayed state. Fuse both passes over i.
            for (std::size_t j = 0; j < S; ++j) {
                sk[j] = 0.0F;
            }
            for (std::size_t i = 0; i < S; ++i) {
                const float ki = kt[i];
                float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    srow[j] *= gDecay;
                    sk[j]   += srow[j] * ki;
                }
            }

            // Gated prediction error d[j] = (v[j] - sk[j]) * beta.
            for (std::size_t j = 0; j < S; ++j) {
                d[j] = (vt[j] - sk[j]) * b;
            }

            // Rank-1 update s[i,j] += k[i] * d[j], and read out from the
            // UPDATED state o[j] = sum_i s[i,j] * qs[i]. Two passes: the
            // readout must see the full updated matrix.
            for (std::size_t i = 0; i < S; ++i) {
                const float ki = kt[i];
                float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    srow[j] += ki * d[j];
                }
            }
            for (std::size_t j = 0; j < S; ++j) {
                ot[j] = 0.0F;
            }
            for (std::size_t i = 0; i < S; ++i) {
                const float qi = qs[i];
                const float* srow = s + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    ot[j] += srow[j] * qi;
                }
            }
        }
    }
}

void gatedDeltaNetChunk(const float* q,
                        const float* k,
                        const float* v,
                        const float* gLog,
                        const float* beta,
                        float*       state,
                        float*       out,
                        std::size_t  T,
                        std::size_t  H,
                        std::size_t  S,
                        std::size_t  chunkSize) {
    const float qScale = 1.0F / std::sqrt(static_cast<float>(S));
    if (chunkSize == 0) {
        chunkSize = 64;
    }
    const std::size_t C = chunkSize;

    // Per-chunk / per-head scratch, all sized for the maximum chunk width.
    std::vector<float> s0(S * S);      // chunk-start state snapshot [i*S + j]
    std::vector<float> gcum(C);        // cumulative log-decay within chunk
    std::vector<float> egc(C);         // exp(gcum)
    std::vector<float> u(C * S);       // (S0^T k_a)[j]        row a
    std::vector<float> uq(C * S);      // (S0^T qs_a)[j]       row a
    std::vector<float> qs(C * S);      // scaled query         row a
    std::vector<float> amat(C * C);    // strict-lower gated Gram A[a,m]
    std::vector<float> kq(C * C);      // (k_m . qs_a)         [a*C + m]
    std::vector<float> d(C * S);       // solved delta vectors row a

    for (std::size_t h = 0; h < H; ++h) {
        float* st = state + h * S * S;

        for (std::size_t c0 = 0; c0 < T; c0 += C) {
            const std::size_t cs = std::min(C, T - c0);   // this chunk's width

            // Snapshot the chunk-start state: both the RHS and the readout
            // reference S0 while the update below overwrites `st`.
            std::copy(st, st + S * S, s0.begin());

            // Cumulative (inclusive) log-decay and its exp within the chunk.
            float run = 0.0F;
            for (std::size_t a = 0; a < cs; ++a) {
                run += gLog[(c0 + a) * H + h];
                gcum[a] = run;
                egc[a]  = std::exp(run);
            }

            // Per-token: scaled query, u = S0^T k_a, uq = S0^T qs_a.
            for (std::size_t a = 0; a < cs; ++a) {
                const float* qa = q + ((c0 + a) * H + h) * S;
                const float* ka = k + ((c0 + a) * H + h) * S;
                float* qsa = qs.data() + a * S;
                float* ua  = u.data() + a * S;
                float* uqa = uq.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    qsa[j] = qa[j] * qScale;
                    ua[j]  = 0.0F;
                    uqa[j] = 0.0F;
                }
                for (std::size_t i = 0; i < S; ++i) {
                    const float ki   = ka[i];
                    const float qsi  = qsa[i];
                    const float* row = s0.data() + i * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        ua[j]  += row[j] * ki;
                        uqa[j] += row[j] * qsi;
                    }
                }
            }

            // Gram-derived matrices: A[a,m] = beta_a (k_a.k_m) exp(G_a-G_m)
            // strictly below the diagonal; kq[a,m] = (k_m . qs_a).
            for (std::size_t a = 0; a < cs; ++a) {
                const float* ka = k + ((c0 + a) * H + h) * S;
                const float* qsa = qs.data() + a * S;
                const float  ba = beta[(c0 + a) * H + h];
                for (std::size_t m = 0; m < cs; ++m) {
                    const float* km = k + ((c0 + m) * H + h) * S;
                    float kk = 0.0F;
                    float kqv = 0.0F;
                    for (std::size_t i = 0; i < S; ++i) {
                        kk  += ka[i] * km[i];
                        kqv += km[i] * qsa[i];
                    }
                    kq[a * C + m] = kqv;
                    amat[a * C + m] =
                        (m < a) ? ba * kk * std::exp(gcum[a] - gcum[m]) : 0.0F;
                }
            }

            // Forward-substitution solve of (I + A) d = r, r_a = beta_a
            // (v_a - exp(G_a) u_a). (I + A) is unit lower-triangular.
            for (std::size_t a = 0; a < cs; ++a) {
                const float* va = v + ((c0 + a) * H + h) * S;
                const float  ba = beta[(c0 + a) * H + h];
                const float* ua = u.data() + a * S;
                float* da = d.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    da[j] = ba * (va[j] - egc[a] * ua[j]);
                }
                for (std::size_t m = 0; m < a; ++m) {
                    const float amm = amat[a * C + m];
                    const float* dm = d.data() + m * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        da[j] -= amm * dm[j];
                    }
                }
            }

            // Output: o_a = exp(G_a) uq_a + sum_{m<=a} exp(G_a-G_m)(k_m.qs_a) d_m.
            for (std::size_t a = 0; a < cs; ++a) {
                float* oa = out + ((c0 + a) * H + h) * S;
                const float* uqa = uq.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    oa[j] = egc[a] * uqa[j];
                }
                for (std::size_t m = 0; m <= a; ++m) {
                    const float w  = std::exp(gcum[a] - gcum[m]) * kq[a * C + m];
                    const float* dm = d.data() + m * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        oa[j] += w * dm[j];
                    }
                }
            }

            // Carry the state: S' = exp(G_{cs-1}) S0
            //                       + sum_m exp(G_{cs-1}-G_m) k_m d_m^T.
            const float gLast = gcum[cs - 1];
            const float eLast = std::exp(gLast);
            for (std::size_t i = 0; i < S; ++i) {
                const float* s0row = s0.data() + i * S;
                float* strow = st + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    strow[j] = eLast * s0row[j];
                }
            }
            for (std::size_t m = 0; m < cs; ++m) {
                const float* km = k + ((c0 + m) * H + h) * S;
                const float* dm = d.data() + m * S;
                const float  wm = std::exp(gLast - gcum[m]);
                for (std::size_t i = 0; i < S; ++i) {
                    const float ki = km[i] * wm;
                    float* strow = st + i * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        strow[j] += ki * dm[j];
                    }
                }
            }
        }
    }
}

void deltanetChunkCumGate(const float* gLog,
                          float*       gCum,
                          std::size_t  T,
                          std::size_t  H,
                          std::size_t  chunkSize) {
    const std::size_t C = chunkSize ? chunkSize : 64;
    for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t c0 = 0; c0 < T; c0 += C) {
            const std::size_t cs = std::min(C, T - c0);
            float run = 0.0F;
            for (std::size_t a = 0; a < cs; ++a) {
                run += gLog[(c0 + a) * H + h];
                gCum[(c0 + a) * H + h] = run;
            }
        }
    }
}

void deltanetKktSolveInverse(const float* k,
                             const float* beta,
                             float*       a0,
                             std::size_t  T,
                             std::size_t  H,
                             std::size_t  S,
                             std::size_t  chunkSize) {
    const std::size_t C = chunkSize ? chunkSize : 64;
    const std::size_t nChunks = (T + C - 1) / C;

    // Strict-lower ungated Gram of one chunk: lt[a*C + m] = beta_a (k_a.k_m).
    std::vector<float> lt(C * C);

    for (std::size_t c = 0; c < nChunks; ++c) {
        const std::size_t c0 = c * C;
        const std::size_t cs = std::min(C, T - c0);
        for (std::size_t h = 0; h < H; ++h) {
            float* a0c = a0 + ((c * H) + h) * C * C;
            std::fill(a0c, a0c + C * C, 0.0F);

            // Build strict-lower ungated Gram.
            for (std::size_t a = 0; a < cs; ++a) {
                const float* ka = k + ((c0 + a) * H + h) * S;
                const float  ba = beta[(c0 + a) * H + h];
                for (std::size_t m = 0; m < a; ++m) {
                    const float* km = k + ((c0 + m) * H + h) * S;
                    float kk = 0.0F;
                    for (std::size_t i = 0; i < S; ++i) {
                        kk += ka[i] * km[i];
                    }
                    lt[a * C + m] = ba * kk;
                }
            }

            // Invert the unit lower-triangular L = I + strictLower(lt).
            // X = L^-1 is unit lower-triangular:
            //   X[a,a] = 1;  X[a,m>a] = 0;
            //   X[a,m] = -sum_{p=m..a-1} lt[a,p] X[p,m]   (a > m)
            for (std::size_t a = 0; a < cs; ++a) {
                a0c[a * C + a] = 1.0F;
                for (std::size_t m = 0; m < a; ++m) {
                    float acc = 0.0F;
                    for (std::size_t p = m; p < a; ++p) {
                        acc += lt[a * C + p] * a0c[p * C + m];
                    }
                    a0c[a * C + m] = -acc;
                }
            }
        }
    }
}

void deltanetChunkForward(const float* q,
                          const float* k,
                          const float* v,
                          const float* gCum,
                          const float* beta,
                          const float* a0,
                          float*       state,
                          float*       out,
                          std::size_t  T,
                          std::size_t  H,
                          std::size_t  S,
                          std::size_t  chunkSize) {
    const float qScale = 1.0F / std::sqrt(static_cast<float>(S));
    const std::size_t C = chunkSize ? chunkSize : 64;

    std::vector<float> s0(S * S);      // chunk-start state snapshot
    std::vector<float> egc(C);         // exp(gCum_a)
    std::vector<float> ieg(C);         // exp(-gCum_a)
    std::vector<float> u(C * S);       // (S0^T k_a)[j]
    std::vector<float> uq(C * S);      // (S0^T qs_a)[j]
    std::vector<float> qs(C * S);      // scaled query
    std::vector<float> rp(C * S);      // exp(-gCum_m) r_m
    std::vector<float> d(C * S);       // solved delta vectors

    for (std::size_t h = 0; h < H; ++h) {
        float* st = state + h * S * S;

        for (std::size_t c0 = 0; c0 < T; c0 += C) {
            const std::size_t c  = c0 / C;
            const std::size_t cs = std::min(C, T - c0);
            const float* a0c = a0 + ((c * H) + h) * C * C;

            std::copy(st, st + S * S, s0.begin());

            for (std::size_t a = 0; a < cs; ++a) {
                const float g = gCum[(c0 + a) * H + h];
                egc[a] = std::exp(g);
                ieg[a] = std::exp(-g);
            }

            // qs, u = S0^T k_a, uq = S0^T qs_a.
            for (std::size_t a = 0; a < cs; ++a) {
                const float* qa = q + ((c0 + a) * H + h) * S;
                const float* ka = k + ((c0 + a) * H + h) * S;
                float* qsa = qs.data() + a * S;
                float* ua  = u.data() + a * S;
                float* uqa = uq.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    qsa[j] = qa[j] * qScale;
                    ua[j]  = 0.0F;
                    uqa[j] = 0.0F;
                }
                for (std::size_t i = 0; i < S; ++i) {
                    const float ki  = ka[i];
                    const float qsi = qsa[i];
                    const float* row = s0.data() + i * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        ua[j]  += row[j] * ki;
                        uqa[j] += row[j] * qsi;
                    }
                }
            }

            // r'_m = exp(-gCum_m) r_m,  r_m = beta_m (v_m - exp(gCum_m) u_m).
            for (std::size_t m = 0; m < cs; ++m) {
                const float* vm = v + ((c0 + m) * H + h) * S;
                const float  bm = beta[(c0 + m) * H + h];
                const float* um = u.data() + m * S;
                float* rpm = rp.data() + m * S;
                for (std::size_t j = 0; j < S; ++j) {
                    rpm[j] = ieg[m] * bm * (vm[j] - egc[m] * um[j]);
                }
            }

            // d_a = exp(gCum_a) * sum_{m<=a} a0[a,m] r'_m   (a0 lower-tri).
            for (std::size_t a = 0; a < cs; ++a) {
                float* da = d.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    da[j] = 0.0F;
                }
                for (std::size_t m = 0; m <= a; ++m) {
                    const float w = a0c[a * C + m];
                    const float* rpm = rp.data() + m * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        da[j] += w * rpm[j];
                    }
                }
                for (std::size_t j = 0; j < S; ++j) {
                    da[j] *= egc[a];
                }
            }

            // Output: o_a = exp(G_a) uq_a
            //             + sum_{m<=a} exp(G_a-G_m)(k_m.qs_a) d_m.
            for (std::size_t a = 0; a < cs; ++a) {
                float* oa = out + ((c0 + a) * H + h) * S;
                const float* uqa = uq.data() + a * S;
                const float* qsa = qs.data() + a * S;
                for (std::size_t j = 0; j < S; ++j) {
                    oa[j] = egc[a] * uqa[j];
                }
                for (std::size_t m = 0; m <= a; ++m) {
                    const float* km = k + ((c0 + m) * H + h) * S;
                    float kq = 0.0F;
                    for (std::size_t i = 0; i < S; ++i) {
                        kq += km[i] * qsa[i];
                    }
                    const float w   = egc[a] * ieg[m] * kq;
                    const float* dm = d.data() + m * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        oa[j] += w * dm[j];
                    }
                }
            }

            // Carry: S' = exp(G_{cs-1}) S0 + sum_m exp(G_{cs-1}-G_m) k_m d_m^T.
            const float eLast = egc[cs - 1];
            for (std::size_t i = 0; i < S; ++i) {
                const float* s0row = s0.data() + i * S;
                float* strow = st + i * S;
                for (std::size_t j = 0; j < S; ++j) {
                    strow[j] = eLast * s0row[j];
                }
            }
            for (std::size_t m = 0; m < cs; ++m) {
                const float* km = k + ((c0 + m) * H + h) * S;
                const float* dm = d.data() + m * S;
                const float  wm = eLast * ieg[m];
                for (std::size_t i = 0; i < S; ++i) {
                    const float ki = km[i] * wm;
                    float* strow = st + i * S;
                    for (std::size_t j = 0; j < S; ++j) {
                        strow[j] += ki * dm[j];
                    }
                }
            }
        }
    }
}

namespace {
// Numerically-stable softplus: log(1 + exp(x)).
inline float softplus(float x) {
    return x > 0.0F ? x + std::log1p(std::exp(-x))
                    : std::log1p(std::exp(x));
}
} // namespace

void deltanetGate(const float* alpha,
                  const float* ssmA,
                  const float* ssmDt,
                  float*       gLog,
                  std::size_t  T,
                  std::size_t  H) {
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            const float sp = softplus(alpha[t * H + h] + ssmDt[h]);
            // ssm_a (GGUF SSM_A_NOSCAN) is the pre-computed decay coefficient
            // A = -exp(A_log), stored negative. llama.cpp multiplies softplus
            // by it directly (qwen35moe.cpp: gate = softplus * ssm_a). Do NOT
            // apply -exp() again here — that double-exponentiates the decay,
            // making the recurrent state forget far too slowly.
            gLog[t * H + h] = ssmA[h] * sp;
        }
    }
}

void sigmoidInPlace(float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = 1.0F / (1.0F + std::exp(-y[i]));
    }
}

void gatherHeadsFromChannels(const float* src,
                             float*       dst,
                             std::size_t  T,
                             std::size_t  offset,
                             std::size_t  srcHeads,
                             std::size_t  dstHeads,
                             std::size_t  S,
                             std::size_t  convTotalWidth) {
    for (std::size_t t = 0; t < T; ++t) {
        const float* srcTok = src + t * convTotalWidth + offset;
        float*       dstTok = dst + t * dstHeads * S;
        for (std::size_t hd = 0; hd < dstHeads; ++hd) {
            const std::size_t srcHead = hd % srcHeads;
            const float* srcHeadPtr = srcTok + srcHead * S;
            float*       dstHeadPtr = dstTok + hd * S;
            for (std::size_t s = 0; s < S; ++s) {
                dstHeadPtr[s] = srcHeadPtr[s];
            }
        }
    }
}

void causalConv1dSilu(const float* convInput,
                      const float* kernel,
                      float*       out,
                      std::size_t  T,
                      std::size_t  channels,
                      std::size_t  kernelSize) {
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            float acc = 0.0F;
            for (std::size_t kk = 0; kk < kernelSize; ++kk) {
                // convInput row (t + kk) — the state-prepended input
                // (time-major, channel contiguous). kernel is channel-major
                // (GGUF ssm_conv1d.weight [d_conv, conv_dim], ne0=d_conv
                // contiguous → weight[c*K + kk], matching ggml_ssm_conv).
                acc += convInput[(t + kk) * channels + c] *
                       kernel[c * kernelSize + kk];
            }
            // SiLU: acc * sigmoid(acc).
            out[t * channels + c] = acc / (1.0F + std::exp(-acc));
        }
    }
}

void l2NormInPlace(float*      x,
                   std::size_t rows,
                   std::size_t dim,
                   float       eps) {
    for (std::size_t r = 0; r < rows; ++r) {
        float* row = x + r * dim;
        double sumSq = 0.0;
        for (std::size_t j = 0; j < dim; ++j) {
            sumSq += static_cast<double>(row[j]) * static_cast<double>(row[j]);
        }
        // ggml_l2_norm: scale = 1 / max(sqrt(sumSq), eps).
        const float scale =
            1.0F / std::fmax(std::sqrt(static_cast<float>(sumSq)), eps);
        for (std::size_t j = 0; j < dim; ++j) {
            row[j] *= scale;
        }
    }
}

} // namespace mimirmind::compute
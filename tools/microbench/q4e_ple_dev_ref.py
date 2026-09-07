#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# Qwen4-Exp PLE device-forward reference for the two NEW kernels (5.27 I-4c):
#   (1) signed-sqrt gate + gated value  (Qwen4ExpTextPLELayer.forward core)
#   (2) dilated depthwise causal conv1d + silu  (_short_conv)
# grouped-RMSNorm (already HC-verified) and the projections (BF16 GEMM) are NOT
# re-tested here. numpy f32; dumps for q4e_ple_dev_parity.cu.
#
#   gate:  key_normed[T,hc,d], query_normed[T,hc,d], value[T,d] -> gated[T,hc,d]
#     g = (key*query).sum(-1)/sqrt(d); g = sign(g)*sqrt(max(|g|,1e-6))
#     gated = sigmoid(g)[...,None] * value[:,None,:]
#   conv:  x[T,hc*d], w[hc*d, K], zero state -> out[T,hc*d]
#     inpad = [zeros(state_len) | x] per channel; state_len=(K-1)*dilation
#     out[t,c] = silu( sum_k w[c,k] * inpad[t + k*dilation, c] )

import os
import sys

import numpy as np


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q4e_ple_dev_dump"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4242)

    T, d, hc = 5, 2560, 4
    hcd = hc * d
    K, dilation = 4, 3
    state_len = (K - 1) * dilation   # 9

    key_normed = rng.standard_normal((T, hc, d)).astype(np.float32)
    query_normed = rng.standard_normal((T, hc, d)).astype(np.float32)
    value = rng.standard_normal((T, d)).astype(np.float32)
    convw = (0.1 * rng.standard_normal((hcd, K))).astype(np.float32)
    xconv = rng.standard_normal((T, hcd)).astype(np.float32)

    # (1) gate
    g = (key_normed.astype(np.float64) * query_normed).sum(-1) / np.sqrt(d)   # [T,hc]
    g = np.sign(g) * np.sqrt(np.maximum(np.abs(g), 1e-6))
    s = 1.0 / (1.0 + np.exp(-g))                                              # [T,hc]
    gated = (s[..., None] * value[:, None, :]).astype(np.float32)             # [T,hc,d]

    # (2) dilated depthwise causal conv + silu
    inpad = np.concatenate([np.zeros((state_len, hcd), np.float32), xconv], 0)  # [state_len+T, hcd]
    out = np.zeros((T, hcd), np.float32)
    for t in range(T):
        acc = np.zeros(hcd, np.float64)
        for k in range(K):
            acc += convw[:, k].astype(np.float64) * inpad[t + k * dilation]
        out[t] = (acc / (1.0 + np.exp(-acc))).astype(np.float32)   # silu

    def dump(n, a):
        a.astype("<f4").tofile(os.path.join(out_dir, n))

    dump("key_normed.bin", key_normed)
    dump("query_normed.bin", query_normed)
    dump("value.bin", value)
    dump("gated.bin", gated)
    dump("convw.bin", convw)
    dump("xconv.bin", xconv)
    dump("convout.bin", out)
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"T {T}\nd {d}\nhc {hc}\nK {K}\ndilation {dilation}\nstate_len {state_len}\n")
    print(f"wrote {out_dir}: T={T} d={d} hc={hc} K={K} dil={dilation}")
    print(f"gated[0,0,:3]={gated[0,0,:3]}  convout[0,:3]={out[0,:3]}")


if __name__ == "__main__":
    main()

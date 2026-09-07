#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# I-3 Hyper-Connections (Qwen4-Exp GatedResidual) reference dump for the
# standalone CUDA parity harness (q4e_hc_parity.cu). Reimplements the exact
# forward math from huggingface/transformers models/qwen4_exp
# (Qwen4ExpTextGatedResidual + grouped Qwen4ExpTextRMSNorm), in float32/numpy —
# no torch/transformers install needed. Deterministic (fixed seed).
#
# Dumps raw little-endian float32 tensors + a manifest into an output dir:
#   H.bin        [T, hc*d]      input hyper-stream tensor
#   wnorm.bin    [hc*d]         hc_norm weight, ALREADY (1+w)-baked (matches our
#                               loader's AddOne on norm.weight)
#   wdown.bin    [lowrank, hc*d]   input_mix_weight_down.weight   (nn.Linear [out,in])
#   wup.bin      [hc*d, lowrank]   input_mix_weight_up.weight
#   wbi.bin      [hc, hc*d]        block_inject_weight.weight
#   mixed.bin    [T, d]         expected mixed output
#   inj.bin      [T, hc]        expected injection weights
# Reference math (use_combine=True), verbatim from the modular file:
#   normed = hc_norm(H)                                   # grouped RMSNorm
#   w = silu(down(normed) / hc); w = sigmoid(up(w))
#   w = w.reshape(T, hc, d)
#   mixed = (w * normed.reshape(T, hc, d)).mean(axis=1)
#   inj = 2*sigmoid(block_inject(normed) / hc)

import os
import struct
import sys

import numpy as np


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def grouped_rmsnorm(x, wbaked, group_size, eps):
    # x: [T, hc*d]; normalize each `group_size` chunk independently, then *wbaked.
    T, hcd = x.shape
    xg = x.reshape(T, hcd // group_size, group_size).astype(np.float64)
    ms = np.mean(xg * xg, axis=-1, keepdims=True)
    out = xg / np.sqrt(ms + eps)
    out = out.reshape(T, hcd)
    return (out * (wbaked.astype(np.float64))).astype(np.float32)


def linear(x, w):
    # nn.Linear semantics: y = x @ w.T, w is [out, in].
    return (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/mimirmind/q4e_hc_dump"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(1234)
    T = 5
    d = 2560
    hc = 4
    lowrank = 320
    eps = 1e-6
    hcd = hc * d

    # Inputs / weights at realistic-ish scale (RMSNorm makes the absolute input
    # scale irrelevant; keep weights small so the low-rank path stays in range).
    H = rng.standard_normal((T, hcd)).astype(np.float32)
    wnorm = (1.0 + 0.02 * rng.standard_normal(hcd)).astype(np.float32)  # already (1+w)
    wdown = (0.02 * rng.standard_normal((lowrank, hcd))).astype(np.float32)
    wup = (0.02 * rng.standard_normal((hcd, lowrank))).astype(np.float32)
    wbi = (0.02 * rng.standard_normal((hc, hcd))).astype(np.float32)

    normed = grouped_rmsnorm(H, wnorm, d, eps)
    w1 = silu(linear(normed, wdown) / hc)          # [T, lowrank]
    w2 = sigmoid(linear(w1, wup))                  # [T, hcd]
    w2r = w2.reshape(T, hc, d)
    mixed = (w2r * normed.reshape(T, hc, d)).mean(axis=1).astype(np.float32)  # [T, d]
    inj = (2.0 * sigmoid(linear(normed, wbi) / hc)).astype(np.float32)        # [T, hc]

    def dump(name, arr):
        arr.astype("<f4").tofile(os.path.join(out_dir, name))

    dump("H.bin", H)
    dump("wnorm.bin", wnorm)
    dump("wdown.bin", wdown)
    dump("wup.bin", wup)
    dump("wbi.bin", wbi)
    dump("mixed.bin", mixed)
    dump("inj.bin", inj)

    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write(f"T {T}\nd {d}\nhc {hc}\nlowrank {lowrank}\neps {eps:.9g}\n")

    print(f"wrote dump to {out_dir}: T={T} d={d} hc={hc} lowrank={lowrank} eps={eps}")
    print(f"mixed[0,:4]={mixed[0,:4]}  inj[0]={inj[0]}")


if __name__ == "__main__":
    main()

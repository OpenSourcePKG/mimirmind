#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# I-6 real-weight parity: recompute the qwen4_exp block-0 attn Hyper-Connections
# GatedResidual from the ACTUAL checkpoint weights + the engine's dumped block-0
# stream-in, and compare to the engine's dumped mixed/inj. Localises whether the
# HC integration (weight layout / (1+w) norm / F32 path) is exact with real
# weights (the isolated harness used random weights).
#
#   engine dumps (/opt/mimirmind/q4e_dbg, via MIMIRMIND_Q4E_DUMP=1):
#     blk0_streamin.bin [T*hcd] f32, blk0_attn_mixed.bin [T*d], blk0_attn_inj.bin [T*hc]
#   run: /opt/mimirmind/.venv-hf/bin/python q4e_hc_realweight_check.py

import json
import struct
import sys

import numpy as np

CKPT = "/opt/mimirmind/models/qwen3.8-flash-next-nvfp4"
DBG = "/opt/mimirmind/q4e_dbg"
LAYER = "model.language_model.layers.0.attn_hyper_connection."
D, HC, LOWRANK, EPS = 2560, 4, 320, 1e-6
HCD = HC * D


def load_st_tensor(name):
    idx = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]
    fn = idx[name]
    with open(CKPT + "/" + fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        e = hdr[name]
        f.seek(base + e["data_offsets"][0])
        raw = f.read(e["data_offsets"][1] - e["data_offsets"][0])
    dt = e["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dt in ("F16",):
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise SystemExit("unhandled dtype " + dt)
    return arr.reshape(e["shape"]).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def grouped_rms(x, wbaked, group):
    T = x.shape[0]
    xg = x.reshape(T, -1, group).astype(np.float64)
    out = xg / np.sqrt(np.mean(xg * xg, -1, keepdims=True) + EPS)
    return (out.reshape(T, -1) * wbaked.astype(np.float64)).astype(np.float32)


def main():
    streamin = np.fromfile(DBG + "/blk0_streamin.bin", dtype="<f4")
    mixed_eng = np.fromfile(DBG + "/blk0_attn_mixed.bin", dtype="<f4")
    inj_eng = np.fromfile(DBG + "/blk0_attn_inj.bin", dtype="<f4")
    T = mixed_eng.size // D
    print(f"T={T}")
    streamin = streamin.reshape(T, HCD)
    mixed_eng = mixed_eng.reshape(T, D)
    inj_eng = inj_eng.reshape(T, HC)

    hc_norm = load_st_tensor(LAYER + "hc_norm.weight")            # [hcd]
    wdown = load_st_tensor(LAYER + "input_mix_weight_down.weight")  # [lowrank, hcd]
    wup = load_st_tensor(LAYER + "input_mix_weight_up.weight")      # [hcd, lowrank]
    wbi = load_st_tensor(LAYER + "block_inject_weight.weight")      # [hc, hcd]
    print(f"hc_norm{hc_norm.shape} wdown{wdown.shape} wup{wup.shape} wbi{wbi.shape}")

    wbaked = 1.0 + hc_norm                    # AddOne (Qwen4ExpTextRMSNorm (1+w))
    normed = grouped_rms(streamin, wbaked, D)
    w1 = silu((normed.astype(np.float64) @ wdown.T.astype(np.float64)) / HC)
    w2 = sigmoid(w1 @ wup.T.astype(np.float64))
    w2r = w2.reshape(T, HC, D)
    mixed_ref = (w2r * normed.reshape(T, HC, D)).mean(1).astype(np.float32)
    inj_ref = (2.0 * sigmoid((normed.astype(np.float64) @ wbi.T.astype(np.float64)) / HC)).astype(np.float32)

    def cmp(tag, a, b):
        mx = np.max(np.abs(a - b))
        rel = mx / (np.max(np.abs(b)) + 1e-9)
        print(f"  {tag:6s} maxAbs={mx:.3e} maxRel={rel:.3e}  ref[0,:3]={b.flatten()[:3]}  eng[0,:3]={a.flatten()[:3]}")

    print("engine vs real-weight reference:")
    cmp("mixed", mixed_eng, mixed_ref)
    cmp("inj", inj_eng, inj_ref)


if __name__ == "__main__":
    main()

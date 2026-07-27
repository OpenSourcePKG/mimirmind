#!/usr/bin/env python3
"""gguf-active-bytes -- active weight bytes read per decode token.

Decode is (mostly) weight-streaming: every token reads the *active* weights
once. For a dense model that is the whole model; for an MoE it is attention +
router + shared + only top_k/n_experts of each expert bank + the tied lm_head.
Divide by a measured ms/tok to get the effective decode bandwidth, or divide
a bandwidth budget by this to get the tok/s roofline.

Pure stdlib GGUF (v2/v3) tensor-info parser -- no deps, no model load.

    python3 tools/gguf-active-bytes.py MODEL.gguf

Companion to tools/roofline-probe.py (which turns this into a tok/s model)
and `prefill_bench --bw` (which measures the DRAM bandwidth ceiling).
"""
import sys
import struct

# ggml type -> (block_elements, block_bytes)
TYPE_SIZE = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rstr(f):
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")


def skip_val(f, vt):
    if vt in (0, 1, 7):
        f.read(1)
    elif vt in (2, 3):
        f.read(2)
    elif vt in (4, 5, 6):
        f.read(4)
    elif vt in (10, 11, 12):
        f.read(8)
    elif vt == 8:
        rstr(f)
    elif vt == 9:                       # array
        et = rd(f, "<I")
        for _ in range(rd(f, "<Q")):
            skip_val(f, et)
    else:
        raise ValueError(f"unknown gguf value type {vt}")


def read_kv(f):
    key = rstr(f)
    vt = rd(f, "<I")
    if vt in (4, 5):
        return key, rd(f, "<i" if vt == 5 else "<I")
    if vt in (10, 11):
        return key, rd(f, "<q" if vt == 11 else "<Q")
    skip_val(f, vt)
    return key, None


def tensor_bytes(dims, typ):
    ne = 1
    for d in dims:
        ne *= d
    be, bb = TYPE_SIZE[typ]
    return (ne // be) * bb


def main(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF file"
        rd(f, "<I")                      # version
        n_tensors = rd(f, "<Q")
        n_kv = rd(f, "<Q")
        meta = {}
        for _ in range(n_kv):
            k, v = read_kv(f)
            if v is not None:
                meta[k] = v
        tensors = []
        for _ in range(n_tensors):
            name = rstr(f)
            nd = rd(f, "<I")
            dims = [rd(f, "<Q") for _ in range(nd)]
            typ = rd(f, "<I")
            rd(f, "<Q")                  # offset
            tensors.append((name, dims, typ))

    n_experts = next((v for k, v in meta.items()
                      if k.endswith("expert_count")), None)
    top_k = next((v for k, v in meta.items()
                  if k.endswith("expert_used_count")), 8)
    frac = top_k / n_experts if n_experts else 1.0

    active = 0
    full = 0
    cats = {}
    for name, dims, typ in tensors:
        b = tensor_bytes(dims, typ)
        full += b
        is_expert = "_exps" in name
        # Expert banks: only top_k/n_experts read per token. Everything else
        # (attention, router, shared FFN, norms) is read fully. token_embd is
        # tied to the lm_head, whose full table is matmul'd for logits every
        # token, so it counts in full too.
        a = b * frac if is_expert else b
        active += a
        cat = ("expert" if is_expert else
               "router" if "gate_inp" in name else
               "attn" if "attn" in name else
               "lm_head/embd" if "token_embd" in name or "output" in name else
               "norm" if "norm" in name else "dense_ffn/other")
        cats[cat] = cats.get(cat, 0) + a

    print(f"model: {path}")
    print(f"  n_experts={n_experts} top_k={top_k} active_frac={frac:.4f}"
          f"  tensors={n_tensors}")
    print(f"  total weights (all resident): {full / 1e9:.2f} GB")
    print(f"  ACTIVE bytes / token:         {active / 1e9:.3f} GB")
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:20s} {v / 1e9:.4f} GB")
    print(f"\n  effective decode BW = {active / 1e9:.3f} GB / (your ms/tok / 1000)")
    print(f"  tok/s roofline      = BW_GBs / {active / 1e9:.3f} * gemv_eff")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: gguf-active-bytes.py MODEL.gguf", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])

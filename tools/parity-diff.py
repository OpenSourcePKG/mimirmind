#!/usr/bin/env python3
"""parity-diff — compare per-block, per-stage hidden-state dumps from
llama-parity-dump and mimirmind.

Usage:
    parity-diff REF_DIR TGT_PREFIX [--threshold 0.5] [--topk 5]

REF_DIR contains llama dumps named `blk{N}-<stage>.bin`.
TGT_PREFIX is the mimirmind dump prefix; matching files are
`<TGT_PREFIX>-blk{N}-<stage>.bin`.

Stages are inspected in the canonical Gemma 4 block execution order
(attn_norm → Qcur_pos → Kcur_pos → ... → l_out). The diff highlights the
*first* stage in each block whose max_abs exceeds the threshold, so you
can isolate which substep introduces a divergence.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np


STAGE_ORDER = [
    "inp_scaled",
    "attn_norm",
    "Qcur_pos",
    "Kcur_pos",
    "Vcur_normed",
    "attn_post_norm",
    "attn_out",
    "ffn_mlp",
    "ffn_moe",
    "ffn_moe_combined",
    "ffn_post_norm",
    "out_scaled",
    "l_out",
]

REF_RE = re.compile(r"^blk(\d+)-(\w+)\.bin$")


def load(path: Path):
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError(f"{path}: truncated header")
        idx, T, d = struct.unpack("<III", header)
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size != T * d:
        raise ValueError(
            f"{path}: data size {data.size} != T*d {T*d}"
        )
    return idx, T, d, data.reshape(T, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("tgt_prefix", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="max_abs threshold (default 0.5)")
    ap.add_argument("--topk", type=int, default=3,
                    help="top-K largest diffs to print per divergent stage (default 3)")
    args = ap.parse_args()

    # Discover (block, stage) tuples from the ref directory.
    found = {}
    for p in args.ref_dir.iterdir():
        m = REF_RE.match(p.name)
        if not m:
            continue
        n     = int(m.group(1))
        stage = m.group(2)
        found.setdefault(n, set()).add(stage)
    if not found:
        print(f"parity-diff: no blk*-*.bin in {args.ref_dir}", file=sys.stderr)
        return 2

    print(f"parity-diff: {len(found)} block(s), stages: "
          f"{sorted({s for ss in found.values() for s in ss})}\n")

    header = f"{'blk':>3} | " + " | ".join(f"{s:>17}" for s in STAGE_ORDER)
    print(header)
    print("-" * len(header))

    first_div = None

    for n in sorted(found.keys()):
        row = [f"{n:>3}"]
        for stage in STAGE_ORDER:
            ref_p = args.ref_dir / f"blk{n}-{stage}.bin"
            tgt_p = (args.tgt_prefix.parent /
                     f"{args.tgt_prefix.name}-blk{n}-{stage}.bin")
            if not (ref_p.exists() and tgt_p.exists()):
                row.append(f"{'-':>17}")
                continue
            try:
                _, T1, d1, a = load(ref_p)
                _, T2, d2, b = load(tgt_p)
            except ValueError as e:
                row.append(f"err:{e}")
                continue
            if (T1, d1) != (T2, d2):
                row.append(f"shape!=")
                continue
            diff = np.abs(a - b)
            ma   = float(diff.max())
            me   = float(diff.mean())
            row.append(f"ma={ma:>5.3f},µ={me:>5.3f}")
            if first_div is None and ma > args.threshold:
                first_div = (n, stage, ma, a, b, diff)
        print(" | ".join(row))

    if first_div is None:
        print(f"\n=> All stages match within threshold {args.threshold:.4f} ✓")
        return 0

    n, stage, ma, a, b, diff = first_div
    print(f"\n=> FIRST DIVERGENCE: blk{n} stage={stage} max_abs={ma:.4f}")
    flat_idx = np.argpartition(diff.flatten(), -args.topk)[-args.topk:]
    flat_idx = flat_idx[np.argsort(-diff.flatten()[flat_idx])]
    for fi in flat_idx:
        t, dp = np.unravel_index(int(fi), diff.shape)
        av, bv = float(a[t, dp]), float(b[t, dp])
        print(f"   t={t:>2} d={dp:>5}  llama={av:>+12.6f}  "
              f"mimir={bv:>+12.6f}  diff={bv-av:>+10.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
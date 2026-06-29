#!/usr/bin/env python3
"""parity-diff — compare per-block hidden-state dumps from llama-parity-dump
and mimirmind. Reports the first block where the absolute difference exceeds
a threshold and the position of the largest error.

Usage:
    parity-diff REF_DIR TGT_PREFIX [--threshold 0.01]

REF_DIR holds llama.cpp dumps named blk{N}.bin.
TGT_PREFIX is the basename mimirmind used, e.g. /tmp/dumps/mimir → expects
files /tmp/dumps/mimir-blk{N}.bin.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


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
    ap.add_argument("ref_dir", type=Path,
                    help="directory with llama dumps (blk{N}.bin)")
    ap.add_argument("tgt_prefix", type=Path,
                    help="mimirmind dump prefix (DIR/PREFIX → PREFIX-blk{N}.bin)")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="max_abs threshold considered a divergence (default 0.01)")
    args = ap.parse_args()

    ref_files = sorted(args.ref_dir.glob("blk*.bin"),
                       key=lambda p: int(p.stem.replace("blk", "")))
    if not ref_files:
        print(f"parity-diff: no blk*.bin in {args.ref_dir}", file=sys.stderr)
        return 2

    print(f"parity-diff: {len(ref_files)} block dump(s)\n")
    print(f"{'blk':>4} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10} "
          f"{'ref@max':>14} {'tgt@max':>14} {'pos':>14}")

    first_div = None
    for ref_p in ref_files:
        n = int(ref_p.stem.replace("blk", ""))
        tgt_p = args.tgt_prefix.parent / f"{args.tgt_prefix.name}-blk{n}.bin"
        if not tgt_p.exists():
            print(f"blk{n:<3d}: missing tgt {tgt_p}")
            continue

        try:
            _, T1, d1, a = load(ref_p)
            _, T2, d2, b = load(tgt_p)
        except ValueError as e:
            print(f"blk{n:<3d}: load error: {e}")
            continue

        if (T1, d1) != (T2, d2):
            print(f"blk{n:<3d}: shape mismatch ref={T1}x{d1} tgt={T2}x{d2}")
            continue

        diff = np.abs(a - b)
        rel  = diff / (np.abs(a) + 1e-9)
        flat = np.unravel_index(int(np.argmax(diff)), diff.shape)
        ref_v, tgt_v = float(a[flat]), float(b[flat])
        ma = float(diff.max())
        print(f"{n:>4} {ma:>10.4f} {float(diff.mean()):>10.4f} "
              f"{float(rel.max()):>10.4f} {ref_v:>14.6f} {tgt_v:>14.6f} "
              f"(t={flat[0]:>2},d={flat[1]:>5})")

        if first_div is None and ma > args.threshold:
            first_div = (n, ma, flat, ref_v, tgt_v)

    if first_div is not None:
        n, ma, (t, d), av, bv = first_div
        print(
            f"\n=> FIRST DIVERGENCE: blk{n} max_abs={ma:.6f} "
            f"at (t={t}, d={d})  llama={av:.6f}  mimir={bv:.6f}"
        )
        return 1
    print("\n=> All blocks match within threshold "
          f"{args.threshold:.4f} ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
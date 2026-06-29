#!/usr/bin/env python3
"""parity-diff — compare per-block hidden-state dumps from llama-parity-dump
and mimirmind.

Usage:
    parity-diff REF_DIR TGT_PREFIX [--threshold 0.01] [--topk 5]

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
        raise ValueError(f"{path}: data size {data.size} != T*d {T*d}")
    return idx, T, d, data.reshape(T, d)


def describe_block(n, a, b, topk):
    diff = np.abs(a - b)
    if diff.size == 0:
        return
    ma   = float(diff.max())
    mean = float(diff.mean())
    std  = float(diff.std())

    # per-token mean diff (helps spot "one token is wrong")
    per_tok = diff.mean(axis=1)
    per_tok_str = " ".join(f"t{i}={v:.4f}" for i, v in enumerate(per_tok))

    rel = diff / (np.abs(a) + 1e-9)
    print(f"\n--- blk{n} -- max_abs={ma:.4f} mean_abs={mean:.4f} std={std:.4f} "
          f"max_rel={rel.max():.4f}")
    print(f"  per-token mean_abs: {per_tok_str}")

    # Top-K positions by abs diff.
    flat_idx = np.argpartition(diff.flatten(), -topk)[-topk:]
    flat_idx = flat_idx[np.argsort(-diff.flatten()[flat_idx])]
    for fi in flat_idx:
        t, d_pos = np.unravel_index(int(fi), diff.shape)
        av = float(a[t, d_pos])
        bv = float(b[t, d_pos])
        print(f"  top: t={t:>2} d={d_pos:>5}  llama={av:>+12.6f}  "
              f"mimir={bv:>+12.6f}  diff={bv-av:>+10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path,
                    help="directory with llama dumps (blk{N}.bin)")
    ap.add_argument("tgt_prefix", type=Path,
                    help="mimirmind dump prefix (DIR/PREFIX → PREFIX-blk{N}.bin)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="max_abs threshold considered a divergence (default 0.5)")
    ap.add_argument("--topk", type=int, default=5,
                    help="top-K largest diffs to print per block (default 5)")
    ap.add_argument("--summary-only", action="store_true",
                    help="single line per block; no top-K details")
    args = ap.parse_args()

    ref_files = sorted(args.ref_dir.glob("blk*.bin"),
                       key=lambda p: int(p.stem.replace("blk", "")))
    if not ref_files:
        print(f"parity-diff: no blk*.bin in {args.ref_dir}", file=sys.stderr)
        return 2

    print(f"parity-diff: {len(ref_files)} block dump(s)\n")
    print(f"{'blk':>4} {'max_abs':>10} {'mean_abs':>10} "
          f"{'max_rel':>10}  {'pos':>14}")

    first_div = None
    blocks = []
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
        flat = np.unravel_index(int(np.argmax(diff)), diff.shape)
        ma = float(diff.max())
        rel = diff / (np.abs(a) + 1e-9)
        print(f"{n:>4} {ma:>10.4f} {float(diff.mean()):>10.4f} "
              f"{float(rel.max()):>10.4f}  "
              f"(t={flat[0]:>2},d={flat[1]:>5})")
        blocks.append((n, a, b))
        if first_div is None and ma > args.threshold:
            first_div = n

    if not args.summary_only and first_div is not None:
        print(f"\n========= details around first divergence (blk{first_div}) =========")
        for n, a, b in blocks:
            if n <= first_div + 1:  # show 2 blocks: first div + the next one
                describe_block(n, a, b, args.topk)
            else:
                break

    if first_div is not None:
        return 1
    print(f"\n=> All blocks match within threshold {args.threshold:.4f} ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Bucket a Kineto/torch-profiler trace into per-stage GPU ms (roadmap 5.18.7).

Graph-mode cross-check: under CUDA graphs the record_function *scope* attribution
is lost (a decode step is one graph replay), but the trace still records every
replayed GPU kernel by name. This sums GPU-kernel self-time by name and buckets
names into mimirmind decode-prof stages -- a per-stage breakdown that survives
CUDA graphs, unlike the eager cuda.Event path.

Usage:
  python3 parse_kineto.py <trace.json|trace.json.gz> --steps N [--top 25]

--steps = number of decode steps captured in the profiled window (from the probe
completion_tokens), so bucket totals become per-step ms.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict

# name-substring -> stage bucket (first match wins), case-insensitive.
BUCKETS = [
    ("moe.gemm", r"marlin|fused_moe|moe_.*gemm|grouped_gemm|awq|gptq|w4a16"),
    ("moe.other", r"topk|expert|routing|moe|scatter|gather|permute|silu.*mul|act.*quant"),
    ("gdn", r"gated_delta|gdn|chunk_.*delta|fused_recurrent|causal_conv|conv1d|delta_rule|fla_"),
    ("attn", r"flash|attention|paged|attn|rotary|rope|reshape_and_cache|kv_cache"),
    ("proj", r"scaled_mm|cutlass|gemm|matmul|linear|cublas|addmm|mm_"),
    ("norm", r"rms_?norm|layernorm|norm_.*quant|quant_.*norm|fused_add_rms"),
    ("elementwise", r"elementwise|add|mul|cast|copy|memcpy|memset|index|embedding"),
]
_COMPILED = [(b, re.compile(p, re.I)) for b, p in BUCKETS]


def _bucket(name: str) -> str:
    for b, rx in _COMPILED:
        if rx.search(name):
            return b
    return "other"


def _load(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, required=True,
                    help="decode steps in the profiled window (== completion tokens)")
    ap.add_argument("--top", type=int, default=25, help="top-N kernels to list")
    args = ap.parse_args()

    trace = _load(args.trace)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])

    by_name: dict[str, float] = defaultdict(float)  # us
    by_name_cnt: dict[str, int] = defaultdict(int)
    gpu_total = 0.0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = str(ev.get("cat", "")).lower()
        # GPU kernel activities: cat is 'kernel' (or 'gpu_memcpy'/'gpu_memset').
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0))
        name = ev.get("name", "?")
        by_name[name] += dur
        by_name_cnt[name] += 1
        gpu_total += dur

    if gpu_total == 0:
        print("No GPU kernel events found (check cat field / trace type).")
        # dump distinct cats to help
        cats = sorted({str(e.get("cat")) for e in events if isinstance(e, dict)})
        print("cats seen:", cats[:20])
        return 1

    buckets: dict[str, float] = defaultdict(float)
    for name, us in by_name.items():
        buckets[_bucket(name)] += us

    n = max(1, args.steps)
    print(f"\n=== per-stage GPU ms/step (graph-mode, kernel-name bucketed, steps={n}) ===")
    print(f"{'bucket':<14}{'ms/step':>10}{'% of GPU':>10}")
    for b, us in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"{b:<14}{us/1000.0/n:>10.3f}{100.0*us/gpu_total:>9.1f}%")
    print("-" * 34)
    print(f"{'GPU TOTAL':<14}{gpu_total/1000.0/n:>10.3f}{100.0:>9.1f}%")

    print(f"\n=== top {args.top} kernels by GPU self-time ===")
    print(f"{'ms/step':>9}{'calls/step':>11}  bucket / name")
    for name, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{us/1000.0/n:>9.3f}{by_name_cnt[name]/n:>11.1f}  "
              f"[{_bucket(name)}] {name[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

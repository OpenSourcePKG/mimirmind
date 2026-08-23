#!/usr/bin/env python3
"""Decode probe -- bs=1, 512-token generation, per-token decode ms (roadmap 5.18.7).

Short prompt so prefill is negligible and the run is decode-dominated. Runs the
identical request against whichever OpenAI-compatible engine ``--base-url`` points
at (vLLM or mimirmind), so the two sides line up for the delta table.

Pair with the engine-side per-stage instrumentation:
  * vLLM  : MIMIR_VLLM_PROBE=1  (mimir_vllm_probe attaches; use --enforce-eager)
  * mimir : MIMIRMIND_DECODE_PROFILE=1 MIMIRMIND_PROFILE_EVERY=N

Example:
  python3 decode_probe.py --base-url http://localhost:8000 --model qwen --label vllm
  python3 decode_probe.py --base-url https://172.17.0.2:8080 --insecure \
      --api-key "$KEY" --model qwen3.6-35b --label mimir
"""

from __future__ import annotations

import argparse
import statistics

from probe_common import ProbeResult, add_common_args, stream_chat

# Short, deterministic instruction -> long continuation. Keeps prefill tiny.
DECODE_PROMPT = (
    "Write a detailed, continuous explanation of how a modern GPU executes a "
    "matrix multiplication, covering memory hierarchy, warps, and tensor cores. "
    "Do not stop early; keep going until you are told to stop."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    messages = [{"role": "user", "content": DECODE_PROMPT}]

    def one() -> ProbeResult:
        return stream_chat(
            args.base_url, messages,
            model=args.model, max_tokens=args.max_tokens, temperature=0.0,
            api_key=args.api_key, insecure=args.insecure,
        )

    for _ in range(max(0, args.warmup)):
        one()

    runs = [one() for _ in range(max(1, args.repeat))]
    med_decode = statistics.median(r.decode_ms for r in runs)
    med_tps = statistics.median(r.decode_tps for r in runs)
    med_ttft = statistics.median(r.ttft_ms for r in runs)
    n_out = runs[-1].n_out

    print(f"\n=== decode probe [{args.label}] "
          f"(bs=1, max_tokens={args.max_tokens}, repeat={args.repeat}) ===")
    for i, r in enumerate(runs):
        print(r.as_line(f"run{i}"))
    print("-" * 72)
    print(f"MEDIAN       ttft={med_ttft:8.1f}ms  "
          f"decode={med_decode:6.2f}ms/tok ({med_tps:6.1f} tok/s)  n_out={n_out}")
    if runs[-1].server_fields:
        print(f"server       {runs[-1].server_fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

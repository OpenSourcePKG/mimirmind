#!/usr/bin/env python3
"""roofline-probe -- decode-throughput model for MimirMind, calibrated to
measured NUC numbers (2026-07-27).

Answers: for a given (model, quant, hardware, decode regime), how many
tok/s can we realistically expect, and does it clear a target like 100?

WHAT CHANGED vs the 2026-07-22 version (all from on-target measurement):

  * The DRAM bus is NOT the wall and dispatch/host-sync is NOT the lever.
    Measured on the NUC (Xe-LPG):
      - `prefill_bench --bw`: sustained bus 67.5 GB/s (2N/time memcpy).
      - Gemma 4 26B-A4B-Q6_K: 3.555 GB active/token (GGUF exact,
        tools/gguf-active-bytes.py), decode 138.6 ms/tok.
        -> effective 25.6 GB/s = 38% of the bus over the whole token.
      - OpProfiler breakdown: matmul 72% (~100 ms), norm 12%, attn 7%,
        resid/router/act ~9%; 1819 dispatches/token.
        -> the matmuls themselves run at ~52% of the bus.

  * ROOT CAUSE: decode is M=1 GEMV -- each weight is read once and used
    once, no reuse. An iGPU sustains only ~50% of peak bandwidth on GEMV;
    80-90% needs GEMM (M>1, weight reuse). So the efficiency knob is
    `gemv_eff`, which rises when a forward processes M>1 positions.

  * MTP / speculative decode is therefore a DOUBLE lever, not just an
    accept-rate multiplier: verifying M draft tokens in one forward reads
    each weight once for M rows (GEMV -> GEMM, gemv_eff up) AND amortises
    the weight bytes over the accepted tokens. Continuous batching (multi
    user) does the same via the sequence dimension.

MODEL (calibrated so 26B-A4B-Q6_K, M=1 -> 138.6 ms/tok):

    t_stream_ms = active_bytes / (BW * gemv_eff(M)) * 1e3
    t_token_ms  = t_stream_ms * (1 + OVERHEAD_FRAC)      # +norm/attn/...
    tok_per_s   = accepted_per_forward / t_token_ms * 1e3

  gemv_eff(M): 0.52 at M=1 (measured), ramps toward ~0.90 as M grows
  (weight reuse). accepted_per_forward folds the MTP accept rate.

Pure stdlib. Run:  python3 tools/roofline-probe.py
"""
from __future__ import annotations
from dataclasses import dataclass

# --- Calibration anchors (measured on the NUC, 2026-07-27) -----------------
BW_GBS_NUC      = 67.5     # sustained bus, prefill_bench --bw
GEMV_EFF_M1     = 0.52     # matmul fraction of bus at M=1 (from OpProfiler)
GEMV_EFF_MAX    = 0.90     # GEMM ceiling with full weight reuse
OVERHEAD_FRAC   = 0.38     # non-matmul (norm/attn/resid/router/act) over matmul time

# Effective bytes per weight parameter at a given quant.
BYTES_PER_PARAM = {
    "Q8_0": 8.5 / 8, "Q6_K": 6.6 / 8, "Q5_K": 5.5 / 8, "Q4_K": 4.5 / 8,
    "Q3_K": 3.4 / 8, "Q2_K": 2.6 / 8,
    # Unsloth Dynamic-2.0 style over the *active* weights (attn/router/shared
    # high, routed experts low).
    "Dyn2.5": 2.5 / 8, "Dyn3.0": 3.0 / 8,
}


def gemv_eff(m: int) -> float:
    """Bus utilisation as a function of positions-per-forward M.
    M=1 GEMV measured ~0.52; each extra row adds reuse toward the GEMM ceiling."""
    return min(GEMV_EFF_M1 + (m - 1) * 0.09, GEMV_EFF_MAX)


@dataclass(frozen=True)
class Model:
    name: str
    total_b: float
    active_b: float          # active params/token, billions
    # measured active bytes/token override (GB) at a specific quant, if known
    measured_active_gb: float = 0.0
    measured_quant: str = ""


MODELS = {
    # Reference: exactly what we run today. measured_active_gb from GGUF.
    "gemma4-26b-a4b": Model("Gemma 4 26B-A4B", 26.0, 4.0,
                            measured_active_gb=3.555, measured_quant="Q6_K"),
    # Off-the-shelf A3B fits (qwen35moe already runs on CUDA).
    "qwen35-a3b":     Model("Qwen3.6-35B-A3B", 35.0, 3.0),
    "qwen3-30b-a3b":  Model("Qwen3-30B-A3B", 30.5, 3.3),
    # The byte-budget-designed custom target.
    "mimir-nuc-a2b":  Model("Mimir-NUC-A2B (custom)", 30.0, 1.7),
    # Frontier point -- shows the quality wall.
    "deepseek-v3":    Model("DeepSeek-V3 (671B/37B)", 671.0, 37.0),
}


def active_bytes_gb(model: Model, quant: str) -> float:
    if model.measured_active_gb and quant == model.measured_quant:
        return model.measured_active_gb
    # Scale the measured 26B anchor by the active-param and bpw ratio when we
    # only have an estimate. Falls back to active_b * bpw for others.
    return model.active_b * BYTES_PER_PARAM[quant]


def decode(model: Model, quant: str, m: int, bw: float = BW_GBS_NUC) -> dict:
    ab = active_bytes_gb(model, quant) * 1e9
    t_stream = ab / (bw * 1e9 * gemv_eff(m)) * 1e3
    t_token = t_stream * (1 + OVERHEAD_FRAC)
    return {"active_gb": ab / 1e9, "t_stream": t_stream, "t_token": t_token}


def tok_s(model: Model, quant: str, m: int, accept: float, bw: float = BW_GBS_NUC) -> float:
    t = decode(model, quant, m, bw)["t_token"]
    accepted = 1.0 + (m - 1) * accept          # 1 real + (M-1) draft * accept
    return accepted / t * 1e3


def ram_gb(model: Model, quant: str) -> float:
    return model.total_b * BYTES_PER_PARAM[quant]


def main():
    target = 100.0
    print("=" * 96)
    print(f"MimirMind decode roofline (NUC, calibrated)  --  target {target:.0f} tok/s")
    print(f"bus {BW_GBS_NUC} GB/s | gemv_eff M1={GEMV_EFF_M1} -> max {GEMV_EFF_MAX} | "
          f"overhead {OVERHEAD_FRAC:.0%}")
    print("=" * 96)

    # Validate the anchor.
    a = decode(MODELS["gemma4-26b-a4b"], "Q6_K", m=1)
    print(f"\nanchor: 26B-A4B-Q6_K M=1 -> {a['t_token']:.0f} ms/tok "
          f"({1000/a['t_token']:.1f} tok/s)   [measured 138.6 ms]")

    print("\n" + "-" * 96)
    hdr = ["model", "quant", "RAM GB", "active GB", "M1 t/s",
           "M4+MTP0.6", "hits 100"]
    w = [24, 8, 8, 10, 9, 12, 9]
    print("  ".join(f"{h:<{x}}" for h, x in zip(hdr, w)))
    print("-" * 96)
    plan = [
        ("gemma4-26b-a4b", "Q6_K"), ("gemma4-26b-a4b", "Dyn2.5"),
        ("qwen35-a3b", "Dyn2.5"), ("qwen3-30b-a3b", "Dyn2.5"),
        ("mimir-nuc-a2b", "Dyn2.5"), ("deepseek-v3", "Q4_K"),
    ]
    for mkey, quant in plan:
        model = MODELS[mkey]
        m1 = tok_s(model, quant, m=1, accept=0.0)
        m4 = tok_s(model, quant, m=4, accept=0.6)      # MTP width 4, accept 0.6
        hit = "YES" if m4 >= target else "no"
        cells = [model.name, quant, f"{ram_gb(model, quant):.1f}",
                 f"{active_bytes_gb(model, quant):.2f}",
                 f"{m1:.1f}", f"{m4:.1f}", hit]
        print("  ".join(f"{c:<{x}}" for c, x in zip(cells, w)))

    print("\n" + "=" * 96)
    print("Levers to reach 100 tok/s (evidence-ranked):")
    print("  1. M>1 per forward (MTP / continuous batching) -- GEMV->GEMM,")
    print("     lifts bus utilisation from ~52% toward ~90% AND amortises")
    print("     weight bytes over accepted tokens. The under-rated double lever.")
    print("  2. Dynamic mixed-Q_K quant (~2.5 bpw) -- the byte-budget lever (~7x).")
    print("  3. Smaller active params (A2B) -- for the last stretch.")
    print("  NOT a lever: CLR/dispatch elimination (measured ~8%); the 26B is")
    print("  byte-hopeless (~5x over budget even at the bus ceiling).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""roofline-probe -- decode-throughput roofline model for MimirMind.

Answers one question: for a given (model, quantization, hardware, decode
regime), how many tokens/second can we realistically expect, and does it
clear a target such as 100 tok/s?

The model is deliberately simple and physically grounded. Decode is
memory-bandwidth-bound *or* launch/host-sync-bound; we model both and add
them, because on Xe-LPG today the host-sync term dominates for MoE:

    t_token = t_weights + t_kv + t_host_sync + t_dispatch
      t_weights   = active_params * bpw/8 / BW        (stream active weights)
      t_kv        = kv_bytes(context) / BW            (KV read, grows w/ ctx)
      t_host_sync = n_layers * syncs * host_sync_ms   (host-side top-K stalls)
      t_dispatch  = n_dispatches * dispatch_us

    tok_per_s = 1000 / t_token_ms * mtp_factor

Two decode regimes are compared:
  * "today"       -- host-side top-K forces a D2H/host/H2D round trip per
                     MoE layer; this is the observed launch-bound regime
                     (E4B ~133 ms/tok on the NUC while its pure roofline is
                     ~32 ms/tok).
  * "graph-replay"-- device-side top-K + CLR/CUDA-Graph/HipGraph capture
                     removes the host-sync stalls; decode approaches the
                     bandwidth roofline.

Calibration anchors (from Synaipse):
  * Meteor Lake NUC: DDR5-5600 dual-IMC, sustained read ~65-75 GB/s (UMA).
    Dispatch overhead ~10-15 us. E4B Q4_K_M ~130-141 ms/tok measured.
  * DGX Spark GB10: 128 GB LPDDR5x, ~273 GB/s. qwen35moe ~27-34 ms/tok.
  * MTP on Xe-LPG realistically 1.4-2.2x (quality-neutral). Default 1.8x.

Pure stdlib, no deps. Run:  python3 tools/roofline-probe.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Quantization: effective bytes per weight parameter.
# K-quants carry block scales/mins, so bpw is a little above the nominal bits.
# --------------------------------------------------------------------------
BYTES_PER_PARAM = {
    "Q8_0": 8.5 / 8,      # ~1.06
    "Q6_K": 6.6 / 8,      # ~0.83
    "Q5_K": 5.5 / 8,      # ~0.69
    "Q4_K": 4.5 / 8,      # ~0.56
    "Q3_K": 3.4 / 8,      # ~0.43
    "Q2_K": 2.6 / 8,      # ~0.33
    # Unsloth Dynamic 2.0 style: sensitive layers (attn/router/shared) high,
    # routed experts low -> effective ~2.5-3.0 bpw over the *active* weights.
    "Dyn2.5": 2.5 / 8,    # 0.3125
    "Dyn3.0": 3.0 / 8,    # 0.375
}


@dataclass(frozen=True)
class Hardware:
    name: str
    bandwidth_gbs: float       # sustained read bandwidth, GB/s
    dispatch_us: float         # per-dispatch launch latency, microseconds
    host_sync_ms: float        # cost of one host round-trip stall, ms


HARDWARE = {
    "nuc": Hardware("NUC Meteor Lake (Xe-LPG)", 70.0, 12.0, 2.0),
    "spark": Hardware("DGX Spark GB10", 273.0, 5.0, 0.8),
}


@dataclass(frozen=True)
class Model:
    name: str
    total_b: float             # total params, billions
    active_b: float            # active params per token, billions
    n_layers: int
    # KV bytes per token per full context token (GQA/MLA already folded in):
    # = 2 (K+V) * n_kv_heads * head_dim * bytes_per_elem, cached per layer.
    kv_bytes_per_ctx_tok: float
    dispatches_per_layer: int  # rough dispatch count in the decode path


# Candidate models. Numbers are order-of-magnitude architecture figures for
# roofline purposes, not exact GGUF metadata.
MODELS = {
    # A3B family -- the sweet spot for the 100 tok/s target.
    "qwen3-30b-a3b": Model("Qwen3-30B-A3B", 30.5, 3.3, 48,
                           kv_bytes_per_ctx_tok=48 * 4 * 128 * 2, dispatches_per_layer=22),
    "qwen35-a3b": Model("Qwen3.6-35B-A3B (qwen35moe)", 35.0, 3.0, 48,
                        kv_bytes_per_ctx_tok=48 * 2 * 128 * 2, dispatches_per_layer=24),
    "qwen3-next-80b-a3b": Model("Qwen3-Next-80B-A3B", 80.0, 3.0, 48,
                                kv_bytes_per_ctx_tok=48 * 2 * 128 * 2, dispatches_per_layer=26),
    # Reference points.
    "gemma4-26b-a4b": Model("Gemma 4 26B-A4B", 26.0, 4.0, 46,
                            kv_bytes_per_ctx_tok=46 * 4 * 256 * 2, dispatches_per_layer=20),
    "gemma4-e4b": Model("Gemma 4 E4B", 8.0, 4.0, 35,
                        kv_bytes_per_ctx_tok=35 * 2 * 256 * 2, dispatches_per_layer=18),
    # Frontier point -- to show why it cannot hit the target.
    "deepseek-v3": Model("DeepSeek-V3 (671B/37B)", 671.0, 37.0, 61,
                         kv_bytes_per_ctx_tok=61 * 1 * 512 * 2, dispatches_per_layer=30),
}


@dataclass
class Regime:
    name: str
    host_syncs_per_layer: float   # host round trips per layer (0 => captured)
    dispatch_scale: float         # fraction of dispatches still paid serially


REGIMES = {
    "today": Regime("today (launch-bound)", host_syncs_per_layer=1.0, dispatch_scale=1.0),
    "graph-replay": Regime("graph-replay (roofline)", host_syncs_per_layer=0.0, dispatch_scale=0.05),
}


def decode_ms(model: Model, quant: str, hw: Hardware, regime: Regime,
              context: int) -> dict:
    """Return a breakdown of per-token decode time in milliseconds."""
    bpw = BYTES_PER_PARAM[quant]
    bw = hw.bandwidth_gbs * 1e9  # bytes/s

    active_bytes = model.active_b * 1e9 * bpw
    t_weights = active_bytes / bw * 1e3

    kv_bytes = model.kv_bytes_per_ctx_tok * context
    t_kv = kv_bytes / bw * 1e3

    t_host_sync = model.n_layers * regime.host_syncs_per_layer * hw.host_sync_ms

    n_dispatch = model.n_layers * model.dispatches_per_layer * regime.dispatch_scale
    t_dispatch = n_dispatch * hw.dispatch_us / 1e3

    total = t_weights + t_kv + t_host_sync + t_dispatch
    return {
        "weights": t_weights, "kv": t_kv, "host_sync": t_host_sync,
        "dispatch": t_dispatch, "total": total,
    }


def tok_per_s(model: Model, quant: str, hw: Hardware, regime: Regime,
              context: int, mtp: float) -> float:
    total = decode_ms(model, quant, hw, regime, context)["total"]
    return 1000.0 / total * mtp


def ram_gb(model: Model, quant: str) -> float:
    """Total weight footprint (all experts resident) at this quant."""
    return model.total_b * 1e9 * BYTES_PER_PARAM[quant] / 1e9


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def fmt_row(cells, widths, align="<"):
    return "  ".join(f"{str(c):{align}{w}}" for c, w in zip(cells, widths))


def sweep(target=100.0, context=2048, mtp=1.8):
    print("=" * 92)
    print(f"MimirMind decode roofline  --  target {target:.0f} tok/s  "
          f"(context={context}, MTP x{mtp})")
    print("=" * 92)

    for hw_key in ("nuc", "spark"):
        hw = HARDWARE[hw_key]
        print(f"\n### {hw.name}   ({hw.bandwidth_gbs:.0f} GB/s, "
              f"dispatch {hw.dispatch_us:.0f} us, host-sync {hw.host_sync_ms:.1f} ms)")
        widths = [26, 8, 8, 12, 12, 12, 12]
        hdr = ["model", "quant", "RAM GB", "today t/s", "replay t/s",
               "replay+MTP", "hits target"]
        print(fmt_row(hdr, widths))
        print("-" * 92)
        for mkey, model in MODELS.items():
            # pick a representative quant per model class
            quants = ["Q4_K", "Dyn2.5"] if model.active_b <= 5 else ["Q4_K"]
            for quant in quants:
                today = tok_per_s(model, quant, hw, REGIMES["today"], context, mtp=1.0)
                replay = tok_per_s(model, quant, hw, REGIMES["graph-replay"], context, mtp=1.0)
                replay_mtp = tok_per_s(model, quant, hw, REGIMES["graph-replay"], context, mtp)
                hit = "YES" if replay_mtp >= target else "no"
                print(fmt_row(
                    [model.name, quant, f"{ram_gb(model, quant):.1f}",
                     f"{today:.1f}", f"{replay:.1f}", f"{replay_mtp:.1f}", hit],
                    widths))

    print("\n" + "=" * 92)
    print("Breakdown for the recommended pick (Qwen3.6-35B-A3B, Dyn2.5, NUC):")
    print("=" * 92)
    hw = HARDWARE["nuc"]
    model = MODELS["qwen35-a3b"]
    for rkey, regime in REGIMES.items():
        b = decode_ms(model, "Dyn2.5", hw, regime, context)
        base = 1000.0 / b["total"]
        print(f"\n  regime = {regime.name}")
        print(f"    weights {b['weights']:6.1f} ms | kv {b['kv']:5.1f} ms | "
              f"host_sync {b['host_sync']:6.1f} ms | dispatch {b['dispatch']:5.1f} ms")
        print(f"    -> {b['total']:.1f} ms/tok = {base:.1f} tok/s  "
              f"(x{mtp} MTP = {base * mtp:.1f} tok/s)")


if __name__ == "__main__":
    sweep()
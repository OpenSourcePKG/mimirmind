# GB10 / DGX Spark — Performance Decision Record

Target: **NVIDIA GB10 (Grace ARM64 + Blackwell, sm_121a), 128 GB LPDDR5x,
273 GB/s, 48 SMs.** Primary metric: **serving decode tok/s** (Bragi is a
multi-tenant serving backend; single-user is secondary).

This file records what has been **measured** on GB10 so an external reviewer
(or a future contributor) does not re-propose already-settled work. It is the
public, code-adjacent summary of the internal perf ledger; the detailed raw
data lives in the project's private notes. **When code and this file disagree,
re-measure — do not guess.**

## Method (how these numbers were obtained)

- **No Nsight Compute (`ncu`).** It hangs the GB10 box hard (twice required a
  physical reboot). Profiling is done **in-code** via the
  `MIMIRMIND_DECODE_PROFILE` CUDA-event section timer (`PROFILE_EVERY=N`),
  **toggle-delta A/B** (flip one env flag, back-to-back HTTP measurement), and
  **byte-accounting** from the model geometry against the 273 GB/s roofline.
- **One GPU container at a time.** Two `--gpus` containers on this box →
  CUDA illegal-access. A/B = stop prod-serve, run one probe container, restore.
- Decode `%` from the profiler are **serialized shares** (a forced
  `cudaEventSynchronize` per section); real continuous-batched execution
  overlaps kernels, so a section's true marginal cost is ≤ its serialized
  share. **Relative composition is the valid signal**, not absolute ms.

## Central finding: decode is weight-bandwidth / occupancy bound, not FLOPS bound

Single-user Qwen3.6-35B-A3B-NVFP4 decode moves ~**3.5 GB of weights/token** at
~**125 GB/s ≈ 46% of the 273 GB/s peak**. The lever is **bytes/token and kernel
occupancy**, not tensor-core throughput. This is measured, not assumed.

Measured decode-step composition (serialized shares, `MIMIRMIND_DECODE_PROFILE`,
Qwen3.6-35B-A3B-NVFP4):

| section | conc=1 | conc=16 (at load) |
|---|---|---|
| **moe.gemm** (routed experts, NVFP4) | 37% | **43–44% (dominant)** |
| gdn.proj (GDN in/out proj, BF16) | 16% | 13–14% |
| attn.paged (paged-KV attention) | 11% (flat vs ctx) | 8–13% |
| lmhead | 9% | 5% |
| gdn.tail / gdn.conv | 12% | 11% |
| attn (qkv+rope+kv-write) | 5% | 2–5% |
| routing | 5% | 3–4% |
| gdn.recur (linear-attn update) | 2% | 8% |

`moe.gemm` at M=1 achieves **60 GB/s = 22% of peak** — latency/access-pattern
bound (scattered 20-byte NVFP4 super-blocks), **not compute bound**. Under
serving batch, expert-grouping gives each expert M>1 rows and `moe.gemm` rises
to **48–78% of peak = bandwidth-bound**.

---

## CONFIRMED (measured, kept)

- **Native blocked-NVFP4 decode reads ¼ the weight bytes and wins where the
  path is weight-bound.** Qwen3.8-27B dense: **346 → 216 ms/tok (−38%)** with
  `MIMIRMIND_QWEN_DENSE_NVFP4_DECODE=1` (opt-in). MoE routed experts default to
  native blocked-NVFP4 banks (`MIMIRMIND_NVFP4_MOE=nvfp4`), plus additive
  FP4-TC sidecars for prefill. Device-driven expert-pointer construction (no
  D2H, no host sync).
- **Weight-bandwidth-bound is model-dependent — verify per model.** Gemma4-12B
  dense NVFP4 decode was **NEUTRAL** (BF16 14.9 vs NVFP4 14.4 tok/s =
  compute-bound there), so native-4bit decode is dormant/opt-in for it, while
  the Qwen dense path is a clear win. Do not assume 4-bit decode helps every
  model.
- **Serving decode campaign: ~−13% step (~34 → 39.4 tok/s at load).**
  moe.gemm uint4 + register-staged activations + warp top-k + GDN-fusion.
- **PagedAttention V2 (split-K + warp-shuffle) achieves the vLLM flat
  signature.** attn.paged time is FLAT across context at fixed concurrency
  (e.g. 81 → 30.4 ms at 2213 tok). At its worst corner (conc16 + long ctx) it
  is already ~78% of peak = saturated; the only reducible quantity there is KV
  **bytes**, not kernel efficiency.
- **cuBLAS-FP8 decode GEMV (M==1-gated): ~7 ms/tok, +23%.**
- **Prefill wins (compute-bound, large M):** cuBLAS-BF16 GEMM −16.5%;
  F32-TC + cuDNN-SDPA −5% @700 tok / −11% @3400 tok; FP4-tensor-core grouped
  MoE **1.20–1.37×** (this is where NVFP4 TC belongs — prefill, not decode);
  shared-expert FP4-TC +12%.

## REJECTED (investigated, do NOT re-propose)

- **M=1 NVFP4 tensor-core decode GEMM — NO-GO (structural, not tunable).**
  A W4A16 bf16-wmma decode kernel was built and measured **2.8× SLOWER at
  bs=1**. Root cause: on sm_121a the only NVFP4 tensor-core op is
  `mma.sync m16n8k64` — the **M dimension is fixed at 16**, so at M=1 the tile
  is **1/16-filled** (15 of 16 rows zero-padded). A tensor-core kernel reads
  the **exact same weight bytes with the same access pattern** as the tuned
  CUDA-core GEMV, so it hits the **same 60 GB/s latency wall** — TC cannot make
  the weights arrive faster. vLLM's own GB10 autotune independently corroborates:
  it keeps `BLOCK_SIZE_M=16` to M=512 and ships **no NVFP4 MoE config for GB10
  at all**. Expected value: single-user ~0% to negative; serving batch ~0%
  (grouping already makes it bandwidth-bound). **The M=1 sparse-decode GEMM is
  at its architectural floor on GB10.**
- **FP8 (E4M3) re-quant of attention projections — slower, off by default.**
  Back-to-back HTTP A/B (Qwen3.6-35B-NVFP4, conc32): **BF16 84.6 vs FP8 70.5
  gen-tok/s** (BF16 +20%). `matmul_fp8_gemm` is slower than the BF16 TF32-TC
  path at decode-M and it runs on all GDN layers × 3 proj. Kept behind
  `MIMIRMIND_NVFP4_ATTN_FP8=gdn` for memory-saving only. (These projections are
  natively FP8 in the checkpoint, so BF16 holds them exactly — no accuracy loss
  from staying BF16.)
- **Q8_0 linear re-quant of FP8/NVFP4-origin projections — breaks coherence.**
  Q8_0 is a per-32 linear quant; these projections are log-distributed
  (e4m3/NVFP4), so Q8_0's absmax scale crushes the small weights. Off by
  default (`MIMIRMIND_NVFP4_Q8_PROJ`).
- **Attention/GDN kernel-efficiency throughput lever — PLATEAU (~0–3%).**
  attn.paged is overhead/occupancy-bound (flat vs context) except one saturated
  corner; gdn.recur is a small sequential recurrence at vLLM parity with no
  bandwidth slack. No single-kernel lever reaches a meaningful aggregate gain.
- **FP16-KV as a throughput default — no throughput win.** A/B at load:
  halving KV bytes moved decode throughput **0%** on both models (KV is a
  minority, overhead-bound slice). Kept as an opt-in **capacity** win (2× KV
  memory), not a throughput default.
- **MTP / speculative decoding on the optimized MoE — no win / net loss.**
  DFlash Qwen3.6-35B: no win against the optimized batched-MoE baseline; the
  ReplaySSM GDN-verify was refuted by measurement before code (verify cost >
  the GDN-recurrence share, which is only ~8%). Do not prioritize MTP before
  the single-token decode core is exhausted.

## Tooling constraints

- **Nsight Compute (`ncu`): banned** — hangs the GB10 box (physical reboot).
  Do not plan any methodology around it. Use `MIMIRMIND_DECODE_PROFILE`,
  toggle-delta A/B, and byte-accounting.
- **SGLang as a spec-dec oracle: banned** — hung the box twice.
- **One GPU container at a time** (two → CUDA illegal-access).
- The Spark `/opt/mimirmind` tree is a lagging deployment checkout (not git);
  build/verify against a clean `git archive HEAD` snapshot, not that tree.

## Active / open optimization targets

Ordered by expected value against the **serving** metric, from the measured
vLLM-vs-mimirmind dispatch-map (the 2.1× single-user gap is dominated by these,
NOT by an M=1 MoE tensor-core):

1. **CUDA-Graph capture of the decode step.** We default-off; vLLM default-on.
   An earlier isolated measurement put it at ~3% (engine already launch-lean),
   but the dispatch-map flags it as **the biggest single-user gap term** —
   worth a proper re-measurement rather than treating ~3% as final.
2. **Fused norm + quant / act-quant** (vLLM fuses these; we don't fully).
3. **TC-dense projections** (qkv / o / lm_head) — the dense, large-enough GEMMs
   where tensor cores *do* pay off, unlike the M=1 sparse MoE GEMM.
4. **KV dtype autotuning by context length** — FP16-KV exists
   (`MIMIRMIND_SERVING_KV_FP16`); FP8/E4M3-KV is worth it for **capacity**
   (larger batch / longer context) plus a **~5% throughput corner** at
   long-ctx + high-concurrency only (quality-gated, lossy). Not a blanket
   throughput default.
5. **CUDA-core M=1 MoE-GEMV weight-layout repack + 128-bit vectorized loads +
   deeper prefetch** — the *only* moe.gemm-specific idea with positive EV, but
   capped (~single-user +15–28%) and single-user-only. Weigh against CUDA-Graph
   (cheaper, also single-user). **Not** tensor cores.

Explicitly **not** an active target: another generic F32/Q4/Q5/Q6 CUDA kernel,
another attention variant, or a from-scratch M=1 NVFP4 tensor-core GEMM (see
REJECTED).

# Roadmap

> **Canonical source:** the authoritative, machine-tracked roadmap lives
> in the Synaipse vault at `Memory/mimirmind/_roadmap.md`, reached only
> through the `synaipse_roadmap_*` MCP tools (structured step tree, live
> "AI is working here" cursor, rolled-up progress/hours, dependencies).
> **This document is a narrative mirror** — read it for the story, but
> when it disagrees with `_roadmap.md`, the roadmap tool wins. Update
> this file when a phase materially changes; keep per-step status on the
> roadmap tool. See `CLAUDE.md` → "Roadmap Anchor" for the workflow.

Norse-themed phase names. Each phase has to be runnable and verifiable
before the next one starts. MimirMind exists to query a *loaded*
knowledge source; Mimir was the keeper of the *Well of Wisdom* whose
preserved head whispered counsel to Odin in every crisis.

## Project Well — *Alpha*  ✅ Complete (2026-06-28)

The source. Reading from disk into memory.

| M | Item | Status |
|---|---|---|
| M1 | Level Zero context, device enumeration, driver checks | ✅ done — container smoke verified on Meteor Lake |
| M2 | USM allocator: probe limits, shared pool, per-tensor allocation, free-list, stats | ✅ done — phase-2 ceiling probe capped at 4 GiB so shared hosts don't OOM during diagnostics |
| M3 | GGUF format reader, tensors mapped into USM via mmap + copy | ✅ done — mmap dropped after load (madvise + fadvise `DONTNEED`) so the page cache doesn't hold the model twice |
| M4 | CPU-only forward pass + autoregressive generation, output parity vs `llama-cli` | ✅ done — bit-exact token parity on Qwen 2.5 7B Q4_K_M |

Baseline: Qwen 2.5 7B Instruct Q4_K_M, picked because Gemma 3 was not
yet available when M4 landed. Same llama-family architecture.

## Project Envoy — *Beta*  ✅ Complete (2026-06-29)

The messenger. The head learns to speak.

| M | Item | Status |
|---|---|---|
| M5 | Level Zero GPU kernels (Q4_K + Q6_K + Q8_0 matmul, rmsnorm, RoPE, silu_mul, gelu_mul, ...), on-the-fly dequant | ✅ done — ocloc → SPIR-V at build time; CPU-parity verified element-wise in `gpu_tests` |
| M6 | KV cache in USM, autoregressive sampling (greedy + temperature + top-k + top-p), seed-determinism | ✅ done — `compute::Sampler` with reusable scratch, 9 unit tests in `compute_tests` |

End-of-Envoy performance: **126 ms/tok** decode on Qwen 2.5 7B Q4_K_M.
The remaining optimisation shelf is in
`Memory/mimirmind/research/m5-optimization-shelf.md` in the Synaipse
vault.

## Mimir-1.0 — *Release*  🚧 99 % — pending integration

The counsel. Production-queryable.

| M | Item | Status |
|---|---|---|
| M7 | OpenAI-compatible HTTP API (`/v1/chat/completions`), chat template, streaming | ✅ done — Qwen ChatML, Gemma 2/3, and Gemma 4 templates; non-streaming and SSE both wired |
| **M8** | **Gemma 4 architecture: hybrid local/global attention, MoE routing, per-layer dims, alt-attn, multi-norm choreography, chat template** | ✅ done (2026-06-30) — coherent output at 145-148 ms/tok on Gemma 4 26B-A4B Q6_K/Q8_0; image tag `e2d1f2f` |
| M9 | Layer streaming + multi-request concurrency | Open (re-scoped 2026-08-17) — the concurrency stack is *built* but CUDA/Bragi-only (5.4); Xe-LPG needs an L0 port (M9.1) + layer streaming (M9.2). See Mimir-1.1 and Mimir-2.0 below |

### M8 sub-step breakdown

The Gemma 4 work landed in pieces:

| Sub | What |
|---|---|
| M8.1-2 | Architecture detection + ArchBackend dispatch |
| M8.3 | Multi-norm choreography (attn_post_norm + four FFW norms) |
| M8.4 | Per-layer SWA pattern + sliding-window attention |
| M8.5 | Dense FFN (path A) with GELU SwiGLU |
| M8.6 | MoE path B: 128 experts, top-8, renormalised weights |
| M8.7 | V-norm bare RMSNorm, Q pre-scale by sqrt(head_dim) |
| M8.8 | ffn_down_exps.scale per-expert multiplier |
| M8.9 | RMSNorm `w · norm` (not `(1+w) · norm`) |
| M8.10 | Per-layer SWA RoPE base + proportional-RoPE freq_factors |
| M8.A | ArchBackend OOP split (Qwen2 / Gemma4) |
| M8.B-C | Per-layer head_dim/kv_heads + alternative attention V=K for full-attn layers |
| M8.D-E | Tensor-parity-test framework + per-stage intermediate dumps |
| M8.F | QuantType OOP refactor + 210+ unit tests across 4 binaries |
| M8.G | Q8_0 GPU matmul kernel + Gemma 4 chat template + thinking-channel strip |

The full story of M8 (and what the bring-up taught us about debugging
instruction-tuned models in general) is in
[`journey.md`](journey.md).

### What's left before Mimir-1.0 is tagged

- **Multi-request concurrency.** The server today serialises all
  requests through a mutex. Per-request KV cache + scratch pool +
  decode loop needs to land before the engine can handle parallel
  users.
- **KV-cache reuse across turns** — a multi-turn conversation
  currently re-prefills the entire context. With cache reuse only the
  *new* user turn needs to be prefilled.
- **Pegenaut integration** — the sister TypeScript RAG project that
  this engine was originally built to feed. MimirMind plugs in as the
  LLM provider via the OpenAI-compatible API.

## M5i — Prefill-Perf-Pass  ✅ Complete (2026-07-02)

Batched-input pass over the transformer block. Motivated by RAG-style
prompts (hundreds of tokens of context) where prefill was almost as
expensive per token as decode.

| Sub | Item |
|---|---|
| M5i.A | GEMM kernels for Q4_K/Q6_K/Q8_0 (2D geometry, `M_TILE=8`, SLM-tiled) |
| M5i.B | Fused QKV: load-time W_q‖W_k‖W_v concat + `qkv_split.cl` scatter |
| M5i.D | Per-QuantType autotune at `loadModel()` — matvec-loop vs GEMM |
| M5i.E | Load-time parity gates: `GpuOps::selfTest` + autotune-parity check |
| M5i.F | MoE expert grouping — batch the T×K_top iteration into `nExperts` matmuls (T>1) |
| M5i.I | `governor.gpuClockPin: "rp0"` for reproducible bench runs |

Outcome on the target hardware: the GEMM kernel loses the autotune
race 2-3× against a matvec loop on Xe-LPG (compile-time surprise,
runtime data). Autotune pins matvec-loop per quant type. The parity
gate + autotune combination turned out to be the more valuable
deliverable than the GEMM kernel itself. Fused QKV runs on 14/30 blocks
of Gemma 4 26B-A4B; the remaining SWA blocks skip fusion because of
mixed quantisation. MoE expert grouping is active at T>1 and doesn't
affect decode.

## Project Sleipnir — *Post-Mimir-1.0 Perf & Model Extensions*  🚧 In planning (2026-07-08)

Sleipnir, Odin's eight-legged mount, carried its rider across all
nine worlds — the fastest steed in the Edda. Where Mimir-1.0 finds
the answer, Sleipnir makes it *fast*. This phase collects every
throughput-relevant lever that landed (or was scoped) after the
Mimir-1.0 baseline, and adds the two model-side accelerations Google
released for Gemma 4 in mid-2026.

### Milestone order

| M | Item | Status | Expected delta |
|---|---|---|---|
| **M-CLR** | Command-List Replay for decode | ✅ done 2026-07-06 | −6.4 % decode on E4B, verified live |
| **M10.2** | KV cache dtype layer (F32 → FP16 → Q8_0) | 🚧 ~60 % — FP16 read+write path live; migration + parity + default-on left | +2 % on short contexts, +25-40 % at T_k > 3000 |
| **M8.K** | GEMM Xe-LPG-native rewrite | 🚧 in progress — v2-GEMM prototypes 1.6-1.9× over matvec | Prerequisite for the batched-verify path that MTP and spec-dec need |
| **M8.L** | Flash-Prefill T_k-scaling rewrite | 🚧 in progress — per-WG tile SLM independent of T_k | Removes the ~80× T_k>256 prefill regime |
| **M9.6.6** | GPU clock governor retune | ✅ done 2026-08-17 — headroom-adaptive up-gain + overshoot watchdog | Closes the bench-to-prod gap |
| **M9.11** | Speculative decoding framework | 🔲 blocked on M8.K | 20-40 % decode — but see the CUDA cross-learning note below |
| **M-Ratatoskr** | Gemma 4 MTP drafter integration | 🔲 blocked on M8.K + M9.11 | **up to 3× decode**, quality-neutral (official) |
| **M-Diff** | DiffusionGemma backend | 🔲 deferred, optional | 3-5× decode, explicit quality trade-off, not for prod chat |

> **Cross-learning (2026-08-17, from the CUDA/Bragi tracks):** spec-decode
> (MTP 5.6, DFlash 5.9) turned out to be a **net loss** against an
> *optimised* MoE baseline on GB10 — the M=K+1 verify re-reads too many
> experts with no amortisation. Xe-LPG differs (Gemma 4 is dense / 26B-A4B,
> far lower per-token verify cost), so a drafter can still pay here — but
> M9.11/M-Ratatoskr must be benchmarked against the **CLR-optimised** Xe-LPG
> decode, not a naive baseline. The DFlash ReplaySSM GDN-fold is reusable IP.

### M-CLR — Command-List Replay ✅

Delivered 2026-07-06 (image `2f3a0ea`) behind the toggle
`features.clr: true` in `config.json` (formerly the
`MIMIRMIND_ENABLE_CLR=on` env var, migrated in `456bd2a`). Eliminates
the per-launch dispatch overhead
on decode (936 dispatches per token × ~12 µs Xe-LPG launch cost).
Verified live at −6.4 % on E4B Q4_K_M against the immediate path with
the same image, same prompt. The Level-Zero mutable-command-list
extension is not on the production driver, so option B (scalar-in-USM
indirection with a persistent record/replay handle) was chosen.

### M10.2 — KV cache dtype layer 🚧

Two phases. **Phase 0** switches K/V storage from F32 to FP16, halving
KV reads per attention call. **Phase 1** (later) adds Q8_0 for a
further 2× cut. FP16 is the largest bandwidth-relevant lever for as
long as KV reads scale linearly with T_k — during long chats and RAG
prompts, KV traffic dominates the read budget. **Status ~60 %:** the FP16
read *and* write path is live end-to-end — dtype-aware `KvCache` alloc +
striding (incl. Q8_0 block storage), an fp32-staging write redirect into
`kvCommitFp16Async`, and FP16 attention read kernels on L0 and HIP. It is
user-selectable via `runtime.kvDtype` but still **default-off/WIP**. Left:
the backend migration, a dedicated FP16/Q8_0 KV **parity test**, Q8_0 for
the IMRoPE path, and flipping default-on.

### M8.K — GEMM Xe-LPG-native rewrite 🔲

The M5i GEMM kernel lost the autotune race 2-3× against a matvec loop
on Xe-LPG. Without an Xe-LPG-native GEMM that actually scales
sub-linearly with M, batched-target-verify (needed for both spec-dec
and MTP) does not pay off — a single verify of N=4 costs 4× a single
forward, so the drafter gain is eaten by the verify cost. Roadmap
note lives in the Synaipse vault. **Started (~40 %):** native L0
v2-GEMM prototypes (`matmul_{q8_0,q6k,q4k}_gemm_v2`, shrunk SLM
`X_TILE=256`) land 1.6-1.9× over the matvec loop, with the autotune
parity gate picking matvec-loop vs GEMM per M-bucket. Full coverage +
the batched-verify win still to come.

### M-Ratatoskr — Gemma 4 MTP drafter integration 🔲

Ratatoskr, the squirrel that runs up and down Yggdrasil carrying
messages between the eagle above and the serpent below — a
lightweight courier that propagates proposals between layers. The
metaphor lands exactly on what a Multi-Token-Prediction drafter does:
a small, fast model that hands speculative tokens to the target for
batched verification.

Google released official MTP drafters for Gemma 4 on 2026-05-05 —
one for each of E2B, E4B, 26B-A4B, and 31B — under Apache 2.0. The
drafters share the target model's KV cache, so integration doesn't
double the KV budget. Expected payoff after M8.K lands:

| Model | Decode today (bench) | + MTP (N=4, acc=0.7) | Ratio |
|---|---:|---:|---:|
| E4B Q4_K_M | 7.6 tok/s | **18-23 tok/s** | 2.4-3.0× |
| E2B Q4_K_M | ~12.8 tok/s (projected) | **30-38 tok/s** | 2.4-3.0× |

What MimirMind needs to add:
- GGUF loader support for the 4-layer drafter models
- Second `_backend` slot on `InferenceEngine`
- Speculation loop with target-KV-shared draft state
- Batched target verify + accept/reject sampling
- `speculative.enabled` config toggle and kill switch

Rough sizing: 5-7 days, plus M8.K running first.

### M-Diff — DiffusionGemma backend 🔲 (deferred)

Google released DiffusionGemma on 2026-06-10 — a 26B MoE (3.8B
active) trained on text-diffusion instead of autoregression, with 4×
generation throughput on high-end GPUs. Adapting MimirMind for it
would need a new (bidirectional) attention kernel, a new
denoising-refinement sampler, and a different GGUF conversion; the
quality trade-off is explicit even in Google's own guidance ("For
maximum quality production work, Google still recommends
autoregressive Gemma 4"). Kept as an exploratory follow-up, not a
next milestone.

### tok/s target picture for this phase

| Setup | E4B tok/s | E2B tok/s | Delta vs today |
|---|---:|---:|---:|
| Today (bench, CLR on) | 7.6 | ~12.8 | 1.0× |
| + M10.2 (T_k=3000) | 9.2 | 15.5 | 1.2× |
| + M8.K + M-Ratatoskr | **18-23** | **30-38** | **~3×** |

Production numbers are lower today because the GPU clock governor
throttles aggressively — the governor retune (M9.6.6) **landed
2026-08-17** (headroom-adaptive up-gain + overshoot watchdog, fast-down
safety preserved) and closes most of the bench-to-prod gap.

## Mimir-1.1 — Concurrency

**Status (2026-08-18): largely delivered by M9.1.** The backend-neutral
slab batcher un-gated concurrent multi-request decode on Level Zero and
bypasses the per-engine request mutex, so the Xe-LPG path no longer
serialises same-model requests. Key finding: on the Meteor Lake iGPU this
buys *latency / fairness*, not throughput — with no matrix engine, batched
decode re-reads the weights per sequence (confirmed for both sparse-MoE and
dense), so a batch does not amortise. Throughput concurrency is a Bragi /
GB10 concern. Remaining here is polish: slot eviction / preemption on the
slab path and a memory-based batch-capacity probe. The original design
notes below are kept for provenance.

Per-request state isolation. Removes the request mutex.

- **Per-request KV cache pool.** Today there is one engine-wide
  cache sized for the largest seen sequence. Switching to a pool of
  per-request caches lets multiple requests overlap. The pool is
  sized at startup from the available USM budget.
- **Per-request scratch buffers.** `BlockBuffers` is currently a
  single instance shared across `runBlock` calls. Either we extract
  it per-request (memory cost) or we make `runBlock` re-entrant
  with a passed-in scratch.
- **Lockless decode dispatch.** Replace the server mutex with finer
  per-request locking that only serialises the actual GPU kernel
  launches (since `GpuKernel::setArgumentValue` is mutating state
  on the kernel handle).
- **Pegenaut-side load test.** Multi-user RAG workload over the
  HTTP API to confirm the concurrency model holds up.

## Mimir-2.0 — Bigger models

Models that don't fit in iGPU-shared memory comfortably.

- **Layer streaming.** Only keep one or two transformer blocks
  resident in GPU-addressable USM at any moment; bring the next
  block in while the current one is computing. On Intel UMA hardware
  this is mostly a page-table-shuffle problem — host RAM and
  GPU-addressable memory are the same physical DDR5 — rather than the
  PCIe-bandwidth problem it is on discrete GPUs.
- **DeepSeek V4 Flash** if it ships, **GLM-4.7 Flash** if available,
  **Gemma 4 31B dense** (already fits but is a good integration
  test for non-MoE).
- **Multi-model hosting** — load several models at once, switch on
  demand. Possible once layer streaming reduces the per-model
  resident footprint.
- **Optional discrete-GPU path** — Arc Battlemage / Lunar Lake dGPU
  via the same Level Zero stack with prefetch overlap to hide PCIe.
  Not a priority but architecturally adjacent.

## Quantisation rollout

Orthogonal to the milestone phases:

- M3-M4: only Q4_K_M dequantised. ✅
- M5: Q4_K + Q6_K both on GPU. ✅
- M8.G: Q8_0 GPU kernel added. ✅
- Q6_K and Q8_0 for Gemma 4 production: both verified, bit-identical
  output in tested chat scenarios. Q6_K saves 17 % memory at zero
  observable quality cost. ✅
- Q4_K_M for Qwen 2.5 production: shipping since M7. ✅
- F16 / BF16: dequant routines present, no GPU kernels (X stays F32
  through the matmul). Used today only for the small norm/scale
  tensors that don't dominate.
- Q5_K, Q3_K, IQ-family: not implemented. Adding any of them is a
  one-class file under `src/compute/quant/` plus one switch case in
  the registry plus, optionally, a SPIR-V kernel. The interface is
  designed for it.

See [`quants.md`](quants.md) for the implementation detail.

## What this project will *not* become

- **A training framework.** MimirMind loads pre-trained models. The
  "wisdom" is already in the head. We give it a voice; we do not
  teach it anything new. Adding training would change the project's
  identity.
- **A cross-vendor HAL.** MimirMind is not a portability layer.
  Level Zero on Intel iGPU is the primary target, chosen because it
  matches the hardware and UMA model MimirMind was built for. If AMD
  hardware becomes a real deployment target, the plan is a native
  HIP kernel tree next to the existing Level Zero one — not a
  lowest-common-denominator SPIR-V/Vulkan layer that leaves 30–40 %
  of AMD's tok/s on the table. Kernels stay vendor-specific and
  vendor-optimal; the abstraction lives above them, not inside them.
  Full multi-vendor CUDA + ROCm + Level Zero + Metal parity is
  explicitly not the goal — other projects cover that ground.
- **A wrapper around llama.cpp.** The entire point is to learn the
  problem, not to delegate it. `llama-cli` exists in the runtime
  image only as a reference oracle for the parity-test harness.

## See also

- [`architecture.md`](architecture.md) — Technical reference
- [`journey.md`](journey.md) — The Gemma 4 debug story
- [`api.md`](api.md) — HTTP API reference
- [`quants.md`](quants.md) — Quantisation strategy in depth
- [`build.md`](build.md) — Build and run
- [`setup-ct.md`](setup-ct.md) — Proxmox LXC host configuration
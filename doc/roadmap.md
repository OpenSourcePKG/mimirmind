# Roadmap

Norse-themed phase names. Each phase has to be runnable and verifiable
before the next one starts. Mimirmind exists to query a *loaded*
knowledge source; Mimir was the keeper of the *Well of Wisdom* whose
preserved head whispered counsel to Odin in every crisis.

## Project Well — *Alpha*  ✅ Complete (2026-06-28)

The source. Reading from disk into memory.

| M | Item | Status |
|---|---|---|
| M1 | Level Zero context, device enumeration, driver checks | ✅ done — container smoke verified on Meteor Lake |
| M2 | USM allocator: probe limits, shared pool, per-tensor allocation, free-list, stats | ✅ done — phase-2 ceiling probe capped at 4 GiB so shared hosts don't OOM during diagnostics |
| M3 | GGUF format reader, tensors mapped into USM via mmap + copy | ✅ done — mmap dropped after load (madvise + fadvise `DONTNEED`) so the page cache doesn't hold the model twice |
| M4 | CPU-only forward pass + autoregressive generation, output parity vs `llama-cli` | ✅ done — bit-genau Token-Parity on Qwen 2.5 7B Q4_K_M |

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
| M9 | Layer streaming + multi-request concurrency | Open — see Mimir-1.1 and Mimir-2.0 below |

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
| M8.F | QuantType OOP refactor + 82 unit tests across 4 binaries |
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
  this engine was originally built to feed. Mimirmind plugs in as the
  LLM provider via the OpenAI-compatible API.

## Mimir-1.1 — Concurrency

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

- **A training framework.** Mimirmind loads pre-trained models. The
  "wisdom" is already in the head. We give it a voice; we do not
  teach it anything new. Adding training would change the project's
  identity.
- **A multi-vendor abstraction.** No CUDA backend, no ROCm backend,
  no Vulkan backend. Mimirmind is built for Intel Level Zero on
  UMA-capable iGPUs and uses that platform as a target, not as one
  of many. Other platforms have plenty of options.
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
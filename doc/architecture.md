# Architecture

> Technical reference. Audience: someone reading the code.

Mimirmind is a single-process inference engine: one model loaded into
shared memory, served via an OpenAI-compatible HTTP endpoint. The
process is split into a small number of clearly-bounded components
that you can understand one at a time.

## Component layout

```
┌────────────────────────────────────────────────────────────────┐
│                          ApiServer                             │
│  /v1/chat/completions, /v1/models, /health, SSE streaming      │
└────────────────────────────────────────────────────────────────┘
                                │
┌────────────────────────────────────────────────────────────────┐
│                       InferenceEngine                          │
│  loadModel · tokenize · generate · stop conditions · sampling  │
└────────────────────────────────────────────────────────────────┘
                                │
         ┌──────────────────────┼─────────────────────┐
         │                      │                     │
┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│  ArchBackend   │    │   GpuMatmul     │    │     GpuOps     │
│  Qwen2Backend  │    │ Q4_K / Q6_K /   │    │  rmsnorm, rope │
│  Gemma4Backend │    │ Q8_0 kernels +  │    │  silu_mul,     │
│  runBlock(...) │    │ CPU fallback    │    │  gelu_mul, ... │
└────────────────┘    └─────────────────┘    └────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
       ┌────────────────┐              ┌────────────────┐
       │   KvCache      │              │ UsmAllocator   │
       │  per-layer K/V │              │ zeMemAllocShared│
       │  ring buffer   │              │ free-list pool  │
       └────────────────┘              └────────────────┘
                                                │
                                  ┌─────────────────────────┐
                                  │       L0Context         │
                                  │ Level Zero driver/dev/  │
                                  │ context handles, RAII   │
                                  └─────────────────────────┘
```

## Target hardware

Intel **Meteor Lake** systems with the integrated **Intel Arc Graphics**
(Xe-LPG) iGPU. Reference development target: Core Ultra laptop /
NUC-class machine with at least 64 GiB of DDR5 shared between CPU and
iGPU. Lunar Lake (Xe2-LPG) is API-compatible and works without code
changes; we have not benchmarked it.

What the iGPU is good at: arithmetic over data that is already in
RAM. What it is *not* good at: pretending to be a discrete GPU with a
small fast pool. Mimirmind plays to the first; many other engines try
to retrofit the second.

## Memory: per-tensor USM, no monolith

The Level Zero loader on Meteor Lake currently caps single
`zeMemAllocShared` calls at ~4 GiB on most driver versions. Mimirmind
takes that constraint at face value and allocates **one block per GGUF
tensor** — 658 of them for Gemma 4 — rather than fighting it with the
relaxed-allocation env vars.

`UsmAllocator` is a segregated-bucket pool: power-of-two buckets from
4 KiB to 1 MiB with a per-bucket free-list, plus a passthrough path for
oversized allocations. The largest single tensor at Q8_0 is the
ffn_gate_up_exps weight at 539 MiB per layer × 30 layers, well under
the cap.

`MIMIRMIND_USM_PROBE_TOTAL_GIB=0` skips the optional phase-2 ceiling
sweep on shared multi-tenant hardware so we don't OOM the host. The
phase-1 per-allocation probe always runs because it's cheap.

The GGUF is `mmap`'d, tensor bytes are `memcpy`'d into USM, then the
mmap is dropped (`madvise(DONTNEED)` + `fadvise(DONTNEED)`) so the
page cache doesn't end up holding the model twice. Total resident
memory after load is the live USM total alone.

## The forward pass

`InferenceEngine::generate` runs a standard prefill-then-decode loop:

1. **Tokenize** the prompt (BPE for Qwen, SentencePiece-style for
   Gemma).
2. **Embedding lookup**: `compute::embeddingLookup` reads one row from
   the (Q-quantised) token-embedding table per token, dequants into a
   F32 buffer.
3. **For each of N transformer blocks**, dispatch to the architecture
   backend's `runBlock(blockIdx, x, T, KvCache&, BlockBuffers&)`.
4. **Final norm + LM head** (`output_norm` × `token_embd^T` for
   weight-tied models like Gemma; separate `output.weight` otherwise).
5. **Sample** one new token from the logits via the configured
   strategy (greedy / temperature / top-k / top-p).
6. Append the sampled token to the KV cache, loop to step 3 with
   T = 1 for decode mode.

Each `runBlock` writes its attention K/V into the per-layer slot of
the KV cache before attention so the next decode step sees them.

## ArchBackend OOP split (M8.A)

`runBlock` is virtual on `ArchBackend`. Two concrete subclasses live
side by side:

- **`Qwen2Backend`** for the llama-family architectures: standard
  attention with no per-head norms, GQA via `nKvHeads`, SwiGLU FFN
  with SiLU, single RMSNorm before each sub-block, ChatML template.

- **`Gemma4Backend`** for the Gemma 4 26B-A4B MoE: per-layer
  head_dim and KV-head count (SWA layers use 8×256, full layers use
  2×512 with alternative `V = K` attention), `attn_q_norm` and
  `attn_k_norm` before RoPE, bare RMSNorm on V before the cache, dense
  + MoE FFN running in parallel with their own normalisation
  choreography, layer-output scaling at the end of each block.

`createArchBackend(architecture, config, weights, ops, gmm)` is the
factory. `isSupportedArchitecture(name)` is an inline header-only
predicate so unit tests can verify dispatch without linking the
concrete backends.

## Quantisation as a polymorphic type (M8.F)

`compute::QuantType` is a polymorphic interface with one singleton
subclass per supported `model::GgmlType`: F32, F16, BF16, Q4_K, Q6_K,
Q8_0. Each class knows its block layout (`blockElements`,
`blockBytes`), its dequantisation routine (`dequantToF32`), and
optionally the SPV module name of a matching GPU matmul kernel
(`gpuMatmulModule`).

The registry is one switch case in `QuantTypeRegistry.cpp`. Adding a
new quantisation type is a one-file add plus that switch case — the
dispatch elsewhere (`compute::matmul`, `GpuMatmul`,
`Gemma4Backend`'s expert-byte calculations) goes through the
interface.

See [`quants.md`](quants.md) for which Q-formats have GPU kernels and
why.

## GPU kernels

Compiled at build time by Intel's `ocloc` from `kernels/*.cl` into
SPIR-V blobs under `build/spv/`. The runtime image bakes those into
`/usr/local/share/mimirmind/spv/`.

**Element-wise + normalisation** (`GpuOps`):
`rmsnorm`, `rmsnorm_gemma` (1+w variant), `rmsnorm_no_weight`,
`add_bias`, `add_residual`, `silu_mul`, `gelu_mul`, `rope_inplace`,
`rope_inplace_ff` (with frequency factors), `mul_scalar`.

**Matmul** (`GpuMatmul`): `matmul_q4k_vec`, `matmul_q6k_vec`,
`matmul_q8_0_vec`. Each follows the same workgroup geometry — 64
threads laid out as 4 sub-groups × 16 lanes, one output element per
sub-group via `sub_group_reduce_add`. The Q6_K kernel additionally
uses Kahan-compensated per-thread accumulation; Q8_0 doesn't need it
(no sub-band scales, lower per-block noise).

All kernels are verified element-wise against CPU reference math
inside the `gpu_tests` binary.

## KV cache

`KvCache` allocates a single contiguous USM block per layer at
`loadModel` time, sized for the max sequence length the caller passes
in. K and V are interleaved per token-position so the attention dot
product reads them with strided access.

The current layout is straight-forward: per layer, per token-position,
contiguous K then contiguous V. There is no sliding-window eviction
yet — sliding-window attention (SWA) is implemented but the cache
itself does not yet drop history outside the window. For Gemma 4's
1024-token SWA layers this means we hold more cache than the model
will read, which costs memory but not correctness.

## Sampling

`compute::Sampler` is a stateful object with a `std::mt19937_64` RNG
and reusable scratch vectors. Greedy (temperature ≤ 0 or topK == 1) is
a plain argmax with no RNG use, bit-identical across runs.
Temperature path scales logits, runs a numerically-stable softmax,
applies top-K cutoff, then top-P (nucleus), then multinomial draw.
Tested on hand-crafted distributions in `compute_tests`.

## HTTP server

`ApiServer` is `cpp-httplib` wrapped around the engine. It exposes
`/health`, `/v1/models`, and `/v1/chat/completions` (both
non-streaming and SSE streaming). The chat-template dispatch happens
on the way *in* (messages → template-encoded token IDs), and the
inverse cleanup (channel-markup stripping for Gemma 4) happens on the
way *out* (decoded text → assistant content).

The server currently serialises all requests through a mutex. Removing
that mutex without breaking the shared `GpuOps` / `GpuMatmul` argument
state is the M9 work item.

## Build pipeline

- Multi-stage Dockerfile: `builder` (toolchain only), `build` (source
  + compile), `llamacpp` (the reference oracle for parity dumps),
  `runtime` (no toolchain, just the binary + SPV blobs).
- CMake ≥ 3.21, Ninja, C++20, strict warnings (`-Wall -Wextra
  -Wpedantic -Wshadow -Wconversion -Wsign-conversion ...`).
- Tests are CMake targets in their own right: `quant_tests`,
  `arch_tests`, `compute_tests`, `gpu_tests`. The first three are pure
  CPU, the last needs `/dev/dri` to run.

See [`build.md`](build.md) for the actual build commands and
host-side prerequisites.

## See also

- [`build.md`](build.md) — Build and run
- [`api.md`](api.md) — HTTP API reference
- [`quants.md`](quants.md) — Quantisation strategy in depth
- [`roadmap.md`](roadmap.md) — What's done and what's next
- [`journey.md`](journey.md) — The Gemma 4 debug story
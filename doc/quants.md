# Quantisation

Mimirmind reads the standard GGUF quantisation formats and runs as much
of the matmul work as possible on the iGPU. This document describes
which formats are supported, how the GPU kernels are structured, and
what happens when a tensor's type does not have a GPU kernel.

## Supported types

| GgmlType | Dequant | GPU matmul | Notes |
|---|---|---|---|
| F32 | trivial passthrough | none (X is already F32) | Used for norm weights, scales |
| F16 | per-element half→float | none | Used for some smaller models |
| BF16 | per-element bf16→float | none | Brain-float fallback |
| Q4_K | 256-elem super-block, 144 B | `matmul_q4k_vec.spv` | Qwen 2.5 dominant format |
| Q6_K | 256-elem super-block, 210 B | `matmul_q6k_vec.spv` (Kahan-accumulated) | Gemma 4 Q6_K format |
| Q8_0 | 32-elem block, 34 B | `matmul_q8_0_vec.spv` | Gemma 4 Q8_0 format |

Each `QuantType` is a singleton subclass in `src/compute/quant/` and is
registered in `QuantTypeRegistry.cpp`. Adding a new type (Q5_K, IQ4_NL,
MXFP4, ...) is a one-file add plus one switch case in the registry —
the dispatch elsewhere goes through the interface.

See `src/compute/QuantType.hpp` for the polymorphic interface.

## GPU kernel design

Every matmul kernel follows the same launch geometry:

- Workgroup: **64 threads** = 4 sub-groups × 16 lanes.
- Each sub-group co-computes **one output element** via
  `sub_group_reduce_add`.
- Each workgroup emits **4 outputs**, so `global_size = ceil(N/4) * 64`.
- Sub-group size pinned at 16 via `intel_reqd_sub_group_size`.

The X-vector (the activation we are matmuling against the quantised
weights) lives in F32 throughout. We tile it into shared local memory
in 1024-element chunks (4 KiB SLM per workgroup) so that all 16 lanes
of a sub-group read the same cache line. The weight bytes stream from
USM-shared memory; on UMA hardware there is no host↔device transfer,
just a pointer dereference.

### Q4_K (`matmul_q4k_vec.cl`)

Super-block of 256 elements in 144 bytes:
- `fp16 d` (scale-of-scales) + `fp16 dmin` (scale-of-mins) = 4 B
- 12 B packed scales: eight 6-bit scales + eight 6-bit mins
- 128 B of 4-bit quants (256 nibbles, low-nibble first)

Per element: `v = d · scale · q - dmin · min` over the relevant
32-element sub-block. The kernel unpacks one super-block per iteration
via `get_scale_min_k4` (which has two cases depending on whether
`j < 4` or not) and accumulates into a per-lane F32 sum.

### Q6_K (`matmul_q6k_vec.cl`)

Super-block of 256 elements in 210 bytes:
- `uint8 ql[128]` — low 4 bits of 256 6-bit quants
- `uint8 qh[64]` — high 2 bits of 256 6-bit quants
- `int8 scales[16]` — one scale per 16-element sub-block
- `fp16 d` — super-block scale

Per element: `q = ((ql_nibble) | (qh_pair << 4)) - 32` (range
`[-32, 31]`), then `v = d · scale[sub_block] · q`.

The kernel uses **Kahan-compensated summation** for the per-lane
accumulator. Without it, the ~K/16 ≈ 176 terms each lane accumulates
for K = 2816 (Gemma 4 d_model) lose ~7 bits of mantissa precision
through fp32 reorder. With Kahan we keep ~24 bits all the way to the
final sub-group reduce.

That said: the Q6_K kernel was the source of the
"precision side-quest" during Gemma 4 bring-up. Per-stage parity
dumps after the Kahan fix showed the per-matmul error was already
smaller than the systematic divergence we were chasing. The Kahan
compensator stays in for correctness; it was not the actual fix for
Gemma 4 output quality. See [`journey.md`](journey.md) for the full
story.

### Q8_0 (`matmul_q8_0_vec.cl`)

Block of 32 elements in 34 bytes:
- `fp16 d` — block scale
- `int8 qs[32]` — 32 signed quants

Per element: `v = d · qs[i]`. The simplest format we support; the
dequant is a single multiply.

The kernel does **not** use Kahan compensation. Q8_0 has no
sub-band scales and no bit unpacking, so per-block dequant noise is
substantially lower than Q6_K. Plain fp32 mad-chain stays well inside
representable range for the ~K/16 terms each lane accumulates.

Adding the Q8_0 kernel was the single largest decode-speed
improvement in the project's history — Gemma 4 went from
2.3 s/tok (with Q8_0 weights running through the CPU fallback) to
145 ms/tok (with the new GPU kernel). On the Q8_0 build of Gemma 4
the biggest weight tensor is `ffn_gate_up_exps` at 539 MiB per layer
× 30 layers — almost all of decode time was waiting on those CPU
matmuls before this kernel landed.

## CPU fallback

Types without a GPU kernel (F32, F16, BF16) fall through to
`compute::matmul`. The CPU path:

1. Dequantises one weight row at a time into a scratch F32 buffer.
2. Computes the dot product against the X-vector in a `double`
   accumulator.
3. Casts the result to F32 at the end.

The double accumulator means the CPU path is **bit-exactly equal to
llama.cpp's CPU output**, modulo the dequant routine itself, which
follows the GGML spec.

Calling `GpuMatmul::matmulAsync(unsupported_type, ...)` transparently
flushes any pending GPU work to preserve ordering, then runs the CPU
matmul synchronously. From the architecture-backend's point of view
there is no behavioural difference; it just gets slower.

## Choosing a quantisation for production

For Gemma 4 26B-A4B specifically (verified empirically on 2026-06-30):

| Quant | Disk | Memory | Decode | Quality |
|---|---:|---:|---:|---|
| Q6_K | 21 GB | 21.3 GiB | 148 ms/tok | bit-identical to Q8_0 in tested greedy chat |
| Q8_0 | 27 GB | 25.0 GiB | 145 ms/tok | bit-identical to Q6_K in tested greedy chat |

**Recommended default: Q6_K.** Saves 17 % of memory (3.7 GiB) at zero
observable quality cost for short-to-medium chat responses. The
per-stage tensor parity diff between Q6_K and Q8_0 weights is
substantial inside the engine, but greedy decoding picks the same
tokens.

For long-context generations (8k+ tokens of model output), Q8_0 might
hold the line longer if cumulative drift starts to mismatch — we have
not measured.

For Qwen 2.5 we use Q4_K_M (the M variant just controls the per-tensor
quant choice; the loader does not care). Q4_K_M sits in ~5 GiB and
runs at 126 ms/tok, which is what we use for the speed baseline.

## Verifying the kernels

`tests/gpu_tests.cpp` (binary: `gpu_tests`, link `cmake --build build
--target gpu_tests`) exercises each GPU kernel against the CPU
reference math element-wise, with hand-crafted Q-block byte patterns
plus a 64-row replicated Gemma-4-d_model-sized matmul. 16 tests, runs
in under a second on the iGPU.

If you change a kernel and `gpu_tests` still passes, the change did
not break parity. If `gpu_tests` fails after a change, the failure
report points at the first divergent output index — usually obvious
enough to bisect the kernel from there.
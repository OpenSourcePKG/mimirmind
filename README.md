<p align="center"><img src="./doc/logo.svg" alt="MimirMind" width="180"></p>

<h1 align="center">MimirMind</h1>

<p align="center"><strong>A from-scratch C++20 inference engine. One codebase serves Gemma 4 and Qwen3-Next across an Intel laptop iGPU, an NVIDIA GB10 Blackwell superchip, and AMD ROCm.</strong></p>

<p align="center">No llama.cpp. No ggml. No PyTorch. No SYCL. Just the OpenAI API and hand-written kernels.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C.svg" alt="C++20">
  <img src="https://img.shields.io/badge/runtime%20deps-zero%20ML%20frameworks-brightgreen.svg" alt="Zero ML framework dependencies">
  <img src="https://img.shields.io/badge/API-OpenAI%20compatible-black.svg" alt="OpenAI-compatible API">
</p>

---

Every line is in this repo — transformer blocks, quant matmul kernels, KV cache, MoE router, linear-attention scan, paged-attention scheduler, HTTP server. `llama.cpp` and friends exist only as the *reference oracle* we check against, bit-for-bit. Point the OpenAI SDK at it and it just works.

Most local-inference stacks wrap someone else's engine and wait on their roadmap. We own the whole stack: when a new architecture or accelerator shows up, we write the kernel and it runs the next day — then we **measure it on the real target and publish the number, good or bad.**

## Highlights

- 🧠 **Newest architectures.** Gemma 4 (26B-A4B MoE, 12B, E-series) and Qwen3-Next / Qwen3.6 35B-A3B — a hybrid of full attention, **GatedDeltaNet linear attention**, and MoE. Plus Qwen 2.5 and Qwen3-Coder-Next. Flagship reasoning models, not toys.
- 🔌 **Drop-in OpenAI API.** `/v1/chat/completions` (streaming SSE), `/v1/embeddings`, `/v1/rerank`, `/v1/audio/*`. Tool / function calling, grammar-constrained JSON, and thinking-mode reasoning channels wired in.
- 🎛️ **Every quant, hand-kernelled.** Q4_K/Q5_K/Q6_K/Q8_0 GGUF *and* **NVFP4** 4-bit tensor-core weights — dequant happens *inside* the matmul, never a round-trip to F32, every kernel verified element-wise against a reference.
- 🏎️ **Serious serving engineering.** Continuous batching, **PagedAttention** (split-K, warp-shuffle), cuDNN fused-flash prefill, **cross-slot prefix sharing** (KV reuse across tenants), FP8/E4M3 KV capacity tier, per-tenant TLS + auth + admission control.
- 🗣️ **Multimodal.** Speech-to-text (Whisper) and neural text-to-speech (Orpheus + SNAC) served through the same OpenAI-compatible endpoints.
- 🔁 **Model-switching daemon (Munin).** Hold several models resident in unified memory and route each request to the right one — no reload, no cold start.
- ✅ **Test-anchored.** Hundreds of unit tests from hand-crafted quant-block byte patterns up through full GPU kernels on real silicon, plus bit-exact parity gates against the reference oracle.
- 🧱 **Pure, auditable C++20.** `std::span`, `std::expected`, `enum class`, RAII over raw handles, one class per file, no exceptions across the driver boundary.

## Runs on real hardware

| Accelerator | Backend | Status |
|---|---|---|
| **Intel Meteor Lake iGPU** (Xe-LPG, Unified Memory) | Level Zero | 🟢 **Mimir-1.0 — released.** Gemma 4 26B MoE, coherent, on a laptop |
| **NVIDIA DGX Spark GB10** (Grace + Blackwell, 128 GB) | CUDA + NVFP4 | 🟡 **Bragi (Mimir-2.0).** Serving-class, multi-tenant (≥64 chats) |
| **AMD RDNA3+** | HIP / ROCm | 🔵 home-lab tier |
| **x86-64 / ARM64** | CPU | ⚪ reference oracle & fallback |

One backend-neutral interface; concrete backends auto-select at runtime. A fourth accelerator is an interface to implement, not a fork.

### The headline: a 26B MoE on a laptop iGPU

Intel Meteor Lake Core Ultra, single integrated GPU, shared DDR5:

| Model | Quant | Memory | Decode |
|---|---|---:|---:|
| Gemma 4 E4B Instruct | Q4_K_M | 2.5 GiB | **132 ms/tok** |
| Gemma 4 26B-A4B Instruct | Q6_K | 21.3 GiB | **148 ms/tok** |
| Gemma 4 26B-A4B Instruct | Q8_0 | 25.0 GiB | **145 ms/tok** |

A 26-billion-parameter MoE at ~dense-7B speed (only ~4 B active/token) — on a consumer iGPU, no discrete card. On GB10 the same engine flips to NVFP4 serving mode with paged prefill and cross-slot KV reuse across many concurrent chats.

## Quick start

```bash
cp config.example.json config.json && $EDITOR config.json
export MIMIRMIND_MODELS_DIR=/path/to/your/models
export MIMIRMIND_CONFIG_HOST=$PWD/config.json
docker compose -f docker-compose.server.yml up -d
```

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}' \
  | jq -r '.choices[0].message.content'
# -> "The capital of France is **Paris**."
```

Native build: `cmake -B build && cmake --build build`. Accelerator setup in [`doc/build.md`](doc/build.md).

## The road (Norse-themed)

**Well** (load) → **Envoy** (kernels) → **🎯 Mimir-1.0** (released: OpenAI HTTP + Gemma 4 on Intel) → **Sleipnir** (Xe-LPG perf) → **🌈 Bragi** (serving-class CUDA on GB10: NVFP4, batching, multi-tenant). Cross-cutting: **Munin** (model-switching daemon), **Heimdall** (auth/TLS), **Loki/Nornir** (LoRA), **Bifröst** (Anthropic↔OpenAI proxy).

We work measure-first: build a lever, run it on the real target, ship it with numbers or shelve it with data. Correctness — bit-exact parity gates — before speed. There's even a book being written alongside: *"From Zero to an Inference Engine."*

---

<p align="center">Apache 2.0 · <em>"Mímir's head Odin takes, and it tells him many tidings true."</em></p>

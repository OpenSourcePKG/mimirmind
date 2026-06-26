<p align="center"><img src="./doc/logo.svg" alt="Mimirmind" width="180"></p>

<h1 align="center">Mimirmind</h1>

<p align="center"><em>Odin carried Mimir's preserved head with him,<br/>seeking counsel from the wisest of all beings.</em></p>

---

A standalone C++ inference engine for **GGUF** language models, built to
exploit the **Unified Memory Architecture (UMA)** of integrated Intel Arc
GPUs through **oneAPI Level Zero** with **Unified Shared Memory (USM)**.

> **Mimirmind loads pre-trained models. It does not train them.**
> The wisdom is already in the head — we only give it a voice.

## Why "Mimir"?

In Norse mythology, Mimir was the keeper of the *Well of Wisdom*. After his
beheading, his preserved head continued to whisper counsel to Odin in every
crisis. Mimir is the archetype of a *loaded* knowledge source: nothing new
is taught to him — the knowledge is already there, waiting to be queried.

That mirrors what this engine does. A GGUF file is the preserved head: a
frozen distillation of training that happened somewhere else. **Mimirmind
is the mechanism that lets it speak again.**

## Target Hardware

Intel **Meteor Lake** systems with the integrated **Intel Arc Graphics**
(Xe-LPG) iGPU. The reference target during development is a recent
high-end Core Ultra laptop / NUC-class machine with at least 64 GB of
DDR5 shared between CPU and iGPU.

The point of this engine is to use UMA properly: **no host↔device copies**.
Model weights, KV-cache, and intermediate tensors all live in a single
`zeMemAllocShared` pool that both the CPU and the iGPU read from directly.
Many small per-tensor allocations side-step the ~4 GB single-allocation cap
that the Level Zero loader currently enforces on Meteor Lake.

## Target Model

**Gemma 4** — including the larger variants (12B Dense, 26B-A4B MoE) that
the mainstream Intel inference stacks (OVMS, LocalAI/SYCL) currently cannot
serve on Meteor Lake.

Gemma 3 (4B, 12B) is brought up first as a *verification baseline*,
cross-checked tensor-for-tensor against `llama.cpp/llama-cli`, before the
Gemma 4 work begins.

## Roadmap

Named after the stages of Mimir's myth. Each phase must be runnable and
verifiable before the next one starts.

### Project Well — *Alpha*

The source. Reading from disk, into memory.

- Level Zero context, device enumeration, driver checks
- USM allocator: shared-memory pool, per-tensor allocation, free-list, stats
- GGUF format reader (tensors mapped directly into USM)
- CPU-only forward pass for Gemma 3 4B, output parity with `llama-cli`

### Project Envoy — *Beta*

The messenger. The head learns to speak.

- GPU kernels in Level Zero: RMSNorm, RoPE, matmul, softmax, SwiGLU
- KV-cache in USM, autoregressive sampling (greedy + temperature)
- Token-for-token parity with `llama-cli` on Gemma 3 4B and 12B

### Mimir-1.0 — *Release*

The counsel. The head is ready to be queried in production.

- OpenAI-compatible HTTP API (`/v1/chat/completions`)
- Gemma 4 architecture: hybrid local/global attention, MoE routing
- Layer streaming for models that exceed comfortable in-memory size

## Tech Stack

- **C++20**, CMake
- **oneAPI Level Zero** loader (`level-zero-dev`)
- **SentencePiece** for tokenization
- **cpp-httplib** + **nlohmann/json** for the HTTP wrapper *(Mimir-1.0 only)*
- **No `llama.cpp` / `ggml` as a runtime dependency.** The point is to learn
  it, not to wrap it. `llama-cli` is used only as a *reference oracle* for
  output verification during development.

## Build & Run

Everything runs inside Docker — the developer machine stays free of the
Intel compute SDK. Only the source tree lives on the host.

```bash
# Iterative dev: edit source on host, rebuild in toolchain container.
# Artefacts land in ./build on the host.
docker compose run --rm builder

# Or, drop into a shell in the toolchain image:
docker compose run --rm builder bash

# Ship the runtime image (multi-stage, fully reproducible):
docker compose build mimirmind

# Run M1 smoke test (Level Zero device enumeration):
docker compose run --rm mimirmind
```

`docker-compose.yml` already passes `/dev/dri` and the `video` + `render`
groups (GIDs 44 + 104) — required for the iGPU to be reachable from the
container. See `doc/setup-ct.md` for host- / LXC-side prerequisites.

## Status

**M0 — Bootstrapping.** Repo skeleton only. No code yet beyond M1's Level
Zero device-enumeration smoke test.

## License

TBD.
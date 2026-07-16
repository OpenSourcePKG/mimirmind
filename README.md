<p align="center"><img src="./doc/logo.svg" alt="MimirMind" width="180"></p>

<h1 align="center">MimirMind</h1>

<p align="center"><strong>A standalone C++20 inference engine for GGUF language models on Intel Arc integrated GPUs — written from scratch, no llama.cpp, no PyTorch, no SYCL.</strong></p>

<p align="center"><em>Odin carried Mimir's preserved head with him, seeking counsel from the wisest of all beings.</em></p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a></p>

---

## What it is

MimirMind runs large GGUF-quantised language models on the **integrated
Intel Arc GPU of Meteor Lake / Lunar Lake** systems through **oneAPI
Level Zero** with **Unified Shared Memory**. It speaks the
**OpenAI Chat Completions API** so existing clients drop in without
modification.

It is a from-scratch implementation: the transformer block, the
quantised matmul kernels, the KV cache, the sampling, the chat-template
renderer, the HTTP server — every line is in this repository. There is
no `llama.cpp`, no `ggml`, no `pytorch`, no `transformers` in the
runtime path. Those projects exist only at the boundary of this one as
the *reference oracle* used to verify bit-level output parity during
development.

## Why this exists

Mainstream inference stacks for Intel iGPUs (OpenVINO Model Server, the
LocalAI SYCL backend, IPEX-LLM) have a shared blind spot: the **Gemma 4
26B-A4B Mixture-of-Experts** model, which Google released as their
flagship "open-weights reasoning" model in 2026. The hardware that
should run it — a recent Core Ultra laptop with 64 GiB of shared
DDR5 — is *capable*. The stacks just haven't shipped support for the
gemma4 architecture yet.

MimirMind was built to close that gap on the hardware that was already
on the desk. Along the way it turned out to be a remarkably clean way
to use the Intel iGPU's Unified Memory Architecture for what it is good
at, instead of treating it like a small discrete GPU.

## Headline numbers

On an Intel Meteor Lake Core Ultra, single iGPU, 58 GiB of shared
memory budget, bench mode (`governor.gpuClockPin: "rp0"`,
`features.clr: true` in `config.json`):

| Model | Quant | Memory | Decode | Output verified |
|---|---|---:|---:|---|
| Qwen 2.5 7B Instruct | Q4_K_M | 4.4 GiB | **126 ms/tok** | Bit-exact vs llama-cli |
| **Gemma 4 E4B Instruct** | **Q4_K_M** | **2.5 GiB** | **132 ms/tok** | Greedy match vs reference |
| **Gemma 4 26B-A4B Instruct** | **Q6_K** | **21.3 GiB** | **148 ms/tok** | Greedy match vs reference |
| **Gemma 4 26B-A4B Instruct** | **Q8_0** | **25.0 GiB** | **145 ms/tok** | Greedy match vs reference |

That's a 26-billion-parameter MoE running at ~7 tokens per second on a
consumer integrated GPU — about the speed of a dense 7B model, because
only 4 B parameters are active per token. The Gemma 4 E4B variant
delivers comparable throughput at ~1/8 the memory footprint, making it
the sweet spot for interactive workloads on constrained hardware.

## What makes it different

**Built for UMA, not adapted.** Discrete-GPU inference engines treat
host RAM and GPU VRAM as separate worlds connected by a PCIe straw.
On Meteor Lake there is no straw — the CPU and the iGPU read from the
same physical DDR5. MimirMind allocates every weight, every KV-cache
row, and every scratch buffer through `zeMemAllocShared`, then hands
the *same* pointer to GPU kernels and to CPU helpers. There are no
host↔device copies.

**Per-tensor allocation, no monolith.** The Level Zero loader on
Meteor Lake currently caps single allocations at ~4 GiB. Many engines
work around this by enabling relaxed-allocation env vars and praying.
MimirMind allocates one block per GGUF tensor — 658 of them for
Gemma 4 — through a segregated-bucket free-list allocator. The largest
single tensor (`ffn_gate_up_exps` at 539 MiB for Q8_0) sits well under
the cap and the model loads in 25 GiB without any
runtime-flag gymnastics.

**Per-quantisation GPU kernels.** Q4_K, Q6_K, and Q8_0 each have a
dedicated SPIR-V kernel compiled at build time by `ocloc`. The kernels
dequantise on-the-fly inside the matmul — we never materialise a
21-GiB tensor as 80 GiB of F32. Each one is verified element-wise
against a `double`-accumulator CPU reference inside the test binary.

**OpenAI-API on the wire, native engine underneath.** Existing
clients (LangChain, the OpenAI Python SDK, your shell-script with
`curl`) point at it without modification. Streaming SSE works. Chat
templates for Qwen (ChatML), Gemma 2/3 (`<start_of_turn>`), and
Gemma 4 (`<|turn>` + thinking-channel markup) are wired into the
server and dispatched by the model's reported architecture.

**210+ unit tests, four binaries.** `quant_tests`, `arch_tests`,
`compute_tests`, and `gpu_tests` exercise the engine from
hand-crafted Q-block byte patterns up through full GPU kernels
on the actual iGPU. The whole suite runs in seconds.

**Pure C++20.** Modern idioms (`std::span`, `std::expected`,
`enum class`, RAII through Level Zero handles). One class per file.
No exceptions across the Level Zero boundary. Trivially auditable.

## Try it

On a host with an Intel iGPU and a Q-quantised Gemma 4 or Qwen GGUF,
either build the image locally (`docker compose build`) or point
`MIMIRMIND_IMAGE` at a pre-built image in your own registry. The
compose file defaults to `mimirmind:latest`.

```bash
# Copy the example config and edit it for your host — model path,
# governor/thermal, feature toggles, etc.
cp config.example.json config.json
$EDITOR config.json

# Point the compose at the models dir + your config, then start.
export MIMIRMIND_MODELS_DIR=/path/to/your/ggufs
export MIMIRMIND_CONFIG_HOST=$PWD/config.json
# Optional: pull from your own registry instead of the default local tag.
# export MIMIRMIND_IMAGE=your-registry.example/mimirmind:latest
docker compose -f docker-compose.server.yml up -d
```

Then talk to it like it's GPT-4:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }' | jq -r '.choices[0].message.content'
# -> "The capital of France is **Paris**."
```

Streaming with `"stream": true` produces SSE just like the official API.

For the full setup, including the host-side `/dev/dri` passthrough
quirks, see [`doc/build.md`](doc/build.md).

## Configuration

Every runtime knob lives in `config.json`. There are no `MIMIRMIND_*`
env vars any more — copy `config.example.json` and edit for your host.
The loader fails fast on a missing file or unknown fields (typo
protection).

**Precedence:** CLI flag > `config.json` > compiled default.

**CLI overrides** (per invocation): `--config PATH`, `--model PATH`,
`--port N`, `--log-level`, `--log-file`. Sampling flags (`--prompt`,
`--temperature`, `--top-k`, `--top-p`, `--seed`) are per-run and don't
belong in the file.

**Config sections:**

| Section | What it controls |
|---|---|
| `models[]` | Loadable model entries (id + path). A standalone mimirmind worker binds to one model at start; **Munin** (see below) holds every `loadOnStart:true` entry resident in USM at once, and attached workers pick one by id. `speculative` additionally binds a draft alongside the target inside the same worker. |
| `server` | Port, log level, log file. |
| `runtime` | KV dtype (`f32`/`q8_0`), max context tokens, USM probe cap, SPV dir, preserve-thinking. Per-model overrides via `models[].runtime`. |
| `features` | `clr`, `flashPrefill`, `fusedQkv`, `moeGroup`, `gemm` (auto/force/disable), `gemmV2`, `gemmMinM`, `dp4a`. |
| `speculative` | Enable + target/draft model ids + `n` (draft tokens per verify round). |
| `governor` | `gpuClockPin`, `tickLog`/`tickLogFile` (per-tick NDJSON sink), fan settings, and the inline thermal profile (formerly a separate `--thermal-profile` file). |
| `diagnostics` | `parityDump`, `traceBlock0`, `traceDecodeFile`, `traceOpTimes`, `gpuBench`, `regressionAlert`. |

The compose file's header carries the complete `MIMIRMIND_*` → JSON
mapping table for anyone migrating an existing deployment.

## Status

**Mimir-1.0 — Release.** Gemma 4 (26B-A4B MoE, E4B) and Qwen 2.5 all
work end-to-end through the OpenAI HTTP API. All 210+ unit tests green;
the Level-Zero / SPV-kernel side is verified against CPU reference math
on real hardware. Command-List Replay (M-CLR) is live in production
(−6.4 % decode on E4B, verified 2026-07-06).

**Project Sleipnir** is the ongoing post-release performance and
model-extension phase. In flight:

- **M10.2 — KV-Cache dtype layer** (F32 → FP16 → Q8_0). Halves KV reads
  per attention call — the largest bandwidth-relevant lever for long
  contexts. Phase 0 foundation committed locally.
- **M-Ratatoskr — Gemma 4 MTP-drafter integration**. Google's official
  Multi-Token-Prediction drafters (released 2026-05-05) paired with our
  E2B/E4B/26B-A4B/31B targets. Projected 3× decode speedup once the
  batched-verify GEMM path (M8.K) lands as a prerequisite.

What remains before Mimir-1.0 is "tagged" rather than just "working":

- **Multi-request concurrency** — the server today serialises requests
  through a mutex. Per-request KV cache + scratch pool needs to land
  before the engine can handle parallel users.
- **KV-cache reuse across turns** for multi-turn chats without re-prefill.
- **Pegenaut integration** — the sister TypeScript RAG project that
  this engine was originally built to feed.

## What's coming

**Project Sleipnir: Speed.** Odin's eight-legged mount. The
post-Mimir-1.0 performance phase, targeting **~3× decode throughput**
on E4B and E2B once M8.K (Xe-LPG-native GEMM) and M-Ratatoskr
(Gemma 4 MTP-drafter integration) land. Bandwidth foundation (M10.2
FP16 KV cache) is in flight now.

**Mimir-1.1: Concurrency.** Per-request KV cache, scratch pool, and a
non-blocking decode loop. Removes the request mutex.

**Mimir-2.0: Bigger models.** Layer streaming for architectures that
don't fit in iGPU-shared memory — DeepSeek V4 Flash, GLM-4.7 Flash, the
Gemma 4 31B dense variant. On UMA hardware this is mostly a
page-table-shuffle problem, not the PCIe-bandwidth problem it is on
discrete GPUs.

**AMD/HIP backend.** The compute layer is being lifted behind a
`ComputeContext` abstraction on `main`, and a HIP/ROCm implementation
targeting RDNA3 (`gfx1101`, RX 7800 XT bring-up rig) is in flight on
`feat/hip-backend-skeleton` — kernel-by-kernel port from the Level-Zero
side, each verified against the CPU-reference math. Meteor-Lake Xe-LPG
remains the primary target; HIP opens a second hardware family without
introducing SYCL or a vendor-neutral graph compiler.

**Pegenaut backend.** MimirMind is the inference half of a TypeScript
RAG stack we're building in parallel. The two will ship as a unit.

See [`doc/roadmap.md`](doc/roadmap.md) for the detailed milestone
breakdown.

## The two ravens

The engine is not alone. Two companion components extend it — both
named after Odin's ravens, each aligned with what it carries.

<p align="center">
  <img src="./doc/logo-munin.svg" alt="Munin" width="120">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./doc/logo-hugin.svg" alt="Hugin" width="120">
</p>

**Munin — memory.** The persistent model-memory daemon. Munin loads
**one or more** GGUF models once into shared USM and keeps them resident
across inference-worker restarts, serving short-lived attached workers
over a chunk-based IPC. A single Munin process holds every
`loadOnStart:true` entry from `config.json` simultaneously, in one shared
USM pool; each attached worker binds to one model by id at attach time.
Cold-restart the worker to swap models, or run multiple workers in
parallel — Munin doesn't reload. Implemented and prod-shaped; lives in
[`src/munin/`](src/munin/) and ships via
[`docker-compose.munin.yml`](docker-compose.munin.yml). Per-request model
switching *from a single worker* is M-Munin.3, still open.
See [`doc/attached-rollout.md`](doc/attached-rollout.md) for the
rollout runbook.

**Hugin — thought.** The input-compression adapter. Hugin flies out to
a long document, extracts its meaning, and returns with a small
compressed representation that the base model consumes as if it were
the original context. Targets 20k-token RAG windows compressed to
64–256 memory tokens, cutting prefill traffic by two orders of
magnitude on UMA hardware. Engineering design in
[`doc/hugin.md`](doc/hugin.md); implementation is M-Hugin,
currently unscheduled.

Munin holds what has been gathered. Hugin flies out to gather it.
Mimir is the wise counsellor they both serve.

## Documentation

| | |
|---|---|
| [`doc/architecture.md`](doc/architecture.md) | The Norse-themed phase breakdown, target hardware, design decisions |
| [`doc/build.md`](doc/build.md) | Build, run, host prerequisites, GPU passthrough |
| [`doc/api.md`](doc/api.md) | OpenAI-compatible HTTP API reference, streaming, errors |
| [`doc/quants.md`](doc/quants.md) | Quantisation strategy: which Q-formats, which GPU kernels, which fallback |
| [`doc/journey.md`](doc/journey.md) | The Gemma 4 debug story and the chat-template-first lesson |
| [`doc/roadmap.md`](doc/roadmap.md) | Detailed milestones, what's done, what's next |
| [`doc/setup-ct.md`](doc/setup-ct.md) | Proxmox LXC host setup for development |

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See also
[`NOTICE`](NOTICE) for attribution requirements when redistributing.

Copyright 2026 Stefan Werfling.

## Acknowledgements

The Norse-themed phase names are not a marketing flourish. Mimir is
the keeper of the *Well of Wisdom* — the archetype of a *loaded*
knowledge source that does not learn anything new, only speaks what
is already in it. That is exactly what an inference engine does to a
trained model. The mythology earned its keep.
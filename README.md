<p align="center"><img src="./doc/logo.svg" alt="Mimirmind" width="180"></p>

<h1 align="center">Mimirmind</h1>

<p align="center"><strong>A standalone C++20 inference engine for GGUF language models on Intel Arc integrated GPUs — written from scratch, no llama.cpp, no PyTorch, no SYCL.</strong></p>

<p align="center"><em>Odin carried Mimir's preserved head with him, seeking counsel from the wisest of all beings.</em></p>

---

## What it is

Mimirmind runs large GGUF-quantised language models on the **integrated
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

Mimirmind was built to close that gap on the hardware that was already
on the desk. Along the way it turned out to be a remarkably clean way
to use the Intel iGPU's Unified Memory Architecture for what it is good
at, instead of treating it like a small discrete GPU.

## Headline numbers (Mimir-1.0)

On an Intel Meteor Lake Core Ultra, single iGPU, 58 GiB of shared
memory budget:

| Model | Quant | Memory | Decode | Output verified |
|---|---|---:|---:|---|
| Qwen 2.5 7B Instruct | Q4_K_M | 4.4 GiB | **126 ms/tok** | Bit-exact vs llama-cli |
| **Gemma 4 26B-A4B Instruct** | **Q6_K** | **21.3 GiB** | **148 ms/tok** | Greedy match vs reference |
| **Gemma 4 26B-A4B Instruct** | **Q8_0** | **25.0 GiB** | **145 ms/tok** | Greedy match vs reference |

That's a 26-billion-parameter MoE running at ~7 tokens per second on a
consumer integrated GPU — about the speed of a dense 7B model, because
only 4 B parameters are active per token.

## What makes it different

**Built for UMA, not adapted.** Discrete-GPU inference engines treat
host RAM and GPU VRAM as separate worlds connected by a PCIe straw.
On Meteor Lake there is no straw — the CPU and the iGPU read from the
same physical DDR5. Mimirmind allocates every weight, every KV-cache
row, and every scratch buffer through `zeMemAllocShared`, then hands
the *same* pointer to GPU kernels and to CPU helpers. There are no
host↔device copies.

**Per-tensor allocation, no monolith.** The Level Zero loader on
Meteor Lake currently caps single allocations at ~4 GiB. Many engines
work around this by enabling relaxed-allocation env vars and praying.
Mimirmind allocates one block per GGUF tensor — 658 of them for
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

**82+ unit tests, four binaries.** `quant_tests`, `arch_tests`,
`compute_tests`, and `gpu_tests` exercise the engine from
hand-crafted Q-block byte patterns up through full GPU kernels
on the actual iGPU. The whole suite runs in seconds.

**Pure C++20.** Modern idioms (`std::span`, `std::expected`,
`enum class`, RAII through Level Zero handles). One class per file.
No exceptions across the Level Zero boundary. Trivially auditable.

## Try it

A built image lives at `mimirmind:latest`. On
a host with an Intel iGPU and a Q-quantised Gemma 4 or Qwen GGUF:

```bash
# Point at your model file and start the server
export MIMIRMIND_MODELS_DIR=/path/to/your/ggufs
export MIMIRMIND_MODEL_PATH=/models/gemma-4-26B-A4B-it-Q6_K.gguf
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

## Status

**Mimir-1.0 — Release.** Gemma 4 and Qwen 2.5 both work end-to-end
through the OpenAI HTTP API. All 82+ unit tests green; the
Level-Zero / SPV-kernel side is verified against CPU reference math
on real hardware.

What remains before Mimir-1.0 is "tagged" rather than just "working":

- **Multi-request concurrency** — the server today serialises requests
  through a mutex. Per-request KV cache + scratch pool needs to land
  before the engine can handle parallel users.
- **KV-cache reuse across turns** for multi-turn chats without re-prefill.
- **Pegenaut integration** — the sister TypeScript RAG project that
  this engine was originally built to feed.

## What's coming

**Mimir-1.1: Concurrency.** Per-request KV cache, scratch pool, and a
non-blocking decode loop. Removes the request mutex.

**Mimir-2.0: Bigger models.** Layer streaming for architectures that
don't fit in iGPU-shared memory — DeepSeek V4 Flash, GLM-4.7 Flash, the
Gemma 4 31B dense variant. On UMA hardware this is mostly a
page-table-shuffle problem, not the PCIe-bandwidth problem it is on
discrete GPUs.

**Pegenaut backend.** Mimirmind is the inference half of a TypeScript
RAG stack we're building in parallel. The two will ship as a unit.

See [`doc/roadmap.md`](doc/roadmap.md) for the detailed milestone
breakdown.

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

TBD — see [`LICENSE`](LICENSE) when it lands. Until then: source
visible for reading, no production deployment outside the development
group.

## Acknowledgements

The Norse-themed phase names are not a marketing flourish. Mimir is
the keeper of the *Well of Wisdom* — the archetype of a *loaded*
knowledge source that does not learn anything new, only speaks what
is already in it. That is exactly what an inference engine does to a
trained model. The mythology earned its keep.
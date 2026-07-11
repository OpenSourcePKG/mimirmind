# Hugin — Input-Compression Adapter for Mimirmind on Xe-LPG

*System-integration design for a Level-Zero / USM inference runtime*

**Author:** Stefan Werfling
**Draft:** 2026-07-10
**Status:** engineering design (M-Hugin, unscheduled)
**Framing:** this is an engineering design document, not a research
paper. The compressor architecture, training objective, and per-chunk
compression pattern are ported from published open-source work — COCOM
(RAG-side blueprint), PISCO (per-chunk pattern), ICAE (distillation
recipe), LLMLingua (optional pre-filter). The contribution of the work
described here is the integration of these known patterns onto our
specific hardware stack (Xe-LPG iGPU, Level-Zero, USM, GGUF, Munin,
pegenaut Chroma), and the measurement of the compound speed stack on
that stack. No published system covers this integration.
**Companion:** [`munin-persistent-model-daemon`](../../../synaipse-vault/Memory/mimirmind/research/munin-persistent-model-daemon.md) (persistence layer),
[`roadmap-speculative-decoding`](../../../synaipse-vault/Memory/mimirmind/research/roadmap-speculative-decoding.md) (decode-side counterpart),
[`mtp-diffusiongemma-status-2026-07`](../../../synaipse-vault/Memory/mimirmind/research/mtp-diffusiongemma-status-2026-07.md)

---

## Overview

Retrieval-augmented generation (RAG) on integrated-GPU hardware is
prefill-bound, not compute-bound. On the Meteor Lake Xe-LPG target
platform of mimirmind, a 20 000-token RAG context spends the majority of
its wall-clock time streaming attention keys, values, and dequantised
weights across a shared LPDDR5X bus that is also serving the CPU. This
document specifies **Hugin**, an input-compression adapter that sits
between the embedding layer and the first transformer block of a frozen
autoregressive base model and compresses a long token sequence into a
fixed, small set of *memory tokens* — typically 64 to 256 — that the
base model consumes as if they were ordinary input embeddings.
Because the base model is not modified, Hugin is a strict inference-time
optimisation: no target-model retraining, no GGUF format change, no
change to the sampling loop, no change to the KV-cache layout. The
algorithmic idea itself is not new — a well-populated 2023-2026
literature already covers input-side context compression via
Q-Formers, gist tokens, autoencoded prompts, and RAG-embedding
adapters, and several open-source implementations exist (COCOM,
500xCompressor, PISCO, xRAG, ICAE, LLMLingua). The contribution of
this proposal is not the compressor architecture but the **system
integration path** for such a compressor on a Level-Zero /
Unified-Shared-Memory runtime targeting an iGPU class where the
dominant cost is *memory traffic*, not floating-point throughput, and
where reducing the effective prefill length by two orders of magnitude
collapses both the attention quadratic term and the per-layer KV-cache
footprint. No published system implements input-side compression on
this hardware stack (Xe-LPG + Level-Zero + GGUF adapters + persistent
USM daemon), and that gap is the target of Hugin.
The proposal treats Hugin as one element of a **compound speed stack**
combining input-side compression (Hugin, per-chunk following PISCO,
optionally preceded by LLMLingua token pruning), output-side
acceleration (Gemma-4 MTP drafters), KV-bandwidth reduction (M10.2
FP16 KV cache), and prefix caching on the compressed memory prefix
(RadixAttention). Stacking the levers is the point: the isolated
Hugin lever yields roughly 4× on typical 20 k-token RAG contexts, but
the stack yields 15×–20× end-to-end wall-clock and 28×–44×
time-to-first-token on the same requests. The paper lays out a
three-milestone build path (M-Hugin.1 through M-Hugin.3) including a
distillation training regime informed by COCOM and 500xCompressor,
a GGUF-compatible adapter format, and an integration schema with
the existing Munin persistence layer.

The name follows the Norse pattern already established in the project.
Odin's two ravens are **Huginn**, *thought*, who flies out to gather
information from the world, and **Muninn**, *memory*, who holds what
has been gathered. Munin is our persistent model-memory daemon that
keeps the weights resident in RAM across restarts. Hugin, complementarily,
is the layer that *flies out* to a document, extracts thought from it,
and returns with a small compact representation that the wise
counsellor (Mimir) can then act on.

---

## 1. Introduction

### 1.1 The prefill bottleneck on RAG

Mimirmind's target production workload is Retrieval-Augmented Generation
served to the sister project *pegenaut*. In a typical RAG call, the
user's turn is preceded by 3 to 15 retrieved document chunks totalling
between 3 000 and 20 000 tokens. The autoregressive base model — Gemma 4
26B-A4B-it in production, Gemma 4 E4B on the perf-shelf — must ingest
every retrieved token *once*, populating a KV-cache tensor with per-layer
key and value projections, before a single response token can be sampled.

On our target hardware the observed cost of this ingestion is
dominated by:

- **KV-cache writes.** For a 20 000-token context on Gemma 4 26B-A4B,
  40 layers with 4 KV heads of dimension 256 in F16 produce
  `20 000 · 40 · 2 · 4 · 256 · 2 B ≈ 3.2 GiB` of KV traffic.
- **Attention key/value reads.** The per-token attention pass streams
  the entire per-layer key and value block across the shared bus each
  time a new query is projected. In the prefill this is not quadratic
  in tokens (it is one large batched matmul), but it *is* linear in
  context length.
- **Weight streaming.** Dequant-on-the-fly of Q6_K, Q8_0 and Q4_K
  block-quantised weights competes for LPDDR5X bandwidth with the
  KV traffic above.

The Perf-Regression-Ledger in the Synaipse vault records a live
1331-token prefill on L0_TARGET_HOST at 5.9 s wall-clock, 141 J
energy, with a Q8_0 A/B win of −13.7 % wall-clock over F32 (session
note `project_session_state.md`). Linear extrapolation places a
20 000-token prefill in the tens-of-seconds range even after every
kernel optimisation the perf shelf has produced this quarter.

Every kernel-side lever we have shipped — the GEMM prefill rewrite
(ADR `gemm-prefill-rewrite`), the FlashAttention Q8_0-GQA prefill
kernel (commits `10e28a4`, `f33ac05`, `cde6e22`, `c18d720`), the
Command-List-Replay decode path (ADR `2026-07-06-command-list-replay`),
the KV-dtype foundation (`M10.2`) — attacks a constant factor in
the same equation. All of them help. None of them reduces the *number
of tokens* the base model has to ingest.

Hugin attacks that number directly. If a 20 000-token document can be
losslessly-enough represented by 256 memory tokens, prefill work is
reduced by a factor of 78 on token count *and* the corresponding KV
traffic vanishes with it. This paper is about whether that is
achievable, how much quality is at risk, and how to build it inside
the mimirmind runtime.

### 1.2 Position of the contribution

The proposal is deliberately narrow. Hugin is:

- **Not a new base model.** The Gemma 4 target weights are frozen.
- **Not a KV-cache prefetch or session-persist scheme.** Those are
  separate levers being explored in `M9.11 speculative decoding` and
  in the roadmap for session-KV-cache across turns.
- **Not a diffusion decoder.** DiffusionGemma is a parallel research
  branch documented in the `mtp-diffusiongemma-status` note; it
  attacks the decode side, Hugin attacks the prefill side.
- **Not a change to the tokenizer** in the SentencePiece sense.
  The user-facing token stream is unchanged; Hugin operates one
  level below, on token *embeddings*.

Hugin is a **learned input-side compressor**, structurally most
similar to the Q-Former in BLIP-2, the latent bottleneck in
Perceiver IO, the "gist tokens" of Mu et al. (2024), and In-Context
Autoencoders (Ge et al., 2024). Its novelty for this project is not
the mathematical shape — the shape is well-explored in the vision-
language and long-context literature — but the *system integration*
on a Level-Zero / USM iGPU stack whose bottleneck profile is
qualitatively different from the discrete-GPU CUDA world in which
those prior works were measured.

### 1.3 Scope — what we build, what we reuse

**We do not invent a new compression method.** The compressor,
training objective, and per-chunk pattern are lifted directly from
published open-source work (Section 3). What this document specifies
is engineering-integration work, not algorithmic novelty.

**What we build:**

- **Runtime integration.** A `HuginAdapter` class inside
  `InferenceEngine`, sharing `UsmAllocator` and `L0Context` with
  the base model; SPIR-V kernels for the encoder's cross-attention
  and projection layers built the same way as the existing Q4_K /
  Q6_K / Q8_0 kernels (ocloc at build time).
- **GGUF adapter format** — a small manifest extension so the
  existing loader (M3) can consume Hugin adapters alongside the
  base model, including a strict tokenizer-hash + residual-
  dimension compatibility check at load time.
- **`config.json` integration** — a single `hugin` block controlling
  opt-in, adapter path, memory-token count and per-chunk caching.
- **Munin coupling** — adapter weights are IPC-shared like the
  base-model weights, surviving mimirmind worker restarts.
- **Pegenaut Chroma integration** — a new
  `POST /v1/hugin/encode` endpoint plus a Chroma-side chunk-
  ingestion pipeline that stores per-chunk memory tokens with an
  adapter-version tag.
- **Compound speed-stack orchestration** — code paths that let
  Hugin compose cleanly with the already-shipped speed levers
  (MTP drafters, KV-F16, FlashAttention Q8_0-GQA, RadixAttention-
  style prefix cache).
- **Distillation training pipeline**, run on a discrete-GPU host
  (not on target), producing GGUF adapters that ship to the
  mimirmind runtime.
- **Benchmarking harness + Perf-Ledger entries** validating each
  step of the stack against the uncompressed baseline on
  L0_TARGET_HOST.

**What we reuse (do not re-derive):**

- Encoder architecture: **COCOM-shaped** cross-attention read-out
  with learnable latent queries; sizing follows COCOM's small
  configuration.
- Per-chunk cache-friendly compression pattern: **PISCO** — one
  fixed set of memory tokens per RAG chunk, combined at query time.
- Training objective: **ICAE**-style self-distillation using the
  frozen base as its own teacher; no gold answers required.
- Optional pre-filter for extreme contexts: **LLMLingua** token
  pruning as a first stage before Hugin encoding.
- Positional treatment, attention masking, and Layer-1-injection
  are directly the BLIP-2 / Compressed Context Memory pattern.

**Effort implication.** Because the algorithmic components are
ported not designed, the training pipeline is a distillation loop
against the frozen base — not a research programme. Estimated
total effort is **5–6 weeks** including target-hardware benchmark,
not the 9 weeks a from-scratch adapter design would take.

---

## 2. Background

### 2.1 Xe-LPG memory hierarchy on Meteor Lake

The target platform, documented in the Synaipse note
`Meteor Lake — Dual-IMC Memory Layout (L0_TARGET_HOST)`, is an
integrated Xe-LPG GPU sharing LPDDR5X-7467 with the CPU. The two
independent 64-bit IMCs deliver approximately 89 GB/s aggregate
bandwidth to any single client that saturates them, but that budget
is contested by the co-tenant pegenaut TypeScript stack running on
the same host. In practice the sustained bandwidth available to
inference is closer to 55–70 GB/s, and the Xe-LPG dispatch overhead
per Level-Zero command has been empirically measured at 10–15 μs
(lesson `xelpg_dispatch_overhead`), not the 40 μs originally
assumed.

The consequence for prefill is that the arithmetic intensity of
the transformer layers, in ops-per-byte, is not the limiting
resource. Q8_0-quantised weights already stream at close to the
peak sustained bandwidth. Any workload proportional to token count
scales linearly with wall-clock time; the only way to shift that
line is to reduce the token count.

### 2.2 Current prefill cost, measured

Numbers taken from the perf ledger and session notes on the
production host:

| Model                     | Ctx tokens | Prefill wall | Prefill / tok | Energy |
|---------------------------|-----------:|-------------:|--------------:|-------:|
| Gemma 4 E4B Q4_K_M        |        306 |     ~0.5 s   |     ~1.6 ms   |   ~10 J |
| Gemma 4 26B-A4B Q8_0      |       1331 |      5.9 s   |     ~4.4 ms   |  141 J |
| Gemma 4 26B-A4B Q6_K      |       3400 |     ~14 s    |     ~4.1 ms   |  ~330 J |
| (extrapolated, same)      |     20 000 |     ~80 s    |      ~4 ms    | ~1900 J |

The E4B row is bench-mode (rp0, CLR-ON) on a warm system. The
26B-A4B rows include a shared-tenant governor penalty. The
per-token prefill cost is remarkably stable across context lengths
in the mid-range, which is the fingerprint of a bandwidth-bound
kernel: bytes-per-token is a constant, wall-clock scales with token
count.

For a pegenaut RAG turn assembling five 4 000-token document chunks,
the extrapolated 80 seconds of prefill dwarfs the tens-of-milliseconds
per decode token that follows. A user perceives the entire request
as "slow", even though every generated token after the first is
served in near-real-time. This is the target of Hugin.

### 2.3 What has already been tried

- **Q8_0 activation quantisation** — landed, −13.7 % wall on prefill,
  live in production.
- **FlashAttention Q8_0-GQA prefill kernel** — landed on head branch,
  K-tile autotune reconstruction just merged (commit `c18d720`).
- **GEMM-for-prefill** — landed as `perf/gemm-prefill`, replaced the
  matvec loop when M > 1 on cooperative Q4_K/Q6_K paths.
- **Command-List-Replay for decode** — landed as `M-CLR`, decode-side
  only (structurally incompatible with MoE prefill; see
  `lesson_moe_clr_incompatible`).
- **KV-dtype F16 foundation** — landed as `M10.2 Phase 1a`, halves
  the KV bandwidth term.
- **Speculative decoding (M9.11) and MTP drafters** — designed and
  measured; the analysis in `roadmap-speculative-decoding` shows
  those levers help decode, not prefill.

All of these are constant-factor kernel improvements. None reduces
the number of tokens entering the base model. Hugin is the first
proposal on the shelf that changes the token count.

---

## 3. Related Work

The idea of compressing a long context into a small set of dense
vectors that a frozen language model can consume is not new; the
literature since 2023 has produced at least four distinct families
of methods, and Hugin sits at their intersection.

**Perceiver and Perceiver IO (Jaegle et al., 2021, 2022).** A learned
cross-attention read-out from a large input into a small latent
array. Perceiver established that a fixed-size latent bottleneck can
carry enough information for downstream tasks provided the read-out
is bidirectional and multi-layer. Perceiver IO extended the pattern
with a symmetric write-out, showing that arbitrary output structures
can be generated from the latent state.

**BLIP-2 Q-Former (Li et al., 2023).** A 32-query trainable module
that turns a frozen ViT's patch tokens into 32 dense vectors and
feeds them, via a linear projection, to the input side of a frozen
LLM. Structurally this is the closest published analogue to what
Hugin proposes: a small, trainable bridge that reshapes a large
frozen-encoder output into a small sequence that a frozen decoder
LLM ingests as regular input embeddings.

**Prefix-Tuning and Prompt-Tuning (Li & Liang, 2021; Lester et al.,
2021).** Established that a small number of trainable "virtual
tokens" prepended to the input can steer a frozen LLM. Hugin
generalises this from a fixed learned prefix to an input-dependent
learned prefix: the memory tokens are a function of the document,
not a constant.

**Gisting (Mu, Li, Goodman, 2024).** A prompt-compression scheme that
inserts special "gist" tokens during a fine-tuning phase and uses
attention masking to force the model to store all information
about the preceding prompt inside those gists. At inference time
the pre-gist tokens are discarded. This is the first work to show
that compression ratios of 26× can be achieved with minimal
downstream loss on a range of instruction-following benchmarks.

**In-Context Autoencoders / AutoCompressors (Ge et al., 2024;
Chevalier et al., 2023).** Autoencoder-style methods that train a
small encoder to produce k dense "summary vectors" from a long
document such that a frozen decoder can reconstruct the document,
or answer questions about it, from those k vectors alone.
Compression ratios of 4×–15× are reported without downstream
degradation on QA. ICAE's reference implementation and the
AutoCompressors code are both public; AutoCompressors is tied to
Llama-2 / OPT-2.7B and has been unmaintained since 2024.

**Compressed Context Memory (Kim et al., ICLR 2024).** Trains a
conditional LoRA that dynamically produces compressed memory of
attention KVs during interactive use. Public PyTorch reference at
[snu-mllab/context-memory](https://github.com/snu-mllab/context-memory).
Structurally similar to Hugin in that the base model is frozen and
only a small adapter is trained; different in that the compression
target is the KV cache itself rather than a Layer-1 memory-token
prefix.

**LLMLingua / LongLLMLingua (Jiang et al., Microsoft, 2023-2026).**
Not a soft-embedding compressor but a *token-selection* compressor
using a small language model to identify and drop low-information
tokens. Reaches roughly 20× compression at a ~1.5 pp accuracy
drop and is the most widely production-deployed prompt-compression
tool as of 2026. It composes with Hugin (token pruning first, then
memory-token compression on the reduced sequence) but is not a
substitute — LLMLingua's compression ceiling is roughly one order
of magnitude below what soft-embedding methods reach.

**xRAG (Cheng et al., NeurIPS 2024).** The extreme end of the
spectrum: a whole retrieved document collapsed to a *single* soft
token that is spliced into the LLM input. Demonstrates that
current embedding models already carry enough information for
short-answer QA when the LLM is fine-tuned to consume that single
vector. Establishes the outer bound of what is achievable but at
severe cost in the LLM-side fine-tuning (which contradicts our
frozen-base contract).

**COCOM (Rau et al., 2024).** Context Embeddings for Efficient
Answer Generation in RAG. Reports a *measured* end-to-end inference
speedup of **5.69×** on RAG QA, versus similar compression baselines.
This is currently the closest published number to the wall-clock
estimate in Section 6.4 of this paper and is the most useful
external calibration point available. The method is closely related
to Hugin's Variant A design.

**500xCompressor (Li & Briscoe, ACL 2025).** Trains a soft-prompt
compressor with 0.3 % additional trainable parameters, reaching
compression ratios up to **480×** while retaining 62–73 % of the
base LLM's downstream capability. This establishes that
compression at Hugin's target range (200×–300×) is in principle
publication-supported, at a documented quality cost that is not
trivial but is not catastrophic either.

**PISCO (2025).** Encoder-decoder jointly trained with variable
memory-token counts per retrieved chunk, targeting RAG
specifically. Anticipates our Mode-2 pegenaut-Chroma integration
(§5.6): per-chunk memory tokens computed at ingest time and stored
alongside the chunk. PISCO's contribution is showing that per-chunk
memory tokens can be combined at query time without cross-chunk
retraining, which is exactly the assumption Mode 2 rests on.

**Q-RAG (Sun et al., ICLR 2026, oral).** Reinforcement-learning
approach that trains a lightweight embedder agent for multi-step
retrieval in latent space while keeping the LLM frozen. Different
task framing — it is a *retrieval* method not a *compression*
method — but the frozen-LLM adapter pattern is the same. Public
code at [griver/Q-RAG](https://github.com/griver/Q-RAG).

**Multi-Token Prediction drafters and speculative decoding
(mimirmind notes).** Attacks the decode side. Fully orthogonal to
Hugin — they can compose.

**Prefix caching / RadixAttention (Zheng et al., 2024).** Reuses the
already-computed KV state of a shared prompt prefix across many
requests. This is a *storage* optimisation and applies even for
uncompressed prompts; it composes cleanly with Hugin because
Hugin-compressed inputs are shorter and therefore cheaper to
prefix-cache. Hugin does not replace prefix caching, it multiplies
its budget.

**T5, BART (Raffel et al., 2020; Lewis et al., 2020).** Encoder-
decoder architectures with cross-attention. If we were free to
change the base model, this would be the textbook approach. We
are not free to change the base model.

Positioning of Hugin within this space:

- Compression ratio target: 50×–300×, more aggressive than Gisting
  (26×) or ICAE (4×–15×), aligned with COCOM's operating point,
  well inside the envelope 500xCompressor has published as
  achievable.
- Frozen base: yes, like BLIP-2, Prefix-Tuning, Compressed Context
  Memory, and COCOM; unlike T5/BART, unlike xRAG (which requires
  base-LLM fine-tuning).
- Input-dependent: yes, unlike Prefix-Tuning; like Gisting, BLIP-2,
  and every other work in this survey.
- Consumption interface: standard input embeddings, no cross-
  attention modification; unlike Perceiver IO's separate decoder,
  like BLIP-2, Gisting, ICAE and COCOM.
- Per-chunk cache-friendly: yes, following PISCO's pattern.

**Honest positioning statement.** The algorithmic content of Hugin
is a straightforward instantiation of the input-side soft-embedding
compressor pattern established in 2023–2024 and refined in 2024–2026
by COCOM, PISCO and 500xCompressor. This proposal's contribution
lies not in a new compression method but in the **first system
integration of such a method on a Level-Zero / USM iGPU inference
stack with a GGUF-serialised adapter, a persistent USM daemon
(Munin) for zero-downtime adapter reload, and a Chroma-side pre-
compression pipeline paired with the pegenaut sister project.** No
published open-source implementation targets this hardware and
runtime combination, and none of the existing implementations
(ICAE, COCOM, PISCO, Compressed Context Memory) is deployable on
Xe-LPG through their supplied code paths.

---

## 4. Method

### 4.1 Overall architecture

Hugin inserts a single new component into the base model's forward
graph:

```
Documents ── Tokeniser ── Embedding lookup ─┐
                                            ▼
                                    ┌─────────────┐
                                    │    Hugin    │   trainable
                                    │  Encoder E  │
                                    └──────┬──────┘
                                           │  m memory tokens (m ≪ n)
                                           ▼
     User turn ── Tokeniser ── Embedding lookup ─┐
                                                 ▼
                                    [ memory | user ]  concat on
                                                       sequence axis
                                                 │
                                                 ▼
                                        Transformer Layer 1
                                                 ▼
                                        Transformer Layer 2
                                                 ▼
                                             ... (frozen) ...
                                                 ▼
                                        Transformer Layer L
                                                 ▼
                                             LM head
                                                 ▼
                                             logits
```

Formally, given a document token sequence $\mathbf{d} \in \mathcal V^n$
and a user turn $\mathbf{u} \in \mathcal V^k$, and writing $W_E \in
\mathbb R^{|\mathcal V| \times d}$ for the base model's frozen
embedding matrix:

$$
\mathbf X_d = W_E[\mathbf d] \in \mathbb R^{n \times d}, \quad
\mathbf X_u = W_E[\mathbf u] \in \mathbb R^{k \times d}
$$

Hugin's encoder $E_\theta$, with trainable parameters $\theta$,
produces $m$ *memory embeddings*:

$$
\mathbf M = E_\theta(\mathbf X_d) \in \mathbb R^{m \times d}
$$

where $m$ is a fixed hyperparameter (candidates: 64, 128, 256) and
$m \ll n$. The base model $F_\phi$, frozen with pretrained weights
$\phi$, is then invoked on the concatenation of memory embeddings
and the user's token embeddings:

$$
\hat{\mathbf y} = F_\phi([\mathbf M; \mathbf X_u])
$$

Only $\theta$ is trained. $\phi$ and $W_E$ are frozen.

The memory embeddings share the base model's residual-stream
dimension $d$ (2304 for Gemma 4 26B-A4B, 3072 for Gemma 4 31B dense).
This is what allows them to be consumed by Layer 1 without any
change to Layer 1 itself.

### 4.2 Design variants and the selected design

The original chat that motivated this paper considered three
variants; I summarise them here for the record and then justify
the selection.

**Variant A — Multiple document tokens, single-shot.** The document
is passed through a small encoder that emits $m$ tokens. Those
tokens are prepended to the user turn. This is the design shown in
Section 4.1. It is the simplest to implement, the cheapest to
serve, and the closest structural match to how the base model
already consumes its input. It is the **selected design.**

**Variant B — Encoder / cross-attention decoder (T5-shaped).** The
document is encoded into $m$ latents and the base model reads them
via cross-attention layers inserted between existing self-
attention blocks. This is architecturally cleaner in the sense
that it separates document and query streams, but it requires
inserting new attention layers into a frozen decoder — meaning
either extensive training of the injected cross-attention weights,
or a wholesale replacement of the decoder with an encoder-decoder
model, which contradicts the "frozen base" contract. Rejected.

**Variant C — Multi-stage adapter with re-injection at deeper
layers.** A small encoder produces $m$ latents before Layer 1; a
second adapter re-injects a possibly-different set of latents
between an early group of layers (say Layers 1–8) and a late
group (Layers 9–L). Rationale is that early layers encode more
syntactic structure and late layers more semantic; conditioning
each group with a different projection of the document could
capture more of the input's information than a single-stage
compression. Variant C stays query-agnostic and therefore remains
Chroma-cache-friendly.

**Variant D — Query-conditioned single-stage compression.** The
document *and* the user's question are fed jointly into the Hugin
encoder; the latent queries cross-attend into
`[Doc-Embeddings ; Query-Embeddings]` and produce memory tokens
that already reflect what the question is asking. Expected quality
win over Variant A on extract-verbatim, numeric-lookup, and
multi-hop reasoning; expected quality parity elsewhere. Cost: the
memory tokens become a function of the query — Chunk-caching in
Chroma (Mode 2, §5.6) no longer applies to any query that triggers
Variant D, and each request pays the full ~1.1 s encoder step.
Related published work: LongLLMLingua's query-aware mode, PISCO's
variable-per-chunk mode, [Query-Conditioned Selector for RAG
Compression](https://arxiv.org/pdf/2602.15856).

**Variants A, C, and D coexist at runtime.** The three variants
attack different quality/speed points, and the M-Hugin design
loads all three adapters when they are available and lets the
runtime select per request via an autotune mechanism (§4.4).
Variant B is not implementable under the frozen-base contract and
is not built.

**Trade-off summary:**

| Variant | Query-aware | Chroma-cacheable | Encode/req | Best at            |
|---------|:-----------:|:----------------:|:----------:|--------------------|
| A       |     no      |       yes        |   ~1.1 s   | general purpose    |
| C       |     no      |       yes        |   ~1.5 s   | long docs, complex |
| D       |    **yes**  |    **no**        |   ~1.1 s   | extract/numeric    |

### 4.3 The Hugin encoder

The encoder $E_\theta$ is a small transformer-encoder-style block
with cross-attention read-out into a fixed-size latent array,
following Perceiver's original design more closely than BLIP-2's:

1. **Learnable latent queries.** $m$ latent vectors $\mathbf Q \in
   \mathbb R^{m \times d_h}$, initialised randomly, are the only
   sequence-length-independent state in the encoder.
2. **Cross-attention read-out.** A stack of $L_E$ blocks alternates
   between (a) cross-attention from $\mathbf Q$ into $\mathbf X_d$
   using the standard scaled-dot-product form and (b)
   self-attention within $\mathbf Q$.
3. **Projection.** A final linear layer maps from the encoder's
   inner dimension $d_h$ up (or down) to the base model's residual
   dimension $d$, producing $\mathbf M$.

Sizing candidates for the first prototype:

| Parameter          | Small     | Mid       | Large      |
|--------------------|----------:|----------:|-----------:|
| $L_E$ (blocks)     | 4         | 6         | 8          |
| $d_h$              | 512       | 768       | 1024       |
| Heads              | 8         | 12        | 16         |
| $m$                | 128       | 256       | 256        |
| Parameter count    | ~15 M     | ~55 M     | ~140 M     |

The "small" configuration is small enough to keep the encoder
step cheap on the iGPU (see the performance model in Section 6);
the "large" configuration is what we would expect to need if
compression to $m = 64$ is required for a very long context.

### 4.4 Training regime

The base model is frozen. Only $\theta$ is trained.

**Objective.** Self-distillation from the frozen base model, along
the same lines as AutoCompressors and ICAE:

1. Sample a document $\mathbf d$ (up to 32 k tokens) and a related
   query-answer pair $(\mathbf u, \mathbf y)$ from a RAG-style
   corpus.
2. Compute a *teacher* target: run the frozen base model on the
   uncompressed sequence $[\mathbf d; \mathbf u]$ and record the
   logits at the answer positions, call them
   $\ell_{\text{teacher}}(\mathbf y_t \mid \mathbf d, \mathbf u,
   \mathbf y_{<t})$.
3. Compute a *student* prediction: run Hugin on $\mathbf d$ to
   produce $\mathbf M$, then run the frozen base model on
   $[\mathbf M; \mathbf u]$ and record the logits at the answer
   positions.
4. Minimise the token-level KL divergence between teacher and
   student logits at each answer position, plus a small standard
   cross-entropy term against the gold answer.

$$
\mathcal L(\theta) = \sum_t \Big[
   \underbrace{\mathrm{KL}\big(
      p_{\text{teacher}}(\cdot \mid \mathbf d, \mathbf u, \mathbf y_{<t})
      \Vert
      p_{\text{student}}(\cdot \mid E_\theta(\mathbf d), \mathbf u,
                                   \mathbf y_{<t})
   \big)}_{\text{distillation}}
   + \lambda \underbrace{\big(-\log p_{\text{student}}(\mathbf y_t)\big)}_{\text{gold}}
\Big]
$$

with $\lambda$ typically 0.1–0.3.

**Why distillation.** The teacher and the student share the same
frozen decoder $F_\phi$; the only difference is the input side.
The KL term therefore reduces to *"produce memory tokens such that
$F_\phi$'s next-token distribution matches what it would have been
if $F_\phi$ had seen the full document."* This is the target we
actually care about at inference time. It is not sensitive to the
gold answer's exact wording, and it works even when the RAG corpus
has no gold answers at all — an uncurated dump of documents plus
generated queries suffices.

**Curriculum.** Start with short documents (256–1024 tokens) and
increase progressively to 32 k. Rationale is Section 3 of the
Gisting paper: the compression head is far easier to learn on
short inputs, and the learned latent-query patterns transfer.

**Data.** For a first prototype, the FineWeb-Edu corpus plus
synthetic query generation via the base model itself is sufficient
to reach a proof-of-concept on Gemma 4 E4B. For a Gemma 4 26B-A4B
production run, the pegenaut RAG corpus itself is the
distribution-appropriate training source.

**Where training happens.** *Not on mimirmind's target hardware.*
Hugin's training is a several-GPU-day job on a discrete-GPU host
(a rented H100 or an owned RTX card) using PyTorch. Only the
resulting adapter weights are shipped back to the mimirmind
runtime, converted to GGUF, and loaded through the existing
weight-loader path. This preserves the project rule that mimirmind
does not train (see CLAUDE.md).

### 4.5 Positional and vocabulary considerations

Two subtleties matter for correctness inside a decoder-only base
model.

**Positional encoding of the memory tokens.** Gemma 4 uses RoPE
with per-layer sliding-window base frequencies. The memory tokens
occupy positions $0, 1, \dots, m-1$ in the base model's position
counter and the user turn occupies positions $m, m+1, \dots,
m+k-1$. This means the *effective* positional distance between
the last memory token and the first user token is 1, not $n$. This
is exactly the property we want: the model should treat memory
tokens as if they were an immediate, adjacent context, not as a
distant preamble. There is no leakage of the "original" positions
$0, \dots, n-1$ into the memory tokens; those positions are
absorbed by the Hugin encoder's own positional encoding, which is
independent of the base model's.

**Vocabulary.** Memory tokens do not correspond to any entry in
the base model's vocabulary. They are embeddings in $\mathbb R^d$
that never pass through the vocabulary projection. The LM head at
the top of the base model reads only from the final-layer
positions of the *user turn* (and any tokens generated during
decode), so it never has to invert memory embeddings back to
vocabulary items. This is analogous to how BLIP-2's Q-Former
outputs are consumed.

**Attention mask.** During prefill of $[\mathbf M; \mathbf X_u]$
the standard causal mask applies: the user turn attends to the
memory tokens (which precede it) but not vice versa. During
decode, generated tokens attend to both. Memory tokens do not
attend to each other autoregressively during base-model prefill —
their contents were already fixed by Hugin's own internal encoder
pass — so the attention block treats them as a fully bidirectional
prefix, which is again the same regime as BLIP-2.

### 4.6 Multiple documents

For RAG with $C$ retrieved chunks, two strategies are compatible
with the design:

- **Concatenate then compress.** Hugin sees the concatenation of
  all chunks and emits a single set of $m$ memory tokens. Cheaper
  at inference, but the encoder must handle up to $Cn$ input
  tokens.
- **Compress then concatenate.** Hugin is run once per chunk,
  producing $m$ memory tokens per chunk; the base model sees
  $Cm$ memory tokens followed by the user turn. Slightly less
  compressed but much friendlier to caching: each chunk's memory
  tokens are a pure function of that chunk, so they can be
  precomputed once and reused across every query that retrieves
  the same chunk.

The **compress-then-concatenate** variant is the natural pairing
with pegenaut's ChromaDB retrieval store, which already indexes
per-chunk. The Hugin memory tokens can be stored *alongside* the
chunk in Chroma, retrieved together with it, and delivered to
mimirmind without a further Hugin invocation at request time.
This is a substantial deployment simplification: at request time
Hugin does not need to run at all for cache hits, and only runs
for the (small) fraction of chunks freshly ingested since the
last index update.

### 4.7 Runtime variant selection (Autotune)

Variants A, C and D each dominate a different point on the
quality × speed × cacheability surface. Rather than pick one
statically, the runtime loads all three adapters (when available)
and picks per request via an **Autotune** layer, following the
project's established pattern (`GpuMatmul::autotune`, the K-tile
autotune for the Q8_0-GQA prefill kernel).

**Decision signals collected at request time:**

- `n_doc`: total retrieved-document token count.
- `n_answer_est`: expected answer-length bucket (from `max_tokens`
  or a small prior).
- `query_kind`: cheap heuristic classifier — regex + keyword rules —
  returning one of `{paraphrase, extract_verbatim, numeric,
  multi_hop}`. Cost budget: below 500 μs per request.
- `chroma_cache_hit`: were the retrieved chunks pre-compressed with
  Variant-A or Variant-C memory tokens already stored in Chroma?
- `session_quality_signal`: rolling window on user reprompts,
  clarification patterns, or explicit thumbs-down; if it rises
  above a threshold, escalate to higher-quality variants.

**Decision logic (short-circuit, in order):**

```
if n_doc < config.hugin.min_document_tokens:
    variant = NONE           # uncompressed, base model directly

elif chroma_cache_hit and query_kind in {paraphrase, multi_hop}:
    variant = cached_variant  # A or C, whichever was ingested
                              # → 0 s encode

elif query_kind in {extract_verbatim, numeric}
     and n_doc > config.hugin.query_conditioned_threshold:
    variant = D               # query-conditioned, live encode
                              # trades Chroma cache for quality

elif n_doc > config.hugin.long_context_threshold:   # e.g. 12 000
    variant = C               # multi-stage, best on very long docs

else:
    variant = A               # default general-purpose
```

**Autotune calibration.** A winner cache indexed by
`(adapter_version, ctx_bucket, query_kind_bucket, hardware_id)`
stores the variant that produced the lowest wall-clock for a
given quality gate on the pegenaut internal QA slice. The cache
is populated by a startup bench (three warm-ups, ten measurement
iterations per candidate) analogous to
[`GpuMatmul::autotune`](../src/compute/GpuMatmul.hpp), and
persisted in a small `hugin_autotune.json` sitting alongside
`config.json`. Munin holds the file across worker restarts.

**Prod-quality safeguards** — codified from
`lesson_dp4a_autotune_prod_hazard`:

1. **Never auto-pick on wall-clock delta alone.** The winner-cache
   requires the candidate to be within a 3 pp quality margin of the
   baseline on a small held-out set; only *then* is wall-clock
   the tiebreaker. Wall-clock wins that come at unacknowledged
   quality cost were the exact failure mode of the DP4A autotune.
2. **Prior weighting for Chroma-cacheable variants.** Variants A
   and C keep the compound stack (Mode 2) intact; the autotune
   must add a fixed cost penalty (e.g. +200 ms) to Variant D's
   score to reflect the loss of the Chroma-cache path unless a
   query-kind signal overrides it. Without this prior, small
   bench noise would flip the winner to D and silently collapse
   the compound-stack TTFT gain.
3. **Kill switch.** `MIMIRMIND_HUGIN_VARIANT={auto|A|C|D|off}`
   env override for both dev and prod triage. Ledger entry
   required per switch flip.
4. **Windowed adaptive re-selection.** If a session's
   `session_quality_signal` degrades, escalate one variant tier
   (A → C, or A/C → D) for the remainder of the session; log the
   escalation.

**Encoder invocation strategy at runtime:**

| Variant chosen | Encoder run at query time? | KV-cache prefix reusable? |
|:--------------:|:--------------------------:|:-------------------------:|
| A (from Chroma)|             no             |            yes            |
| C (from Chroma)|             no             |            yes            |
| A (live)       |            yes             |            yes            |
| C (live)       |            yes             |            yes            |
| D              |         yes, always        |            no             |
| NONE           |             no             |            n/a            |

Two adapters (`A` and `C`) can share the same Chroma pre-
compression pipeline because they are both query-agnostic;
Variant D is by construction incompatible with Chroma pre-
compression, and lives entirely on the live-encode path.

**Effort:** the runtime-selection layer is a small piece of code
(~2 days including query-kind classifier and safeguards). The
weight of the milestone comes from training three adapters
instead of one — see M-Hugin.2 revision in §9.

---

## 5. System integration in Mimirmind

Hugin is a runtime-time-only optimisation that must fit into a
codebase whose architecture is already fixed. This section maps
the design onto the concrete component layout documented in
`doc/architecture.md`.

### 5.1 Component placement

```
┌────────────────────────────────────────────────────────────────┐
│                          ApiServer                             │
│  ChatCompletionHandler (POST /v1/chat/completions, JSON+SSE)   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     RequestDispatcher                          │
│  routes to engine, wires spec-dec if enabled                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     InferenceEngine                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  HuginAdapter    (new — optional, per-engine)            │  │
│  │    encode(document_tokens) → memory_embeddings           │  │
│  │    owns its own SPIR-V kernels for cross-attn / MHA      │  │
│  │    reuses UsmAllocator + L0Context of the engine         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ArchBackend (Qwen2 / Gemma4 / Gemma4E4B / Gemma4Moe / Dense)  │
│    runBlock(...) — unchanged                                   │
└────────────────────────────────────────────────────────────────┘
```

The HuginAdapter is a **new class**, siblings of `ArchBackend`, that
implements a single method: given a batch of document token IDs, it
returns a tensor of shape `[m, d]` living in USM. That tensor is
then written directly into the base model's residual-stream input
buffer at positions $0 \dots m-1$, and the base model's `runBlock`
is invoked on the concatenated `[m + k]`-length sequence.

The base-model `ArchBackend` implementations do not need to change:
from their perspective the input residual stream simply has $m + k$
positions instead of $n + k$.

### 5.2 GGUF-compatible adapter format

Hugin's weights are stored as a single GGUF file with a manifest
tag identifying it as a Hugin adapter and referencing the base
model's residual dimension and tokenizer hash. The existing GGUF
loader (built in M3) is used with only two additions:

- A new `AdapterKind::HUGIN` enum value.
- A new `HuginManifest` structure recording $m$, $L_E$, $d_h$,
  the base-model residual dimension $d$, and the SHA-256 of the
  base tokenizer for compatibility checking.

Adapters are shipped as small independent files, on the order of
50–500 MiB depending on encoder size. They can be swapped without
rebuilding or restarting the container: `POST /v1/admin/reload`
(defined in the Munin proposal, Section "Hot-Reload") triggers a
reload of the currently-active adapter path from `config.json`.

Rejecting an incompatible adapter (wrong $d$, wrong tokenizer
hash) at load time is essential — silent mismatches would degrade
quality invisibly, which is exactly the failure mode described in
`lesson_dp4a_autotune_prod_hazard`. The compatibility check
belongs at the loader boundary, not inside the encode path.

### 5.3 Config schema

Following the `config.json` migration already landed in commit
`456bd2a`, Hugin is opt-in via a single config block:

```jsonc
{
  "hugin": {
    "enabled": true,
    "min_document_tokens": 512,
    "long_context_threshold": 12000,
    "query_conditioned_threshold": 4000,
    "adapters": {
      "A": "/models/adapters/hugin-gemma4-26b-a4b-A-m128.gguf",
      "C": "/models/adapters/hugin-gemma4-26b-a4b-C-m128.gguf",
      "D": "/models/adapters/hugin-gemma4-26b-a4b-D-m128.gguf"
    },
    "memory_tokens": 128,
    "autotune": {
      "enabled": true,
      "winner_cache_path": "/etc/mimirmind/hugin_autotune.json",
      "quality_gate_pp": 3.0,
      "variant_D_cost_penalty_ms": 200,
      "startup_bench": true
    },
    "cache": {
      "enabled": true,
      "backend": "chroma",
      "chroma_url": "http://pegenaut-chroma:8000",
      "collection": "hugin-memory-v1",
      "ingest_variants": ["A", "C"]
    }
  }
}
```

`min_document_tokens` is the threshold below which Hugin is skipped
and the base model is fed the raw text directly. For a 200-token
system prompt or a two-line follow-up question, compression is
strictly a loss. `long_context_threshold` and
`query_conditioned_threshold` are the cutoffs the autotune uses in
its decision logic (§4.7). `ingest_variants` lists which variants
the Chroma pre-compression pipeline pre-computes and stores per
chunk — Variant D is deliberately absent because it is query-
conditioned.

For dev / triage, `MIMIRMIND_HUGIN_VARIANT={auto|A|C|D|off}`
short-circuits the autotune and pins a variant.

### 5.4 Interaction with the Munin persistence daemon

Munin (see `research/munin-persistent-model-daemon.md`) allocates
model weights in USM once at startup and exports IPC handles to the
worker processes. The Hugin adapter weights are compatible with this
model: they are loaded once by Munin, exported as an additional
IPC-shared tensor block, and attached by the worker at the same
time as the base model tensors.

The Hugin adapter's KV / activation scratch is per-engine and
allocated by the worker at startup, exactly the same way the base
model's scratch is allocated. No cross-process state.

### 5.5 Interaction with the KV cache and existing kernels

The base model's KV cache sees only `m + k` positions per request.
KV-cache size at 20 000 raw tokens compressed to 256 memory tokens
plus a 512-token user turn shrinks from 3.2 GiB to
`(256 + 512) / 20000 · 3.2 GiB ≈ 123 MiB` — a factor of 26. This
frees KV budget for either longer decodes, more concurrent
sessions, or KV-dtype upgrades to F16 (M10.2) without pressure.

The FlashAttention Q8_0-GQA prefill kernel (commits `10e28a4` and
following) is invoked with a shorter sequence length. Because the
kernel is O(seq²) in shared-memory tile traffic per head, the
speedup from shorter sequences compounds with FlashAttention's
constant factor; the two levers do not cancel.

RoPE application is unchanged — memory tokens receive positions
$0 \dots m-1$ and are RoPE-encoded just like ordinary tokens.

### 5.6 Interaction with pegenaut

Pegenaut is the intended consumer. Two integration modes:

**Mode 1 — transparent.** Pegenaut sees no change. Mimirmind
receives the full RAG prompt, Hugin compresses the retrieved
document chunks internally, and pegenaut is oblivious. This mode
requires the smallest amount of new code but does not exploit
per-chunk caching in Chroma.

**Mode 2 — pre-compressed.** Pegenaut, at document-ingest time,
sends each chunk to a new mimirmind endpoint
`POST /v1/hugin/encode` and stores the returned memory tokens
alongside the chunk in Chroma. At query time it sends a request
that includes *memory tokens* instead of *document text*, and
mimirmind bypasses Hugin encoding for that chunk entirely.

Mode 2 is a small extension of the OpenAI-compatible endpoint
(a new fields `content_kind: "hugin_memory"` on message parts)
but yields the largest wall-clock win because Hugin's own encoder
step is amortised across every query that hits the same chunk.
Recommended as the M-Hugin.3 target.

---

## 6. Performance model

I derive expected wall-clock improvements from the measured
bandwidth-bound cost profile of the current runtime, following
the same style as the analyses in
`research/roadmap-speculative-decoding` and
`research/roadmap-gemm-xelpg-native-rewrite`.

### 6.1 The three cost terms of prefill

For a token sequence of length $T$ passed once through the base
model at prefill, the wall-clock cost decomposes as:

$$
t_{\text{prefill}}(T) \approx
    \underbrace{L \cdot 2 T d^2 / B}_{\text{QKVO+FFN weights}} +
    \underbrace{L \cdot 2 T^2 d / B}_{\text{attention KV traffic}} +
    \underbrace{L \cdot c_{\text{kernel}}}_{\text{dispatch}}
$$

where $L$ is the number of layers, $d$ the residual dimension,
$B$ the sustained bandwidth to the GPU, and $c_{\text{kernel}}$
the fixed dispatch cost per kernel (10–15 μs on Xe-LPG per
`lesson_xelpg_dispatch_overhead`). For Gemma 4 MoE, the first
term is *per active expert*, not per layer, and its effective
constant is smaller than dense.

At $T = 20{,}000$ on Gemma 4 26B-A4B the second term (attention
KV traffic) dominates; the first term is roughly proportional to
$T$ but the second grows quadratically. Below $T \approx 4{,}000$,
the first term dominates and prefill scales linearly in $T$.

### 6.2 Effect of compression to $m$ memory tokens

Hugin replaces $T$ with $m + k$ where $k$ is the user turn length,
typically 100–1000. Substituting for a 20 000-token document and
$m = 256$, $k = 512$:

- First term shrinks by factor $20000/(256+512) \approx 26$.
- Second term shrinks by factor $(20000)^2/(256+512)^2 \approx 680$.
- Third term is unchanged (still $L$ kernels dispatched, same
  fixed cost).

Because the second term is the dominant one at long contexts, the
overall prefill wall-clock speedup at 20 000 tokens is bounded
above by roughly 300–500× *before* accounting for the encoder
cost, and above by roughly 26× at short contexts where the second
term is not yet dominant.

### 6.3 The encoder cost

Hugin's own encoder is a small transformer: $L_E$ layers on $n$
input tokens with residual dimension $d_h$. Its cost is
approximately:

$$
t_{\text{encode}}(n) \approx L_E \cdot (2 n d_h^2 + 2 n^2 d_h) / B
$$

For the "small" configuration ($L_E = 4$, $d_h = 512$) on
$n = 20{,}000$:

- Weight term: $4 \cdot 2 \cdot 20000 \cdot 512^2 / (55 \text{ GB/s})
  \approx 750$ ms
- Attention term (uncompressed within the encoder): $4 \cdot 2 \cdot
  20000^2 \cdot 512 / (55 \text{ GB/s}) \approx 30$ s

The attention term inside Hugin's own encoder is the killer if we
run naïve full self-attention on the raw document. Two mitigations
are structurally available:

1. **Perceiver-style read-out only.** The encoder does not
   self-attend among the document tokens; it only cross-attends
   *from* the $m$ latent queries *into* the document tokens. The
   attention cost is $O(m n)$, not $O(n^2)$. For $m = 256$ and
   $n = 20{,}000$, this is a factor of 78 reduction — critical.
   This is the design chosen in Section 4.3.
2. **Chunked document processing.** Feed the encoder chunks of
   $n_c = 2048$ tokens at a time and pool latents. This is the
   fallback if the read-out design does not converge quality-wise.

With the Perceiver-style read-out, Hugin's encoder cost is
approximately:

$$
t_{\text{encode}}(n) \approx L_E \cdot (2 n d_h^2 + 2 m n d_h) / B
$$

For the "small" config with $L_E = 4$, $d_h = 512$, $m = 256$,
$n = 20{,}000$, on $B = 55$ GB/s:

- Weight term: 750 ms as above
- Cross-attention term: $4 \cdot 2 \cdot 256 \cdot 20000 \cdot 512 /
  55\,\text{GB/s} \approx 380$ ms

Total encoder cost: **~1.1 s** for a 20 000-token document at this
configuration.

### 6.4 Break-even and total wall-clock estimate

Uncompressed 20 000-token prefill on 26B-A4B (from ledger
extrapolation): **~80 s**.

Hugin-compressed prefill of $256 + 512 = 768$ tokens on the same
model: linearly extrapolated from the 1 331-token / 5.9 s row,
approximately **3.4 s**.

Encoder step: **~1.1 s** (see 6.3), skippable entirely if the
document was pre-compressed via Mode 2.

**Live-Mode total:** $1.1 + 3.4 = 4.5$ s per request.
**Pre-compressed Mode total:** $3.4$ s per request.

Speedup vs uncompressed baseline: **~18×** (live) to **~24×**
(pre-compressed) on 20 000-token contexts.

Below 4 000 tokens the wins shrink significantly, and below 2 000
tokens the encoder cost may exceed the savings; the
`min_document_tokens` guard in the config (Section 5.3) exists
exactly for this regime.

### 6.5 Prediction hazard notice

The performance model above rests on the same assumptions that
have burned the project before (see `lesson_dp4a_autotune_prod_hazard`
and `lesson_xelpg_dp4a_compute_ratio`): namely that memory bandwidth
is the dominant cost and that Xe-LPG dispatch overhead is well-
characterised. Both assumptions hold *on the paths we have already
measured*, but neither has been measured for the specific kernel
shape a Hugin encoder would use. Before committing to a build path
we should run a small SPIR-V microbenchmark that reproduces the
encoder's inner loop on realistic tensor shapes and validates the
750 ms weight term. This is the same discipline that
`research/roadmap-speculative-decoding` codified in its
"attention-share validation before build" gate.

### 6.6 The compound speed stack

Hugin in isolation is a ~4× lever on end-to-end wall-clock for a
long-context RAG request. That is not the point of building it.
The point is that Hugin composes with the already-shipped and
already-planned speed levers, and the compound behaviour is
substantially better than any single lever.

The stack in composition order, from input to output:

1. **Optional LLMLingua token pruning.** For contexts above
   ~10 000 tokens, drop low-information tokens with a small
   selector model first. Typical documented ratio 15×–20× at
   ~1.5 pp accuracy loss. Reduces the input to Hugin's encoder,
   which is what dominates Hugin-live-mode cost.
2. **Per-chunk Hugin compression.** Each retrieved chunk
   compressed independently to $m$ memory tokens; cached in
   Chroma alongside the chunk (Section 5.6, PISCO pattern).
3. **RadixAttention-style prefix cache** on the memory-token
   prefix. Even when Hugin's memory tokens change per request,
   many RAG turns share the system prompt and adapter-scaled
   BOS; those positions stay cached.
4. **KV-cache in F16** (M10.2 Phase 1a, live).
5. **FlashAttention Q8_0-GQA prefill kernel** on the shortened
   sequence (already live).
6. **MTP drafters** on decode (planned M-MTP after M8.K).
7. **Model routing** — E4B for short-answer paths, 26B-A4B for
   long-form. Applies when `stream=true` and the client
   indicates a chat-style rather than analytical use.

**Compound projections** on a 20 000-token / 300-token-decode
request on Gemma 4 26B-A4B Q8_0, prod-governor:

| Stage                                          | Prefill (s) | Decode (s) | Total (s) | Speedup |
|------------------------------------------------|------------:|-----------:|----------:|--------:|
| Baseline (today)                               |          88 |         66 |       154 |    1.0× |
| + Hugin (live mode, $m=256$)                   |         3.1 |         66 |      69.1 |    2.2× |
| + Hugin (pre-compressed, $m=128$)              |         2.0 |         66 |      68.0 |    2.3× |
| + LLMLingua before Hugin (live only)           |         2.1 |         66 |      68.1 |    2.3× |
| + MTP drafters (2.5× decode)                   |         2.0 |       26.4 |      28.4 |    5.4× |
| + LLMLingua + Hugin + MTP + F16 KV             |         1.9 |       24.0 |      25.9 |    5.9× |
| + Model routing to E4B for short answers       |         1.2 |        7.5 |       8.7 | **18×** |

The dominant terms shift as we stack. In the baseline row, prefill
dominates (57 %). After just Hugin, decode dominates (96 %). After
MTP, prefill and decode are balanced (~7 % / ~93 %). This is why
the stack matters: attacking only prefill hits diminishing returns
fast; attacking only decode leaves the huge 88 s prefill floor.

**Time-to-first-token** (TTFT), the subjectively perceived
"responsiveness" metric:

| Stage                                     | 8 k ctx TTFT | 20 k ctx TTFT |
|-------------------------------------------|-------------:|--------------:|
| Baseline                                  |         35 s |          88 s |
| + Hugin live mode                         |        2.4 s |         3.1 s |
| + Hugin pre-compressed                    |        2.0 s |         2.0 s |
| + LLMLingua + Hugin pre-compressed        |        2.0 s |         2.0 s |
| + everything on E4B route                 |        1.3 s |         1.3 s |

At 20 000-token contexts the TTFT compresses from 88 seconds to
around 1.3 seconds — a **~68×** improvement on the metric users
actually perceive.

**How to stage the stack.** The compound-stack numbers only
materialise if the ordering below is respected during M-Hugin:

- M-Hugin.1 delivers Hugin alone → **~2.3× total, ~28× TTFT** on
  20 k contexts.
- Add MTP drafters (independent milestone M-MTP) →
  **~5.4× total**.
- Add pegenaut pre-compression (M-Hugin.3) → **~5.9× total,
  ~44× TTFT**.
- Add model routing (a config-only change on the pegenaut side) →
  **~18× total, ~68× TTFT**.

Model routing is the cheapest lever on this list and is worth
piloting even before Hugin lands, as a control experiment.

**LLMLingua's diminishing return.** Once the encoder is Perceiver-
shaped (cross-attention only, not full self-attention), Hugin's
encode cost is already linear in $n$ and small (~55 μs/token).
LLMLingua would drop 20 000 to ~1 000 tokens before Hugin, which
brings encode from 1.1 s to ~55 ms — a real saving in live mode,
but zero saving in pre-compressed mode because the encode was
amortised anyway. Recommendation: **skip LLMLingua in M-Hugin.1
and M-Hugin.2**, evaluate only if live-mode TTFT for very long
contexts still matters after M-Hugin.3.

**xRAG's 1-token-per-chunk regime.** At $m=1$ per chunk, five
retrieved chunks would occupy five prefix positions. The
theoretical maximum TTFT gain is real but xRAG requires
fine-tuning the base LLM (frozen-base contract broken). Filed as
research follow-up under M-Hugin.4 only if we later gain the
option to fine-tune the base ourselves.

---

## 7. Evaluation

### 7.1 Correctness parity

The primary correctness question is whether $F_\phi([\mathbf M;
\mathbf X_u])$ produces answers of the same quality as
$F_\phi([\mathbf X_d; \mathbf X_u])$. Two metrics:

- **Token-level distributional distance.** KL divergence between
  the base model's next-token distributions at each answer
  position, computed on a held-out RAG QA set. Threshold: median
  KL below the KL between the same base model under two different
  temperature-noise runs on the same input, which acts as a
  natural upper bound for "same for practical purposes".
- **Task-level answer quality.** Exact-match and F1 on:
  - NarrativeQA (long-form document QA)
  - HotpotQA (multi-hop, retrieval-augmented)
  - a held-out slice of the pegenaut production corpus

Acceptance: task-level quality within 2 percentage points of the
uncompressed baseline for the largest tested compression ratio.

### 7.2 Reference oracle for kernel parity

The Hugin encoder is new code producing new SPIR-V kernels. The
project's standing rule (CLAUDE.md, "reference oracle") is that
new kernels are element-wise validated against `llama-cli`. Hugin
does not have a `llama-cli` reference — no equivalent exists in
the upstream — so the reference for correctness during
implementation is a **CPU-only float64 implementation of the same
encoder graph** in a `gpu_tests` binary, executed on the same
weights. Deviation tolerance: 1 e-5 relative per element on Q8_0-
quantised weights, 1 e-3 for post-softmax activations.

### 7.3 Wall-clock parity

Bench-mode runs on L0_TARGET_HOST with governor pinned per
`operations.md`, three warm-ups, ten measurement iterations,
report p50 and p95. Ledger entry required per the standing
`feedback_perf_ledger` rule.

Comparison targets:

- Baseline: current tip, no Hugin, no spec-dec, live production
  kernels.
- Hugin only, live mode.
- Hugin only, pre-compressed mode.
- Hugin + MTP drafters (Section 8 of `mtp-diffusiongemma-status`).

### 7.4 Quality-loss stress cases

Compression is known to hurt on *precise-quote-retrieval*
questions and *exact-numeric* questions. A dedicated stress
subset of the evaluation covers:

- Paraphrase questions ("what does the document say about X?")
- Extract-precise-string questions ("copy the section-3 heading
  verbatim")
- Numeric questions ("what is the deadline listed in appendix B?")
- Multi-hop reasoning across multiple compressed chunks

If Hugin passes paraphrase and multi-hop but fails extract-verbatim
and numeric, that is a real deployment constraint that should be
documented and reflected in a routing heuristic: for questions that
look extractive, run uncompressed.

---

## 8. Risks and Limitations

### 8.1 Quality risk

Aggressive compression loses information. At 300× compression the
information budget per memory token is fewer than a dozen source
tokens' worth of entropy. Perceiver and BLIP-2 both report
smooth, gradual degradation with compression ratio; Gisting
reports a knee around 26×; ICAE around 15×. Hugin's target ratios
(50×–300×) are aggressive by every published baseline, and we
should expect the first prototype to require moderating to
$m \geq 512$ or 8×–32× ratios before quality is production-
acceptable.

The correct framing is that Hugin *widens* the RAG operating
envelope: contexts we today refuse to serve because prefill takes
80 s become servable, possibly with a small quality trade-off.
Contexts we already serve well can continue to be served
uncompressed via the `min_document_tokens` guard.

### 8.2 Training complexity

Training an adapter for a specific base model is a several-GPU-day
job that must be redone whenever the base model changes. For a
project targeting Gemma 4 as its production model this is bounded,
but the ADR-level decision to build Hugin should account for the
cost of maintaining trained adapters across Gemma 4 26B-A4B,
Gemma 4 E4B, and any future Gemma 4 31B dense.

### 8.3 Deployment complexity

Mode 2 (pre-compressed chunks in Chroma) requires coordination
between mimirmind and pegenaut: a chunk-ingestion pipeline, an
adapter-version tag on each stored memory-token blob, and an
invalidation strategy when the adapter is retrained. This is
manageable but non-zero engineering.

### 8.4 Prediction hazard

Section 6.5 already flags this. Every performance number in this
paper is a model, not a measurement. The project's track record
with prediction-driven perf work is mixed (DP4A, M5f.5). The M-
Hugin.1 milestone must include a bandwidth-microbenchmark gate
*before* the training investment is committed.

### 8.5 Interaction with speculative decoding

Hugin compresses the input; speculative decoding accelerates the
output. They are structurally compatible — the base model's forward
pass is unchanged in both — but the two levers together compound
the KV-cache size delta, since spec-dec temporarily doubles KV
storage for the draft. This is a good problem to have, but should
be measured, not assumed.

### 8.6 Multi-turn state

Once memory tokens are established for a document, the natural
question is whether they should persist across turns of a
conversation. The answer is yes, and the M-Hugin.3 milestone
extends the `session-kv-cache` roadmap note with exactly this:
memory tokens are stored in the session state alongside the
KV-cache, and only the user's new turn is prefilled fresh each
round.

### 8.7 Vocab and base-model compatibility

The Hugin adapter is bound to a specific base model's embedding
matrix and residual dimension. Switching base models invalidates
the adapter. This is not a bug — it is the same constraint as
prefix-tuning — but it means the adapter GGUF must carry an
enforced tokenizer-hash and residual-dimension check that the
loader honours *before* the encoder ever runs. Falling back to
uncompressed on mismatch is preferable to silent garbage.

---

## 9. Milestone Breakdown — M-Hugin

Phase: post-Mimir-1.0. Placement in the roadmap after Munin (which
delivers zero-downtime deploy, a prerequisite for iterating on
adapter versions in production).

### M-Hugin.1 — Runtime integration + Variant A on Gemma 4 E4B (1.5–2 weeks)

Effort scoped to a runtime-integration milestone because the
encoder architecture and training recipe are ported from COCOM.

- Bandwidth-microbenchmark gate: reproduce the 750 ms weight term
  from Section 6.3 on real Xe-LPG kernels. Blocker for the rest
  of the milestone.
- `HuginAdapter` class + SPIR-V kernels for cross-attention and
  projection.
- GGUF adapter format + loader integration (compatibility check
  by tokenizer hash and residual dimension).
- CPU-only reference-oracle test rig in `gpu_tests`.
- Load an already-trained COCOM-style checkpoint (if available for
  a small model with Gemma tokenizer) or train a first small
  Variant-A adapter following COCOM's published training recipe
  on FineWeb-Edu (1–3 GPU-days on a rented H100).
- Wall-clock parity report against uncompressed E4B on 4 000-,
  8 000-, and 16 000-token contexts.
- Go / no-go decision: does the adapter reach quality parity
  within 2 pp at 8× compression?

### M-Hugin.2 — Variant A + C for Gemma 4 26B-A4B, Autotune framework (3–3.5 weeks)

- Train Variant-A "mid" adapter for Gemma 4 26B-A4B on the
  pegenaut RAG corpus using ICAE-style self-distillation.
  Objective and curriculum lifted from ICAE / AutoCompressors
  code, no new training research.
- Train Variant-C adapter (multi-stage, re-injection at Layer 8).
  Reuses the same corpus and objective; adds a second small
  adapter block trained after the first has converged.
- **Autotune framework** (§4.7): winner cache, startup bench,
  three-signal decision logic (initially covering A vs. C vs.
  uncompressed), `hugin_autotune.json` file layout, Munin-shared
  persistence.
- Query-kind classifier stub (regex + keyword, cost budget
  <500 μs); only paraphrase vs. long-doc split is exercised in
  this milestone.
- `config.json` `hugin` block (§5.3) with `adapters.{A,C}` slots.
- Live-mode A/B on L0_TARGET_HOST against production baseline
  for A alone, then A+C+autotune together; ledger entry with
  wall-clock delta, energy delta, quality delta on the pegenaut
  internal QA set.
- Roll-out policy: `MIMIRMIND_HUGIN_VARIANT=auto` disabled by
  default in the first ledger round, then flipped after
  quality-gate confirmation.

### M-Hugin.3 — Variant D + query-conditioned autotune (2 weeks)

- Train Variant-D adapter (query-conditioned, single-stage) on
  the pegenaut RAG corpus. Training data includes
  (document, query, answer) triples so the encoder learns which
  document content matters given which question.
- Extend query-kind classifier to distinguish `paraphrase`,
  `extract_verbatim`, `numeric`, `multi_hop`; small held-out
  labelled set on the pegenaut corpus for calibration.
- Extend autotune to include Variant D with the safeguards from
  §4.7 (quality-gate margin, cost penalty prior, kill switch).
- A/B on the four query-kind buckets vs. the M-Hugin.2 baseline;
  ledger entry.

### M-Hugin.4 — Pre-compressed mode + compound stack integration (1.5–2 weeks)

- `POST /v1/hugin/encode` endpoint on mimirmind.
- Pegenaut ChromaDB ingestion pipeline extension: on chunk
  insert, call `/v1/hugin/encode` for `ingest_variants=[A, C]`,
  store memory tokens alongside chunk content, tag with adapter
  version.
- Query path: pegenaut retrieves memory tokens along with (or
  instead of) text, passes them as `content_kind: "hugin_memory"`;
  autotune's `chroma_cache_hit` signal wires to this.
- Cache invalidation policy on adapter retrain.
- Full compound-stack A/B: autotune-selected Hugin +
  MTP + F16 KV + model-routing on short-answer paths. Ledger
  entry with the full stack numbers from §6.6.

**Total scheduled effort: ~8–9.5 weeks.**

The step up from the earlier ~5–6-week estimate is entirely from
adding Variants C and D plus the autotune framework, which was
the explicit request; the algorithmic components are still
ported, not designed.

### M-Hugin.5 — Two-stage hybrid (contingent, not scheduled)

Only if the pegenaut-corpus quality analysis under M-Hugin.3
shows a clear gap between Variant A/C (cacheable but
query-agnostic) and Variant D (query-aware but not cacheable)
that motivates a **two-stage hybrid**: a large query-agnostic
stage-1 latent set stored in Chroma per chunk, and a small
query-conditioned stage-2 compressor run at query time on the
combined stage-1 latents + query. Estimated 3–4 weeks. Filed as
research, not scheduled.

---

## 10. Alternatives Considered

### 10.1 Chunked prefill with early exit

Feed only the top-K retrieved tokens per query into the base
model, ranked by a small classifier. Simpler than Hugin, but
throws away RAG signal outside the top-K; no notion of "compressed
whole-document context".

### 10.2 Retrieval at every layer (RETRO-style)

Retrieve chunks and cross-attend to them at every layer of the
base model. Requires modifying the base model — violates the
frozen-base constraint. Documented as future research if we ever
control our own base model.

### 10.3 Longer-context base model + FlashAttention 3

Just serve longer contexts natively with a better attention
kernel. Attacks the constant factor, not the linear-in-tokens
term. The FlashAttention Q8_0-GQA prefill kernel already landed
(commit `10e28a4`) and its win is measured in the 20 % range, not
in the 20× range Hugin targets.

### 10.4 KV-cache reuse via prefix caching (RadixAttention)

Reuse KV state across requests that share a prefix. Composes with
Hugin, does not replace it. Adds substantial storage cost for
long shared prefixes; Hugin reduces that storage cost by the
compression ratio.

### 10.5 Ship llama.cpp

Not a serious alternative given the CLAUDE.md project charter,
but worth naming as the null hypothesis: llama.cpp does not
implement anything Hugin-like today, and its RAG story on Xe-LPG
is not competitive.

### 10.6 Do nothing beyond what already ships

Rely entirely on the constant-factor kernel work (FlashAttention,
Q8_0, KV-F16, GEMM-prefill, CLR). Prefill on 20 k-token contexts
stays at roughly one minute. RAG turns above 10 k tokens remain
subjectively unusable, and pegenaut cannot deliver interactive-
feeling responses for corpus-heavy queries. This is the current
state and the null hypothesis against which all M-Hugin ledger
entries are measured.

---

## 11. Summary

Retrieval-augmented generation on integrated GPUs is a
prefill-bound workload whose dominant cost is memory traffic
proportional to the number of tokens entering the base model.
Every kernel-level optimisation mimirmind has landed this quarter
attacks a constant factor in that equation; none reduces the
token count.

**Hugin** attacks that token count by porting a well-established
input-side soft-embedding compression pattern (COCOM / PISCO /
ICAE / Compressed Context Memory) onto our specific runtime:
Level-Zero, USM, GGUF loader, Munin persistence, pegenaut Chroma.
The algorithmic content is not new — the field has published
adapters at compression ratios matching or exceeding our targets
since 2023 — but no open-source implementation targets this
hardware and runtime combination.

The scheduled work is engineering integration in four milestones
(~8–9.5 weeks), plus a distillation training pass per adapter
variant run off-target on a rented discrete GPU. Three adapter
variants coexist at runtime — Variant A (single-stage,
query-agnostic, Chroma-cacheable), Variant C (multi-stage with
Layer-8 re-injection, query-agnostic, Chroma-cacheable) and
Variant D (single-stage, query-conditioned, not cacheable) — with
per-request selection driven by an Autotune layer following the
project's existing `GpuMatmul::autotune` pattern. Every milestone
is gated by a measured comparison against the uncompressed
baseline on L0_TARGET_HOST and a ledger entry per the standing
perf-ledger rule.

The isolated Hugin lever yields **~2.3× end-to-end wall-clock**
and **~28× TTFT** on 20 000-token RAG contexts. The point of
building Hugin is not the isolated number but the compound stack
it enables: composed with the already-shipped KV-F16 foundation,
FlashAttention Q8_0-GQA, RadixAttention-style prefix cache, plus
the planned MTP drafters and pegenaut-side model routing, the
stack projects **~18× end-to-end** and **~68× TTFT** on the same
requests. Those compound numbers are the target of the project.

Nothing about this proposal is publication-worthy, and it is not
intended as one. It is an engineering path onto our existing
runtime that turns unusable long-context RAG (80-second prefill)
into a responsive interaction (~1–3-second TTFT), by combining
levers that individually already exist.

---

## 12. Recommended rollout for a single-maintainer project

The full M-Hugin.1–.4 build path is ~8–9.5 weeks of focused
engineering plus rented GPU time for three adapter retrains. For
a single-maintainer project, that is a large block to commit
before the underlying assumptions are measured. The phased
rollout below reduces risk by ordering the work so that each step
either delivers user-visible value or produces a hard measurement
that gates the next step.

### Phase 0 — Model routing (1–2 days, immediate)

Pegenaut-side configuration change: route short-answer requests
to Gemma 4 E4B, long-form to Gemma 4 26B-A4B. No mimirmind code
change; no adapter; no training.

- Expected win: **3–4× on short-answer wall-clock**, no quality
  cost on chat-style traffic.
- Doubles as a control experiment: does routing alone close
  enough of the UX gap that Hugin is no longer the biggest lever?

**Exit condition:** ledger entry with routing on / routing off
delta on the actual pegenaut traffic mix.

### Phase 1 — Traffic instrumentation (2–3 days)

Log per-request context length and query kind on pegenaut for
1–2 weeks. Collect the distribution.

**Decision gate:**

- Median context < 4 k tokens → **do not build Hugin.** Invest
  the time in MTP drafters and prefix caching instead.
- Median context 4–8 k → build **Variant A only + Chroma Mode 2**,
  skip C, D, and autotune. Saves ~4 weeks.
- Median context > 12 k → build the full A + C + autotune stack
  as specified in §9; the compound stack numbers justify it.
- Query kind is dominantly extract-verbatim / numeric → Variant D
  becomes a first-class requirement, not an M-Hugin.3 add-on.

### Phase 2 — Bandwidth microbenchmark gate (2–3 days)

Before any adapter training or C++ integration, reproduce the
§6.3 encoder-inner-loop cost on real Xe-LPG kernels. A ~50-line
SPIR-V microbench measuring the two dominant kernel shapes
(cross-attention read-out, weight-term matmul) on realistic
tensor shapes.

**Decision gate:**

- Weight term close to the predicted 750 ms → the §6 wall-clock
  model holds, proceed.
- Weight term substantially higher (say > 2 s) → the compound
  stack numbers do not hold. Recompute before committing weeks
  of work. This is exactly the guardrail
  `lesson_dp4a_autotune_prod_hazard` and
  `lesson_xelpg_dp4a_compute_ratio` codified for prior perf work.

### Phase 3 — Munin first (Prerequisite)

Adapter iteration in production without zero-downtime reload is
untenable — every retrain becomes a 90-second outage. Munin
(see `research/munin-persistent-model-daemon`) is not optional
before Hugin; it is a prerequisite. Build the Hot-Reload option
(the smaller Option B in the Munin proposal) at minimum before
starting M-Hugin.2.

### Phase 4 — Hugin minimal (~4–5 weeks)

Ship only Variant A + Chroma Mode 2:

- M-Hugin.1 unchanged (~1.5–2 W)
- Compressed M-Hugin.2: only Variant A on 26B-A4B, no autotune,
  no Variant C, no config `adapters.{C,D}` slots (~2 W)
- Compressed M-Hugin.4: pre-compressed mode + pegenaut Chroma
  integration + compound-stack A/B (~1.5 W)

Expected wins on the shipped path: **~2–3× total wall-clock,
~28× TTFT on 20 k contexts**. All numbers from §6.6 rows 1–2.
Not the 18× headline number, but 80 % of the value at 50 % of
the effort.

### Phase 5 — Autotune + Variant C/D (only if Phase 4 justifies it)

Once Variant A is in production for a few weeks and ledger
entries show real quality-vs-speed data on the pegenaut traffic:

- If quality is fine and traffic contexts are getting longer →
  add Variant C (multi-stage) for the very long docs.
- If quality gaps appear on extract-verbatim / numeric →
  add Variant D + query-conditioned autotune.
- If quality is uniformly fine → the compound stack is already
  as good as it gets without base-model changes; do not add C/D.

**Estimated effort at this stage:** 3–4 weeks incremental,
scoped by which of C/D actually justify themselves in the data.

### Kill criteria for the whole M-Hugin track

- Phase 1 shows median context ≤ 4 k tokens → drop Hugin, MTP
  and prefix caching cover the rest.
- Phase 2 microbench shows Xe-LPG encoder cost > 3× the
  prediction → drop Hugin, base-model longer-context work
  becomes the better lever.
- Phase 4 A-only ledger shows < 1.5× total wall-clock gain →
  do not proceed to C/D. Recover the sunk cost as documentation
  of what did not work on this class of hardware, and stop.

The point of this ordering is that no phase is a wasted step
even if the next phase is dropped: Phase 0 delivers user-visible
value directly; Phases 1–2 produce measurements you want anyway;
Phase 3 (Munin) is on the roadmap independently; Phase 4 is
where you first commit meaningful engineering hours, and only
after three cheap gates have said yes.

---

## References

Note: URLs are provided where the reference is a public web
document. Internal Synaipse notes are linked at the top of this
paper.

- Jaegle, A. et al. (2021). *Perceiver: General Perception with
  Iterative Attention.* ICML.
- Jaegle, A. et al. (2022). *Perceiver IO: A General Architecture
  for Structured Inputs and Outputs.* ICLR.
- Li, J., Li, D., Savarese, S., Hoi, S. (2023). *BLIP-2:
  Bootstrapping Language-Image Pre-training with Frozen Image
  Encoders and Large Language Models.* ICML.
- Li, X.L., Liang, P. (2021). *Prefix-Tuning: Optimizing
  Continuous Prompts for Generation.* ACL.
- Lester, B., Al-Rfou, R., Constant, N. (2021). *The Power of
  Scale for Parameter-Efficient Prompt Tuning.* EMNLP.
- Mu, J., Li, X.L., Goodman, N. (2024). *Learning to Compress
  Prompts with Gist Tokens.* NeurIPS 2023.
- Ge, T. et al. (2024). *In-Context Autoencoder for Context
  Compression in a Large Language Model.* ICLR.
- Chevalier, A., Wettig, A., Ajith, A., Chen, D. (2023).
  *Adapting Language Models to Compress Contexts.* EMNLP.
- Zheng, L. et al. (2024). *SGLang: Efficient Execution of
  Structured Language Model Programs / RadixAttention.*
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer
  Learning with a Unified Text-to-Text Transformer (T5).* JMLR.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence
  Pre-training for Natural Language Generation.* ACL.
- Google AI (2026-05-05). *Accelerating Gemma 4: MTP Drafters.*
  <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>
- Google AI (2026-06-10). *DiffusionGemma: 4× Faster Text
  Generation.*
  <https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/>

Internal mimirmind / Synaipse references (see `.mcp.json` for
access):

- `Memory/mimirmind/research/munin-persistent-model-daemon.md`
- `Memory/mimirmind/research/roadmap-speculative-decoding.md`
- `Memory/mimirmind/research/mtp-diffusiongemma-status-2026-07.md`
- `Memory/mimirmind/research/perf-regression-ledger.md`
- `Memory/mimirmind/decisions/target-model-quant.md`
- `Memory/mimirmind/decisions/config-json-migration.md`
- `Memory/mimirmind/decisions/2026-07-06-command-list-replay.md`
- `Memory/mimirmind/decisions/gemm-prefill-rewrite.md`
- `Memory/mimirmind/lessons/lesson_xelpg_dispatch_overhead.md`
- `Memory/mimirmind/lessons/lesson_dp4a_autotune_prod_hazard.md`

---

## Appendix A — Notational summary

| Symbol         | Meaning                                                    |
|----------------|------------------------------------------------------------|
| $n$            | Length of the input document token sequence                |
| $k$            | Length of the user turn token sequence                     |
| $m$            | Number of Hugin memory tokens (fixed hyperparameter)       |
| $d$            | Base-model residual-stream dimension                       |
| $d_h$          | Hugin encoder internal dimension                           |
| $L$            | Number of base-model transformer layers                    |
| $L_E$          | Number of Hugin encoder blocks                             |
| $\phi$         | Frozen base-model weights                                  |
| $\theta$       | Trainable Hugin encoder weights                            |
| $F_\phi$       | Base-model forward pass                                    |
| $E_\theta$     | Hugin encoder forward pass                                 |
| $W_E$          | Base-model token-embedding matrix                          |
| $\mathbf M$    | Memory token embeddings, shape $[m, d]$                    |
| $B$            | Sustained memory bandwidth, ≈ 55 GB/s on target            |

## Appendix B — Etymology

Odin's two ravens, *Huginn* (thought) and *Muninn* (memory),
appear together in *Grímnismál* stanza 20 of the *Poetic Edda*:

> Huginn ok Muninn fljúga hverjan dag
> Jörmungrund yfir;
> óumk ek of Hugin, at hann aftr né komi-t,
> þó sjámk meirr um Munin.

Odin fears Huginn will not come back — the thought that flies out
into the world may be lost. He fears more for Muninn — memory,
harder to replace once gone. The naming choice for this proposal
reflects the same asymmetry: Munin (persistence) is the
irreplaceable long-lived state; Hugin (compression) is
recomputable if lost.
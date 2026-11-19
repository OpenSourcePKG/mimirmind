# Hugin — Model Surgery Companion for MimirMind

*Inspect, locate and edit a frozen model through MCP; every edit is a Rune.*

**Version:** hugin 0.5.0 (2026-11)
**German version:** [`hugin.de.md`](hugin.de.md)
**Roadmap:** track 7 (M-Hugin + Futhark) in the Synaipse roadmap

---

## 1. What Hugin is

Hugin is a separate C++ process next to `mimirmind`. It loads the same base
model with the same engine code and exposes it as an **MCP server on
stdio**. An agent (Claude, through Claude Code) drives it with tools: look
at weights and activations, find where a behaviour lives, try an
intervention, and write the result down as a **Rune**.

The rules Hugin works under:

- **The base model file is never modified.** Every change is a Rune file,
  bound to the base model by a fingerprint, applied by **Futhark** after
  load, and removed again by taking the Rune out of the config.
- **Preview equals Rune.** Every preview tool runs the same engine path the
  Rune will use in serving. What a preview shows is what the staged Rune
  does.
- **Hugin never activates anything.** It stages, gates and commits Runes.
  Turning one on is a config change (`models[].runes`) plus a restart,
  done by a human in a maintenance window.
- **No Python, no training framework.** Hugin is built from the engine's
  own sources; the forward is the serving forward.

The name follows the Norse theme: Huginn (*thought*) flies out and comes
back with what it found; Muninn (*memory*, our model-residency daemon)
keeps what is there.

Not yet built (roadmap 7): adapter training with its own C++ backward core
(LoRA-SFT, DAgger, GKD, pruning + healing) and the input-compression
encoder (design: [`hugin-input-compression.md`](hugin-input-compression.md)).

---

## 2. Build and run

Hugin is a CMake target of the main tree:

```bash
cmake --build build --target hugin
```

On a CUDA build it can open NVFP4 checkpoints; a CPU build covers GGUF
models (useful for fast validation on small models such as Llama 3.2 1B).

```text
hugin [--model <file.gguf | nvfp4-checkpoint-dir>] [--tokenizer-gguf <file.gguf>]
      [--rune-dir <dir>]      staged/committed/rejected Runes + audit log (default ./runes)
      [--single-engine]       hold base OR rune variant, never both
```

- `--single-engine` is for models too large to hold twice (Kolibri-1 on
  128 GB). Comparisons then evaluate the base on all sets first and load
  the variant once.
- On CUDA, start Hugin with the serving environment of the model
  (`MIMIRMIND_*` flags) so its forward matches production.
- Only one GPU process at a time on GB10: stop serving before a GPU
  Hugin session.

Register it with Claude Code:

```bash
claude mcp add hugin -- /path/to/hugin --model /path/to/model.gguf
```

Without `--model`, call `open_model` first.

---

## 3. Runes and Futhark

A **Rune** is a safetensors file. The manifest sits in the header
`__metadata__` under `mimirmind.rune`; the tensors are the payloads. The
manifest records name, base fingerprint, provenance (tool, parameters) and
a list of ops.

| Op kind | What it does | Applied as |
|---|---|---|
| `write_slice` | Overwrite a row/column range of a tensor | weight write at load |
| `scale` | Multiply rows or columns (e.g. one head's O-proj columns) | weight write at load |
| `mask` | Disable MoE experts: router logits → −inf before top-K | load-time mask (overlay) |
| `steer` | x += v after block L, every row, every path | residual overlay |
| `steer_project` | x −= α (x·u) u, u unit, 0 < α ≤ 1 | residual overlay |
| `steer_if` | add or project, only rows with cos(h_Lc, c) > θ | residual overlay + device gate |
| `lowrank`, `write_quant_blocks` | defined in the format | refused at load (not yet applied) |

Futhark, the applier inside `mimirmind`:

1. checks the base fingerprint, op applicability, target type and shape,
   the per-op tensor hash chain and payload sizes, for every Rune in stack
   order, **before** anything is written;
2. writes weight ops into the loaded weights (before any derived weight copy
   is built) and installs overlays in the arch backend;
3. logs one line per Rune and per overlay (`futhark` log tag);
4. refuses loudly — never silently skips — when a backend lacks an op
   (for example `steer_if` on Level Zero or HIP).

Activation is a config entry per model:

```json
{ "id": "kolibri1", "runes": ["/path/to/runes/committed/my_rune.rune"] }
```

Removing the entry and restarting returns to the exact base model.

---

## 4. Lifecycle: stage → gate → commit

```text
preview tools ──► rune_stage_*  ──► staged/
                                       │ rune_gate (base vs. rune, Futhark load path)
                      red ◄────────────┤
                 rejected/             │ green
                                       ▼
                                  rune_commit ──► committed/ ──► config + restart (human)
                                       ▲
                                  rune_rollback (committed → staged → rejected)
```

- **`rune_stage_*`** writes the Rune and a sidecar with parameters and
  history. Guardrails per kind (relative steer norm, head fraction, scale
  ceiling, projection α range). Steer kinds additionally (7.12):
  - no steering after block 0; a dose above 0.5 of the residual norm
    needs an `override_reason`;
  - **held-out evidence**: the direction (`steer`, `steer_project`) or the
    condition (`steer_if`) must have been fitted with `holdout_positive` /
    `holdout_negative` prompts — at least 4 per class, held-out AUC ≥
    `min_holdout_auc` (0.8), and for a condition at most
    `max_holdout_negative_fire` (5 %) of every gated row of the held-out
    negatives above the threshold. Held-out prompts that repeat a fitting
    prompt are refused;
  - the shuffled-label null is reported but not gated on: with
    d_model ≫ prompts, in-sample AUC and probe val accuracy reach 1.0 on
    random labels too, so they say nothing on their own;
  - **causality**: `causality_check` must have judged exactly this edit
    (vector at this α, or this set of head edits) at least `weak` — effect
    ≥ 1.5× the 95th percentile of random edits of the same size (random
    directions of equal norm, or random other heads); `causal` additionally
    needs ≥ 3× and an effect that grows over 25 / 50 / 100 % dose. Head
    edits also need a measured dose ≤ 0.5 (residual change / residual norm
    at the edited block);
    the baseline for a vector is the mean difference of its own fitting
    prompts under shuffled labels at the same norm — random directions in
    R^d are too easy to beat. Measure with a `target` token where possible:
    KL only shows *that* the output moves, and a real French steer on
    Llama 3.2 1B was `causal` (3.1×) on the target but not above the
    baseline on KL. The latest check of an edit is the one the gate uses;
    its metric is recorded in the provenance;
  - any failed check can be overridden with an `override_reason`, which is
    written into the provenance.
- **`rune_gate`** runs a `target` set (what must improve) and a `control`
  set (what must not regress) on the base and on the Rune-applied model.
  Green iff:
  - target accuracy does not drop and accuracy or mean logprob improves;
  - control accuracy drops by at most `max_control_accuracy_drop`
    (default 0) and mean logprob by at most `max_control_logprob_drop`
    (default 0.05);
  - no new non-finite items;
  - free-generation checks: at most `max_check_regressions` (default 0)
    checks that pass on the base and fail with the Rune, per set.

  Red moves the Rune to `rejected/` unless `auto_reject` is false.
- **`rune_commit`** only accepts a green Rune whose file is byte-identical
  to what the gate evaluated, and returns the config line.
- Every step is appended to `audit.jsonl` in the Rune directory. Rejected
  Runes stay on disk.

A green gate is necessary, not sufficient. Check the Rune on the real
prompts of the deployment (system prompts, tools, RAG context) before
activation — Kolibri's `de_heads_top64_x2` passed its gate and still
flipped English RAG users to German under the production identity prompt.

---

## 5. Goldsets

`eval`, `eval_preview`, `rune_gate` and `steer_if_preview` share one item
format (inline `items` or a `.json` / `.jsonl` file via `path`):

| Item | Correct iff |
|---|---|
| `{prompt, expect}` | every expected token is the argmax (teacher-forced) |
| `{prompt, choices[], answer}` | the answer has the highest total logprob |
| `{prompt, generate:{max_tokens, lang, max_other_markers, match[], no_match[]}}` | greedy KV-cached generation passes every check: answer language (`de` / `en`), no code-mixing, fact regexes |

Language checks count function words and need at least three marker words;
very short answers get no language verdict, so read the answer diffs as
well.

Prompts may contain the model's special tokens (`<|im_start|>` …); they
are parsed as special tokens, so chat-formatted items reach the forward
exactly as in serving.

---

## 6. Tool reference

### Model and weights

| Tool | Purpose |
|---|---|
| `open_model` | Open a GGUF file (weights + forward) or an NVFP4 checkpoint (forward only, CUDA) |
| `model_info` | Architecture, base fingerprint, tensor count, bytes per type |
| `list_tensors` | Tensors with type, shape, size and whether a Rune can patch them |
| `tensor_stats` | Mean, std, min/max, L2, NaN/Inf, per-row norms (host dequant) |
| `tensor_hash` | fnv1a64 content hash — the hash a Rune op must carry |

### Forward and inspection

| Tool | Purpose |
|---|---|
| `tokenize` | Ids and pieces with the model's tokenizer |
| `next_token` | Top-k next-token distribution and entropy |
| `perplexity` | Teacher-forced per-token logprob, NLL, perplexity |
| `logit_lens` | Residual after each block through final norm + lm_head |
| `activations` | Residual statistics per block (norm, update, cosine, top dims) |
| `attention_pattern` | One block's attention weights per head, rebuilt from the engine's post-RoPE Q/K |
| `expert_usage` | MoE routing actually taken (stage tap at the router selection) |
| `causal_trace` | Activation patching clean → corrupt per (block, position) |
| `causality_check` | Is a located edit (steer vector at α, or head edits) more than a random edit of the same size? Dose sweep + verdict causal / weak / epiphenomenal / undetermined + measured dose (§4) |
| `edit_search` | Multi-objective edit search (own TPE, no Optuna): head count k / scale / block window, or steer α; target goldset up, control KL down, no free-generation regression; returns the Pareto front. Stage a front point with `rune_stage_heads` / `rune_stage_steer` + `search` + `trial` (parameters and seed in the provenance) |
| `eval` | Goldset eval, optionally against Runes (deltas) |
| `generate` | Greedy continuation with optional steer / head edits (slow, stateless) |

### Interventions (preview only, nothing written)

| Tool | Purpose |
|---|---|
| `steer_vector` | Mean residual difference positive − negative at one block |
| `steer_list` | Steering vectors of this session |
| `steer_preview` | Next-token shift (and text) with α · vector added |
| `ablate_preview` | Zero or scale attention heads; per-head scan |
| `eval_preview` | Goldset eval with steer and/or head edits on every forward |
| `condition_vector` | Fit the gate of a conditional steer (see §7) |
| `steer_if_preview` | Projection / conditional steer through the real overlay (see §7) |

### Runes

| Tool | Purpose |
|---|---|
| `rune_stage_steer` | Steering vector → `steer` Rune |
| `rune_stage_steer_if` | → `steer_if` (with condition) or `steer_project` Rune |
| `rune_stage_ablate` | Head ablation → 0/1 column `scale` on `attn_output` |
| `rune_stage_heads` | Head scaling → column `scale` (exact on Q8_0 / FP8 block scales for 0 or powers of two) |
| `rune_stage_experts` | Expert disable → `mask` on the MoE router |
| `rune_gate` | Target/control gate, base vs. Rune (§4) |
| `rune_commit` | Green + byte-identical → `committed/`, returns the config line |
| `rune_rollback` | committed → staged → rejected |
| `rune_list` | All Runes with status, kind and gate summary |
| `rune_inspect` | Decode a Rune manifest |
| `rune_check` | Dry-run Runes against the loaded model with Futhark's own checks |
| `rune_compare` | Same input on base and on Rune-applied model: KL, top-1 agreement, perplexity |

---

## 7. Conditional and projection steering (7.11)

A plain `steer` adds the same vector to every row. That fits an
unconditional goal ("always a bit more formal"), but not a conditional one
("answer in German *when* the system prompt asks for it") — the constant
shift also moves the cases that must stay as they are.

`steer_if` adds a gate. At a condition block Lc ≤ L the engine scores each
row:

```text
s_r = cos(h_r at Lc, c)            c = unit condition direction
x_r += α · [s_r > θ] · v           (add)   or
x_r −= α · [s_r > θ] · (x_r·u) u   (project), at block L
```

Design decisions:

- **Per row, not per sequence.** CAST (IBM, ICLR 2025) decides once per
  sequence at prefill. A per-row gate needs no row → sequence map, runs
  entirely on the device (no host sync) and works unchanged on every
  forward path: prefill, decode, continuous batching with ragged rows, and
  CUDA graphs. Consequence: the condition is fitted on the rows that
  produce the answer, not only on the end of the prompt.
- **New op kinds, not new fields.** An older Futhark would ignore an
  unknown field and apply the steer unconditionally; an unknown op kind is
  refused.
- **Device ops** behind `IRuneSteerOps` (`rowDot`, gated row add) on CPU and
  CUDA. Level Zero and HIP return no implementation, so such a Rune fails
  at load there.
- Projection α is limited to (0, 1]: α > 1 would reflect the component and
  install the opposite behaviour.

Workflow:

1. `steer_vector` — the direction to add (or to project out).
2. `condition_vector` — a binary linear probe (the 8.23 softmax trainer) on
   L2-normalised residuals at block Lc, positive vs. negative prompts.
   `positions: "generated"` (default) uses the rows that generate the
   base model's first `max_tokens` answer tokens. Returns direction,
   threshold, train/val accuracy, per-class score ranges and per-prompt
   fire rates. `calibrate_negatives: true` scores every row the gate will
   see on the negative prompts (all prompt rows plus generated answer rows)
   and raises the threshold above their maximum — required when the
   direction is fitted on `last` rows, because the gate runs on every row.
3. `steer_if_preview` — installs the overlay exactly as the Rune would
   (resolved through the real Rune applier) and reports base vs. edited
   text plus fire rates over prompt and answer rows. Optional goldset.
4. `rune_stage_steer_if` → `rune_gate` → `rune_commit`.

Validation on Llama 3.2 1B (CPU): a "cooking" condition at block 4
separated held-out prompts (val accuracy 0.93); a French steer gated on it
switched cooking answers to French while all other answers stayed
byte-identical to the base; the staged Runes reproduced the preview texts
exactly (4/4 and 2/2).

---

## 8. Limits

- NVFP4 checkpoints: forward tools only (no host weight inspection), CUDA
  build required.
- `generate` is stateless and slow; `steer_if_preview` and generate-items
  in `eval` use the KV-cached serving decode.
- One GPU process at a time on GB10; Hugin sessions on a large model need a
  maintenance window.
- Keep the prompts used to fit steer and condition vectors disjoint from
  the gate goldsets, and use the same questions in both classes of a
  condition (only the property under test may differ). On Kolibri-1 a
  "German instruction present" condition looked perfectly separable
  (val 1.00) until the fitting questions were made disjoint; then it fell
  to chance (0.57).
- Steering and head edits are cheap levers. Failures that need new
  knowledge or a different policy belong to adapter training (roadmap
  7.4/7.5), not to a Rune of this kind.

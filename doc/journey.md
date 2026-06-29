# The Gemma 4 debug story

> A long day, a lot of false starts, and one specific lesson that
> would have saved most of them.

This is a development post-mortem more than a technical reference. If
you want to know how Mimirmind works, read [`architecture.md`](architecture.md).
If you want to know what *not* to do next time you bring up a new
model, keep reading.

## Setting

End of June 2026. The engine had been running Qwen 2.5 cleanly for
weeks. Gemma 3 had been brought up in pieces. The next target was the
Gemma 4 26B-A4B Instruct — Google's mixture-of-experts flagship, a
26-billion-parameter model where only 4 B parameters activate per
token. The architecture has a number of gemma4-specific quirks:

- Hybrid attention: 25 sliding-window layers (head_dim 256, 8 KV heads)
  interleaved with 5 full-attention layers (head_dim 512, 2 KV heads).
- Q-K normalisation per head before RoPE.
- `f_attention_scale = 1.0` instead of the usual `1/sqrt(head_dim)`,
  which the engine handles by pre-scaling Q.
- Dense + MoE FFN in parallel with a five-norm choreography around them.
- 128 experts, top-8 routing with renormalised weights and a per-expert
  output scale.
- A proportional-RoPE `freq_factors` table applied only on the
  full-attention layers.

We had the model file. We had the architecture spec. We had a parity
test harness against `llama-cli`. We sat down to make it talk.

## The first sound was garbage

Greedy decode, the prompt was `Hello, world!`, the model emitted:

```
`1\11-;-``
```

Random ASCII soup. Not a single recognisable word. The natural
hypothesis: the architecture is wrong somewhere.

Over the next several hours we found and fixed ten gemma4-specific
bugs, every one of them real:

| # | Bug |
|---|---|
| 1 | The V projection needs a bare RMSNorm (no learned weight) before going into the KV cache. |
| 2 | Q is pre-scaled by `sqrt(head_dim)` so the attention's internal `1/sqrt(head_dim)` cancels. |
| 3 | The `ffn_down_exps.scale` per-expert multiplier was being dropped on the floor. |
| 4 | Gemma 4 uses standard `y = w · norm(x)`, not the `(1+w) · norm(x)` variant we'd inherited from Gemma 3. |
| 5 | Sliding-window layers use a different RoPE base (10000) than full-attention layers (1e6). |
| 6 | The proportional-RoPE `freq_factors` table applies only to full-attention layers. |
| 7 | Per-layer head_dim and KV-head count — SWA: 256×8, full: 512×2. |
| 8 | Full-attention layers can omit `attn_v.weight` entirely; when they do, V is derived from the raw K projection. |
| 9 | The tokenizer's `add_space_prefix` flag is `false` for Gemma 4 (Gemma 2/3 default to true). This was the single biggest single-fix improvement. |
| 10 | The MoE router's input gets a specific normalise → multiply-by-scale → multiply-by-`1/sqrt(d_model)` chain. |

Each fix made the output measurably better. The progression went:

```
\`1\\11-;-\`\`               (start)
तपणे...kali...               (+ standard norm)
ेंेंกว่า로...inglyingly       (+ embedding scale)
true true true true ...     (+ V-norm + Q pre-scale)
true greatest true greatest  (+ ffn_down_exps.scale)
own than-m-9-0-0-0-0-1- 0-  (+ standard w·norm)
ownwriters-1-1-11- ...      (+ per-layer SWA + freq_factors)
,,,!!!!!!!!!!!!!!!!!         (+ per-layer dims + alt-attn)
\n\n!!!!!!!!!!!!!!!!!!!     (+ tokenizer add_space_prefix=false)
```

Every fix narrowed the failure mode. The structural divergence at
block 0 dropped from `mean_abs = 0.12` to `mean_abs = 0.005`. The
parity-dump tooling we built to track this — the
`llama-parity-dump` tool wired into a `cb_eval` callback inside
llama.cpp, plus a per-stage tensor diff helper — became the single
most useful piece of infrastructure in the codebase.

And the model still produced garbage.

## The precision side-quest

By the late afternoon the architectural fixes had run their course.
The remaining failure mode looked like cumulative numerical drift:
block-0 `ffn_mlp` showed sparse outliers up to `max_abs = 3.81` even
though the mean was `0.029` and the inputs (post-RoPE Q and K) were
within `0.05` of the llama-cli reference. Across 30 blocks the drift
accumulated to `max_abs = 12`, well past the threshold at which the
greedy argmax would flip.

The natural hypothesis: Q6_K dequant precision is the bottleneck. We
spent the next several hours on it:

1. **Kahan-compensated accumulation** in the Q6_K matmul kernel.
   Helped marginally — the per-stage diff at block 0 stayed at 3.81 —
   but did push the greedy output from `\n\n!!!` to ` it it it it`.
   "Slightly better" felt like progress.
2. **A higher-precision model build** — the Q8_0 variant of Gemma 4
   instead of Q6_K. 27 GB download. Loaded it. Block-0 `ffn_mlp` diff
   dropped from 3.81 to 1.21, exactly as you'd expect from a
   higher-precision quantisation. Greedy output went to
   ` the possible.<turn|><channel|><|image><|image>×15`.
3. **Re-ran the per-stage parity dump** on Q8_0. Final-block drift
   was basically unchanged. The end-to-end diff stayed at 12-ish.

Wait. The first three tokens — `the`, `possible`, `.` — are
*real English words*. After three plausible tokens, the model
suddenly emits `<turn|>` and then a long run of `<|image>` tokens.
Those are special tokens from Gemma 4's vocabulary.

It is generating *its own chat markup*.

## The actual bug

Gemma 4 is an instruction-tuned model. Its training data is full of
the chat-format wrapping: every example begins with
`<|turn>user\n{content}<turn|>\n<|turn>model\n` and the model has
learned to emit `<turn|>` to signal end-of-turn, `<|channel>` to
delimit a thinking channel, and so on.

When we fed it the prompt `What is the capital of France?` as raw
completion text — no chat template, no turn markers — the model had
no idea what context it was supposed to be in. After a few tokens of
plausible completion it slipped into the trained chat-markup pattern
that dominated its training set: `<turn|>`, `<channel|>`,
`<|image>`. From the outside it looked like garbage. From the
model's point of view it was doing exactly what it was trained to do.

We implemented the chat template properly:

```
<bos><|turn>user
What is the capital of France?<turn|>
<|turn>model
```

(Note Gemma 4 dropped Gemma 2/3's symmetric `<start_of_turn>` /
`<end_of_turn>` for the asymmetric `<|turn>` / `<turn|>` pair.
Same algorithm, different literals.)

First run with chat template applied:

```
<|channel>thought
<channel|>The capital of France is **Paris**.<turn|>
```

13 tokens, hit `<turn|>`, perfect output. The `<|channel>` /
`<channel|>` wrapper at the start is Gemma 4's thinking-channel
markup — emitted even when thinking mode is off, contents empty in
that case. We strip it from the OpenAI-API response in
`ChatTemplate::cleanResponse`.

Then we went back to Q6_K — the same model we'd been fighting all day —
and re-ran with the chat template:

```
<|channel>thought
<channel|>The capital of France is **Paris**.<turn|>
```

**Bit-identical to Q8_0.** Same tokens. Same speed. The precision
side-quest had not been false — Q6_K does have more dequant noise
than Q8_0, and the parity-dump numbers don't lie — but the precision
was *never the reason the model was producing garbage*.

## What the architectural fixes were worth

All ten of the gemma4-specific architecture bugs were real. The
parity test showed them, the per-stage diffs proved them out, every
single one had to be fixed before the model would produce coherent
text *with* the chat template. We were not chasing ghosts. The
architecture was actually broken, and we actually fixed it.

What we did wrong was the *order*. We assumed that the visible
failure mode — garbage tokens out of greedy decode — would be cured
when the architecture matched. With a base model (a completion model
that has never been fine-tuned for chat) that assumption is sound.
With an instruction-tuned model, the assumption is false: even a
perfectly-implemented forward pass produces "garbage" if the prompt
format doesn't match what the model expects.

If we had implemented the chat template *first* and only then started
debugging the architecture, every fix would have produced visibly
coherent output (just *wrong* coherent output until the math was
right). The architecture bugs would have been just as real, but the
failure mode would have been "the model says Berlin is the capital of
France" rather than "the model emits ASCII soup", and the debug
process would have stayed grounded in observable language. Instead we
spent hours interpreting nonsense.

## The lesson

> When an instruction-tuned model produces garbage, check the chat
> template *before* you check anything else.

Specifically, the order we should always follow on a new model
bring-up from now on:

1. **Tokenizer sanity check** — `encode(decode(s)) == s` for a few
   strings. Make sure `<bos>`, `<eos>`, and the model's chat-format
   special tokens are present in the vocab.
2. **Chat template implemented** — pull the literal Jinja template
   from the HF model card, identify the special tokens, wire them up.
   Confirm that `findToken()` returns non-negative IDs for every
   token the template needs.
3. **Single chat-formatted prompt** — a basic factual question. If
   the model gives a plausible but wrong answer, the chat path is
   working and you can now debug the architecture. If the model
   gives garbage, the chat template is still wrong.
4. **Only then** start the parity test, the architecture audit, the
   precision investigation.

Steps 1-3 take about 30 minutes. Skipping them cost us 12 hours.

## What we kept from the long way around

- **The parity test harness** — `llama-parity-dump` with `cb_eval`
  callbacks dumping per-stage tensors plus the `parity-diff.py`
  helper. We will use this on every future model bring-up. Even now
  that we know precision wasn't the bug, the harness is the right
  way to *prove* that.
- **The QuantType polymorphic refactor** — was triggered by the
  precision investigation but is genuinely cleaner code, and made
  the Q8_0 kernel addition a one-line wiring change.
- **82 unit tests** — pure-CPU dequant, registry, factory, softmax,
  multiHeadAttention, MoE routing, and 14 GPU-vs-CPU kernel
  parity tests. All of these were written during the side-quest
  and they catch regressions cleanly.
- **Kahan in the Q6_K kernel** — stays in. It's correct. It just
  wasn't the fix we needed it to be.
- **The Gemma 4 chat-template implementation** — including the
  asymmetric turn markers, the thinking-channel stripping, the
  system-message folding into the first user turn. That part is
  genuinely Mimir-1.0-shippable.

## And the result

Gemma 4 26B-A4B Instruct, Q6_K, on consumer Intel Arc Graphics
(Meteor Lake):

```
What is the capital of France?
→ The capital of France is **Paris**.
```

145 ms per token. About 7 tokens per second of a 26-billion-parameter
mixture-of-experts model running on an integrated GPU. The same
hardware that, this morning, was producing `\n\n!!!`.

Days like this are the price of admission for projects like this.
The lesson is cheap if you read it once and write it down.

---

Reference notes: `Memory/mimirmind/lessons/chat-template-first.md` and
`Memory/mimirmind/status/phase-mimir-1.0-m8.md` in the project's
Synaipse vault carry the original session log if you want the
unedited version.
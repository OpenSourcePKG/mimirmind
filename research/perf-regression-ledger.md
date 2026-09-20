# Perf-Regression Ledger

Post-deploy performance entries. Newest first. Box = Spark GB10 (xd-ki-pkg1),
model qwen3.6-35B-A3B-NVFP4 unless noted.

## 2026-09-21 — GDN conv+split fusion (5.18.21.6) — DEPLOYED, ~3% prefill

**Change:** commit `2419978`. New `gdn_conv_split_fuse` kernel folds the standalone
conv1d-silu (`gdn.conv.k`) into the warp post-conv-prep (`gdn.split`): each q/k/v read
becomes an inline causal conv1d+SiLU reading `convInput` directly, so the `qkvMixed`
intermediate never touches HBM and one launch is dropped. Same conv Σ-order + SiLU + the
same shared-memory L2 tree as the two-kernel path → output matches baseline. Gated behind
`MIMIRMIND_GDN_CONV_SPLIT_FUSE` (default OFF), ragged serving prefill + conv-batch-pack +
S≤1024; falls back otherwise.

**Deploy:** flag made durable in `prod_recreate.sh` (backup `prod_recreate.sh.pre-convsplit`);
prod recreated, `MIMIRMIND_GDN_CONV_SPLIT_FUSE=1` live, PROD_OK.

**A/B + prod validation (@~3.3-3.5k tok, full prod flags):**
- Stage-pair: `gdn.conv.k` 6.16 + `gdn.split` 7.56 = **13.72 ms/chunk** (baseline) →
  `gdn.conv.k` ABSENT + `gdn.split` 7.82 = **7.82 ms/chunk** (fused) = ~5.9 ms/chunk saved.
- Prod warm prefill **0.542 ms/tok** vs 0.561 baseline = **~3% faster/token**.
- Parity: coherence OK, Walter-Moers byte-identical head + no loop (both A/B phases + prod).

**Rollback:** `MIMIRMIND_GDN_CONV_SPLIT_FUSE=0` (env) or restore `prod_recreate.sh.pre-convsplit`.
Context: this is the one compute win of the 5.18.21 prefill campaign; the remaining ~3x vs
vLLM is at the CUTLASS sm120 floor (MoE-GEMM) + distributed micro-inefficiency, not reachable.

## 2026-09-20 — FRESH vLLM prefill head-to-head (5.18.21.5) — 3.6x gap is REAL, NOT stale

**Measurement:** prefill_h2h.sh, client-timed TTFT (max_tokens=1, temp=0, per-run nonce → no
prefix-cache), ~3468tok, clk 2405MHz, sanctioned swap.
- **mimirmind** current prod (4.7.1 + glue-fusion + cuDNN-paged + all prod flags): **~1945ms** (0.56 ms/tok)
- **vLLM** nvcr 26.06, same NVFP4 model: **~541ms** (0.156 ms/tok)
- **Gap = 3.6x** — reproduced on CURRENT prod; the 09-15 anchor was NOT stale.

**Redirect:** attn is NOT the main lever (O(T²) slope ⇒ attn ~20-30% at 3468tok; zeroing it still
leaves ~2.5-3x). The 3.6x is dominated by the LINEAR terms — MoE-glue (dgemm/prep/scatter/silu/gather
~38%) + GDN/mamba (~31%), each ~3x slower/token than vLLM. MoE-GEMM is already parity (3 neutral
levers). vLLM's edge = fused MoE glue + optimized mamba chunked-scan. Next real levers: GDN prefill
(Phase 2) + MoE-glue fusion. FlashInfer attn kernel = dead end on GB10 (sm_121=FA2 class).
Note: research/2026-09-20-attn-prefill-oracle-flashinfer-not-a-lever-gb10.

## 2026-09-20 — CUTLASS v4.4.2 → v4.7.1 bump (5.18.21.2) — PERF-NEUTRAL

**Change:** `third_party/cutlass` git checkout v4.4.2 → v4.7.1 (vLLM's version), rebuild
`mimirmind_cutlass_moe` + relink `mimirmind`. Built clean — NO API breaks (RasterOrderOptions
detail-ns + CollectiveBuilder/KernelScheduleAuto compatible; `[100%] Built target mimirmind` rc=0).

**Serve-A/B (ab_cutlass471.sh, prod config.serve.json, clock 2405MHz confirmed under load):**
- Warm prefill **171 / 173 / 172 ms / 319tok** = IDENTICAL to 4.4.2's 170–172ms → **perf-neutral**.
- Coherence OK; Walter-Moers 1194 chars, no loop → numerics stable under the new schedule.

**Verdict:** REFUTES the "4.4.2 KernelScheduleAuto resolves to a weaker sm120 FP4 schedule"
hypothesis (was the "highest-ceiling" delta-2 lever). 3rd consecutive neutral MoE-GEMM result
(after w13-fusion + glue-fusion) → the prefill gap vs vLLM is NOT in the MoE GEMM; it is
distributed (confirms Phase-0). Prod currently LIVE on the 4.7.1 binary. No baseline-binary
backup (root-owned build dir); revert = `git checkout v4.4.2` + rebuild. keep-vs-revert: user decision.

## 2026-09-20 — MoE glue-fusion: fuse per-expert gather into TC act-quant (5.18.21.2)

**Deploy:** commit 922f778 (branch feat/moe-tc-w13-fusion-raster; build-cudnn binary
live on prod via docker start re-exec). The standalone `rt.gather` (~4%/9.2ms of warm
prefill) is fused INTO the TC-path act-quant via an optional `srcMap` on
`moe_act_quant_nvfp4_gather_rows` — reads ungathered `moeInput` at `rowSrcTok[logical]`,
skips the `xComp` intermediate. Bit-identical by design; `rt.gather` retained only for
the non-TC scalar decode path (M=1 `runMoeFfn`).

**Serve-A/B (ab_gluew13.sh, sanctioned swap):**
- `rt.gather` GONE from all TC-prefill profiler lines (0 occurrences); decode M=1 keeps
  it (0.18ms, intended).
- PROD-config warm prefill = **170–172ms / 319tok** (cold 629ms) = 0.536ms/tok vs baseline
  0.543ms/tok (215ms/396tok) → **on-par, bit-identical, NO regression**. TTFT win in the
  noise at this prompt size; net = clean structural simplification (fewer launches, no
  intermediate buffer).
- Coherence OK ("Hauptstadt Frankreich = Paris"); Walter-Moers open-ended = no degeneration
  loop (rep-penalty prod default, still load-bearing).

**Gotcha:** profile prefill TTFT on prod `config.serve.json`, NOT `config.gdbprofile.json`
— the latter's GDN-chunked-TC path is degraded (`gdn.k2=212ms/52%`) and drowns any MoE
delta. `c067f8f` (w13 gate+up fusion, gateup=1) = perf-neutral, likely env-gated OFF.
Full detail: research/2026-09-20-qwen36-serve-prefill-runtime-profile-cutlass-tc-already.

## 2026-09-17 — string `pattern` (regex) grammar enforcement (5.31)

**Deploy:** commit d5872d1. A JSON-Schema string `pattern` is compiled to EBNF via
xgrammar's OWN regex converter `RegexToEBNF` (the one Grammar::FromRegex uses;
forward-declared, symbol in libxgrammar.a) and spliced as the value rule — covers
char classes/quantifiers/groups/alternation/anchors/\d\w\s (converter emits
\u-escapes FromEBNF accepts, sidestepping hand-EBNF char-class limits). Precedence
over maxLength; convert failure falls through (never breaks tool-calling). Validated
standalone (^[A-Z]{2}[0-9]{4}$, \d{3}-\d{4}, [a-z]+@..., ^(cat|dog|fish)$, \w+,
^\s*$) + deployed RELEASE: ^[A-Z]{2}[0-9]{4}$ from junk prompt -> "AB1234" (matches,
det.); ^(red|green|blue)$ from "purple" -> "red"; int-range+enum+float+chat clean.
Backup mimirmind.pre-pattern. Structured-output coverage now: enum, type, integer
range, float upper-bound, maxLength, pattern — see reference/2026-09-17-structured-
output-two-enforcement-paths-profile-toggle.

## 2026-09-17 — number (float) upper-bound grammar enforcement (5.31)

**Deploy:** commit 40884ea. Extends the integer range to `type: number`: a decimal
fraction is unbounded so llama.cpp does NOT range-bound floats; we enforce the
meaningful direction (value <= max) exactly — integer-part range + top fraction
<= 0.frac(max) via a digit-wise numFracLeEbnf; lower side superset-safe (never
rejects a valid value). Base value_num capped 20 int + 15 frac digits. Validated
standalone ([0,1],[0,1.5],[0,0.99],[1,10.25],[0,1000] — fixed an
alternation-precedence bug where the optional fraction bound only to the last
integer-range alternative) + deployed RELEASE: temp maximum:2 -> temp=2.0 (model
asked for 5, det.); maximum:1.5 -> 0.93; integer-range/enum/chat clean. Backup
mimirmind.pre-floatrange.

## 2026-09-17 — integer min/max exact grammar range enforcement (5.31, llama.cpp-style)

**Deploy:** commit 7b2625b. Compile an integer `minimum`/`maximum` into an EXACT
digit-wise EBNF range (llama.cpp `build_min_max_int` approach: split [lo,hi] into
prefix-aligned subranges → per-subrange digit patterns) inside the schema-driven
value rule. xgrammar/our XML EBNF don't enforce numeric ranges for free (vLLM falls
back to the regex-FSM `outlines` backend for that); we do it in-grammar. Base
value_int also capped at 20 digits (any 64-bit value) so an UNbounded integer can't
run away to a 140-digit number. Validated standalone vs xgrammar ([0,20],[1,255],
[3,17],[10,9999]... exact, no leading zeros) + deployed RELEASE: count maximum:20 ->
count in [1,20] (10, det.); count no-maximum -> 20 digits (was 140); enum+typed clean.
Two enforcement paths documented (in-grammar vs FSM/outlines) with a planned
per-model-profile toggle: reference/2026-09-17-structured-output-two-enforcement-
paths-profile-toggle. Backup mimirmind.pre-intrange.

## 2026-09-17 — default frequency penalty 0 — fixes short/looping list answers (5.31)

**Deploy:** commit 6563b88 (branch; user pushes). Root cause of "answer always too
short/degenerate" (chat-trace 1ecfc47d, POST all earlier fixes): the default
frequency penalty (0.5, window 512) punishes RECURRING tokens, but a list/table
answer legitimately repeats structure ("- ","(","Roman",years) → qwen3.6 drifts off
the clean list format, truncates early, and even loops on its own ("...Ein Roman für
Erwachsene! Und Tiere! Und Kinder!..." x4). A/B (clean context, temp=0): freq 0.5 →
loop + 357 tok short; freq ~0 → full clean deterministic list (437 tok, 22 works, no
loop). qwen3.6 generation_config ships NO frequency_penalty; vLLM default is 0 —
mimirmind imposed an aggression the model never declared. Fix: frequencyPenaltyDefault
0.5→0 + antiLoopFrequency 0.5→0 (LlmConfig); exact-token repetition penalty 1.10 kept
as loop guard. Validated on deployed RELEASE with the NEW default (no client override):
complete clean deterministic list; plain chat clean. Backup mimirmind.pre-freqpenalty.

NOTE — this is the THIRD over-aggressive anti-loop mechanism found to degrade qwen3.6
output (after the answer-floor temp lift and, earlier, greedy-vs-sampling): the pattern
is that reactively-added loop-safety (temp lift + freq penalty) hurts quality more than
it helps; align with the model's generation_config + keep only the mild repetition
penalty. Also open: `count` still degenerates to a 140-DIGIT integer (type fix made it
digits-only but [0-9]+ has no length bound) — needs an integer digit-length cap
(mimirmind safety) and/or per-param bounds in Pegenaut's MCP tool config.

## 2026-09-16 — answer-floor temp-lift OFF by default (per-model config) — fixes tool-answer quality (5.31)

**Deploy:** commit 9993b3a (branch; user pushes). Root cause of bad tool-answers
(short / broken markdown / hallucinated `<result_list>`,`<ref_list>` tags /
non-deterministic, chat-trace 15c39a8a): the non-thinking anti-loop answer floor
lifted a greedy (temp=0) answer to min(generation_config.temp=1.0, cap 0.7) = temp
0.7 top_p 0.95 top_k 20. On qwen3.6 (3B-A) summarising a ~5 KB tool result, temp>0
garbles formatting. A/B (reconstructed round-1 x3): temp 0.7 AND 0.3 messy +
non-deterministic; **temp 0 + penalties = clean, deterministic (526 tok identical),
loop-free** (the rep+freq penalties run independently and already break the greedy
loop the lift targeted).

**Fix:** temp-lift cap moved from a hardcoded server constant to per-(machine,model)
config — LlmConfig.answerFloorTempCap default 0 = lift OFF (greedy stays greedy;
penalties carry loop-protection). Re-enable per checkpoint via HW-fingerprint model
overlay (ProbePicks.applyAnswerFloorTempCap key MIMIRMIND_ANSWER_FLOOR_TEMP_CAP ->
_config.answerFloorTempCap, per-model, no global env). MIMIRMIND_ANSWER_FLOOR_TEMP_CAP
still overrides per-request.

**Validation (deployed RELEASE):** round-1 tool-answer x3 IDENTICAL clean markdown;
"answer sampling floor" log gone; plain chat + tool-call (whitespace/type/schema
fixes) regressions clean. Backup mimirmind.pre-answerfloor. Rollback: env
MIMIRMIND_ANSWER_FLOOR_TEMP_CAP=0.7 (restore old lift) or restore backup binary.

## 2026-09-16 — tool-arg value rules now COMPILED FROM THE JSON SCHEMA (roadmap 5.31, vLLM-style)

**Deploy:** replaced the hardcoded type switch with a schema-driven generator
(`valueRuleFromSchema`) — the vLLM "feed the schema, type/enum come for free"
approach, adapted to the RAW (unquoted) Qwen-XML value: enum→exact literal set,
integer/number/boolean→typed base rule, string+maxLength→bounded `[^<]{0,N-1}`,
else→permissive whitespace-safe `value`. SchemaParams carries each property's full
subschema; each key's rule compiled once. Commit af615a6 (branch; user pushes).
Forced RELEASE recompile (rm root-owned .o in container), backed up prev binary to
mimirmind.pre-schemafix, prod_recreate (PROD_OK 135s).

**Validation (deployed RELEASE):**
- ENUM: convert_temp(unit: enum[celsius,fahrenheit,kelvin]) x3 → unit is EXACTLY an
  enum literal (UNIT_IN_ENUM 3/3), value a real int, no invented keys. An enum
  value now cannot ramble at all (grammar-enforced from the schema).
- websearch regression: count integer-or-absent, query leads with content+author,
  no non-schema keys; plain chat coherent.

**Why not literal xgrammar FromJSONSchema:** vLLM emits JSON tool calls and hands the
schema to xgrammar's JSON compiler; qwen3.6 uses the Qwen-Coder XML format whose wire
value has no JSON quotes, so a JSON-string grammar (quoted) wouldn't match — we read
the schema keywords ourselves. Also xgrammar's own JSON-Schema path does NOT enforce
maxLength/pattern/numeric-range (documented gap), so enum+type are the reliable wins;
unbounded-string rambling is model quality, not grammar (same as vLLM). Rollback:
restore mimirmind.pre-schemafix + prod_recreate, or MIMIRMIND_TOOL_GRAMMAR=0.

## 2026-09-16 — tool-arg integer/number/boolean type-constraint LIVE (roadmap 5.31, follow-up)

**Deploy:** follow-up to the whitespace fix. `ToolCallConstraint.cpp` used one
string `value` rule for ALL params → an integer param (`count`) could decode into
a 150-digit string / text ramble; the model also invented non-schema params
(`language`,`read_top` in chat-trace 38a5af75). Fix: JSON-Schema-type-aware value
rules — integer `"-"? [0-9]+`, number `"-"? [0-9]+ ("." [0-9]+)?`, boolean
`"true"|"false"`; string/array/object keep the permissive rule. Commit 0078262
(branch fix/tool-arg-degeneration-salvage-resample; user pushes). Forced RELEASE
recompile (rm the root-owned .o in the builder container), `VALUE_INT_IN_BINARY`
verified via strings, prod-binary backed up to mimirmind.pre-countfix, deployed
via prod_recreate (PROD_OK 155s).

**Validation (deployed RELEASE, Walter-Moers x5, tool_choice=required):**
- `count` integer-or-absent 5/5 (never a degenerate string).
- `extra_nonschema_keys=none` 5/5 — invented params gone (grammar strictly
  enumerates schema keys).
- query still leads with content + author name; weather-tool + plain-chat clean.

**Regression watch:** string params can still RAMBLE (877-char query; a weather
`city` full of think-aloud reasoning) — model output quality, not
grammar-expressible. Rollback: restore mimirmind.pre-countfix + prod_recreate, or
`MIMIRMIND_TOOL_GRAMMAR=0`.

## 2026-09-16 — tool-arg whitespace-value grammar fix LIVE (roadmap 5.31)

**Deploy:** `ToolCallConstraint.cpp` value rule `value ::= [^<]+` →
`value ::= [!-;=-~] [^<]*` (leading char must be printable-ASCII content; kills
whitespace-only / newline-leading required args). Commit 399f8b4 (branch
fix/tool-arg-degeneration-salvage-resample; user pushes). Root cause found live
via cuda-gdb on the serve process (chat-trace 2bd53443, "Liste die Werke von
Walter Moers"): the old rule forbade only the EMPTY value, so the 3B-active model
on a forced (tool_choice=required) OOD tool call shipped `{"query":"\n"}` ~40-60%
of the time — even through the grammar-masked salvage. Forced RELEASE recompile
(deleted the root-owned .o inside the builder container — a plain `cmake --build`
had relinked the stale restored .o), `FIX_IN_BINARY`/`OLD_GONE` verified via
strings; deployed via prod_recreate (preflight-gated, PROD_OK 140s).

**Validation (deployed RELEASE binary, Walter-Moers x5, tool_choice=required):**
- 5/5 queries WHITESPACE_ONLY=False, 5/5 contain the author name (was ~40-60%
  dead pre-fix). Plain-chat coherence intact; no regression.
- cuda-gdb harness on the debug binary earlier showed 2/3 need NO salvage at all
  (first-pass query now leads with content) + a tool_choice=auto control where
  the model answers directly (no forced search) — the OOD forcing was the trigger.

**Regression watch:** integer params (e.g. `count`) still share the string value
rule → can degenerate to a long digit/ramble string (follow-up: type-aware value
rule). Rollback: `MIMIRMIND_TOOL_GRAMMAR=0` (disables the whole tool grammar).
Evidence: Synaipse bugs/2026-09-16-toolarg-degeneration-grammar-value-rule-
debugger-confirmed + solutions/2026-09-16-toolarg-whitespace-value-grammar-fix-
validated.

## 2026-09-16 — qwen4_exp multi-chat serving bounded GATE (roadmap 5.27.14)

**Context:** qwen4_exp (Qwen3.8-Flash-Next NVFP4, arch qwen4_exp: Hyper-Connections
+ PLE n-gram + MoE-512, blocks=48 d_model=2560) batched serving forward landed
(5.27.11, HC+PLE per-slot correct). This is the post-diag-cleanup validation:
the MIMIRMIND_Q4E_DIAG/q4ediag/dumpNorm surface was removed (4 files, 126 LOC),
clean CUDA rebuild PASS (0 errors), binary boots + loads (SERVE_UP ~370s).

**Correctness (all PASS on the cleanup-rebuilt binary, serving-class + batcher on,
GROUPED_MOE=3, config.serve.qwen4exp.json maxActive=64 ctx=8192):**
- Needle-in-haystack retrieval: PASS (extracted the planted code from ~3k-tok filler).
- 8 concurrent (6 distinct + 2 dup): all coherent, SERVE_ALIVE, DUP_IDENTICAL
  (slot-consistency in real serving).
- Tool-calling (Qwen3-Coder XML path): auto→tool_call, required→tool_call,
  round-trip→NL answer, no-tool control→no tool_call. Fully functional.

**Perf (conc32 anchor, 32×96 tok, temp=0, one-GPU swap; qwen3.6 = restored prod):**
- qwen4_exp: TTFT 501 ms, 32/32 OK in 213.9 s → **14.4 tok/s aggregate**
- qwen3.6:   TTFT 205 ms, 32/32 OK in  21.5 s → **142.7 tok/s aggregate**
- → qwen4_exp **~9.9× slower aggregate throughput, ~2.4× worse TTFT** at conc32.

**Read:** correctness is production-grade; throughput is NOT serving-viable as-is.
The ~10× gap is far beyond the "expected slower" (x4-stream HC activations) — prime
suspect is the **PLE host-gather** (16 rows/token gathered from the ~44 GiB NVMe
host-mmap table, per slot, per decode step) serialising every step under concurrency.
qwen4_exp stays an experimental quality-ceiling bet, NOT a throughput serving target;
README/doc/roadmap serving mirrors intentionally NOT updated. Unlocking it needs the
PLE gather optimised (batched device-side gather / resident table) + decode levers
(roadmap 5.27.9 I-8). Prod qwen3.6 restored clean (32/32, 142.7 tok/s post-restore).

**Bounded gate — deferred (user choice):** long soak (bounded-mem, 0-err), conc64
anchor, deep quality A/B vs qwen3.6. Diag surface removed + gate committed here;
those remain as a follow-up on-box run. Evidence: /opt/mimirmind/qwen4exp_gate_527_14.log.

## 2026-09-15 — HW profile re-key (fingerprint drift) restores cuDNN-paged prefill

**Deploy:** the 5.19 Layer-2 perf profile had silently stopped loading — the HW
fingerprint drifted 78f59405170233c2 -> a68b756bff8519e8 (CUDA 13.3 update), so
`loadProbePicks` found no match ("per-HW profile not deployed") and prod ran with
NONE of the profile flags. Re-keyed configs/hw/a68b756bff8519e8 (copy + fingerprint
field) + restored MOE_DECODE_REG value 1->4 (commit 3ebbb2e). Graceful
`docker restart -t 60 mimirmind-serve`; profile now loads (cuDNN SDPA + paged
prefill, F32-TC, cuBLAS-FP8, MMQ+TC, MoE silu-fuse, MoE decode-reg mode 4), serve
healthy + coherent.

**Win (isolated A/B, dedicated container, ~4184-tok ctx, cache-busted, only
ATTN_CUDNN_PAGED toggled, steady reps):**
- ATTN_CUDNN_PAGED=0: ~4.96 s prefill TTFT
- ATTN_CUDNN_PAGED=1: ~2.00 s prefill TTFT  → **-60% (2.48x)** at 4.2k ctx
(documented -42% @3438 tok; the win grows with context length). Prod had silently
lost this ~2.5x prefill win since the CUDA-13.3 drift.

**Full-power vLLM prefill A/B (after mains power-cycle, both ~40W, ~4.2k ctx,
cooldown-spaced):** mimirmind (cuDNN-paged) ~1.90s vs vLLM ~0.527s = **3.6× prefill
gap** — down from the ~6× on 2026-09-14 (that run had cuDNN-paged silently OFF via
the profile drift). The re-key ~halved the prefill gap. Full power itself barely
moved mimir prefill (2.0→1.9s @4.2k); the residual 3.6× is structural kernel
efficiency (5.18.1). DECODE is bandwidth-bound (~14-19 tok/s, power-independent:
same at 16W and 40W) — the ~13-vs-35 "regression" was a power/methodology artefact,
NOT code (roadmap 5.18.20b resolved).

**Regression watch:** LESSON — after any CUDA/driver update, re-check the HW
fingerprint; the per-HW profile silently stops loading and prod runs degraded.
Also: the DGX Spark clamps power under SUSTAINED load (bursts ~40W ok, continuous
decode degrades) — space perf probes with cooldown.
Rollback: remove configs/hw/a68b756bff8519e8 + restart (reverts to no-profile).
Evidence: Synaipse `research/2026-09-15-single-user-decode-14-vs-35-and-hw-profile-not-loading`.
NOTE: single-user M=1 decode ~2.5x drop (13 vs 35) is SEPARATE, unfixed (5.18.20b bisect).

## 2026-09-14 — Full A/B vs vLLM (anchor datapoint, not a deploy)

**Setup:** same nvfp4 ckpt, GB10 @2405 MHz (0.80× throttle, equal both sides),
one-GPU swap. mimirmind = build-cudnn + prod flags, single-model config.ab.json
(ctx 8192, maxActive 32). vLLM = nvcr 26.06, `--max-model-len 8192 --gpu-mem-util
0.9 --max-num-seqs 64`. Runner `/opt/mimirmind/ab_full_vs_vllm.sh`.

**Speed (mimir → vLLM, factor):**
- single-user decode 512-gen: ~14 tok/s (bimodal 22/14/13) → 75.6 tok/s  (~3.4–5.3×)
- single-user prefill-TTFT ~2.3k ctx: ~2.2–2.4 s → 0.37 s  (~6×)
- conc16 agg / TTFT: 30.2 / 25.1 s  vs  257.6 / 3.17 s  (8.5× / 7.9×)
- conc32 agg / TTFT: 36.5 / 48.0 s  vs  297.0 / 6.31 s  (8.1× / 7.6×)
- conc64: mimir 46.6 tok/s, 64/64 OK, TTFT 92.6 s; **vLLM CRASHED 0/64** (timeout/
  conn-reset) — mimir admission holds where vLLM defaults fall over.

**Deviations found:** (1) prefill scheduling still ~linear TTFT despite MIXED_STEP
default-on → #1 lever (roadmap 5.21.13/5.21.14); (2) decode below historical (~14 vs
~35) — suspected anti-loop sampling-floor tax (5.18.17); (3) BUG: enable_thinking:true
garbled @temp=0 (5.30). Output: mimir think=0 = cleanest/correct; vLLM leaks the
"Here's a thinking process:" preamble (Fall A model-behaviour, confirmed on-box).

**Mixed-step confirm (5.21.13):** the A/B ran mixed-step OFF (default-off since the
08-24 multi-model illegal-access revert; env-only). Dedicated single-model re-run
with MIXED_STEP=1+MIXED_BATCH=1: agg +35–48% (30.2→44.4 / 36.5→53.9 / 46.6→62.9),
decode_est +30–68%, but **TTFT stays ~linear** (24.7/45.2/87.1 s) → scheduling is a
throughput lever, NOT a TTFT fix. TTFT gap = prefill compute throughput (912 vs
vLLM ~6095 tok/s, ~6.7×) → roadmap 5.18.18 (masked MoE prefill). Mixed-step prod
enablement blocked by the multi-model illegal-access → roadmap 5.21.14.

Evidence: Synaipse `research/2026-09-14-full-ab-vllm-vs-mimirmind-path-to-parity`.

## 2026-09-13 — GDN prefix-cache DEFAULT-ON (roadmap 5.28.1)

**Deploy:** ContinuousBatcher warm-slot GDN prefix reuse flipped to server-side
default-ON for SSM/GatedDeltaNet backends (opt-out `MIMIRMIND_GDN_PREFIX_CKPT=0`,
instant, no rebuild). Includes recurrent-only checkpoint trim (per-checkpoint
image 83→63 MiB dense-pack). Binary: build-cudnn @ 2026-09-13 16:18.

**Win (TTFT on shared-prefix continuation / RAG / multi-turn):**
- Serial RAG, shared system+context prefix, distinct query:
  - 787 prompt_tok:  667 → 326 ms  (−51%)
  - 1537:           1478 → 650 ms  (−56%)
  - 3037:           4005 → 978 ms  (−76%)
  - 5312:          10117 → 767 ms  (−92%)
- Concurrent RAG (2048-tok prefix): 4×4 −76%, 8×3 −75% (hit-rate holds under batching).
- Multi-turn chat (8×4): 9550 → 3377 ms (−65%), 24/24 turns hit.
- Live post-cutover probe: cold 1.075 s → warm 0.277 s (HIT resume@512, LCP 697).

**Correctness:** contamination gate identical (OTTER×3) OFF & ON; 0 HTTP errors
across 21k (soak) + 5k (re-soak) + bench requests; coherent.

**Memory:** 3h soak (fat) bounded — MemAvailable converged to a flat plateau
(~33.5 GiB free under worst-case constant churn of 12 distinct 1.5k prefixes),
0 leak; post-stop preflight 118 GiB clean reclaim. 45-min re-soak (trimmed)
bounded, 0 errors. Ring cap = 8 ckpts/slot × ~63 MiB (recurrent-only).

**Regression watch:** none. Pure-attention backends unaffected (gated on
backendNeedsSsmScratch). Rollback = `MIMIRMIND_GDN_PREFIX_CKPT=0` + recreate.

Evidence: Synaipse `research/gdn-prefix-cache-hitrate-bench-2026-09-13`,
`research/gdn-prefix-cache-soak-2026-09-13`. Branch feat/5.28.1-gdn-prefix-cache
(needs commit+push: ServingSession.cpp, ContinuousBatcher.{cpp,hpp}).

---

## 2026-09-17 — Thinking-mode sampling: derail fix (per-model thinking preset from profile)

**Symptom (quality, not perf):** `enable_thinking=true` on qwen3.6 (35B-A3B
NVFP4, GB10) DERAILED — a long reasoning chain drifted off-topic mid-thought
(measured: a transitivity logic prompt produced unrelated "Frank Zappa songs"
garbage; a photosynthesis prompt emitted 2800-char reasoning then an EMPTY
answer via finish=length). Non-thinking (server default) was fine and on par
with the vLLM oracle.

**Root cause:** the server thinking floor lifted a greedy request to the model's
`samplingTempDefault` — sourced from `generation_config.json`, which for qwen3.6
ships a GENERIC `do_sample` `temperature=1.0`. The Qwen3 model card's REASONING
preset is temp=0.6 (top_p 0.95, top_k 20), NOT 1.0. temp=1.0 over a 500+ token
chain on a 3B-ACTIVE MoE drifts and derails. (The vLLM "coherent thinking" run
was coherent only because it ran greedy temp=0, not because vLLM samples smarter.)

**Fix (ADR-aligned, profile-driven — no hardcoded per-arch const):** new
`LlmConfig.thinkingTemp/thinkingTopP/thinkingTopK`, populated from the per-model
HW-fingerprint overlay (`MIMIRMIND_THINKING_TEMP/_TOP_P/_TOP_K`) via
ProbeConfig/InferenceEngine. The thinking floor now prefers the thinking preset
BEFORE generation_config BEFORE the generic escape values. qwen3.6 overlay pins
0.6 / 0.95 / 20. Files: LlmConfig.hpp, ProbeConfig.{hpp,cpp}, InferenceEngine.cpp,
ChatCompletionHandler.cpp, configs/hw/{a68b756bff8519e8,78f59405170233c2}/
model-overlays/qwen3.6.json. Ops override: env MIMIRMIND_THINKING_TEMP etc.

**Deploy gotcha:** the live GB10 fingerprint is now **a68b756bff8519e8** (was
78f59405170233c2 pre-CUDA-13.3; the fp folds toolkit/driver version). Overlay
placed under BOTH; only a68b is live. Verified in logs: "per-model overlay FOUND
for 'qwen3.6' — merged" + "profile applied: thinking sampling -> temp=0.6
top_p=0.95 top_k=20".

**Validation (deployed Release, temp=0 request → floor lifts to 0.6):**
- Logic (transitivity): derail GONE — correct answer ("Anna>Ben>Carla").
- Reasoning (train time): "3 Stunden" correct, 46s→21s.
- Photosynthesis / 5 languages: coherent & COMPLETE at max_tokens=2000
  (finish=stop); the earlier empty answers at max_tokens=700 were a token-budget
  artifact of the test (reasoning ate the budget), not the sampling fix.

**Note:** thinking stays server-default-OFF (correct for simple Q&A: think-off
6-24s vs think-on 21-128s and needs answer-budget headroom). This fix only makes
opt-in thinking COHERENT instead of derailing. No perf regression on the default
non-thinking path (guarded by `cr.enableThinking`).

---

## 2026-09-17 — Honesty floor: default OFF, per-model overlay (quality, not perf)

**Change:** the server honesty floor (8.19.11 — prepend a "hedge, don't
confabulate" system prompt when a request ships NO system and NO tools) now
defaults **OFF** and is a per-(machine,model) overlay toggle
(`LlmConfig.honestyFloor`, overlay flag `MIMIRMIND_HONESTY_FLOOR`), not an
env-default-ON server constant.

**Why:** oracle A/B (2026-09-17, qwen3.6 vs vLLM, plain chat) — the floor made
the model REFUSE long-tail knowledge questions ("Liste die Werke von Walter
Moers") that the un-floored vLLM answers at par. Over-refusal is worse than the
confabulation it was meant to curb; grounded/agentic flows carry a system prompt
or tools and never hit the floor anyway.

**Files:** LlmConfig.hpp, ProbeConfig.{hpp,cpp}, InferenceEngine.cpp,
ChatCompletionHandler.cpp. Commit <pending>. Ops override: env
MIMIRMIND_HONESTY_FLOOR=1 re-enables per-request; an overlay can re-enable it for
a checkpoint that over-confabulates.

**Validation (deployed Release, no system + no tools, temp=0):**
- Walter Moers: was REFUSE → now ATTEMPTS (frames correctly, lists; confabulates
  specific titles exactly as vLLM does — model ceiling, needs the search tool).
- Photosynthesis / capital-of-Australia: unchanged, correct → no regression.
- Long-tail (Liechtenstein heads of gov): attempts + self-hedges ("Fürst vs
  Ministerpräsident is a separate office") though confabulates specifics.

**Regression watch:** grounded RAG / agentic / parity / needle flows unaffected
(all carry a system prompt or tools → floor never applied, byte-identical). The
tradeoff (more confabulation on ungrounded long-tail facts) is accepted and
matches the vLLM oracle; the real remedy is the (now-fixed) search tool path.

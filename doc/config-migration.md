# Config migration

Schema is additive — every new field lands with a safe default and
older `config.json` files keep working unchanged. This note collects
the deltas since the last widely-deployed config so operators can
adopt new capabilities intentionally rather than by accident.

## `models[].backend` — per-model backend selection

Landed 2026-07-19 (branch `feat/hip-backend-skeleton` after backend-
pool merge).

### What

New optional string field on every `models[]` entry. Pins the
`InferenceEngine` for that model to a specific device from
`BackendPool::discoverAll()`. Recognised tokens:

| Token          | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `""` / unset   | delegate to `BackendPool::select(Auto)` — first available GPU |
| `"auto"`       | same as unset — explicit for clarity                          |
| `"cpu"`        | force the reference CPU backend                               |
| `"l0"` / `"l0:0"` | LevelZero device (Intel iGPU) — index defaults to 0        |
| `"hip"` / `"hip:0"` | HIP device (AMD dGPU)                                    |

Unknown tokens fail loud at boot with the list of available pool
entries printed to `stderr`.

### Why

Before this field, `MIMIRMIND_BACKEND` env-var picked ONE backend for
the whole process. On a dual-GPU host (iGPU + Radeon) the second card
was invisible to the runtime.

Per-model selection makes speculative decoding across two backends
practical — target on the fat dGPU, draft on the iGPU, one worker
process:

```json
"models": [
  { "id": "primary", "path": "/models/gemma-4-e4b.gguf",   "backend": "hip:0" },
  { "id": "draft",   "path": "/models/gemma-2-2b-q4km.gguf", "backend": "l0:0" }
]
```

### Rollout

- Existing configs: no change required. The default (unset) preserves
  `autoSelect` behaviour — an L0-only build picks L0, a HIP-only build
  picks HIP, a dual build picks the first entry (L0 by enum order).
- Dual-GPU deployments: set `backend` explicitly on each model. Verify
  with `curl http://<worker>/v1/system/info | jq '.backend_pool'` —
  every entry the process sees is listed there, and
  `.engine_backend.token` on `/v1/system/status` per-model reflects
  the actual binding.

### Config-schema check

`ServeMode` runs the pool lookup with the exact token before
`loadModel` runs. So a typo (`"hip:11"` on a single-device box)
surfaces before the model bytes are read, not mid-generation.

## `speculative.drafter` + n-gram knobs

Landed earlier on `feat/ngram-spec-decode` (separate branch, may or
may not be part of the same PR train — check the merge base).

### What

`speculative` block gets three new optional fields:

```json
"speculative": {
  "enabled":   false,
  "drafter":   "model",     // NEW — "model" | "ngram"
  "target":    "primary",
  "draft":     "draft",     // still required when drafter == "model"
  "n":         4,
  "ngramMinK": 2,           // NEW — PLD needle-length lower bound
  "ngramMaxK": 3            // NEW — PLD needle-length upper bound
}
```

- `drafter: "model"` (default) preserves the pre-existing behaviour —
  requires `draft` to resolve against a `models[]` entry with a vocab-
  compatible tokenizer.
- `drafter: "ngram"` engages Prompt-Lookup Decoding — draft tokens
  come from an in-context n-gram scan over the committed history. No
  second model, no vocab check, ~zero draft cost per round. Best on
  RAG-style prompts with citation repetition.

### Why

Model-draft spec-dec has been in the tree since M9.11.4 but never
enabled in prod — the accept-rate baseline (`todos/m9-11-accept-rate-
baseline.md`) is still open. N-gram spec removes the baseline blocker
because it needs no second model and no vocab check.

### Rollout

- Off by default. Existing configs keep working.
- To try n-gram spec on a repetitive workload:

  ```json
  "speculative": {
    "enabled": true,
    "drafter": "ngram",
    "target":  "primary",
    "n":       4
  }
  ```

- `spec_rate` shows up in the `/v1/chat/completions` server logs per
  request. If it stays below ~0.3 on prod traffic, disable — the
  target-batched-verify overhead exceeds the draft savings there.

## What's NOT changed

- The top-level `runtime`, `features`, `governor`, `diagnostics`
  sections are unchanged in this window. Everything you already have
  in `config.json` keeps working.
- No fields have been removed or renamed. Every migration in this
  document is opt-in.

## Verification helper

After deploy, `/v1/system/info` shows the resolved shape:

```bash
curl -s http://<worker>:8080/v1/system/info | jq '{
  backend_pool,
  engine_backend,
  speculative_decoding
}'
```

- `backend_pool` — every entry `BackendPool::discoverAll` found.
  Empty means no backends compiled in and available.
- `engine_backend` — which of those the primary engine bound to.
- `speculative_decoding` — `status`, `drafter`, and per-drafter fields
  (`draft_model_arch` for model, `ngram_min_k`/`ngram_max_k` for
  n-gram).

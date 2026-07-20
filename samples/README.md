# `/samples` — reproducible bench prompts

Deterministic prompt library that feeds:

- `scripts/bench-vs-baseline.sh` — A/B server bench across the full
  matrix.
- `scripts/bench-length-sweep.sh` — single-endpoint length sweep for
  prefill-scaling analysis.
- `tests/prefill_bench.cpp` via `--prompt-file` — attention-kernel
  micro-bench driven by real prompt sizes.

## Factor matrix

| Axis            | Values                                                                 |
|-----------------|------------------------------------------------------------------------|
| **length**      | `120`, `250`, `512`, `1k`, `2k`, `4k`, `8k`, `16k`, `32k` (target tokens) |
| **shape**       | `instruct`, `rag`, `needle`, `code`, `summarize`, `chat-multiturn`     |
| **language**    | `en`, `de` (EN ≈ 4.0 chars/tok, DE ≈ 3.0 chars/tok — Llama/Gemma BPE)   |
| **decode**      | per-shape `max_new_tokens` list — see `manifest.json` → `decode_targets` |

Sparse on purpose: a 32k-token instruction prompt makes no sense; a
120-token needle-in-haystack has nowhere to hide the needle. See
`SHAPE_LENGTHS` in `generate.py` for the exact allowed combinations.

## Layout

```
samples/
├── README.md              ← this file
├── generate.py            ← deterministic generator (stdlib only)
├── manifest.json          ← machine-readable index of every file
├── prompts/
│   ├── en/{shape}-{bucket}.{txt,json}
│   └── de/{shape}-{bucket}.{txt,json}
└── chats/
    ├── en/chat-multiturn-{bucket}.jsonl
    └── de/chat-multiturn-{bucket}.jsonl
```

- `.txt` — raw prompt, single user turn.
- `.json` — needle files. `{ "prompt": "...", "needle": "CODE-XXXX",
  "position_pct": 0.42 }`. Recall check: response must contain the
  `needle` string verbatim.
- `.jsonl` — chat files. One JSON message per line
  (`{"role": ..., "content": ...}`); pass the whole array to
  `/v1/chat/completions.messages`.

## Manifest schema (`manifest.json`)

```jsonc
{
  "version": 1,
  "seed": 20260720,
  "chars_per_token": {"en": 4.0, "de": 3.0},
  "shapes":          ["chat-multiturn", "code", "instruct", "needle", "rag", "summarize"],
  "lengths":         ["120", "250", "512", "1k", "2k", "4k", "8k", "16k", "32k"],
  "languages":       ["en", "de"],
  "decode_targets":  { "instruct": [32, 128], "rag": [128, 512], ... },
  "entries": [
    {
      "id":             "en/rag-4k",
      "path":           "prompts/en/rag-4k.txt",
      "shape":          "rag",
      "lang":           "en",
      "length_bucket":  "4k",
      "token_target":   4096,
      "char_count":     16478,
      "decode_targets": [128, 512],
      "type":           "prompt"
    },
    { "id": "en/needle-4k", "path": "prompts/en/needle-4k.json",
      "type": "needle", "needle_field": "needle", "prompt_field": "prompt",
      "position_pct": 0.42, ... }
  ]
}
```

## Regeneration

```
python3 samples/generate.py
```

Same seed → identical bytes. Content is procedural fake-facts
(templated regions, products, agencies) — no real-world knowledge, so
a memorised passage can never boost a decode. Adjust `LENGTHS`,
`SHAPE_LENGTHS`, or `DECODE_TARGETS` at the top of `generate.py` and
re-run.

## Consuming the samples

### Full A/B matrix

```
scripts/bench-vs-baseline.sh \
    http://host-candidate:8080 http://host-baseline:8080 \
    --shapes rag,needle,code --lengths 512,1k,2k,4k --shots 3
```

Emits a `shape × lang × bucket × decode` table with median wall time,
server-side `prefill_ms` / `decode_ms` (when the endpoint reports
them), a `+/-%` delta column, and needle-recall for needle rows.
Trailing JSON blob is ledger-paste-ready.

### Prefill length sweep (single endpoint)

```
scripts/bench-length-sweep.sh http://host:8080 \
    --shape rag --lang en --decode 32 --shots 3
```

`--decode 32` keeps decode work roughly constant across lengths, so
the wall slope reads as prefill scaling. Swap `--shape` for
`needle`/`summarize`/`chat-multiturn` to profile different scan
patterns.

### Kernel-level with `prefill_bench`

```
docker compose run --rm mimirmind \
    /usr/local/bin/prefill_bench \
        --prompt-file /src/samples/prompts/en/rag-16k.txt \
        --cpt 4.0 --tq 32 --kvdtype q8_0
```

`--prompt-file` reads the file (auto-extracts `.prompt` from needle
JSON), derives `T_k = ceil(bytes / --cpt)`, and appends the file
basename to the ledger label so runs across lengths stay
distinguishable.

## Token-count caveats

- Length targets are **approximate**. The generator sizes files by
  chars/token; the real BPE tokenizer will be off by a few percent.
  Empirically the derived count lands within ±5 % of the target for
  both EN and DE across the whole bucket range.
- The Gemma tokenizer treats DE compounds more coarsely than English,
  hence the tighter `3.0` chars/tok ratio. If you retune, adjust
  `CHARS_PER_TOKEN` in `generate.py` and regenerate.
- For byte-exact reproducibility across sessions, don't hand-edit the
  files — always regenerate from `generate.py`.
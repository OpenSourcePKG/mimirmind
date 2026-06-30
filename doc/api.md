# HTTP API

OpenAI-compatible Chat Completions, plus a tiny health endpoint and a
model-discovery endpoint. The server listens on port 8080 by default;
override via `MIMIRMIND_PORT` in the environment.

## `GET /health`

Liveness probe. Returns `200 OK` with a small JSON body once the model
is fully loaded into USM.

```bash
curl -s http://localhost:8080/health
# -> {"status":"ok"}
```

Before the model finishes loading the server is not bound on the port
yet, so this is a meaningful readiness check: if `/health` returns
`200`, the engine is warm.

## `GET /v1/models`

Lists the single model that is loaded into this process. Mimirmind is
single-model per process by design; the response shape mirrors the
OpenAI API so existing client libraries don't choke.

```bash
curl -s http://localhost:8080/v1/models | jq
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "google_gemma-4-26B-A4B-it-Q6_K",
      "object": "model",
      "created": 1782774169,
      "owned_by": "mimirmind"
    }
  ]
}
```

The `id` is derived from the GGUF filename (stem, no extension). Use
that as the `model` field in chat-completion requests when you want to
echo it back in the response; the server does not verify it against
the loaded model.

## `POST /v1/chat/completions` — non-streaming

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise C++ engineer."},
      {"role": "user", "content": "Why is std::vector<bool> special?"}
    ],
    "temperature": 0.0,
    "max_tokens": 200
  }' | jq
```

Response:

```json
{
  "id": "chatcmpl-vm78Pba4mDbGVF1mGK99HUYf",
  "object": "chat.completion",
  "created": 1782774169,
  "model": "google_gemma-4-26B-A4B-it-Q6_K",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "std::vector<bool> is special because ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens":     34,
    "completion_tokens": 142,
    "total_tokens":      176
  }
}
```

### Supported request fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `messages` | array of `{role, content}` | (required) | `role ∈ {system, user, assistant}` |
| `model` | string | server default | Echoed back; not validated |
| `temperature` | float | 0.0 | 0 = greedy |
| `top_p` | float | 1.0 | Nucleus sampling cutoff |
| `top_k` | int | 0 | 0 = disabled |
| `seed` | int | 0 | 0 = non-deterministic |
| `max_tokens` | int | 512 | Hard cap on generated tokens |
| `max_completion_tokens` | int | (alias for `max_tokens`) | OpenAI's current name |
| `stop` | string \| string[] | none | Decoder stops if any stop string appears |
| `stream` | bool | false | Switches to SSE — see below |

### `finish_reason` values

- `"stop"` — model hit an architecture-specific stop token
  (`<|im_end|>` for Qwen, `<end_of_turn>` for Gemma 2/3, `<turn|>`
  for Gemma 4) or a user-provided `stop` string.
- `"length"` — `max_tokens` was reached.

## `POST /v1/chat/completions` — streaming (SSE)

Add `"stream": true` and read the response with `curl -N`:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a one-line fun fact"}],
    "stream": true,
    "max_tokens": 80
  }'
```

Each emitted event is a single Server-Sent Event chunk:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1782774169,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1782774169,"model":"...","choices":[{"index":0,"delta":{"content":"Honey"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1782774169,"model":"...","choices":[{"index":0,"delta":{"content":" never"}}]}

...

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1782774169,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Running behind a reverse proxy

SSE breaks easily behind a reverse proxy with default settings. The
two failure modes are:

- **Buffering**: the proxy collects 30-50 tokens worth of chunks
  before flushing, killing the live-stream experience.
- **HTTP/1.0 fallback**: the proxy speaks HTTP/1.0 to the upstream,
  which has no chunked transfer encoding, so the proxy buffers the
  full response in memory and the client sees nothing until the
  generation is complete.

The server sets `X-Accel-Buffering: no` and `Cache-Control: no-cache`
on every SSE response, which nginx and other proxies honour — but a
handful of explicit knobs are still required for things to work
end-to-end:

```nginx
upstream mimirmind {
    server 127.0.0.1:8080;
    keepalive 16;
}

server {
    listen 443 ssl http2;
    server_name mimirmind.example.com;

    # ... ssl_certificate / ssl_certificate_key ...

    # Talk HTTP/1.1 to the upstream so chunked transfer works.
    # Default is 1.0 — without this the proxy buffers the whole body.
    proxy_http_version 1.1;
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    location /v1/chat/completions {
        proxy_pass http://mimirmind;

        # Buffering off — the actual SSE-critical knob.
        proxy_buffering           off;
        proxy_request_buffering   off;   # forward POST body immediately
        proxy_cache               off;

        # Empty Connection header lets nginx do upstream keep-alive.
        proxy_set_header Connection "";

        # Long generations stall the stream — extend timeouts.
        proxy_read_timeout        3600s;
        proxy_send_timeout        3600s;
        proxy_connect_timeout     30s;

        chunked_transfer_encoding on;
    }

    location / {
        proxy_pass http://mimirmind;
        proxy_read_timeout 60s;
    }
}
```

#### Common pitfalls

1. **`proxy_http_version 1.1` omitted.** Default is HTTP/1.0. No
   chunked encoding → full-response buffering → tokens arrive in one
   delayed burst at the end. Easy to spot: time-to-first-byte ≈
   total-generation-time.
2. **`proxy_buffering` left on.** Even with HTTP/1.1, the default
   4-8 KiB output buffers coalesce chunks. Symptom: tokens arrive in
   visible bursts of ~30-50 instead of streaming one at a time.
3. **`proxy_read_timeout` too low.** Default 60 s. Long prompts +
   slow generations → 504 mid-stream. Bump it to 3600 s.

#### Cloudflare in front

Cloudflare's default proxying buffers SSE responses and strips
`X-Accel-Buffering`. Workarounds:

- Set `Cache-Control: no-transform` on the response (CF respects
  this) — the server does *not* currently set it; either add an
  `add_header` directive in nginx or proxy through a `cloudflared`
  tunnel with the gRPC/SSE-aware mode enabled.
- Or terminate Cloudflare with a "DNS only" record for the API
  hostname and let nginx do everything itself.

## Error responses

Error responses follow the OpenAI shape:

```json
{
  "error": {
    "message": "messages must not be empty",
    "type":    "invalid_request_error",
    "code":    null
  }
}
```

Common HTTP status codes:

- `400` — Invalid request body, missing required field, unsupported
  role, model not loaded, chat template unsupported for the model's
  architecture.
- `500` — Engine threw mid-generation (rare; check
  `MIMIRMIND_LOG_FILE` for the stack).
- `503` — Server is in shutdown.

## Architecture-specific notes

### Qwen 2 / 2.5 / 3

Standard ChatML template (`<|im_start|>` / `<|im_end|>`). When no
`system` message is provided, the engine injects Qwen's default system
message ("You are Qwen, created by Alibaba Cloud. You are a helpful
assistant.") to match the HF Jinja template behaviour. Override by
passing your own `system` message as the first turn.

### Gemma 2 / 3

Symmetric `<start_of_turn>` / `<end_of_turn>` markers. Roles map
`user → user`, `assistant → model`. System messages are folded into
the first user turn separated by a blank line, because Gemma was not
trained with a separate system role.

### Gemma 4

Asymmetric `<|turn>` / `<turn|>` markers. Roles same as Gemma 2/3.
Plus a thinking-channel wrapper that the model emits at the start of
every response: `<|channel>thought\n<channel|>`. With thinking mode
off the channel is empty.

The wrapper is stripped on both paths so OpenAI-compatible clients
never see it:

- **Non-streaming**: `ChatTemplate::cleanResponse` runs over the
  finished string and removes the wrapper.
- **Streaming**: `ResponseCleaner` is a stateful per-token filter in
  the SSE callback that swallows the wrapper tokens and the trailing
  whitespace before any `delta.content` chunk is emitted.

Both paths produce the same visible content.

## Client compatibility

The OpenAI Python SDK works out of the box with `base_url` pointing
at the server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",   # required by the SDK, not checked by mimirmind
)

resp = client.chat.completions.create(
    model="google_gemma-4-26B-A4B-it-Q6_K",
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
```

LangChain `ChatOpenAI`, the LlamaIndex `OpenAI` adapter, and any
shell-script using `curl` work the same way.
)
## Prefix-cache (M9.1)

The engine keeps the KV state of its last request around between
calls. When the next request's prompt starts with the same token
sequence, those leading tokens skip prefill — only the new suffix
runs through the transformer.

In a typical chat, the OpenAI request format means every follow-up
turn sends the *entire* conversation back to the server. The bulk
of those tokens are unchanged, and that's where the cache pays off.

Stats reported in the server log:

```
chat.completion id=... prompt=147 cached=132 new=20 prefill=480ms decode=...
```

`prompt=147` is what was tokenised; `cached=132` was reused from the
previous turn; only the trailing 15 prompt tokens + 20 generated
tokens hit the GPU. Prefill drops from ~15 s to ~0.5 s for a 4-turn
follow-up against Gemma 4 26B.

### When the cache does not match

The cache is keyed on **decoded token ids**. Anything that makes the
new prompt's tokens diverge from the previous turn's decoded tokens
caps the cached length at the divergence point. The two common
causes:

1. **Different system prompt or earlier user turn.** Expected
   behaviour — the cache should not apply across unrelated chats.
   A single-slot cache means parallel users compete; whichever
   request lands second only matches the shared prefix (often just
   the system message).
2. **Response cleaning strips the model's own markup.** For Gemma 4
   the engine decodes `<|channel>thought<channel|>` before the
   visible answer. By default the server strips that wrapper before
   sending the response, but the client then echoes back only the
   cleaned text on the next turn — the encoded prompt no longer
   matches the cached tokens past the first assistant turn.

The second case is the one to know. See **preserve-thinking** below.

### `MIMIRMIND_PRESERVE_THINKING`

Set this env var on the server (`MIMIRMIND_PRESERVE_THINKING=1`) to
keep the thinking-channel wrapper in the assistant response on both
the non-streaming and streaming paths. Clients then receive the raw
decoded text including `<|channel>thought\n<channel|>`. When they
send it back as part of the next turn's `messages`, the encoded
tokens match the cache exactly and prefix reuse extends all the way
through the previous assistant turn — typical 5× prefill speedup
for multi-turn chats.

Trade-off: clients see the wrapper and either need to know what to
do with it or render it as-is. Pegenaut's chat UI is wrapper-aware
and will hide it when this mode is enabled.

Default (off) keeps responses clean for unsuspecting OpenAI clients
but caps the cache reuse at the boundary of the first assistant
turn. Either choice is correct; pick based on whether your client
controls both ends of the conversation.

### Cache scope and lifecycle

- **Pre-allocated, fixed size.** On first generate() the engine
  allocates KV state for `MIMIRMIND_MAX_CONTEXT_TOKENS` (default
  8192) tokens. This is the per-request hard cap on
  `prompt_tokens + max_tokens`; exceeding it returns an error
  rather than silently dropping the cache. The fixed size is
  what makes multi-turn prefix reuse actually work — without it,
  a growing prompt across turns would reallocate-and-reset the
  cache every turn.
- **One slot.** The engine holds the KV state of exactly one
  conversation at a time. A request whose prompt does not share a
  prefix with the cached tokens evicts the previous slot
  (`cached=0` in the log) and replaces it.
- **No cross-process state.** The cache lives in process memory.
  Restarting the server wipes it.
- **Concurrency.** The handler mutex serialises generate() calls,
  so the cache is naturally consistent. Multi-slot cache + per-
  request KV is M9.3 (not yet implemented).
- **Errors.** If `generate()` throws partway through, the cache is
  reset before the exception propagates — no half-written KV
  survives into the next request.

### Sizing the cache

KV-cache size grows linearly with `MIMIRMIND_MAX_CONTEXT_TOKENS`.
For Gemma 4 26B (30 layers, hybrid SWA/full attention with
1024/2048 kv-dim), the rough cost is **~430 KiB per token**
across all layers:

| MAX_CONTEXT_TOKENS | KV cache size | Good for |
|---|---|---|
| 2048 | ~870 MiB | Short single-turn prompts only |
| 4096 | ~1.7 GiB | Default-ish chat history |
| **8192** (default) | **~3.4 GiB** | **Long chat, modest RAG context** |
| 16384 | ~6.9 GiB | RAG with large doc context |
| 32768 | ~13.7 GiB | Long-document analysis |

The number is rough — actual cost depends on the model's KV head
dimensions. The exact bytes are logged at startup
(`kvcache: pre-allocated for N tokens`). For mimirmind running
alongside ~22 GiB of Gemma 4 26B weights on a 64 GiB UMA host,
sizes up to ~16K are comfortable.

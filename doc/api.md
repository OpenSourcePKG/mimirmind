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

If you are running the server behind nginx or another reverse proxy,
SSE streaming requires:

```nginx
location / {
    proxy_pass http://mimirmind:8080;
    proxy_buffering off;          # critical for SSE
    proxy_cache off;
    proxy_set_header Connection "";
    proxy_http_version 1.1;
}
```

Without `proxy_buffering off` the proxy will batch the SSE chunks and
the client only sees the response after the full generation completes,
which defeats the point.

### What the streaming path does NOT do (yet)

The Gemma 4 thinking-channel markup (`<|channel>thought\n<channel|>`)
that appears at the start of every response is stripped from the
non-streaming response in `ChatTemplate::cleanResponse`. The
streaming path emits raw deltas — the wrapper is contained in the
first few tokens, so most clients can ignore it, but a stateful
streaming stripper is on the TODO list.

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
off the channel is empty; the wrapper is stripped from the
non-streaming response content before it reaches the wire.

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
"""Shared streaming client for the vLLM/mimirmind per-stage probes (roadmap 5.18.7).

Both engines expose an OpenAI-compatible ``/v1/chat/completions`` endpoint with SSE
streaming, so the *same* probe runs against either -- that symmetry is the whole
point of the delta table. Stdlib only (urllib) to avoid installing anything into
the box GPU container.

Measured, per request:
  * ttft_ms      -- wall time until the first streamed content token (== prefill).
  * decode_ms    -- mean wall ms per subsequent token (inter-token).
  * decode_tps   -- 1000 / decode_ms.
  * n_out        -- streamed output tokens (content deltas counted).
  * server_*     -- any usage / timing fields the server reports (best effort).
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProbeResult:
    ttft_ms: float
    decode_ms: float
    decode_tps: float
    n_out: int
    wall_ms: float
    first_text: str = ""
    server_fields: dict = field(default_factory=dict)

    def as_line(self, label: str) -> str:
        return (
            f"{label:<12} ttft={self.ttft_ms:8.1f}ms  "
            f"decode={self.decode_ms:6.2f}ms/tok ({self.decode_tps:6.1f} tok/s)  "
            f"n_out={self.n_out:4d}  wall={self.wall_ms:8.1f}ms"
        )


def _ctx(insecure: bool) -> Optional[ssl.SSLContext]:
    if not insecure:
        return None
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def stream_chat(
    base_url: str,
    messages: list[dict],
    *,
    model: str = "default",
    max_tokens: int = 512,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    insecure: bool = False,
    timeout: float = 600.0,
) -> ProbeResult:
    """Fire one streaming chat completion and time TTFT + per-token decode."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    t_prev: Optional[float] = None
    inter_token_ms: list[float] = []
    n_out = 0
    first_text = ""
    server_fields: dict = {}

    with urllib.request.urlopen(req, timeout=timeout, context=_ctx(insecure)) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("usage"), dict):
                server_fields["usage"] = obj["usage"]
            for extra in ("timings", "prefill_ms", "decode_ms"):
                if extra in obj:
                    server_fields[extra] = obj[extra]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if not piece:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
                first_text = piece[:40]
            else:
                inter_token_ms.append((now - t_prev) * 1000.0)  # type: ignore[operator]
            t_prev = now
            n_out += 1

    t_end = time.perf_counter()
    ttft_ms = ((t_first or t_end) - t0) * 1000.0
    decode_ms = (sum(inter_token_ms) / len(inter_token_ms)) if inter_token_ms else 0.0
    return ProbeResult(
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        decode_tps=(1000.0 / decode_ms) if decode_ms > 0 else 0.0,
        n_out=n_out,
        wall_ms=(t_end - t0) * 1000.0,
        first_text=first_text,
        server_fields=server_fields,
    )


def add_common_args(parser) -> None:
    parser.add_argument("--base-url", required=True,
                        help="Engine root, e.g. https://172.17.0.2:8080 or http://localhost:8000")
    parser.add_argument("--model", default="default", help="Model id the server expects")
    parser.add_argument("--api-key", default=None, help="Bearer token if the server requires auth")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification (self-signed)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before timing")
    parser.add_argument("--repeat", type=int, default=3, help="Timed repetitions (median reported)")
    parser.add_argument("--label", default="engine", help="Row label for the printed line")

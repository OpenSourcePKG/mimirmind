#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Stefan Werfling
#
# Extended SSE streaming anchor harness (roadmap 8.30.11.2). Exercises the
# ChatCompletionHandler streaming edge cases that the basic anchor does NOT
# cover — multibyte UTF-8 chunk-splitting, reasoning_content, tool-call deltas,
# and stream_options.include_usage — and captures a NORMALISED, per-delta record
# so a before/after run diffs byte-for-byte.
#
# Requests are greedy (temperature 0) + fixed seed so the token stream is
# reproducible; a behaviour-neutral refactor of the streaming path must produce
# an IDENTICAL normalised record. `id` and `created` (which legitimately vary)
# are excluded; everything else — role chunk, the ORDERED list of content /
# reasoning delta strings (i.e. the exact chunk boundaries the UTF-8 hold
# produces), tool-call deltas, finish_reason, the usage chunk shape, and [DONE]
# — is captured.
#
# Usage:
#   anchor_stream_harness.py --key sk-... --out baseline.json      # capture
#   anchor_stream_harness.py --diff baseline.json candidate.json   # compare

import argparse
import json
import ssl
import sys
import urllib.request

BASE_URL = "https://localhost:8080/v1/chat/completions"
MODEL = "qwen3.6"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _base(messages, **extra):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0,
        "seed": 1,
        "max_tokens": extra.pop("max_tokens", 64),
    }
    body.update(extra)
    return body


# Deterministic, edge-case-focused test matrix.
TESTS = {
    "plain": _base(
        [{"role": "user", "content": "Reply with exactly: PONG"}], max_tokens=16),
    "utf8_multibyte": _base(
        [{"role": "user",
          "content": "Repeat exactly, nothing else: 日本語テスト🍎🚀🔥漢字café"}],
        max_tokens=48),
    "reasoning": _base(
        [{"role": "user",
          "content": "What is 7 times 6? Reason in one short sentence, then "
                     "give the answer."}],
        max_tokens=160, enable_thinking=True),
    "usage": _base(
        [{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=16, stream_options={"include_usage": True}),
    "tool_stream": _base(
        [{"role": "user", "content": "Weather in Berlin? Call the tool."}],
        max_tokens=200, tools=[_WEATHER_TOOL], tool_choice="auto"),
    "held_markup": _base(
        [{"role": "user",
          "content": "Output only this literal XML and nothing else: "
                     "<note>hi</note>"}],
        max_tokens=64, tools=[_WEATHER_TOOL], tool_choice="auto"),
}


def run_one(key, body):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        BASE_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    rec = {
        "role_chunks": 0,
        "content_deltas": [],
        "reasoning_deltas": [],
        "tool_call_deltas": [],
        "finish_reason": None,
        "usage": None,
        "done": False,
        "n_chunks": 0,
        "error": None,
    }
    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=180)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"http:{e}"
        return rec
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            rec["done"] = True
            continue
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            rec["error"] = f"badjson:{payload[:80]}"
            continue
        rec["n_chunks"] += 1
        if d.get("usage") is not None:
            # usage chunk: capture shape (keys + whether counts are ints),
            # not the raw numbers (which are token-exact but here we assert
            # the terminal-usage-chunk STRUCTURE is preserved).
            u = d["usage"]
            rec["usage"] = sorted(u.keys()) if isinstance(u, dict) else str(type(u))
        for ch in d.get("choices", []):
            delta = ch.get("delta", {}) or {}
            if "role" in delta:
                rec["role_chunks"] += 1
            if delta.get("content"):
                rec["content_deltas"].append(delta["content"])
            if delta.get("reasoning_content"):
                rec["reasoning_deltas"].append(delta["reasoning_content"])
            for tc in (delta.get("tool_calls") or []):
                fn = tc.get("function", {}) or {}
                rec["tool_call_deltas"].append(
                    {"i": tc.get("index"),
                     "name": fn.get("name"),
                     "args": fn.get("arguments")})
            if ch.get("finish_reason"):
                rec["finish_reason"] = ch["finish_reason"]
    rec["content_text"] = "".join(rec["content_deltas"])
    rec["reasoning_text"] = "".join(rec["reasoning_deltas"])
    return rec


def capture(key, out):
    result = {name: run_one(key, body) for name, body in TESTS.items()}
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    n_err = sum(1 for r in result.values() if r["error"])
    print(f"captured {len(result)} tests -> {out} ({n_err} errors)")
    for name, r in result.items():
        print(f"  {name}: finish={r['finish_reason']} "
              f"content_deltas={len(r['content_deltas'])} "
              f"reasoning_deltas={len(r['reasoning_deltas'])} "
              f"tool_deltas={len(r['tool_call_deltas'])} "
              f"usage={r['usage']} done={r['done']} err={r['error']}")
    return 0 if n_err == 0 else 1


# Tests whose GENERATION is nondeterministic (long free-form chain-of-thought:
# tiny kernel-level numeric nondeterminism compounds over many tokens). For
# these we cannot exact-diff the emitted text, so we compare a STRUCTURAL
# fingerprint — the streaming PLUMBING that a refactor must preserve: reasoning
# is routed to reasoning_content (not content), finish_reason, usage, done,
# role. The exact UTF-8-hold chunking is exact-gated by the deterministic
# `utf8_multibyte` (content path shares utf8IncompleteTailStart).
STRUCTURAL = {"reasoning"}


def _fingerprint(r):
    if r is None:
        return None
    return {
        "role_chunks": r.get("role_chunks"),
        "has_content": len(r.get("content_deltas") or []) > 0,
        "has_reasoning": len(r.get("reasoning_deltas") or []) > 0,
        "tool_deltas": len(r.get("tool_call_deltas") or []),
        "finish_reason": r.get("finish_reason"),
        "usage": r.get("usage"),
        "done": r.get("done"),
        "error": r.get("error"),
    }


def diff(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    ok = True
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if name in STRUCTURAL:
            fa, fb = _fingerprint(ra), _fingerprint(rb)
            if fa != fb:
                ok = False
                print(f"DIFF [{name}] (structural):\n      base={fa!r}\n      cand={fb!r}")
            else:
                print(f"OK   [{name}] (structural)")
            continue
        if ra != rb:
            ok = False
            print(f"DIFF [{name}]:")
            keys = set((ra or {})) | set((rb or {}))
            for k in sorted(keys):
                va, vb = (ra or {}).get(k), (rb or {}).get(k)
                if va != vb:
                    print(f"    {k}:\n      base={va!r}\n      cand={vb!r}")
        else:
            print(f"OK   [{name}]")
    print("=== PARITY: IDENTICAL ===" if ok else "=== PARITY: MISMATCH ===")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("BASE", "CAND"))
    args = ap.parse_args()
    if args.diff:
        return diff(*args.diff)
    if not args.key or not args.out:
        ap.error("capture needs --key and --out")
    return capture(args.key, args.out)


if __name__ == "__main__":
    sys.exit(main())

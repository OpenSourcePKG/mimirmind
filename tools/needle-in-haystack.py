#!/usr/bin/env python3
"""needle-in-haystack — Q8_0-KV recall gate for the M10.2.0 design ticket.

Sends N synthetic long-context prompts to a live mimirmind server. Each
prompt is a haystack of unrelated factoids with a single SECRET_CODE
needle at a deterministic-but-random position, followed by a query
asking the model to reproduce the code. Success = the response
contains the needle token.

Usage:
    tools/needle-in-haystack.py --url http://host:8080 \\
        [--count 20] [--context-chars 16000] [--seed 42] \\
        [--model any] [--max-tokens 32] [--json OUT.json]

The suite is the Commit-8 acceptance gate for KvDtype::Q8_0: kernel-
level parity (gpu_tests attention_*_q8_0_parity, tol 5e-3) is
necessary but not sufficient because per-block absmax bleed can
survive fixture-level tolerance while collapsing K-channel outlier
recall in real prompts (see design ticket
`Memory/mimirmind/todos/m10-2-0-kv-cache-q8-0-design.md`,
"Warum reicht Kernel-Level-Parity NICHT").

Design defaults follow the ticket: 20 prompts × ~4k tokens (≈ 16k
chars for English), pass gate at ≥ 95 % recall. Exit code 0 on pass,
1 on fail, so this can gate CI/deploy pipelines.

Detected server dtype (via /v1/system/info.kv_cache.dtype) is
labelled on every output line so a comparison run across F32 / FP16 /
Q8_0 is trivially triageable.

No third-party deps; stdlib only. Deterministic under a fixed --seed
so a regression can be reproduced verbatim.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


# The needle format is unambiguous: 8-char alphanumeric with a
# specific prefix/suffix pattern that the filler never emits, so a
# grep-for-substring check has zero false positives.
NEEDLE_PREFIX = "CODE-"


# Filler paragraphs are procedural fake-facts. Templated so no real-
# world knowledge leaks in; deterministic under the seed.
FILLER_TEMPLATES = [
    "In the region of {region}, the annual production of {product} reached "
    "{n} units in year {year}, according to the {agency} statistical bureau.",
    "The {agency} concluded that {product} exports to {region} grew by "
    "{n} percent between {year} and {year2}, driven by policy reforms.",
    "A {year} survey by the {agency} found that {n} percent of {region} "
    "households own at least one {product}, up from a decade prior.",
    "During the {year} fiscal year, the {agency} allocated {n} million in "
    "grants to {product}-related research in {region}, prioritising SMEs.",
    "By {year2}, the projected demand for {product} in {region} is expected "
    "to exceed {n} thousand units, per the {agency}'s baseline scenario.",
]

REGIONS   = ["Aquileia", "Belrion", "Cathenor", "Dornhelm", "Ellisar",
             "Faralind", "Gorwyn", "Halmuth", "Ithelion", "Jarvath",
             "Kernewic", "Lomoria", "Mardun", "Nevaris", "Ostmark"]
PRODUCTS  = ["photonic sensors", "biosynthetic wool", "modular gearboxes",
             "carbon batteries", "ceramic bearings", "hyperspectral cameras",
             "hydrogel packaging", "nanoporous filters", "phase-change alloys",
             "shape-memory springs"]
AGENCIES  = ["Northern Trade Council", "Federated Metrics Office",
             "Continental Registry", "Alliance Bureau of Analysis",
             "Union of Regional Assemblies"]


@dataclass
class NeedleResult:
    idx: int
    needle: str
    position_pct: float
    prompt_chars: int
    latency_ms: float
    response: str
    passed: bool


def gen_needle(rng: random.Random) -> str:
    """Deterministic 8-char alphanumeric needle."""
    alphabet = string.ascii_uppercase + string.digits
    return NEEDLE_PREFIX + "".join(rng.choices(alphabet, k=8))


def gen_filler_paragraph(rng: random.Random) -> str:
    tmpl = rng.choice(FILLER_TEMPLATES)
    return tmpl.format(
        region  = rng.choice(REGIONS),
        product = rng.choice(PRODUCTS),
        agency  = rng.choice(AGENCIES),
        n       = rng.randint(3, 987),
        year    = rng.randint(1965, 2020),
        year2   = rng.randint(2025, 2050),
    )


def build_prompt(rng: random.Random, target_chars: int) -> tuple[str, str, float]:
    """Return (prompt, needle, position_pct)."""
    needle = gen_needle(rng)
    needle_line = (
        f"Please remember: the SECRET CODE for this session is {needle}. "
        "Keep this in mind for the question at the end."
    )

    # Build filler until we're close to the target.
    fillers: list[str] = []
    body_len = 0
    while body_len < target_chars:
        p = gen_filler_paragraph(rng)
        fillers.append(p)
        body_len += len(p) + 2  # + "\n\n"

    # Choose a random insertion index (paragraph-granular so we never
    # split a filler paragraph). Reject depth 0 and depth n so the
    # needle is genuinely inside the haystack.
    n_fillers = len(fillers)
    insert_at = rng.randint(1, max(1, n_fillers - 1))
    position_pct = 100.0 * insert_at / n_fillers
    fillers.insert(insert_at, needle_line)

    header = (
        "Below is a text you must read carefully. After the text, you will "
        "be asked to recall a specific fact.\n\n"
        "--- BEGIN TEXT ---\n\n"
    )
    footer = (
        "\n\n--- END TEXT ---\n\n"
        "Question: What is the SECRET CODE mentioned in the text? "
        "Answer with only the code itself, nothing else."
    )
    prompt = header + "\n\n".join(fillers) + footer
    return prompt, needle, position_pct


def query_server(url: str, prompt: str, model: str,
                 max_tokens: int, timeout_s: float) -> tuple[str, float]:
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a careful reader. Follow the user's "
                        "instructions precisely."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens":  max_tokens,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = payload["choices"][0]["message"]["content"]
    return text, latency_ms


def fetch_server_dtype(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(
                url.rstrip("/") + "/v1/system/info", timeout=5.0) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        return info.get("kv_cache", {}).get("dtype")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url",           default="http://localhost:8080")
    ap.add_argument("--model",         default="any")
    ap.add_argument("--count",         type=int,   default=20)
    ap.add_argument("--context-chars", type=int,   default=16000,
                    help="approx target haystack size; ≈ 4000 tokens for English")
    ap.add_argument("--max-tokens",    type=int,   default=32)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--timeout",       type=float, default=180.0)
    ap.add_argument("--pass-gate",     type=float, default=0.95,
                    help="minimum recall fraction to exit 0")
    ap.add_argument("--json",          type=str,   default=None,
                    help="optional path to write full result JSON")
    args = ap.parse_args()

    dtype = fetch_server_dtype(args.url) or "unknown"
    print(f"# needle-in-haystack | server={args.url} kv_dtype={dtype} "
          f"count={args.count} ctx≈{args.context_chars}ch "
          f"seed={args.seed} gate={int(args.pass_gate * 100)}%")

    rng = random.Random(args.seed)
    results: list[NeedleResult] = []

    for i in range(args.count):
        prompt, needle, pos_pct = build_prompt(rng, args.context_chars)
        try:
            response, latency = query_server(
                args.url, prompt, args.model,
                args.max_tokens, args.timeout)
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            print(f"[{i+1:02d}/{args.count}] REQUEST FAILED needle={needle} "
                  f"err={exc}")
            results.append(NeedleResult(
                idx=i, needle=needle, position_pct=pos_pct,
                prompt_chars=len(prompt), latency_ms=0.0,
                response=str(exc), passed=False))
            continue
        passed = needle in response
        tag = "PASS" if passed else "FAIL"
        # Trim response for terminal readability but keep enough to see
        # what the model actually said on a fail.
        resp_shown = response.strip().replace("\n", " ")[:80]
        print(f"[{i+1:02d}/{args.count}] {tag} needle={needle} "
              f"pos={pos_pct:5.1f}% prompt={len(prompt):>6}ch "
              f"latency={latency:6.0f}ms resp='{resp_shown}'")
        results.append(NeedleResult(
            idx=i, needle=needle, position_pct=pos_pct,
            prompt_chars=len(prompt), latency_ms=latency,
            response=response, passed=passed))

    n_pass    = sum(1 for r in results if r.passed)
    recall    = n_pass / max(1, len(results))
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]

    print("")
    print(f"# recall     : {n_pass}/{len(results)} = {recall * 100:.1f}%")
    if latencies:
        print(f"# latency ms : mean={statistics.mean(latencies):.0f} "
              f"p50={statistics.median(latencies):.0f} "
              f"max={max(latencies):.0f}")
    print(f"# kv_dtype   : {dtype}")
    print(f"# gate       : {int(args.pass_gate * 100)}% "
          f"→ {'PASS' if recall >= args.pass_gate else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "url":         args.url,
                "kv_dtype":    dtype,
                "seed":        args.seed,
                "count":       args.count,
                "context_chars": args.context_chars,
                "recall":      recall,
                "pass_gate":   args.pass_gate,
                "results": [
                    {"idx": r.idx, "needle": r.needle,
                     "position_pct": r.position_pct,
                     "prompt_chars": r.prompt_chars,
                     "latency_ms":   r.latency_ms,
                     "response":     r.response,
                     "passed":       r.passed}
                    for r in results
                ],
            }, f, indent=2)

    return 0 if recall >= args.pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
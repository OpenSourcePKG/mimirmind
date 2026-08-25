#!/usr/bin/env python3
"""Grounded probe -- ~2316-token context, TTFT/prefill measurement (roadmap 5.18.7).

Mirrors the grounded-RAG anchor used for the prefill gap: a long retrieved context
plus a question, streamed, timing wall-clock to the first token (== prefill/TTFT).
Only a few output tokens are requested, so the number reflects prefill, not decode.

Runs the identical request against either OpenAI-compatible engine (vLLM or
mimirmind). Pair with engine-side instrumentation:
  * vLLM  : VLLM_TORCH_PROFILER_DIR=<dir> + /start_profile,/stop_profile  (Layer A trace)
            or MIMIR_VLLM_PROBE=1 (Layer B/C, --enforce-eager).
  * mimir : MIMIRMIND_DECODE_PROFILE=1 (prefill sections), SSE prefill_ms.

Example:
  python3 grounded_probe.py --base-url http://localhost:8000 --model qwen \
      --ctx-tokens 2316 --label vllm
"""

from __future__ import annotations

import argparse
import statistics

from probe_common import ProbeResult, add_common_args, stream_chat

# A deterministic, self-consistent passage so the answer is actually grounded.
_FACT_SENTENCES = [
    "The Mimir inference engine loads pre-trained weights and does not train models.",
    "On the DGX Spark platform Mimir targets the Blackwell GB10 with 128 GB of unified LPDDR5x memory.",
    "The unified memory on that platform delivers roughly 273 gigabytes per second of bandwidth.",
    "Weights on the serving platform are stored in the NVFP4 four-bit format.",
    "The Meteor Lake target instead uses the Intel Xe-LPG integrated GPU through oneAPI Level Zero.",
    "The serving build is code-named Bragi and supports at least sixty-four concurrent chats.",
    "Speculative decoding on the serving path uses native multi-token prediction heads.",
    "The reference oracle during development is llama-cli, used only for tensor and logits parity.",
    "The project forbids using llama.cpp or ggml as a runtime dependency.",
    "The strategic target model is Gemma 4, with Gemma 3 kept only as a verification baseline.",
]


def _build_context(target_tokens: int) -> str:
    """Assemble filler + fact sentences until ~target_tokens (~4 chars/token)."""
    target_chars = target_tokens * 4
    chunks: list[str] = []
    i = 0
    while sum(len(c) + 1 for c in chunks) < target_chars:
        fact = _FACT_SENTENCES[i % len(_FACT_SENTENCES)]
        chunks.append(f"[doc {i:04d}] {fact}")
        i += 1
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--ctx-tokens", type=int, default=2316,
                    help="Approximate context length in tokens (~4 chars/token)")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="Small: we time prefill/TTFT, not decode")
    args = ap.parse_args()

    context = _build_context(args.ctx_tokens)
    question = ("Based only on the context above, what memory bandwidth does the "
                "serving platform provide, and what weight format does it use?")
    messages = [
        {"role": "system", "content": "Answer strictly from the provided context."},
        {"role": "user", "content": f"{context}\n\n---\n\n{question}"},
    ]

    def one() -> ProbeResult:
        return stream_chat(
            args.base_url, messages,
            model=args.model, max_tokens=args.max_tokens, temperature=0.0,
            api_key=args.api_key, insecure=args.insecure,
        )

    for _ in range(max(0, args.warmup)):
        one()

    runs = [one() for _ in range(max(1, args.repeat))]
    med_ttft = statistics.median(r.ttft_ms for r in runs)

    approx_chars = len(context)
    print(f"\n=== grounded probe [{args.label}] "
          f"(~{args.ctx_tokens} ctx-tokens, {approx_chars} chars, repeat={args.repeat}) ===")
    for i, r in enumerate(runs):
        print(r.as_line(f"run{i}"))
    print("-" * 72)
    print(f"MEDIAN TTFT (prefill)  {med_ttft:8.1f}ms")
    print(f"first token text       {runs[-1].first_text!r}")
    if runs[-1].server_fields:
        print(f"server                 {runs[-1].server_fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

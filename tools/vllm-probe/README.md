# vllm-probe — measured per-stage vLLM vs mimirmind delta (roadmap 5.18.7)

Instrument a running vLLM engine (v0.22.1 / Qwen3-Next, GB10) with our own
per-stage timers so we can build an apples-to-apples ms table against
mimirmind's `decode-prof`, ranked by absolute ms gap → ordered lever list with a
hard vLLM target per stage. The dispatch map says *which* kernel; this says *how
many ms*.

Zero vLLM source edits: `mimir_vllm_probe.py` monkeypatches a small registry of
target methods on import. `sitecustomize.py` auto-imports it when the dir is on
`PYTHONPATH`. Stage names mirror mimirmind decode-prof so rows line up.

## Files

| file | role |
|------|------|
| `mimir_vllm_probe.py` | Layer A record_function spans + Layer B/C cuda.Event timers; data-driven target registry (logs bind/MISS per stage) |
| `sitecustomize.py`    | zero-edit activation hook (auto-import via PYTHONPATH) |
| `probe_common.py`     | stdlib OpenAI-compatible streaming client (TTFT + per-token decode ms) |
| `decode_probe.py`     | bs=1, 512-gen decode probe (decode-dominated) |
| `grounded_probe.py`   | ~2316-ctx grounded probe (TTFT/prefill) |

Both probe scripts hit `/v1/chat/completions` and run identically against vLLM
or mimirmind — that symmetry is the point.

## Constraints (Synaipse lessons — do not violate)

- **No `ncu`** (hangs the Spark box). Timing is in-code cuda.Event / Kineto only.
- **One GPU container at a time.** Swap: stop `mimirmind-serve` → run vLLM →
  probe → stop vLLM → restore mimirmind-serve → **re-check perf flags**.
- **vLLM only** (no SGLang). Target tree: `~/Projekte/github/vllm-box` (v0.22.1,
  HEAD `0decac0`) == NGC `nvcr.io/nvidia/vllm:26.06-py3`.

## Box protocol

1. Copy this dir onto the box (e.g. `/opt/mimirmind/tools/vllm-probe`).
2. Activate on the vLLM side (no source edit):
   ```
   export PYTHONPATH=/opt/mimirmind/tools/vllm-probe:$PYTHONPATH
   export MIMIR_VLLM_PROBE=1            # Layer B/C live log
   export MIMIR_VLLM_PROBE_EVERY=64
   export MIMIR_VLLM_PROBE_VERBOSE=1    # print each bound target
   # Layer A trace (graph-safe, no eager needed):
   export VLLM_TORCH_PROFILER_DIR=/tmp/vllm-kineto
   ```
   On first start, read the `[mimir-vllm-probe] bound N/M` line. Any `MISS`
   means a qualname drifted in this build — edit `_TARGETS` in
   `mimir_vllm_probe.py` to match, re-run.
   For clean Layer B/C absolute numbers, start vLLM with `--enforce-eager`
   (ratios stay valid; absolutes from the Layer A Kineto trace).
3. Run the two probes against vLLM:
   ```
   python3 decode_probe.py   --base-url http://localhost:8000 --model <id> --label vllm
   python3 grounded_probe.py --base-url http://localhost:8000 --model <id> --label vllm --ctx-tokens 2316
   ```
4. Stop vLLM → restore `mimirmind-serve` → verify perf flags are on.
5. Run the same two probes against mimirmind with its own profiler:
   ```
   # server env: MIMIRMIND_DECODE_PROFILE=1 MIMIRMIND_PROFILE_EVERY=64
   python3 decode_probe.py   --base-url https://<mimir> --insecure --api-key "$KEY" --model <id> --label mimir
   python3 grounded_probe.py --base-url https://<mimir> --insecure --api-key "$KEY" --model <id> --label mimir --ctx-tokens 2316
   ```
6. Build the delta table: engine-side per-stage ms (vLLM Kineto/probe-log vs
   mimir decode-prof) next to each other, ranked by absolute gap. Record it as a
   Synaipse research note and feed the measured targets back into roadmap 5.18.x.

Detailed rationale, injection-site map, and stage mapping:
Synaipse note *"Plan: vLLM mit eigenen Debug-Ausgaben instrumentieren"* (2026-08-23).

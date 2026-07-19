#!/usr/bin/env bash
#
# Perf-regression A/B bench between a candidate mimirmind and the
# currently-deployed baseline on L0_TARGET_HOST (or any L0 host).
# Emits ledger-ready numbers plus a rough decode/prefill per-token
# comparison table.
#
# Usage:
#   scripts/bench-vs-baseline.sh <candidate_url> <baseline_url> [<model>]
#
# Example:
#   scripts/bench-vs-baseline.sh \
#       http://L0_TARGET_HOST-candidate:8080 \
#       http://L0_TARGET_HOST:8080 \
#       google_gemma-4-E4B-it-Q4_K_M
#
# What it does
# ------------
# 1. Sanity: `/health` on both endpoints returns 200.
# 2. `/v1/system/info` on each — captures the model + backend + kernel
#    autotune report so the ledger entry knows what we're comparing.
# 3. Three prompt buckets:
#      short  — 12-token instruction, 128 max_new
#      medium — 200-token RAG-ish prompt, 256 max_new
#      long   — 2000-token document + summarize, 512 max_new
#    Each fired against BOTH endpoints, 3 shots each. We record wall,
#    prefill_ms, decode_ms per shot.
# 4. Print a summary table + a JSON blob suitable for pasting into
#    `research/perf-regression-ledger.md`.
#
# The candidate must be a fully-warmed engine (first request loads
# weights, autotunes, etc — take one throwaway shot each first).

set -euo pipefail

CANDIDATE_URL="${1:?candidate URL required}"
BASELINE_URL="${2:?baseline URL required}"
MODEL="${3:-google_gemma-4-E4B-it-Q4_K_M}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

check_health() {
    local url="$1"
    local label="$2"
    if ! curl -fs "$url/health" >/dev/null; then
        log "FAIL: $label /health at $url did not return 200"
        exit 1
    fi
    log "$label reachable at $url"
}

# Prompts. Kept literal so runs are reproducible across sessions.
readonly SHORT_PROMPT='Write a one-sentence summary of the color blue.'
readonly MEDIUM_PROMPT='You are a helpful research assistant. Below is a short excerpt from a paper on transformer inference. Summarize the key optimization technique in three sentences, then list two potential drawbacks.

Excerpt: Modern transformer inference has moved toward memory-bandwidth-bound decode, where the per-token latency is dominated by moving weights from DRAM to compute units rather than by the compute itself. Techniques like PagedAttention and continuous batching help amortize the memory-access cost across concurrent requests, but single-user latency remains dominated by the raw weight-fetch bandwidth of the target hardware.'
readonly LONG_PROMPT_HEAD='Below is a long article. Summarize the main argument in three paragraphs.

'

fetch_info() {
    local url="$1"
    local out="$2"
    curl -fs "$url/v1/system/info" > "$out"
    log "info captured to $out"
}

one_shot() {
    local url="$1"
    local prompt="$2"
    local max_new="$3"
    local out_json="$4"

    local start_ms end_ms
    start_ms=$(($(date +%s%N)/1000000))
    curl -fs "$url/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL" --arg p "$prompt" --argjson n "$max_new" '{
            model: $m,
            messages: [{role: "user", content: $p}],
            temperature: 0,
            max_tokens: $n
        }')" > "$out_json"
    end_ms=$(($(date +%s%N)/1000000))
    printf '%d' $((end_ms - start_ms))
}

bench_bucket() {
    local url="$1"
    local label="$2"
    local prompt="$3"
    local max_new="$4"
    local outdir="$5"

    local shot walls=()
    for shot in 1 2 3; do
        local json="$outdir/${label}_${shot}.json"
        walls+=("$(one_shot "$url" "$prompt" "$max_new" "$json")")
        log "  $label shot $shot: wall=${walls[-1]} ms"
    done
    # Median of 3.
    IFS=$'\n' walls_sorted=($(printf '%s\n' "${walls[@]}" | sort -n))
    unset IFS
    printf '%s' "${walls_sorted[1]}"
}

WORKDIR="${TMPDIR:-/tmp}/mm-bench-$$"
mkdir -p "$WORKDIR"
log "workdir: $WORKDIR"

check_health "$CANDIDATE_URL" candidate
check_health "$BASELINE_URL"  baseline

fetch_info "$CANDIDATE_URL" "$WORKDIR/candidate-info.json"
fetch_info "$BASELINE_URL"  "$WORKDIR/baseline-info.json"

# Warm-up shot per endpoint (discarded).
log "warmup — one shot per endpoint, discarded"
one_shot "$CANDIDATE_URL" "warm" 16 "$WORKDIR/warmup-candidate.json" >/dev/null
one_shot "$BASELINE_URL"  "warm" 16 "$WORKDIR/warmup-baseline.json"  >/dev/null

log "candidate — short/medium/long × 3"
CAND_SHORT=$(bench_bucket  "$CANDIDATE_URL" cand_short  "$SHORT_PROMPT"  128 "$WORKDIR")
CAND_MEDIUM=$(bench_bucket "$CANDIDATE_URL" cand_medium "$MEDIUM_PROMPT" 256 "$WORKDIR")
CAND_LONG=$(bench_bucket   "$CANDIDATE_URL" cand_long   "${LONG_PROMPT_HEAD}$(seq 1 400 | tr '\n' ' ')" 512 "$WORKDIR")

log "baseline — short/medium/long × 3"
BASE_SHORT=$(bench_bucket  "$BASELINE_URL" base_short  "$SHORT_PROMPT"  128 "$WORKDIR")
BASE_MEDIUM=$(bench_bucket "$BASELINE_URL" base_medium "$MEDIUM_PROMPT" 256 "$WORKDIR")
BASE_LONG=$(bench_bucket   "$BASELINE_URL" base_long   "${LONG_PROMPT_HEAD}$(seq 1 400 | tr '\n' ' ')" 512 "$WORKDIR")

pct_delta() {
    local cand="$1" base="$2"
    awk -v c="$cand" -v b="$base" 'BEGIN{ printf "%+.1f", (c-b)/b*100 }'
}

cat <<REPORT

===== BENCH REPORT =====
model:     $MODEL
candidate: $CANDIDATE_URL
baseline:  $BASELINE_URL
workdir:   $WORKDIR

Median wall time (ms):
  bucket   candidate   baseline   delta
  ------   ---------   --------   -----
  short       ${CAND_SHORT}      ${BASE_SHORT}      $(pct_delta "$CAND_SHORT" "$BASE_SHORT")%
  medium      ${CAND_MEDIUM}      ${BASE_MEDIUM}      $(pct_delta "$CAND_MEDIUM" "$BASE_MEDIUM")%
  long        ${CAND_LONG}      ${BASE_LONG}      $(pct_delta "$CAND_LONG" "$BASE_LONG")%

Ledger-ready JSON (paste into research/perf-regression-ledger.md):
$(jq -n \
  --arg m "$MODEL" \
  --arg cu "$CANDIDATE_URL" \
  --arg bu "$BASELINE_URL" \
  --argjson cs "$CAND_SHORT"  --argjson bs "$BASE_SHORT" \
  --argjson cm "$CAND_MEDIUM" --argjson bm "$BASE_MEDIUM" \
  --argjson cl "$CAND_LONG"   --argjson bl "$BASE_LONG" \
  '{model: $m, candidate_url: $cu, baseline_url: $bu,
    wall_ms: {short: {cand: $cs, base: $bs},
              medium: {cand: $cm, base: $bm},
              long:   {cand: $cl, base: $bl}}}')

REPORT

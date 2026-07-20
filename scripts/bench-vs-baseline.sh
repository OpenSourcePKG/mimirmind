#!/usr/bin/env bash
#
# Perf-regression A/B bench between a candidate mimirmind and the
# currently-deployed baseline on any two /v1/chat/completions
# endpoints. Emits ledger-ready numbers plus a per-bucket
# candidate-vs-baseline delta table.
#
# Usage:
#   scripts/bench-vs-baseline.sh <candidate_url> <baseline_url> [<model>]
#       [--shapes <csv>]     restrict to shapes (e.g. rag,needle,code)
#       [--lengths <csv>]    restrict to length buckets (e.g. 512,1k,2k)
#       [--langs <csv>]      restrict to languages (en,de)
#       [--shots N]          shots per bucket (default 3, median wins)
#       [--decode-only]      pick just the first decode_target per shape
#       [--samples <dir>]    samples/ root (default: <repo>/samples)
#       [--manifest <path>]  manifest.json path (default: <samples>/manifest.json)
#
# The candidate must be a fully-warmed engine (first request loads
# weights, autotunes, etc — we take one throwaway shot each first).
#
# Default matrix is the full /samples manifest, which is ~60 files
# × 2 endpoints × 3 shots × 1-2 decode-targets = a few hundred
# requests. Use --shapes / --lengths to scope down.

set -euo pipefail

CANDIDATE_URL=""
BASELINE_URL=""
MODEL="google_gemma-4-E4B-it-Q4_K_M"
SHAPES_FILTER=""
LENGTHS_FILTER=""
LANGS_FILTER="en,de"
SHOTS=3
DECODE_ONLY=0
SAMPLES_DIR=""
MANIFEST_PATH=""

# Repo root — this script lives in <repo>/scripts/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shapes)    SHAPES_FILTER="$2"; shift 2;;
        --lengths)   LENGTHS_FILTER="$2"; shift 2;;
        --langs)     LANGS_FILTER="$2"; shift 2;;
        --shots)     SHOTS="$2"; shift 2;;
        --decode-only) DECODE_ONLY=1; shift;;
        --samples)   SAMPLES_DIR="$2"; shift 2;;
        --manifest)  MANIFEST_PATH="$2"; shift 2;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \?//'
            exit 0;;
        *)  positional+=("$1"); shift;;
    esac
done

if [[ ${#positional[@]} -lt 2 ]]; then
    echo "usage: $0 <candidate_url> <baseline_url> [<model>] [--flags]" >&2
    exit 2
fi
CANDIDATE_URL="${positional[0]}"
BASELINE_URL="${positional[1]}"
[[ ${#positional[@]} -ge 3 ]] && MODEL="${positional[2]}"

SAMPLES_DIR="${SAMPLES_DIR:-$REPO_ROOT/samples}"
MANIFEST_PATH="${MANIFEST_PATH:-$SAMPLES_DIR/manifest.json}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "manifest not found: $MANIFEST_PATH" >&2
    echo "regenerate with:  python3 $SAMPLES_DIR/generate.py" >&2
    exit 2
fi

for bin in jq curl; do
    command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 2; }
done

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

check_health() {
    local url="$1" label="$2"
    if ! curl -fs "$url/health" >/dev/null; then
        log "FAIL: $label /health at $url did not return 200"
        exit 1
    fi
    log "$label reachable at $url"
}

fetch_info() {
    local url="$1" out="$2"
    curl -fs "$url/v1/system/info" > "$out" 2>/dev/null || echo '{}' > "$out"
}

# Turn a JSONL chat file into a JSON messages array.
chat_messages_json() {
    local path="$1"
    jq -sc '.' "$path"
}

# Prompt payload from a single manifest entry (returns JSON messages array
# for chat/prompt, or the extracted needle prompt as a single-message
# array for shape=needle).
messages_for_entry() {
    local entry_json="$1"
    local shape lang path type
    shape=$(jq -r '.shape' <<<"$entry_json")
    lang=$( jq -r '.lang'  <<<"$entry_json")
    path=$( jq -r '.path'  <<<"$entry_json")
    type=$( jq -r '.type'  <<<"$entry_json")
    local abs="$SAMPLES_DIR/$path"

    case "$type" in
        chat)
            chat_messages_json "$abs";;
        needle)
            jq -c '[{role:"user", content:.prompt}]' "$abs";;
        prompt|*)
            jq -Rs -c --arg _l "$lang" \
                '[{role:"user", content: .}]' "$abs";;
    esac
}

# Fire one request. Returns wall-time ms on stdout. Response body
# written to $out_json (server timing fields, if present, live there).
one_shot() {
    local url="$1" messages_json="$2" max_new="$3" out_json="$4"
    local start_ms end_ms
    start_ms=$(($(date +%s%N)/1000000))
    curl -fs "$url/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL" --argjson msgs "$messages_json" \
                     --argjson n "$max_new" '{
                model: $m,
                messages: $msgs,
                temperature: 0,
                max_tokens: $n
            }')" > "$out_json"
    end_ms=$(($(date +%s%N)/1000000))
    printf '%d' $((end_ms - start_ms))
}

# Extract prefill_ms / decode_ms if server exposes them (mimirmind
# does, via /v1/chat/completions.usage.timings.*). Empty string if
# unavailable — the wall time is still the primary signal.
extract_timing() {
    local file="$1" field="$2"
    jq -r "(.usage.timings.\"$field\" // .timings.\"$field\" // \"\")" \
        "$file" 2>/dev/null || printf ''
}

# Median of an odd list of integers on stdin.
median() {
    sort -n | awk '
        { a[NR]=$1 }
        END {
            if (NR == 0) print 0;
            else if (NR % 2) print a[(NR+1)/2];
            else printf "%.0f", (a[NR/2] + a[NR/2+1]) / 2;
        }'
}

pct_delta() {
    local cand="$1" base="$2"
    awk -v c="$cand" -v b="$base" 'BEGIN{
        if (b == 0) { print "n/a"; exit }
        printf "%+.1f", (c-b)/b*100
    }'
}

in_csv() {
    local needle="$1" csv="$2"
    [[ -z "$csv" ]] && return 0
    IFS=',' read -ra parts <<<"$csv"
    for p in "${parts[@]}"; do [[ "$p" == "$needle" ]] && return 0; done
    return 1
}

WORKDIR="${TMPDIR:-/tmp}/mm-bench-$$"
mkdir -p "$WORKDIR"
log "workdir:  $WORKDIR"
log "manifest: $MANIFEST_PATH"

check_health "$CANDIDATE_URL" candidate
check_health "$BASELINE_URL"  baseline

fetch_info "$CANDIDATE_URL" "$WORKDIR/candidate-info.json"
fetch_info "$BASELINE_URL"  "$WORKDIR/baseline-info.json"

# Filter manifest entries per user flags.
ENTRIES_JSON=$(jq -c \
    --arg shapes "$SHAPES_FILTER" \
    --arg lengths "$LENGTHS_FILTER" \
    --arg langs   "$LANGS_FILTER" '
      def csv_ok(v; csv):
        (csv | if . == "" then null
                          else split(",") end) as $list
        | ($list == null) or (any($list[]; . == v));
      .entries[]
      | select(csv_ok(.shape;         $shapes))
      | select(csv_ok(.length_bucket; $lengths))
      | select(csv_ok(.lang;          $langs))
    ' "$MANIFEST_PATH")

if [[ -z "$ENTRIES_JSON" ]]; then
    log "no manifest entries match filter"
    exit 1
fi

# Warm-up shot per endpoint (discarded).
log "warmup — one 16-tok shot per endpoint, discarded"
WARM_MSG='[{"role":"user","content":"warm"}]'
one_shot "$CANDIDATE_URL" "$WARM_MSG" 16 "$WORKDIR/warmup-cand.json" >/dev/null
one_shot "$BASELINE_URL"  "$WARM_MSG" 16 "$WORKDIR/warmup-base.json" >/dev/null

# Results accumulator: newline-separated rows.
RESULTS_TSV="$WORKDIR/results.tsv"
{
    printf 'shape\tlang\tbucket\tdecode\tcand_ms\tbase_ms\tdelta_pct\t'
    printf 'cand_prefill\tbase_prefill\tcand_decode\tbase_decode\tneedle_ok\n'
} > "$RESULTS_TSV"

n_entries=$(echo "$ENTRIES_JSON" | wc -l)
log "matrix: $n_entries entries × $SHOTS shots × 2 endpoints"

idx=0
while IFS= read -r entry; do
    idx=$((idx+1))
    shape=$(jq -r '.shape'         <<<"$entry")
    lang=$( jq -r '.lang'          <<<"$entry")
    bucket=$(jq -r '.length_bucket' <<<"$entry")
    id=$(jq -r '.id'                <<<"$entry")
    type=$(jq -r '.type'            <<<"$entry")

    # decode_targets can be multiple; --decode-only picks the first.
    decode_list=$(jq -c '.decode_targets' <<<"$entry")
    if [[ "$DECODE_ONLY" == "1" ]]; then
        decode_list=$(jq -c '[.[0]]' <<<"$decode_list")
    fi

    msgs=$(messages_for_entry "$entry")

    for decode in $(jq -r '.[]' <<<"$decode_list"); do
        cand_walls=(); base_walls=()
        cand_prefill=""; base_prefill=""
        cand_decode="";  base_decode=""
        needle_hits=0

        needle_expected=""
        if [[ "$type" == "needle" ]]; then
            needle_expected=$(jq -r '.needle' \
                "$SAMPLES_DIR/$(jq -r '.path' <<<"$entry")")
        fi

        log "[$idx/$n_entries] $id decode=$decode"

        for shot in $(seq 1 "$SHOTS"); do
            cj="$WORKDIR/cand-${id//\//_}-d${decode}-s${shot}.json"
            bj="$WORKDIR/base-${id//\//_}-d${decode}-s${shot}.json"
            cw=$(one_shot "$CANDIDATE_URL" "$msgs" "$decode" "$cj")
            bw=$(one_shot "$BASELINE_URL"  "$msgs" "$decode" "$bj")
            cand_walls+=("$cw"); base_walls+=("$bw")

            # capture timing fields from the first shot only.
            if [[ "$shot" == "1" ]]; then
                cand_prefill=$(extract_timing "$cj" prefill_ms)
                base_prefill=$(extract_timing "$bj" prefill_ms)
                cand_decode=$( extract_timing "$cj" decode_ms)
                base_decode=$( extract_timing "$bj" decode_ms)
            fi

            if [[ -n "$needle_expected" ]]; then
                resp=$(jq -r '.choices[0].message.content // ""' "$cj")
                [[ "$resp" == *"$needle_expected"* ]] && \
                    needle_hits=$((needle_hits+1))
            fi
        done

        cand_med=$(printf '%s\n' "${cand_walls[@]}" | median)
        base_med=$(printf '%s\n' "${base_walls[@]}" | median)
        delta=$(pct_delta "$cand_med" "$base_med")
        needle_ok=""
        if [[ "$type" == "needle" ]]; then
            needle_ok="${needle_hits}/${SHOTS}"
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$shape" "$lang" "$bucket" "$decode" \
            "$cand_med" "$base_med" "${delta}%" \
            "$cand_prefill" "$base_prefill" \
            "$cand_decode" "$base_decode" \
            "$needle_ok" >> "$RESULTS_TSV"
    done
done <<<"$ENTRIES_JSON"

echo
echo "===== BENCH REPORT ====="
echo "model:     $MODEL"
echo "candidate: $CANDIDATE_URL"
echo "baseline:  $BASELINE_URL"
echo "workdir:   $WORKDIR"
echo
column -t -s $'\t' "$RESULTS_TSV"

echo
echo "Ledger-ready JSON (paste into research/perf-regression-ledger.md):"
jq -Rn --arg m "$MODEL" \
       --arg cu "$CANDIDATE_URL" \
       --arg bu "$BASELINE_URL" \
       --rawfile tsv "$RESULTS_TSV" '
    ($tsv | split("\n") | map(select(length > 0)) | .[1:] |
        map(split("\t") | {
            shape:         .[0],
            lang:          .[1],
            bucket:        .[2],
            decode:        (.[3]|tonumber),
            cand_ms:       (.[4]|tonumber),
            base_ms:       (.[5]|tonumber),
            delta:         .[6],
            cand_prefill:  .[7],
            base_prefill:  .[8],
            cand_decode:   .[9],
            base_decode:   .[10],
            needle_ok:     .[11]
        })) as $rows
    | {model: $m, candidate_url: $cu, baseline_url: $bu, rows: $rows}
'
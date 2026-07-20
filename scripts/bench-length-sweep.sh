#!/usr/bin/env bash
#
# Single-endpoint length sweep. Iterates one shape × one language
# across the /samples length buckets and emits a per-length
# wall / prefill / decode ms table. Purpose: isolate how prefill
# scales with prompt length on a target host without dragging a
# baseline endpoint or the full shape × decode-target matrix along
# with it.
#
# Usage:
#   scripts/bench-length-sweep.sh <url> [<model>]
#       [--shape rag|instruct|needle|code|summarize|chat-multiturn]
#                           default: rag
#       [--lang en|de]      default: en
#       [--decode N]        max_new_tokens (default: first per-shape target)
#       [--shots N]         shots per length (default 3, median wins)
#       [--samples <dir>]   samples/ root (default: <repo>/samples)
#       [--manifest <path>] manifest.json (default: <samples>/manifest.json)
#
# Example:
#   scripts/bench-length-sweep.sh http://host:8080 gemma-4-E4B \
#       --shape rag --lang en --decode 32
#
# The default --decode 32 makes decode-cost roughly constant across
# lengths, so the wall-time slope is essentially prefill scaling.

set -euo pipefail

URL=""
MODEL="google_gemma-4-E4B-it-Q4_K_M"
SHAPE="rag"
LANG="en"
DECODE=""
SHOTS=3
SAMPLES_DIR=""
MANIFEST_PATH=""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shape)    SHAPE="$2";    shift 2;;
        --lang)     LANG="$2";     shift 2;;
        --decode)   DECODE="$2";   shift 2;;
        --shots)    SHOTS="$2";    shift 2;;
        --samples)  SAMPLES_DIR="$2";  shift 2;;
        --manifest) MANIFEST_PATH="$2"; shift 2;;
        -h|--help)  sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0;;
        *) positional+=("$1"); shift;;
    esac
done

if [[ ${#positional[@]} -lt 1 ]]; then
    echo "usage: $0 <url> [<model>] [--flags]" >&2
    exit 2
fi
URL="${positional[0]}"
[[ ${#positional[@]} -ge 2 ]] && MODEL="${positional[1]}"

SAMPLES_DIR="${SAMPLES_DIR:-$REPO_ROOT/samples}"
MANIFEST_PATH="${MANIFEST_PATH:-$SAMPLES_DIR/manifest.json}"

for bin in jq curl; do
    command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 2; }
done

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "manifest not found: $MANIFEST_PATH" >&2
    echo "regenerate: python3 $SAMPLES_DIR/generate.py" >&2
    exit 2
fi

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# Sanity ping.
curl -fs "$URL/health" >/dev/null \
    || { log "FAIL: /health at $URL"; exit 1; }
log "reachable: $URL"

# Pull matching entries in bucket order. jq sorts strings
# lexicographically, which mis-orders "1k" vs "512". We sort ourselves
# using the token_target int.
ENTRIES=$(jq -c --arg shape "$SHAPE" --arg lang "$LANG" '
    [.entries[]
     | select(.shape == $shape and .lang == $lang)]
    | sort_by(.token_target) | .[]
' "$MANIFEST_PATH")

if [[ -z "$ENTRIES" ]]; then
    log "no entries for shape=$SHAPE lang=$LANG"
    exit 1
fi

WORKDIR="${TMPDIR:-/tmp}/mm-sweep-$$"
mkdir -p "$WORKDIR"

# Warm-up.
log "warmup — 16-tok shot discarded"
curl -fs "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" '{
        model: $m, temperature: 0, max_tokens: 16,
        messages: [{role: "user", content: "warm"}]}')" \
    > "$WORKDIR/warmup.json"

one_shot() {
    local msgs="$1" max_new="$2" out="$3"
    local t0 t1
    t0=$(($(date +%s%N)/1000000))
    curl -fs "$URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg m "$MODEL" --argjson msgs "$msgs" \
                     --argjson n "$max_new" '{
                model: $m, temperature: 0,
                messages: $msgs, max_tokens: $n}')" > "$out"
    t1=$(($(date +%s%N)/1000000))
    printf '%d' $((t1 - t0))
}

messages_for() {
    local entry="$1"
    local type; type=$(jq -r '.type' <<<"$entry")
    local path; path=$(jq -r '.path' <<<"$entry")
    local abs="$SAMPLES_DIR/$path"
    case "$type" in
        chat)   jq -sc '.' "$abs";;
        needle) jq -c '[{role:"user", content:.prompt}]' "$abs";;
        *)      jq -Rs -c '[{role:"user", content: .}]' "$abs";;
    esac
}

median() {
    sort -n | awk '
        { a[NR]=$1 }
        END {
            if (NR == 0) print 0;
            else if (NR % 2) print a[(NR+1)/2];
            else printf "%.0f", (a[NR/2] + a[NR/2+1]) / 2;
        }'
}

printf 'bucket\tt_tok\tchars\tdecode\twall_ms\tprefill_ms\tdecode_ms\tms_per_tok\n'
while IFS= read -r entry; do
    bucket=$( jq -r '.length_bucket' <<<"$entry")
    tok=$(    jq -r '.token_target'  <<<"$entry")
    chars=$(  jq -r '.char_count'    <<<"$entry")
    dec="$DECODE"
    [[ -z "$dec" ]] && dec=$(jq -r '.decode_targets[0]' <<<"$entry")
    msgs=$(messages_for "$entry")

    walls=()
    prefill=""
    decode_ms=""
    for s in $(seq 1 "$SHOTS"); do
        out="$WORKDIR/${bucket}-s${s}.json"
        w=$(one_shot "$msgs" "$dec" "$out"); walls+=("$w")
        if [[ "$s" == "1" ]]; then
            prefill=$(  jq -r '(.usage.timings.prefill_ms // .timings.prefill_ms // "")' "$out")
            decode_ms=$(jq -r '(.usage.timings.decode_ms  // .timings.decode_ms  // "")' "$out")
        fi
    done
    med=$(printf '%s\n' "${walls[@]}" | median)
    per_tok=""
    [[ -n "$decode_ms" && "$decode_ms" != "null" && "$dec" -gt 0 ]] && \
        per_tok=$(awk -v d="$decode_ms" -v n="$dec" 'BEGIN{printf "%.2f", d/n}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$bucket" "$tok" "$chars" "$dec" \
        "$med" "$prefill" "$decode_ms" "$per_tok"
done <<<"$ENTRIES" | column -t -s $'\t'

log "done. workdir: $WORKDIR"
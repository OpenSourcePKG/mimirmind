#!/usr/bin/env bash
#
# Back-to-back parity check for a mimirmind /v1/chat/completions endpoint.
#
# Runs a greedy (temperature 0, fixed seed) completion so the output is
# deterministic, then fingerprints it (md5). Use it to prove a deploy did
# not change behaviour — capture a baseline BEFORE, compare AFTER — without
# depending on the Spark being reachable for anything more than a curl.
# Read-only: it never mutates server state.
#
# Solves the verification half of roadmap 8.14: "is this a scripted, one-command
# parity verdict any operator can run" instead of a remembered ritual.
#
# Usage:
#   scripts/verify-parity.sh <url> [<url2>] [flags]
#     one url   -> print the fingerprint (a pre-deploy baseline)
#     two urls  -> compare them, exit 1 on mismatch (pre-vs-post, or A/B)
#
# Flags (env override in parens = default):
#   --model NAME    (MODEL=primary)
#   --prompt TEXT   (PROMPT="What is the capital of France? Answer in one word.")
#   --system TEXT   (SYSTEM="")               optional system message
#   --tokens N      (MAX_TOKENS=16)
#   --seed N        (SEED=42)
#   --key KEY       (MIMIRMIND_API_KEY)       bearer token if the endpoint needs auth
#   --secure        verify TLS (default: -k, self-signed accepted)
#   -h, --help
#
# A URL may be a bare origin (https://localhost:8080) — the path
# /v1/chat/completions is appended automatically.
#
set -euo pipefail

MODEL="${MODEL:-primary}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one word.}"
SYSTEM="${SYSTEM:-}"
MAX_TOKENS="${MAX_TOKENS:-16}"
SEED="${SEED:-42}"
API_KEY="${MIMIRMIND_API_KEY:-}"
CURL_TLS="-k"

urls=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2;;
        --prompt)  PROMPT="$2"; shift 2;;
        --system)  SYSTEM="$2"; shift 2;;
        --tokens)  MAX_TOKENS="$2"; shift 2;;
        --seed)    SEED="$2"; shift 2;;
        --key)     API_KEY="$2"; shift 2;;
        --secure)  CURL_TLS=""; shift;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 0;;
        -*)        echo "unknown flag: $1" >&2; exit 2;;
        *)         urls+=("$1"); shift;;
    esac
done

[[ ${#urls[@]} -ge 1 ]] || { echo "usage: $0 <url> [<url2>] [flags]  (see --help)" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 3; }

# Build the request body deterministically with jq (no quoting hazards).
build_body() {
    local msgs='[]'
    [[ -n "$SYSTEM" ]] && msgs="$(jq -nc --arg c "$SYSTEM" '[{role:"system",content:$c}]')"
    msgs="$(jq -nc --argjson m "$msgs" --arg c "$PROMPT" '$m + [{role:"user",content:$c}]')"
    jq -nc \
        --arg model "$MODEL" \
        --argjson messages "$msgs" \
        --argjson max_tokens "$MAX_TOKENS" \
        --argjson seed "$SEED" \
        '{model:$model, messages:$messages, temperature:0, top_p:1, seed:$seed,
          max_tokens:$max_tokens, stream:false}'
}

# Return "<md5>  <completion text>" for one endpoint.
fingerprint() {
    local url="$1"
    [[ "$url" == *"/v1/chat/completions" ]] || url="${url%/}/v1/chat/completions"
    local auth=()
    [[ -n "$API_KEY" ]] && auth=(-H "Authorization: Bearer $API_KEY")
    local resp content
    resp="$(build_body | curl $CURL_TLS -sS --max-time 300 \
        -H 'Content-Type: application/json' "${auth[@]}" \
        -X POST "$url" --data-binary @- )" || { echo "REQUEST_FAILED"; return 1; }
    content="$(jq -er '.choices[0].message.content' <<<"$resp" 2>/dev/null)" || {
        echo "BAD_RESPONSE: $(jq -c '.error // .' <<<"$resp" 2>/dev/null || echo "$resp" | head -c 200)"
        return 1
    }
    local md5; md5="$(printf '%s' "$content" | md5sum | cut -d' ' -f1)"
    printf '%s\t%s' "$md5" "$content"
}

log() { printf '\033[1;34m[parity]\033[0m %s\n' "$*"; }

log "model=$MODEL seed=$SEED tokens=$MAX_TOKENS  prompt=\"$PROMPT\""

if [[ ${#urls[@]} -eq 1 ]]; then
    out="$(fingerprint "${urls[0]}")" || { echo "$out" >&2; exit 1; }
    md5="${out%%$'\t'*}"; txt="${out#*$'\t'}"
    log "${urls[0]}"
    printf '  fingerprint: %s\n' "$md5"
    printf '  completion : %s\n' "$txt"
    exit 0
fi

# Two endpoints: compare.
a="$(fingerprint "${urls[0]}")" || { echo "A: $a" >&2; exit 1; }
b="$(fingerprint "${urls[1]}")" || { echo "B: $b" >&2; exit 1; }
amd5="${a%%$'\t'*}"; atxt="${a#*$'\t'}"
bmd5="${b%%$'\t'*}"; btxt="${b#*$'\t'}"
log "A ${urls[0]}  -> $amd5"
log "B ${urls[1]}  -> $bmd5"
if [[ "$amd5" == "$bmd5" ]]; then
    log "PARITY OK (bit-identical greedy completion)"
    printf '  completion: %s\n' "$atxt"
    exit 0
else
    log "PARITY MISMATCH"
    printf '  A: %s\n' "$atxt"
    printf '  B: %s\n' "$btxt"
    exit 1
fi
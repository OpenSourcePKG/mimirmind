#!/usr/bin/env bash
#
# Verifies that mimirmind's Munin attached-mode path still works after
# the GgufReader-split refactor (feat/backend-pool commit 1a01520).
# The refactor moved `loadTensorsIntoChunks` out of GgufReader.cpp into
# a separate translation unit in mimirmind_core_l0. Munin is the sole
# caller. If the split broke ownership semantics we'd see it as:
#   - missing tensor pointers after attach
#   - segfault on first request
#   - Munin healthz reporting mismatched fingerprint
#
# Run this against a live Munin daemon (L0_TARGET_HOST in prod).
#
# Usage:
#   scripts/verify-munin-attach.sh <worker_bin_path> <munin_socket>
#
# Example:
#   scripts/verify-munin-attach.sh \
#       /usr/local/bin/mimirmind \
#       /var/run/munin/munin.sock

set -euo pipefail

WORKER="${1:?worker binary path required}"
SOCKET="${2:?munin socket path required}"
MODEL_ID="${3:-primary}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

if [[ ! -x "$WORKER" ]]; then
    log "FAIL: worker binary not executable at $WORKER"
    exit 1
fi
if [[ ! -S "$SOCKET" ]]; then
    log "FAIL: munin socket not present at $SOCKET"
    exit 1
fi

log "worker: $WORKER"
log "socket: $SOCKET"
log "model:  $MODEL_ID"

# The worker exits cleanly if given `smoke` — it loads the model,
# runs the M4d/e generate step (backend-neutral), prints token count,
# and quits. That's enough to prove attach + tensor pointers + first
# forward all work.
log "running worker in attached-mode smoke ..."
if ! "$WORKER" smoke \
       --attach "unix:$SOCKET" \
       --model  "/models/${MODEL_ID}.gguf" \
       --prompt 'Hello.' \
       2>&1 | tee /tmp/mm-attach-smoke.log
then
    log "FAIL: attached-mode smoke did not exit 0"
    log "log tail:"
    tail -20 /tmp/mm-attach-smoke.log >&2
    exit 1
fi

# Sanity checks on the log — the interesting lines are:
#   MuninClient::attach for id='primary' returned ...
#   loadModelAttached ... ready
#   [M4d/M4e] starting full forward + decode
#   ... total time ...
for pattern in \
    "attaching to Munin" \
    "loadModelAttached" \
    "loadModel: ready" \
    "starting full forward" \
    "Project Well + Envoy smoke test passed"
do
    if ! grep -q "$pattern" /tmp/mm-attach-smoke.log; then
        log "FAIL: log does not contain expected marker: $pattern"
        exit 1
    fi
    log "  ✓ $pattern"
done

log "OK — Munin attached-mode smoke passed. GgufReader split is compatible."

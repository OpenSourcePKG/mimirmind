#!/usr/bin/env bash
# L0-IPC-Testrig — one-shot runner.
#
# Usage:
#   tools/l0-ipc-testrig.sh                 # uses ./build/l0_ipc_testrig
#   tools/l0-ipc-testrig.sh /path/to/binary # explicit binary
#
# Meant to be run inside the mimirmind builder container (which has
# /dev/dri passthrough and the L0 loader). Typical invocation from
# the repo root on the target host (L0_TARGET_HOST):
#
#   docker compose run --rm builder bash -lc '
#     cmake --build /src/build --target l0_ipc_testrig &&
#     /src/tools/l0-ipc-testrig.sh /src/build/l0_ipc_testrig
#   '
#
# Exits 0 on both processes reporting PASS, non-zero otherwise. Prints
# both logs to stderr for triage.

set -u
set -o pipefail

BINARY="${1:-./build/l0_ipc_testrig}"
SOCKET="${MIMIRMIND_L0IPC_SOCKET:-/tmp/l0-ipc-testrig.sock}"

if [[ ! -x "$BINARY" ]]; then
    echo "l0-ipc-testrig.sh: binary not executable at $BINARY" >&2
    exit 2
fi

rm -f "$SOCKET"

echo "=== L0-IPC-Testrig ===" >&2
echo "binary: $BINARY"        >&2
echo "socket: $SOCKET"        >&2

OWNER_LOG="$(mktemp -t l0ipc-owner.XXXXXX.log)"
ATTACHER_LOG="$(mktemp -t l0ipc-attacher.XXXXXX.log)"
trap 'rm -f "$OWNER_LOG" "$ATTACHER_LOG"' EXIT

# Owner in background, attacher in foreground so we get its exit code.
"$BINARY" owner "$SOCKET" > "$OWNER_LOG" 2>&1 &
OWNER_PID=$!

# Wait for the socket to appear (up to 5s). If the owner crashes
# before creating it, we bail early instead of hanging in attacher.
for _ in $(seq 1 50); do
    if [[ -S "$SOCKET" ]]; then
        break
    fi
    if ! kill -0 "$OWNER_PID" 2>/dev/null; then
        echo "l0-ipc-testrig.sh: owner exited before creating socket" >&2
        echo "--- owner log ---"                                      >&2
        cat "$OWNER_LOG"                                              >&2
        exit 1
    fi
    sleep 0.1
done

"$BINARY" attacher "$SOCKET" > "$ATTACHER_LOG" 2>&1
ATTACHER_RC=$?

wait "$OWNER_PID"
OWNER_RC=$?

echo "--- owner log ---"    >&2
cat "$OWNER_LOG"            >&2
echo "--- attacher log ---" >&2
cat "$ATTACHER_LOG"         >&2

echo "owner exit: $OWNER_RC   attacher exit: $ATTACHER_RC" >&2

if [[ $OWNER_RC -eq 0 && $ATTACHER_RC -eq 0 ]]; then
    echo "TESTRIG: PASS" >&2
    exit 0
fi
echo "TESTRIG: FAIL" >&2
exit 1
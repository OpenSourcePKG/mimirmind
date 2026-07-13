#!/usr/bin/env bash
# L0-IPC-Testrig — one-shot runner across all three USM allocation kinds.
#
# Usage:
#   tools/l0-ipc-testrig.sh                 # uses ./build/l0_ipc_testrig
#   tools/l0-ipc-testrig.sh /path/to/binary # explicit binary path
#
# Iterates through shared / host / device kinds. Prints per-kind
# PASS/FAIL and an overall summary. Meant to be run inside the
# mimirmind builder or runtime container (with /dev/dri passthrough
# and the L0 loader). Typical invocation on the target host:
#
#   docker run --rm --device /dev/dri --group-add 44 --group-add 104 \
#       --entrypoint /usr/local/bin/l0-ipc-testrig.sh \
#       docker-registry.example.com/mimirmind:latest \
#       /usr/local/bin/l0_ipc_testrig
#
# Exit code is 0 only if at least one variant reports PASS (any
# working kind is enough for M-Munin). Non-zero if all three FAIL.

set -u
set -o pipefail

BINARY="${1:-./build/l0_ipc_testrig}"
SOCKET="${MIMIRMIND_L0IPC_SOCKET:-/tmp/l0-ipc-testrig.sock}"

if [[ ! -x "$BINARY" ]]; then
    echo "l0-ipc-testrig.sh: binary not executable at $BINARY" >&2
    exit 2
fi

echo "=== L0-IPC-Testrig ==="  >&2
echo "binary: $BINARY"         >&2
echo "socket: $SOCKET"         >&2

any_pass=0
declare -A results

run_one() {
    local kind=$1
    echo                                                   >&2
    echo "========== kind = $kind ==========" >&2
    rm -f "$SOCKET"

    local owner_log attacher_log
    owner_log="$(mktemp -t l0ipc-${kind}-owner.XXXXXX.log)"
    attacher_log="$(mktemp -t l0ipc-${kind}-attacher.XXXXXX.log)"

    "$BINARY" owner "$SOCKET" --kind "$kind" > "$owner_log" 2>&1 &
    local owner_pid=$!

    for _ in $(seq 1 50); do
        if [[ -S "$SOCKET" ]]; then break; fi
        if ! kill -0 "$owner_pid" 2>/dev/null; then
            echo "owner exited before creating socket" >&2
            echo "--- owner log ---"                   >&2
            cat "$owner_log"                           >&2
            results[$kind]="FAIL (owner-early-exit)"
            rm -f "$owner_log" "$attacher_log"
            return 1
        fi
        sleep 0.1
    done

    "$BINARY" attacher "$SOCKET" --kind "$kind" > "$attacher_log" 2>&1
    local a_rc=$?

    wait "$owner_pid"
    local o_rc=$?

    echo "--- owner log ---"    >&2
    cat "$owner_log"            >&2
    echo "--- attacher log ---" >&2
    cat "$attacher_log"         >&2
    echo "owner exit: $o_rc   attacher exit: $a_rc" >&2

    rm -f "$owner_log" "$attacher_log"

    if [[ $o_rc -eq 0 && $a_rc -eq 0 ]]; then
        results[$kind]="PASS"
        any_pass=1
        return 0
    fi
    results[$kind]="FAIL (owner=$o_rc attacher=$a_rc)"
    return 1
}

run_one shared || true
run_one host   || true
run_one device || true

rm -f "$SOCKET"

echo                                                 >&2
echo "========== SUMMARY =========="                  >&2
for k in shared host device; do
    printf '  %-7s : %s\n' "$k" "${results[$k]:-not-run}" >&2
done

if [[ $any_pass -eq 1 ]]; then
    echo "TESTRIG: PASS (at least one kind works)" >&2
    exit 0
fi
echo "TESTRIG: FAIL (all three kinds failed)" >&2
exit 1
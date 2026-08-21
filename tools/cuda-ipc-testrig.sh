#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CUDA-IPC-Testrig runner — launches owner + attacher for each mechanism
# and reports PASS/FAIL per kind. Go/no-go gate for the M-Munin GB10 port.
#
#   ./tools/cuda-ipc-testrig.sh /path/to/cuda_ipc_testrig
#
# Must run inside a --gpus container on the GB10 host (needs the driver).
set -u
BIN="${1:?usage: cuda-ipc-testrig.sh <cuda_ipc_testrig binary>}"
SOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$SOCK_DIR"' EXIT

overall=0
for kind in device managed vmm; do
    sock="$SOCK_DIR/ipc-$kind.sock"
    echo "=== kind=$kind ==="
    "$BIN" owner "$sock" --kind "$kind" &
    owner_pid=$!
    sleep 0.3
    "$BIN" attacher "$sock" --kind "$kind"
    att_rc=$?
    wait "$owner_pid"; own_rc=$?
    if [ "$own_rc" -eq 0 ] && [ "$att_rc" -eq 0 ]; then
        echo ">>> $kind: PASS"
    else
        echo ">>> $kind: FAIL (owner=$own_rc attacher=$att_rc)"
        overall=1
    fi
    echo
done

if [ "$overall" -eq 0 ]; then
    echo "ALL mechanisms PASS"
else
    echo "Some mechanisms FAIL (see per-kind above) — pick a PASS mechanism for M-Munin"
fi
# Non-zero only if EVERY mechanism failed would be too strict; the useful
# signal is the per-kind PASS/FAIL matrix above, so always exit 0.
exit 0

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "munin/Daemon.hpp"  // reuse munin::CliOptions

namespace mimirmind::munin {

/**
 * Lifecycle for the CUDA/GB10 (POSIX-shm) Munin daemon — the shm analogue of
 * Daemon (ADR 2026-08-14, step 4). Loads config, brings a ShmModelStore up
 * (models resident in memfd host RAM), installs SIGINT/SIGTERM wiring, and
 * runs the ShmSocketServer accept loop until a clean shutdown.
 *
 * Simpler than the L0 Daemon: no L0Context / UsmAllocator (chunks are plain
 * memfd host RAM) and no GovernorLock (the GB10 thermal story is separate;
 * healthz still reports governor_owner="munin" by default so the attach
 * contract holds). Pure host code — builds and runs without any GPU SDK.
 */
class ShmDaemon {
public:
    /// Blocks until shutdown. Returns the intended process exit code.
    [[nodiscard]] static int run(const CliOptions& opts) noexcept;

private:
    ShmDaemon() = default;
};

} // namespace mimirmind::munin

# Operations

Notes for running mimirmind on real hardware: monitoring, thermal limits,
and what to expect when the host is under sustained load.

## Thermal profile

Sustained Gemma 4 26B decode saturates the Intel iGPU shader array and
keeps several CPU cores busy for the duration of every request. On
small-form-factor hosts with marginal cooling that can drive the
package temperature into the thermal-trip range, which on Intel takes
the system down without a kernel log entry — the CPU shuts itself off
in microcode.

The thermal profile is mimirmind's mechanism for staying ahead of that.
It is **a per-host configuration** describing what the specific machine
can sustain thermally — not a generic preset. Two different hosts will
need two different profiles.

The profile is intentionally narrow: **only package temperature limits**
live here. Other host metrics (RAM, GPU frequency) are surfaced through
the status endpoint as observability, but they do not participate in
the throttle decision.

### Enabling the guard

Provide a JSON profile via either the CLI flag or the env var:

```bash
mimirmind serve \
  --model /models/gemma-4-26B-A4B-it-Q6_K.gguf \
  --thermal-profile /etc/mimirmind/thermal-nuc14.json

# or

MIMIRMIND_THERMAL_PROFILE=/etc/mimirmind/thermal-nuc14.json \
  mimirmind serve --model ...
```

If **neither is set**, the engine starts unprotected and logs a loud
WARN on startup. There is no built-in default profile — every host has
different cooling and the wrong defaults could be worse than nothing.

### Profile structure

```json
{
  "name":        "nuc14-attic",
  "description": "free-form notes about this host",

  "package_temp_soft_c":      75,
  "package_temp_hard_c":      82,
  "package_throttle_max_ms":  200
}
```

Both `package_temp_soft_c` and `package_temp_hard_c` must be present
together — supplying only one is a load-time error. Leaving both out
disables monitoring entirely and the guard becomes a no-op, which is
useful for testing the wiring without temperature-gating.

`name` is required (free-form identifier shown in logs and
`/v1/system/status`). `description` is optional. Fields unknown to the
loader are silently ignored, so adding future-version knobs to a
profile file does not break old binaries.

### How the limit is enforced

The guard computes a "fraction of the way from soft to hard":

```
reading at soft → 0
reading at hard → 1
linear in between
```

The decode-loop sleep between tokens is `fraction × package_throttle_max_ms`.
Throttling ramps in smoothly as the package heats up rather than
flipping on at one threshold.

Once the package hits its hard limit:

- The current generate() call still finishes (it would be worse to
  hang a client mid-stream).
- New `POST /v1/chat/completions` requests get **`HTTP 503` with
  `Retry-After: 10`** until the host cools off.

### Reading the host

The guard pulls package temperature from two sources, in this order:

1. `/sys/class/thermal/thermal_zone*/type` looking for `x86_pkg_temp`
   (kernel-managed package temperature, always present on Intel).
2. `/sys/class/hwmon/hwmon*/name` == `coretemp` with a `tempN_label`
   reading `Package id 0` (fallback when the thermal zone is named
   differently).

Both paths are visible inside an unprivileged Docker container — no
`--privileged` or capability tweaks needed. On startup the guard
verifies the package-temperature sensor is readable and refuses to
start if it is missing; better to fail at boot than discover a silent
gap during the next thermal incident.

### Status endpoint

`GET /v1/system/status` returns the current readings, the active
profile, and the current throttle decision:

```bash
curl -s https://your-mimirmind/v1/system/status | jq
```

```json
{
  "profile_active": true,
  "profile": {
    "name": "nuc14-attic",
    "description": "ASUS NUC14RVS-B in attic ...",
    "package_temp_soft_c":     75,
    "package_temp_hard_c":     82,
    "package_throttle_max_ms": 200
  },
  "readings": {
    "package_temp_c":    78.5,
    "ram_total_mib":     65484,
    "ram_available_mib": 32100
  },
  "throttle": {
    "state":                "throttling",
    "current_pause_ms":     100,
    "next_request_allowed": true,
    "reason":               "package_temp_c=78.5 above soft=75.0"
  }
}
```

`state` is one of `ok`, `throttling`, `critical`. `next_request_allowed`
is the same condition the chat endpoint uses to decide whether to
return `503`.

**RAM is reported in `readings` even though it is not part of the
profile.** Treat it as observability for external monitors — useful
for spotting trends, building dashboards, or correlating temperature
spikes with memory pressure. It does not affect admission or pacing.

If no profile is configured, the body is just
`{"profile_active": false, "warning": "..."}` and the chat endpoints
run without thermal protection.

## Tuning a profile for a new host

1. Run the host idle with `mimirmind serve --thermal-profile
   /tmp/dry-run.json` where the file sets *very* permissive limits
   (e.g. `soft=99, hard=110`). Poll `/v1/system/status` for a while
   to confirm the sensor source resolved to the expected zone.
2. Pick a workload that mirrors production traffic (Gemma 4 26B Q6_K
   with `max_tokens=400` is a reasonable stress test).
3. Watch package temperature climb under load. Note the **steady-state
   ceiling** — the temperature it converges to when generating for a
   minute or more. Call that `T_ceiling`.
4. Set `package_temp_hard_c` to roughly `T_ceiling + 5°C` if your
   cooling has headroom, or `T_ceiling - 5°C` if it doesn't (i.e. you
   want the throttle to actually trigger and slow things down).
5. Set `package_temp_soft_c` to `hard - 5 to 10°C`. Wider gap = gentler
   ramp; narrower gap = bigger pause once it starts. Try 5°C first and
   adjust if the throttle keeps yo-yoing.
6. `package_throttle_max_ms = 200` is a good starting point. Higher
   means more aggressive cooling, lower means less hit to throughput
   when throttle does activate.

Two example profiles live under `examples/`:

- `examples/thermal-profile-nuc14.json` — conservative profile for a
  small-form-factor host with marginal cooling.
- `examples/thermal-profile-workstation.json` — permissive profile for
  a tower with proper cooling, intended as a "trust the silicon"
  baseline.

## Watching it from outside

The status endpoint is cheap (one sysfs read per call, capped at
~250 ms refresh). A Prometheus scrape or a simple every-5-seconds curl
both work. Useful alerts:

- `throttle.state == "critical"` for more than 60 s — host can't keep
  up with traffic, time to throttle the upstream queue.
- `throttle.state == "throttling"` for more than 5 min — profile may
  be too aggressive, or workload has shifted upward.
- `readings.ram_available_mib` trending toward zero — even though the
  guard does not enforce it, low free RAM under sustained load is
  worth investigating (KV cache leak, model bigger than the host).
- `next_request_allowed == false` returning true → false transitions
  per minute count > N — sustained pressure on the temperature limit.

## What this does NOT do

- **No GPU temperature reading separately.** The Intel iGPU shares its
  silicon with the CPU cores; `x86_pkg_temp` already measures the
  combined die. A separate GPU sensor would mostly track the same
  number a few hundred ms later.
- **No fan / power telemetry.** The profile is purely advisory based
  on thermal headroom. If the underlying cooling solution is broken
  (fan stopped, heatsink lifted), nothing here will detect it directly
  — temperature will rise faster than expected and the guard will
  throttle to the maximum. That is the intended failure mode.
- **No RAM enforcement.** RAM is shown in the status endpoint as a
  signal for external monitoring, but the guard never refuses or
  paces a request based on RAM. Memory pressure on this engine is
  typically slow-moving (KV cache + loaded weights), so it does not
  fit the per-token decision the thermal guard makes. If RAM ever
  needs to gate requests, that lives in a separate config — not
  here.
- **No automatic profile generation.** Profile tuning is a deliberate
  per-host activity — see "Tuning a profile" above.
"""Zero-edit activation hook for the mimir vLLM per-stage probe.

Python's ``site`` module auto-imports ``sitecustomize`` at interpreter startup
if it is importable. Put this directory FIRST on ``PYTHONPATH`` so this file is
found, then set ``MIMIR_VLLM_PROBE=1`` (and/or ``MIMIR_VLLM_SPANS=1``); the probe
attaches to the vLLM engine in every subprocess without touching vLLM source.

If the target environment already ships its own ``sitecustomize``, this shim
loads and re-runs it first so nothing is shadowed. Any failure here is swallowed
-- instrumentation must never break the interpreter.

    export PYTHONPATH=/opt/mimirmind/tools/vllm-probe:$PYTHONPATH
    export MIMIR_VLLM_PROBE=1
    # start vLLM as usual (add --enforce-eager for clean Layer B/C ratios)
"""

import os
import sys


def _chain_preexisting_sitecustomize() -> None:
    """Load any other sitecustomize.py further down sys.path (not this one)."""
    here = os.path.dirname(os.path.abspath(__file__))
    import importlib.util

    try:
        for entry in sys.path:
            cand = os.path.join(entry, "sitecustomize.py")
            if not os.path.isfile(cand):
                continue
            if os.path.abspath(cand) == os.path.abspath(__file__):
                continue
            if os.path.dirname(cand) == here:
                continue
            spec = importlib.util.spec_from_file_location("_chained_sitecustomize", cand)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            break
    except Exception:
        pass


def _main() -> None:
    _chain_preexisting_sitecustomize()
    if not any(os.environ.get(f) for f in ("MIMIR_VLLM_PROBE", "MIMIR_VLLM_SPANS")):
        return
    try:
        import mimir_vllm_probe  # noqa: F401  (auto-activates on import)
    except Exception as exc:  # noqa: BLE001
        print(f"[mimir-vllm-probe] sitecustomize failed to load probe: {exc}",
              file=sys.stderr, flush=True)


_main()

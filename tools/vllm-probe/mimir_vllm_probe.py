"""Import-activated per-stage instrumentation for a vLLM engine (v0.22.1 / Qwen3-Next).

Purpose (mimirmind roadmap 5.18.7): measure, stage-for-stage on the running vLLM
engine, *how many ms* each decode/prefill stage costs on the GB10 box, so a true
apples-to-apples delta table against mimirmind's own `decode-prof` (stage names
mirrored below) can be built. The existing dispatch map says *which* kernel vLLM
picks; this says *how long* it takes.

Design: this module never edits vLLM source. On import it monkeypatches a small,
data-driven registry of target methods, wrapping each with:

  * Layer A  -- a named ``record_function`` span (graph-safe; shows up in the
               Kineto trace produced by ``VLLM_TORCH_PROFILER_DIR``). Active
               whenever ``MIMIR_VLLM_SPANS=1`` (or ``MIMIR_VLLM_PROBE=1``).
  * Layer B  -- ``torch.cuda.Event`` start/stop pairs that accumulate per-stage
               microseconds and periodically print a line mirroring mimirmind's
               ``decode-prof`` output. Active when ``MIMIR_VLLM_PROBE=1``.
  * Layer C  -- for the heavy kernel launch sites, an estimated bytes-moved figure
               is attached so the printed line also reports achieved GB/s, directly
               comparable to the microbench ``%peak`` (273 GB/s GB10 peak).

Because the target tree is not visible from the dev host, the registry is
defensive: every bind logs success/failure so the box operator can see exactly
which stages attached and correct any qualname that drifted between vLLM builds.

Activation (box-side, zero source edit):
    export PYTHONPATH=/path/to/this/dir:$PYTHONPATH   # dir contains this file + the .pth
    export MIMIR_VLLM_PROBE=1                         # Layer B/C live timing log
    export MIMIR_VLLM_SPANS=1                         # Layer A spans (implied by PROBE)
    export MIMIR_VLLM_PROBE_EVERY=64                  # print aggregate every N steps
    # then start the vLLM server / offline LLM as usual (Layer B/C: use --enforce-eager)

Or, if PYTHONPATH auto-activation is not desired, add a single line to the vLLM
entrypoint:  ``import mimir_vllm_probe; mimir_vllm_probe.activate()``

Constraints honored (see Synaipse lessons): no ncu, single GPU container, vLLM only.
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Configuration (env-gated)                                                    #
# --------------------------------------------------------------------------- #


def _flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("", "0", "false", "no", "off")


_PROBE = _flag("MIMIR_VLLM_PROBE")            # Layer B/C: live cuda.Event timing log
_SPANS = _flag("MIMIR_VLLM_SPANS") or _PROBE  # Layer A: record_function spans
_EVERY = int(os.environ.get("MIMIR_VLLM_PROBE_EVERY", "64") or "64")
_VERBOSE = _flag("MIMIR_VLLM_PROBE_VERBOSE")


def _log(msg: str) -> None:
    print(f"[mimir-vllm-probe] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Torch / vLLM span primitive                                                  #
# --------------------------------------------------------------------------- #

try:
    import torch
except Exception:  # pragma: no cover - torch is always present in a vLLM env
    torch = None  # type: ignore


def _resolve_record_function() -> Callable[[str], "object"]:
    """Prefer vLLM's ``record_function_or_nullcontext`` (no-op unless profiling),
    fall back to ``torch.profiler.record_function``, then to a nullcontext."""
    try:
        from vllm.v1.utils import record_function_or_nullcontext  # type: ignore

        return record_function_or_nullcontext
    except Exception:
        pass
    try:
        from torch.profiler import record_function  # type: ignore

        return record_function
    except Exception:
        @contextmanager
        def _null(_name: str):
            yield

        return _null


_record_function = _resolve_record_function()


# --------------------------------------------------------------------------- #
# Per-stage accumulator (Layer B/C)                                            #
# --------------------------------------------------------------------------- #


class _StageAccumulator:
    """Accumulates per-stage GPU time via cuda.Event pairs and prints a periodic
    aggregate that mirrors mimirmind's ``decode-prof`` line format."""

    def __init__(self, every: int) -> None:
        self._every = max(1, every)
        self._pending: list[tuple[str, "torch.cuda.Event", "torch.cuda.Event", Optional[int]]] = []
        self._sum_ms: "OrderedDict[str, float]" = OrderedDict()
        self._bytes: "OrderedDict[str, int]" = OrderedDict()
        self._count: "OrderedDict[str, int]" = OrderedDict()
        self._steps = 0
        self._t_wall0 = time.perf_counter()

    # -- timing lifecycle -------------------------------------------------- #
    @contextmanager
    def time(self, stage: str, nbytes: Optional[int] = None):
        if torch is None or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            stop.record()
            self._pending.append((stage, start, stop, nbytes))

    def step_boundary(self) -> None:
        """Call once per decode step (or per forward) to flush pending events.

        Synchronizing here serializes the stream (like mimirmind's decode-prof),
        so absolute tok/s is depressed -- the *ratios* between stages stay valid.
        """
        if not self._pending:
            return
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            for stage, start, stop, nbytes in self._pending:
                ms = start.elapsed_time(stop)
                self._sum_ms[stage] = self._sum_ms.get(stage, 0.0) + ms
                self._count[stage] = self._count.get(stage, 0) + 1
                if nbytes:
                    self._bytes[stage] = self._bytes.get(stage, 0) + nbytes
        self._pending.clear()
        self._steps += 1
        if self._steps % self._every == 0:
            self._emit()

    # -- reporting --------------------------------------------------------- #
    def _emit(self) -> None:
        if not self._sum_ms:
            return
        n = max(1, self._steps)
        total = sum(self._sum_ms.values()) / n
        parts = []
        for stage, ms_sum in self._sum_ms.items():
            per = ms_sum / n
            frag = f"{stage}={per:.3f}ms"
            nb = self._bytes.get(stage)
            if nb:
                gbps = (nb / n) / (per * 1e-3) / 1e9 if per > 0 else 0.0
                frag += f"({gbps:.0f}GB/s)"
            parts.append(frag)
        _log(
            f"decode-prof steps={self._steps} total={total:.3f}ms/step  "
            + "  ".join(parts)
        )

    def report_final(self) -> None:
        if self._steps == 0 and not self._sum_ms:
            return
        _log("=== final per-stage aggregate ===")
        self._emit()


_accum = _StageAccumulator(_EVERY) if _PROBE else None


# --------------------------------------------------------------------------- #
# Byte-cost estimators for Layer C (GB/s-to-beat)                             #
# --------------------------------------------------------------------------- #


def _numel_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        try:
            total += t.numel() * t.element_size()
        except Exception:
            pass
    return total


# --------------------------------------------------------------------------- #
# Wrapping machinery                                                           #
# --------------------------------------------------------------------------- #


def _wrap_callable(orig: Callable, stage: str, byte_fn: Optional[Callable] = None) -> Callable:
    """Wrap ``orig`` so each call opens a record_function span and (if PROBE)
    a cuda.Event-timed region tagged ``stage``."""

    def wrapper(*args, **kwargs):
        nbytes = None
        if byte_fn is not None and _accum is not None:
            try:
                nbytes = byte_fn(*args, **kwargs)
            except Exception:
                nbytes = None
        if _SPANS and _accum is not None:
            with _record_function(f"mimir.{stage}"), _accum.time(stage, nbytes):
                return orig(*args, **kwargs)
        if _SPANS:
            with _record_function(f"mimir.{stage}"):
                return orig(*args, **kwargs)
        if _accum is not None:
            with _accum.time(stage, nbytes):
                return orig(*args, **kwargs)
        return orig(*args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", stage)
    wrapper.__qualname__ = getattr(orig, "__qualname__", stage)
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
    wrapper._mimir_stage = stage  # type: ignore[attr-defined]
    return wrapper


def _patch_method(qualpath: str, stage: str, byte_fn: Optional[Callable] = None) -> bool:
    """qualpath = 'module.path:ClassName.method' or 'module.path:function'."""
    try:
        mod_name, attr_path = qualpath.split(":", 1)
        import importlib

        mod = importlib.import_module(mod_name)
        parts = attr_path.split(".")
        owner = mod
        for p in parts[:-1]:
            owner = getattr(owner, p)
        leaf = parts[-1]
        orig = getattr(owner, leaf)
        if getattr(orig, "_mimir_stage", None) is not None:
            return True  # already patched (idempotent)
        setattr(owner, leaf, _wrap_callable(orig, stage, byte_fn))
        if _VERBOSE:
            _log(f"bound  {stage:<16} <- {qualpath}")
        return True
    except Exception as exc:  # noqa: BLE001 - defensive across vLLM builds
        _log(f"MISS   {stage:<16} <- {qualpath}  ({type(exc).__name__}: {exc})")
        return False


# --------------------------------------------------------------------------- #
# Step-boundary hook                                                           #
# --------------------------------------------------------------------------- #


def _install_step_boundary() -> bool:
    """Flush the accumulator once per top-level model forward.

    Wraps ``Qwen3NextForCausalLM.forward`` -- the per-token (decode) / per-batch
    (prefill) entry point -- calling ``step_boundary`` after it returns.
    """
    if _accum is None:
        return False
    # Inner model .forward runs exactly once per decode step / prefill batch.
    # Try the arch actually in use first, fall back to related variants.
    candidates = [
        ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model"),
        ("vllm.model_executor.models.qwen3_next", "Qwen3NextModel"),
        ("vllm.model_executor.models.qwen3_next", "Qwen3NextForCausalLM"),
    ]
    import importlib

    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        orig = cls.forward
        if getattr(orig, "_mimir_boundary", None):
            return True

        def _make(orig_fwd):
            def fwd(*args, **kwargs):
                out = orig_fwd(*args, **kwargs)
                try:
                    _accum.step_boundary()
                except Exception:
                    pass
                return out
            fwd._mimir_boundary = True  # type: ignore[attr-defined]
            return fwd

        cls.forward = _make(orig)
        if _VERBOSE:
            _log(f"step-boundary bound <- {mod_name}:{cls_name}.forward")
        return True
    _log("step-boundary hook MISS (no candidate model class found)")
    return False


# --------------------------------------------------------------------------- #
# Target registry -- stage names mirror mimirmind decode-prof                  #
# --------------------------------------------------------------------------- #
#
# (qualpath, mimir-stage, optional byte-estimator). Sub-module forwards are
# wrapped at the class level so their spans nest under the DecoderLayer span.
# Names are from the 5.18.7 plan note; MISS lines on the box show what drifted.

_TARGETS: list[tuple[str, str, Optional[Callable]]] = [
    # -- top / layer scaffolding ------------------------------------------ #
    # Qwen3.6/3.5 MoE (arch Qwen3_5MoeForConditionalGeneration) loads from the
    # qwen3_5 module, whose Qwen3_5DecoderLayer *reuses* the qwen3_next
    # Attention / GatedDeltaNet / SparseMoeBlock classes targeted below.
    ("vllm.model_executor.models.qwen3_5:Qwen3_5DecoderLayer.forward", "layer", None),
    # -- attention path ---------------------------------------------------- #
    # self_attn / linear_attn are attributes; wrap their classes' forward.
    # (qkv/o dense projections live inside these forwards, so no separate span.)
    ("vllm.model_executor.models.qwen3_next:Qwen3NextAttention.forward", "attn", None),
    ("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn:"
     "QwenGatedDeltaNetAttention.forward", "gdn", None),
    # -- MoE / MLP --------------------------------------------------------- #
    ("vllm.model_executor.models.qwen3_next:Qwen3NextSparseMoeBlock.forward", "moe", None),
    # -- Layer C: heavy kernel launch sites (GB/s-to-beat) ----------------- #
    ("vllm.model_executor.layers.fused_moe.experts.marlin_moe:"
     "MarlinExperts.apply", "moe.gemm", None),
    ("vllm.v1.attention.backends.flash_attn:FlashAttentionImpl.forward",
     "attn.flash", None),
]


# --------------------------------------------------------------------------- #
# Activation                                                                   #
# --------------------------------------------------------------------------- #

_ACTIVATED = False


def activate() -> None:
    global _ACTIVATED
    if _ACTIVATED:
        return
    if not (_SPANS or _PROBE):
        return  # nothing requested
    _ACTIVATED = True
    _log(
        f"activating (spans={_SPANS} probe={_PROBE} every={_EVERY}). "
        "Stage names mirror mimirmind decode-prof."
    )
    bound = 0
    for qualpath, stage, byte_fn in _TARGETS:
        if _patch_method(qualpath, stage, byte_fn):
            bound += 1
    _install_step_boundary()
    _log(f"bound {bound}/{len(_TARGETS)} targets. "
         "MISS lines above = qualname drift, edit _TARGETS to match this build.")
    if _accum is not None:
        atexit.register(_accum.report_final)


# Auto-activate on import when any flag is set (supports .pth / PYTHONPATH use).
if _SPANS or _PROBE:
    activate()

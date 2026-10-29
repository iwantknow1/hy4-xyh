#!/usr/bin/env python3
from __future__ import annotations

"""Shared output primitives and helpers for workspace skills and tooling.

Every skill wrapper should emit progress on stderr and a final JSON object on
stdout.  This module provides the canonical implementations so we never
accidentally diverge.  Each skill customises only the *sentinel* prefix;
everything else is identical.

TODO: Move it to a shared common directory once one is established
      These are currently scattered across each skill and should be
      consolidated into a single shared directory.
"""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    """Return an ISO-8601 UTC timestamp string (seconds precision, ``Z`` suffix)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def print_json(data: dict[str, Any]) -> None:
    """Print *data* as a single indented JSON object on stdout."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def emit_progress(
    phase: str,
    message: str,
    *,
    sentinel: str,
    **extra: Any,
) -> None:
    """Write a ``<sentinel><json>`` progress line to stderr.

    Every skill must supply its own ``sentinel`` (e.g.
    ``__WORKSPACE_SERVING_PROGRESS__=``) so the agent harness can route
    progress lines per skill.
    """
    payload: dict[str, Any] = {"phase": phase, "message": message}
    payload.update({k: v for k, v in extra.items() if v is not None})
    sys.stderr.write(
        sentinel + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Lazy loader for the serving skill's _common module
# ---------------------------------------------------------------------------

def load_serving_common(*, module_alias: str, serving_scripts_dir: Path | None = None):
    """Lazily load the serving skill's ``_common.py`` as *module_alias*.

    Returns the loaded module.  Safe to call multiple times — the module is
    cached in ``sys.modules`` on first call.  Benchmark and profiling skills
    both use this so they share SSH primitives without polluting
    ``sys.path``.
    """
    if module_alias in sys.modules:
        return sys.modules[module_alias]

    if serving_scripts_dir is None:
        serving_scripts_dir = (
            Path(__file__).resolve().parents[5]
            / "skills" / "inference-model-skills" / "deployment" / "vllm-ascend-serving" / "scripts"
        )
    src = serving_scripts_dir / "_common.py"
    spec = importlib.util.spec_from_file_location(module_alias, src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load serving common helpers from {src}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module

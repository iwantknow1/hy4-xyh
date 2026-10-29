#!/usr/bin/env python3
from __future__ import annotations

"""POST /start_profile or /stop_profile against a running vllm-ascend service.

Profiler control is part of the *profiling collection* workflow, not part of
service lifecycle, so this client lives inside the profiling-collection skill
rather than the serving skill. The serving skill is intentionally agnostic:
it just passes ``--profiler-config`` through to ``vllm serve`` and never
touches the profiler window.

The POST is executed inside the container via SSH against
``http://127.0.0.1:<port>``, so no SSH tunnel is required. The port is read
from the serving skill's recorded state for that machine; a service must
already be running.

Multi-rank torch profiler setup/finalization can take much longer than an
ordinary inference request, so the timeout is long by default.

Usage:
    python3 profile_control.py --host <ip> --container <name> --action start_profile
    python3 profile_control.py --host <ip> --container <name> --action stop_profile [--timeout 900]

Progress on stderr, final JSON on stdout.
"""

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    add_ssh_connection_args,
    emit_progress,
    load_serving_state,
    print_json,
    resolve_execution_target,
    ssh_exec_target,
)

DEFAULT_TIMEOUT_SECONDS = 600
ALLOWED_ACTIONS = ("start_profile", "stop_profile")


def post_remote_action(target, port: int, action: str, timeout: int,
                       *, conda_env: str | None = None) -> dict[str, Any]:
    """POST http://127.0.0.1:<port>/<action> from inside the container.

    *target* is an ``ExecutionTarget`` (from serving's _common).  When the
    target has a container name set the python script is automatically
    wrapped with ``docker exec``.

    Returns a dict with: ok, status, body. Raises RuntimeError on transport
    failure (SSH problem, container Python missing, etc.).
    """
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action {action!r}; allowed: {ALLOWED_ACTIONS}")

    py = (
        "import json, urllib.error, urllib.request\n"
        f"url = 'http://127.0.0.1:{port}/{action}'\n"
        "req = urllib.request.Request(url, data=None,\n"
        "    headers={'Content-Type': 'application/json'}, method='POST')\n"
        "try:\n"
        f"    with urllib.request.urlopen(req, timeout={int(timeout)}) as resp:\n"
        "        body = resp.read().decode('utf-8', errors='replace')\n"
        "        print(json.dumps({'ok': True, 'status': resp.status, 'body': body}))\n"
        "except urllib.error.HTTPError as exc:\n"
        "    body = exc.read().decode('utf-8', errors='replace')\n"
        "    print(json.dumps({'ok': False, 'status': exc.code, 'body': body}))\n"
        "    raise SystemExit(2)\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
        "    raise SystemExit(3)\n"
    )
    if conda_env:
        script = f"conda run -n {shlex.quote(conda_env)} --no-capture-output /codex/.venv/bin/python -c {shlex.quote(py)}"
    else:
        script = f"/codex/.venv/bin/python -c {shlex.quote(py)}"
    result = ssh_exec_target(target, script, check=False)

    payload: dict[str, Any] | None = None
    stdout = (result.stdout or "").strip()
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = None

    if result.returncode != 0:
        detail: dict[str, Any] = payload or {}
        detail.setdefault("stdout", (result.stdout or "")[-2000:])
        detail.setdefault("stderr", (result.stderr or "")[-2000:])
        detail["rc"] = result.returncode
        raise RuntimeError(
            f"remote {action} on 127.0.0.1:{port} failed: "
            f"{json.dumps(detail, ensure_ascii=False)}"
        )

    return payload or {"ok": True, "status": 200, "body": ""}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    # ---- direct SSH connection ----
    add_ssh_connection_args(p)
    p.add_argument(
        "--action",
        required=True,
        choices=ALLOWED_ACTIONS,
        help="profiler control action to issue",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Timeout in seconds for the control-plane HTTP request. "
            "Multi-rank torch-profiler setup/finalization can take much longer "
            "than an ordinary inference request."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        target = resolve_execution_target(
            host=args.host,
            port=args.ssh_port,
            user=args.user,
            password=args.password,
            key_file=args.key_file,
            container=args.container,
            conda_env=getattr(args, "conda_env", None),
            connection_mode=getattr(args, "connection_mode", "docker"),
        )
        alias = target.alias

        state = load_serving_state(
            alias,
            state_repo_root=target.state_repo_root,
        )
        if state is None:
            print_json({
                "status": "not_found",
                "machine": alias,
                "action": args.action,
                "message": (
                    f"no serving state recorded for {alias}; start the service "
                    "via vllm-ascend-serving first"
                ),
            })
            return 2
        port = state.get("port")
        if not port:
            print_json({
                "status": "not_found",
                "machine": alias,
                "action": args.action,
                "message": "serving state has no port; service may have failed to launch",
            })
            return 2

        emit_progress(
            "profile_control",
            f"posting {args.action} to 127.0.0.1:{port}",
            timeout=args.timeout,
        )
        result = post_remote_action(
            target, int(port), args.action, args.timeout,
            conda_env=getattr(args, "conda_env", None),
        )

        ok = bool(result.get("ok"))
        print_json({
            "status": "ok" if ok else "failed",
            "machine": alias,
            "action": args.action,
            "port": port,
            "http_status": result.get("status"),
            "body": result.get("body"),
            "error": result.get("error"),
        })
        return 0 if ok else 1

    except Exception as exc:
        print_json({
            "status": "failed",
            "host": getattr(args, "host", None),
            "action": getattr(args, "action", None),
            "error": str(exc),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

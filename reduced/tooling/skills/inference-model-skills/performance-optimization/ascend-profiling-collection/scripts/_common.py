#!/usr/bin/env python3
from __future__ import annotations

"""Shared utilities for ascend-profiling-collection scripts.

This module owns helpers that exist *because of profiling*: the SSH +
ascend-env preamble, the local SSH tunnel for sending workload requests, the
profile-control client, and progress / state-dir conventions.

SSH primitives are reused from the serving skill's
``_common`` so we do not maintain a second copy.
"""

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
SERVING_SCRIPTS = ROOT / "skills" / "inference-model-skills" / "deployment" / "vllm-ascend-serving" / "scripts"

# TODO: Move it to a shared common directory once one is established
#       These are currently scattered across each skill and should be
#       consolidated into a single shared directory.

if str(SERVING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SERVING_SCRIPTS))

PROGRESS_SENTINEL = "__WORKSPACE_PROFILING_COLLECTION_PROGRESS__="
COLLECTION_STATE_DIR = ROOT / ".workspace-local" / "ascend-profiling-collection" / "runs"


# ---------------------------------------------------------------------------
# Lazy import of serving _common (single source of truth for SSH primitives)
# ---------------------------------------------------------------------------

from workspace_output import load_serving_common, emit_progress as _emit_progress, print_json, now_utc  # noqa: E402

SERVING = load_serving_common(
    module_alias="workspace_profcoll_serving_common",
    serving_scripts_dir=SERVING_SCRIPTS,
)
SshEndpoint = SERVING.SshEndpoint
ssh_exec = SERVING.ssh_exec
ssh_exec_target = SERVING.ssh_exec_target
_ssh_base_cmd = SERVING._ssh_base_cmd
_ssh_env_for_password = SERVING._ssh_env_for_password
resolve_execution_target = SERVING.resolve_execution_target
resolve_worker_target = SERVING.resolve_worker_target
load_serving_state = SERVING.load_serving_state
ascend_env_preamble = SERVING.ascend_env_preamble
safe_token = SERVING.safe_token
add_ssh_connection_args = SERVING.add_ssh_connection_args


# ---------------------------------------------------------------------------
# Progress / output  (imported from shared lib, sentinel wired here)
# ---------------------------------------------------------------------------



emit_progress = partial(_emit_progress, sentinel=PROGRESS_SENTINEL)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_collection_run_dir(
    *,
    tag: str,
    host: str | None = None,
) -> Path:
    target_token = safe_token(host or "target", fallback="target", max_len=80)
    tag_token = safe_token(tag, fallback="profile", max_len=80)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(10):
        name = (
            f"{ts}_{tag_token}_{target_token}_"
            f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        run_dir = COLLECTION_STATE_DIR / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("failed to allocate a unique profiling collection run directory")


# ---------------------------------------------------------------------------
# Ascend env preamble (mirrors the one in benchmark/_common.py)
# ---------------------------------------------------------------------------

# Backward-compatible alias for callers that don't pass cann_dir yet
ASCEND_ENV_PREAMBLE = SERVING.ascend_env_preamble()


# ---------------------------------------------------------------------------
# Local SSH tunnel for sending workload requests from the local machine
# ---------------------------------------------------------------------------

def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def open_local_tunnel(
    ep,
    remote_port: int,
    *,
    password: str | None = None,
    key_file: str | None = None,
):
    """Open an ephemeral ``ssh -L`` tunnel to ``127.0.0.1:<remote_port>``.

    Yields a dict with ``local_port`` and ``base_url``. Used by the workload
    sender so multimodal payloads (image data URLs) can be assembled locally
    and POSTed without round-tripping through SSH heredocs.

    When *password* or *key_file* is provided they are forwarded to the SSH
    transport so the tunnel works with direct Docker connections that require
    password auth.
    """
    local_port = _find_free_local_port()
    cmd = [
        *_ssh_base_cmd(ep, password=password, key_file=key_file),
        "-o", "ExitOnForwardFailure=yes",
        "-N",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
    ]

    env = dict(os.environ)
    if password:
        env.update(_ssh_env_for_password(password))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.time() + 15
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr is not None else ""
                raise RuntimeError(
                    f"ssh tunnel exited early (rc={proc.returncode}): {stderr[:2000]}"
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                try:
                    sock.connect(("127.0.0.1", local_port))
                    yield {
                        "local_port": local_port,
                        "base_url": f"http://127.0.0.1:{local_port}",
                    }
                    return
                except OSError as exc:
                    last_error = str(exc)
                    time.sleep(0.2)
        raise RuntimeError(
            f"timed out waiting for ssh tunnel to localhost:{local_port} ({last_error})"
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# JSON-emitting subprocess wrapper for sibling scripts
# ---------------------------------------------------------------------------

def call_json_command(cmd: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    """Run ``cmd`` and parse its stdout as JSON.

    Stderr is always relayed so the agent sees progress markers from the
    underlying serving scripts. Raises RuntimeError on non-zero exit
    or non-JSON output, with both streams attached.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd or ROOT),
    )
    stderr_lines: list[str] = []

    def relay_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    thread = threading.Thread(target=relay_stderr, daemon=True)
    thread.start()
    assert proc.stdout is not None
    stdout = proc.stdout.read()
    returncode = proc.wait()
    thread.join(timeout=1)
    stderr = "".join(stderr_lines)
    if returncode != 0:
        raise RuntimeError(
            f"command failed (rc={returncode}): {' '.join(cmd)}\n"
            f"stdout={stdout[:2000]}\nstderr={stderr[:2000]}"
        )
    if not stdout.strip():
        raise RuntimeError(
            f"command produced no output: {' '.join(cmd)}\n"
            f"stderr={stderr[:2000]}"
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"command returned non-JSON output: {' '.join(cmd)}\n"
            f"stdout={stdout[:2000]}"
        ) from exc


# ---------------------------------------------------------------------------
# Convenience wrappers around the serving skill's CLI scripts
# ---------------------------------------------------------------------------

def call_serve_start(extra_args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, str(SERVING_SCRIPTS / "serve_start.py"), *extra_args]
    return call_json_command(cmd)


def call_serve_stop(
    *,
    host: str,
    ssh_port: int = 22,
    user: str = "root",
    password: str | None = None,
    key_file: str | None = None,
    container: str | None = None,
    conda_env: str | None = None,
    connection_mode: str = "docker",
    worker_key_file: str | None = None,
    worker_password: str | None = None,
    worker_container: str | None = None,
    worker_conda_env: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cmd = [sys.executable, str(SERVING_SCRIPTS / "serve_stop.py")]
    cmd.extend(["--host", host])
    if ssh_port != 22:
        cmd.extend(["--ssh-port", str(ssh_port)])
    if user != "root":
        cmd.extend(["--user", user])
    if password:
        cmd.extend(["--password", password])
    if key_file:
        cmd.extend(["--key-file", key_file])
    if container:
        cmd.extend(["--container", container])
    if conda_env:
        cmd.extend(["--conda-env", conda_env])
    if connection_mode != "docker":
        cmd.extend(["--mode", connection_mode])
    if model:
        cmd.extend(["--model", model])
    if worker_key_file:
        cmd.extend(["--worker-key-file", worker_key_file])
    if worker_password:
        cmd.extend(["--worker-password", worker_password])
    if worker_container:
        cmd.extend(["--worker-container", worker_container])
    if worker_conda_env:
        cmd.extend(["--worker-conda-env", worker_conda_env])
    if force:
        cmd.append("--force")
    return call_json_command(cmd)

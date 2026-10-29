#!/usr/bin/env python3
from __future__ import annotations

"""Start a vllm-ascend online service on a workspace-managed remote container.

Usage examples:

    # Fresh start with explicit params
    python3 serve_start.py --host 192.168.1.100 --container my-container \\
        --model /data/models/Qwen3-32B --tp 4 \\
        --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \\
        -- --max-model-len 4096

    # Relaunch with same config
    python3 serve_start.py --host 192.168.1.100 --container my-container --relaunch

    # Relaunch with a new env variable
    python3 serve_start.py --host 192.168.1.100 --container my-container \\
        --relaunch --extra-env VLLM_USE_V1=1

    # Relaunch and remove an old env
    python3 serve_start.py --host 192.168.1.100 --container my-container \\
        --relaunch --unset-env MY_DEBUG

Progress on stderr as __WORKSPACE_SERVING_PROGRESS__=<json>.
Final result on stdout as a single JSON object.
"""

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    check_alive,
    check_health,
    check_models,
    emit_progress,
    load_serving_state,
    load_cluster_state,
    save_cluster_state,
    now_utc,
    print_json,
    probe_npus,
    resolve_execution_target,
    save_serving_state,
    largest_contiguous_free,
    MACHINE_HARDWARE,
    select_devices,
    ssh_exec_target,
    write_serving_history,
    _target_auth,
    ExecutionTarget,
    parse_worker_hosts,
    resolve_worker_target,
    resolve_multi_node_dp_config,
    build_multi_node_dp_args,
)
from workspace_validate import parse_device_csv, require_env_name

RUNTIME_DIR_BASE = ".workspace-runtime/serving"
DEFAULT_HEALTH_TIMEOUT = 3600
DEFAULT_DP_RPC_TIMEOUT = 3600
HEALTH_POLL_INTERVAL = 5


def _safe_machine_segment(machine: str) -> str:
    """Sanitize a machine identifier so it is safe to embed in a directory name.

    Keeps ``[A-Za-z0-9_.-]`` (IPs and hostnames pass through unchanged); any
    other character becomes ``_``.  Empty/None fall back to ``unknown``.
    """
    if not machine:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", machine)


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

def find_free_port(target: ExecutionTarget, exclude: set[int] | None = None) -> int:
    """Find a free TCP port on the remote target, excluding *exclude* ports.

    The exclusion set is embedded into the remote python so two sequential
    allocations (e.g. service port + DP RPC port) can never collide.
    """
    exclude = exclude or set()
    excluded_json = json.dumps(sorted(exclude))
    script = (
        "/codex/.venv/bin/python -c \"\n"
        "import socket, random, json\n"
        "excluded = set(" + excluded_json + ")\n"
        "for _ in range(50):\n"
        "    port = random.randint(30000, 60000)\n"
        "    if port in excluded:\n"
        "        continue\n"
        "    try:\n"
        "        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "        s.bind(('0.0.0.0', port))\n"
        "        s.close()\n"
        "        print(json.dumps({'port': port}))\n"
        "        exit(0)\n"
        "    except OSError:\n"
        "        continue\n"
        "print(json.dumps({'error': 'no free port found'}))\n"
        "exit(1)\n"
        "\""
    )
    result = ssh_exec_target(target, script, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"port discovery failed: {result.stderr[:500]}")
    data = json.loads(result.stdout.strip())
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["port"]


def _parse_devices_csv(value: str) -> set[int]:
    if not value or not value.strip():
        return set()
    return set(parse_device_csv(value) or [])


# ---------------------------------------------------------------------------
# Launch script builder (the core escaping-safe layer)
# ---------------------------------------------------------------------------

def _normalize_json_arg(value: str) -> str:
    """Fix a serve-arg JSON value whose inner quotes were stripped by a
    calling shell (e.g. PowerShell dropping ``"`` inside an inline JSON CLI
    argument), so vLLM's ``--compilation_config`` still receives valid JSON.

    Leaves ordinary args and already-valid JSON untouched.
    """
    v = value.strip()
    if not (v.startswith("{") and v.endswith("}")):
        return value
    try:
        json.loads(v)
        return value
    except Exception:
        pass
    inner = v[1:-1].strip()
    if not inner:
        return value
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    def _quote(x: str) -> str:
        x = x.strip()
        if len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'":
            return '"' + x[1:-1].replace("\\\'", "'").replace('\\"', '"') + '"'
        if re.fullmatch(r"-?\d+(\.\d+)?", x):
            return x
        if x in ("true", "false", "null"):
            return x
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./\-]*", x):
            return '"' + x + '"'
        return x
    try:
        pieces: list[str] = []
        for part in parts:
            key, _, val = part.partition(":")
            pieces.append(_quote(key) + ": " + _quote(val))
        obj = json.loads("{" + ", ".join(pieces) + "}")
        return json.dumps(obj)
    except Exception:
        return value


def build_launch_script(
    *,
    runtime_dir: str,
    model: str,
    served_model_name: str,
    port: int,
    tp: int | None,
    dp: int | None,
    extra_env: dict[str, str],
    extra_args: list[str],
    multi_node_args: list[str] | None = None,
    wrap_script: str = "",
    conda_env: str | None = None,
    cann_dir: str = "/usr/local/Ascend",
    local_ip: str = "",
    comm_iface: str = "",
) -> str:
    """Generate a self-contained bash launch script for vLLM serve.

    When ``multi_node_args`` is non-empty, additionally auto-detects the
    correct NIC name and exports ``HCCL_IF_IP`` / ``GLOO_SOCKET_IFNAME`` /
    ``IP_SOCKET_IFNAME`` / ``HCCL_SOCKET_IFNAME`` for cross-node
    communication.
    """
    lines: list[str] = ["set -e"]
    lines.append('for _pybin in /usr/local/python3.12.13/bin /usr/local/python3/bin; do   if [ -d "$_pybin" ] && ! printf "%s" ":$PATH:" | grep -q ":$_pybin:"; then     export PATH="$_pybin:$PATH";   fi; done; ')

    lines.append(f"mkdir -p {shlex.quote(runtime_dir)}")

    # ---- CANN + ATB environment ----
    # Source the toolkit setup under ascend-toolkit (newer CANN layouts)
    # first, then the classic cann/set_env.sh, then fall back to a manual
    # LD_LIBRARY_PATH export so the driver stays visible.
    toolkit_setup = f"{cann_dir.rstrip('/')}/ascend-toolkit/set_env.sh"
    cann_setup = f"{cann_dir.rstrip('/')}/cann/set_env.sh"
    atb_setup = f"{cann_dir.rstrip('/')}/nnal/atb/set_env.sh"
    lines.append(
        f"if [ -f {shlex.quote(toolkit_setup)} ]; then"
        f"  set +u; source {shlex.quote(toolkit_setup)}; set -u;"
        f" elif [ -f {shlex.quote(cann_setup)} ]; then"
        f"  set +u; source {shlex.quote(cann_setup)}; set -u;"
        f" else"
        f"  export LD_LIBRARY_PATH="
        f'"{cann_dir.rstrip("/")}/driver/lib64/driver'
        f':{cann_dir.rstrip("/")}/driver/lib64'
        f'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}";'
        f" fi"
    )
    lines.append(
        f"if [ -f {shlex.quote(atb_setup)} ]; then"
        f"  set +u; source {shlex.quote(atb_setup)}; set -u;"
        f" fi"
    )

    # ---- conda env validation (conda mode only) ----
    if conda_env:
        lines.append(f"conda run -n {shlex.quote(conda_env)} /codex/.venv/bin/python -c 'print(\"conda_ok\")' > /dev/null 2>&1 || conda run -n {shlex.quote(conda_env)} --no-capture-output /codex/.venv/bin/python -c 'print(\"conda_ok\")' > /dev/null 2>&1")

    for key, value in extra_env.items():
        name = require_env_name(key)
        lines.append(f"export {name}={shlex.quote(value)}")

    # Launch from the runtime dir — NOT from /home/workspace, which would
    # shadow the installed vllm package with the source tree.
    lines.append(f"cd {shlex.quote(runtime_dir)}")

    # Build argv — every token individually quoted for bash safety.
    # Always use "vllm" directly; conda activation is handled inside _serve.sh
    # so the recorded PID is the vllm process itself, not a conda wrapper.
    argv_tokens = ["/codex/.venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", shlex.quote(model)]
    argv_tokens.extend(["--host", "0.0.0.0"])
    argv_tokens.extend(["--port", str(port)])
    if served_model_name:
        argv_tokens.extend(["--served-model-name", shlex.quote(served_model_name)])
    if tp is not None:
        argv_tokens.extend(["--tensor-parallel-size", str(tp)])
    if dp is not None and not multi_node_args:
        # In multi-node mode, data-parallel-size is set by multi_node_args
        # (via build_multi_node_dp_args) — avoid duplicate/conflicting flags.
        argv_tokens.extend(["--data-parallel-size", str(dp)])
    if multi_node_args:
        for arg in multi_node_args:
            argv_tokens.append(shlex.quote(arg))
    for arg in extra_args:
        argv_tokens.append(shlex.quote(_normalize_json_arg(arg)))

    cmd_str = " ".join(argv_tokens)
    stdout_log = f"{runtime_dir}/stdout.log"
    stderr_log = f"{runtime_dir}/stderr.log"
    pid_file = f"{runtime_dir}/pid"

    # Always write the vLLM command as a standalone script for clean quoting
    serve_script = f"{runtime_dir}/_serve.sh"
    lines.append(f"cat > {shlex.quote(serve_script)} << 'WORKSPACE_SERVE_EOF'")
    lines.append("#!/bin/bash")
    # Multi-node HCCL / Gloo env MUST be inside _serve.sh so they survive
    # the docker exec → nohup → bash chain.  (Exports in the outer script
    # may be lost across container boundaries.)
    # Respect user-provided env vars: skip auto-detection for vars already set.
    # Multi-node cross-node comm NIC selection.  Some machine types (e.g. A5)
    # MUST use a dedicated NIC (data0.3001) — its IP/iface is used directly.
    # Otherwise fall back to sibling-NIC auto-detection from the SSH local_ip.
    if multi_node_args and (comm_iface or local_ip):
        if comm_iface:
            lines.append(f'echo "using mandatory comm NIC: {comm_iface} (local_ip={local_ip})"')
            lines.append(f'[ -n "$HCCL_IF_IP" ] || export HCCL_IF_IP={local_ip}')
            lines.append(f'[ -n "$GLOO_SOCKET_IFNAME" ] || export GLOO_SOCKET_IFNAME={comm_iface}')
            lines.append(f'[ -n "$IP_SOCKET_IFNAME" ] || export IP_SOCKET_IFNAME={comm_iface}')
            lines.append(f'[ -n "$HCCL_SOCKET_IFNAME" ] || export HCCL_SOCKET_IFNAME={comm_iface}')
        elif local_ip:
            lines.append(
                f"NIC_NAME=$(ifconfig | grep -B1 'inet {local_ip} ' "
                f"| grep -v 'inet {local_ip} ' | grep -v '^--$' "
                f"| tail -1 | cut -d: -f1 | xargs)"
            )
            lines.append(f'echo "detected NIC: $NIC_NAME"')
            lines.append(f'[ -n "$HCCL_IF_IP" ] || export HCCL_IF_IP={local_ip}')
            lines.append(f'if [ -n "$NIC_NAME" ]; then')
            lines.append(f'  [ -n "$GLOO_SOCKET_IFNAME" ] || export GLOO_SOCKET_IFNAME=$NIC_NAME')
            lines.append(f'  [ -n "$IP_SOCKET_IFNAME" ] || export IP_SOCKET_IFNAME=$NIC_NAME')
            lines.append(f'  [ -n "$HCCL_SOCKET_IFNAME" ] || export HCCL_SOCKET_IFNAME=$NIC_NAME')
            lines.append(f'fi')
    if conda_env:
        lines.append(f'eval "$(conda shell.bash hook)"')
        lines.append(f"conda activate {shlex.quote(conda_env)}")
    lines.append(f"exec {cmd_str}")
    lines.append("WORKSPACE_SERVE_EOF")
    lines.append(f"chmod +x {shlex.quote(serve_script)}")

    if wrap_script:
        # External wrapper: receives serve script path and runtime dir as args.
        # The wrapper decides how to launch (e.g. msprof wrapping, strace, etc.)
        lines.append(
            f"nohup bash {shlex.quote(wrap_script)}"
            f" {shlex.quote(serve_script)} {shlex.quote(runtime_dir)}"
            f" > {shlex.quote(stdout_log)}"
            f" 2> {shlex.quote(stderr_log)}"
            f" </dev/null &"
        )
    else:
        lines.append(
            f"nohup bash {shlex.quote(serve_script)}"
            f" > {shlex.quote(stdout_log)}"
            f" 2> {shlex.quote(stderr_log)}"
            f" </dev/null &"
        )

    lines.append("_PID=$!")
    lines.append("disown $_PID")
    lines.append(f"echo $_PID > {shlex.quote(pid_file)}")
    lines.append("echo $_PID")

    return "\n".join(lines)


def wait_for_devices_free(target: ExecutionTarget, devices: set[int], *, timeout: int = 45) -> bool:
    if not devices:
        return True
    password, key_file = _target_auth(target)
    container = target.container_name
    deadline = time.time() + timeout
    while True:
        try:
            npu_info = probe_npus(target.host_endpoint, password=password, key_file=key_file, container=container)
            busy = {int(dev) for dev in npu_info.get("busy", {}) if str(dev).isdigit()}
            if not (devices & busy):
                return True
        except Exception:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(3)


def read_remote_tail(target: ExecutionTarget, remote_path: str, lines: int = 30) -> str:
    r = ssh_exec_target(target, f"tail -{lines} {shlex.quote(remote_path)} 2>/dev/null || echo '(no log)'", check=False)
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Environment error diagnosis
# ---------------------------------------------------------------------------

_ENV_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("Failed to infer device type", "device-type"),
    ("No module named 'vllm_ascend'", "missing-vllm-ascend"),
    ("No module named 'vllm'", "missing-vllm"),
    ("No module named 'torch_npu'", "missing-torch-npu"),
    ("cannot open shared object file", "missing-so"),
    ("libhccl.so", "missing-so"),
    ("RuntimeError:.*torch_npu", "torch-npu-error"),
    ("ImportError", "import-error"),
    ("ModuleNotFoundError", "module-not-found"),
]


def diagnose_env_failure(
    stderr_tail: str,
    host_id: str,
) -> dict[str, Any] | None:
    """Scan stderr for environment-related errors and return structured recovery guidance.

    Returns a dict with diagnosis details, or None if the error
    doesn't look environment-related.
    """
    if not stderr_tail:
        return None

    matched_tags: list[str] = []
    for pattern, tag in _ENV_ERROR_PATTERNS:
        if pattern in stderr_tail or re.search(pattern, stderr_tail):
            matched_tags.append(tag)

    if not matched_tags:
        return None

    recovery_target = f"--host {host_id}"
    return {
        "error_tags": sorted(set(matched_tags)),
        "cause": "remote Python package version mismatch",
        "recovery_command": (
            f"Ensure the remote container has the correct code and "
            f"dependencies installed. Rebuild vllm and vllm-ascend in "
            f"the correct order with pinned dependencies."
        ),
        "recovery_description": (
            "Re-install vllm and vllm-ascend in the correct order "
            "with pinned dependencies."
        ),
        "warning": (
            "Do NOT run bare `pip install` inside the container. "
            "The container has exact version locks between torch, torch_npu, "
            "vllm, and vllm-ascend. Manual pip install will break the "
            "dependency graph. Use the correct install flags "
            "(--no-deps, --no-build-isolation, VLLM_TARGET_DEVICE=empty, HuaweiCloud pip index)."
        ),
    }


# ---------------------------------------------------------------------------
# Multi-node helpers
# ---------------------------------------------------------------------------

def _preflight_node(target, model: str, label: str):
    """Check SSH connectivity and model path on a single node.

    Returns an error dict on failure, or ``None`` on success.
    """
    emit_progress("preflight", f"[{label}] verifying SSH connectivity")
    r = ssh_exec_target(target, "echo ok", check=False)
    if r.returncode == 255:
        return {
            "host": target.alias,
            "error": f"SSH connection to {target.alias} failed (rc=255)",
            "detail": {
                "ssh_stdout": (r.stdout or "").strip()[-500:],
                "ssh_stderr": (r.stderr or "").strip()[-500:],
            },
        }
    if r.returncode != 0:
        return {
            "host": target.alias,
            "error": f"remote command failed on {target.alias} (rc={r.returncode})",
            "detail": {"stdout": (r.stdout or "").strip()[-500:], "stderr": (r.stderr or "").strip()[-500:]},
        }

    emit_progress("preflight", f"[{label}] checking model path: {model}")
    r = ssh_exec_target(target, f"test -d {shlex.quote(model)} || test -f {shlex.quote(model)}", check=False)
    if r.returncode != 0:
        return {
            "host": target.alias,
            "error": f"model path not found on {label}: {model}",
        }
    return None


def _probe_and_resolve_devices(target, password, key_file, tp, dp, explicit_devices, label):
    """Probe NPUs on a single node and resolve devices.

    Returns ``(devices_csv, npu_info)``. Raises ``RuntimeError`` on device
    selection failures.
    """
    container = target.container_name
    h_ep = target.host_endpoint
    try:
        npu_info = probe_npus(h_ep, password=password, key_file=key_file, container=container)
    except RuntimeError as exc:
        emit_progress("probe-npus", f"[{label}] NPU probe failed (non-fatal): {exc}")
        return (explicit_devices or "", {"devices": [], "free": [], "busy": {}})

    resolved = explicit_devices
    machine_type = (
        npu_info.get("machine_type")
        or target.record.get("host", {}).get("machine_type")
        or target.record.get("container", {}).get("machine_type")
    )
    resolved_devices, device_error = select_devices(
        npu_info, requested_devices=resolved, tp=tp, dp=dp, machine_type=machine_type,
    )
    if device_error:
        raise RuntimeError(f"[{label}] {device_error}")

    if resolved_devices is not None:
        resolved = resolved_devices
    emit_progress(
        "probe-npus", f"[{label}] devices: {resolved}",
        free=npu_info.get("free"),
        busy=list(npu_info.get("busy", {}).keys()),
    )
    return (resolved, npu_info)


def _stop_cluster_service(head_target, worker_targets, head_alias):
    """Stop a previously running multi-node cluster on the given targets."""
    cluster_state = load_cluster_state(head_alias)
    to_stop: list[tuple] = []

    if cluster_state and cluster_state.get("nodes"):
        for node_info in cluster_state["nodes"]:
            host = node_info.get("host", "")
            pid = node_info.get("pid")
            role = node_info.get("role", "unknown")
            if host == head_alias:
                tgt = head_target
            else:
                matching = [w for w in worker_targets if w.alias == host]
                tgt = matching[0] if matching else None
            if tgt and pid:
                to_stop.append((tgt, role, pid))
    else:
        head_state = load_serving_state(head_alias)
        if head_state and head_state.get("pid"):
            to_stop.append((head_target, "head", head_state["pid"]))
        for w in worker_targets:
            ws = load_serving_state(w.alias)
            if ws and ws.get("pid"):
                to_stop.append((w, "worker", ws["pid"]))

    for tgt, role, pid in to_stop:
        if not pid:
            continue
        if not check_alive(tgt, int(pid)):
            emit_progress("stop-existing", f"[{role} {tgt.alias}] already stopped (pid={pid})")
            continue
        emit_progress("stop-existing", f"[{role} {tgt.alias}] stopping (pid={pid})")
        ssh_exec_target(tgt, f"kill -2 {pid} 2>/dev/null || true; sleep 2; kill -15 {pid} 2>/dev/null || true", check=False)
        deadline = time.time() + 20
        while check_alive(tgt, int(pid)) and time.time() < deadline:
            time.sleep(1)
        if check_alive(tgt, int(pid)):
            emit_progress("stop-existing", f"[{role} {tgt.alias}] force-killing (pid={pid})")
            ssh_exec_target(tgt, f"kill -9 {pid} 2>/dev/null || true", check=False)
            time.sleep(2)
        emit_progress("stop-existing", f"[{role} {tgt.alias}] stopped")


def _run_multi_node(args, target, vllm_extra, extra_env, worker_extra_env, model, served_model_name,
                     tp, dp, launch_env, launch_extra_args):
    """Execute the multi-node launch flow. Returns exit code (0 = success)."""
    head_target = target
    head_alias = head_target.alias
    head_ep = head_target.endpoint
    password, key_file = _target_auth(head_target)
    cann_dir = getattr(args, "cann_dir", "/usr/local/Ascend")

    # ---- Phase 1: Parse and resolve worker targets ----
    try:
        worker_hosts = parse_worker_hosts(args.worker_hosts)
    except ValueError as exc:
        print_json({"status": "needs_input", "error": str(exc), "machine": head_alias})
        return 1

    if head_alias in worker_hosts:
        print_json({"status": "needs_input", "error": f"head host {head_alias!r} must not appear in --worker-hosts"})
        return 1

    emit_progress("resolve-target", f"head: {head_alias}, workers: {worker_hosts}")

    worker_targets = []
    for wh in worker_hosts:
        wt = resolve_worker_target(
            wh, head_record=target.record,
            key_file=getattr(args, "worker_key_file", None) or None,
            password=getattr(args, "worker_password", None) or None,
            container=getattr(args, "worker_container", None) or None,
            conda_env=getattr(args, "worker_conda_env", None) or None,
        )
        worker_targets.append(wt)

    # ---- Phase 2: Preflight all nodes ----
    preflight_errors = []
    for tgt, lbl in [(head_target, f"head {head_alias}")] + [(w, f"worker {w.alias}") for w in worker_targets]:
        err = _preflight_node(tgt, model, lbl)
        if err:
            preflight_errors.append(err)

    if preflight_errors:
        print_json({
            "status": "needs_input",
            "error": "preflight check failed on one or more nodes",
            "machine": head_alias,
            "failed_nodes": preflight_errors,
        })
        return 1

    # ---- Phase 3: Stop existing cluster ----
    _stop_cluster_service(head_target, worker_targets, head_alias)

    # ---- Phase 4: Probe NPUs and select devices for multi-node DP ----
    # Ascend NPU: each physical NPU contains 1-2 logical dies.  vLLM's
    # --tensor-parallel-size counts LOGICAL DIES, not physical devices
    # (validated against vllm-0.23.0-tp2-dp2-ep: a single 2-die card accepts
    # --tp 2, so --tp N means N dies).  The script's --tp is in die units.
    #   --tp 2  →  per-node TP=2 dies, dp_local = vllm_dies // 2
    # Dies per physical NPU differs by machine:
    #   A3 (Ascend910): 8 NPU x 2 dies = 16, TP*DP <= 16.
    #   A5 (Ascend950DT): 8 NPU x 1 die  = 8,  TP*DP <= 8.
    # We derive the default TP from the head serving record and refine it from
    # the auto-detected head NPU model right after the device probe below.
    _head_mtype = (
        head_target.record.get("host", {}).get("machine_type")
        or head_target.record.get("container", {}).get("machine_type")
    )
    DIES_PER_DEVICE = (MACHINE_HARDWARE.get((_head_mtype or "").upper()) or {}).get("per_device_dies", 2)
    nnodes = 1 + len(worker_targets)
    # On Ascend a single physical board can expose multiple logical dies that
    # vLLM treats as its tensor-parallel devices.  npu-smi reports the board,
    # not the dies vLLM sees, so we prefer the vLLM-visible device count (from
    # probe_npus) whenever it is available, and fall back to the legacy
    # physical-board model otherwise (e.g. docker mode without /dev/davinci*).
    per_node_tp_devices = tp  # legacy: physical boards required per node for TP
    per_node_tp_dies = (tp * DIES_PER_DEVICE) if tp is not None else None

    # Multi-node: select ALL free devices (tp=0 sentinel), then trim to
    # dp_local_min * per_node_tp_devices so dp_size >= nnodes.
    # Respect user-specified ASCEND_RT_VISIBLE_DEVICES if provided.
    explicit_devices = launch_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    device_select_tp = 0 if worker_targets else tp

    # When True, ASCEND_RT_VISIBLE_DEVICES must NOT be injected from board ids;
    # vLLM auto-discovers its (die) devices on each node.
    auto_visible_devices = False

    head_npu: dict[str, Any] = {}
    try:
        head_devices, head_npu = _probe_and_resolve_devices(
            head_target, password, key_file, device_select_tp, dp, explicit_devices, f"head {head_alias}")
    except RuntimeError as exc:
        print_json({"status": "needs_input", "error": str(exc), "machine": head_alias})
        return 1

    # Refine the per-device die count from the auto-detected head NPU model so
    # A3 (2 dies/NPU) vs A5 (1 die/NPU) sizing is correct even when the record
    # has no explicit machine_type.
    _detected_dies = head_npu.get("per_device_dies")
    if _detected_dies:
        DIES_PER_DEVICE = _detected_dies
        per_node_tp_dies = (tp * DIES_PER_DEVICE) if tp is not None else None
        emit_progress(
            "select-devices",
            f"{head_alias}: detected {DIES_PER_DEVICE} die(s) per NPU "
            f"(machine_type={head_npu.get('machine_type')})",
        )

    worker_devices_list: list[str | None] = []
    worker_npus: list[dict[str, Any]] = []
    for wt in worker_targets:
        wpw, wkf = _target_auth(wt)
        try:
            wd, wn = _probe_and_resolve_devices(wt, wpw, wkf, device_select_tp, dp, None, f"worker {wt.alias}")
            worker_devices_list.append(wd)
            worker_npus.append(wn or {})
        except RuntimeError as exc:
            print_json({"status": "needs_input", "error": str(exc), "machine": head_alias})
            return 1

    def _vllm_count(npu_info: dict[str, Any]) -> int:
        return int(npu_info.get("vllm_device_count") or 0)

    head_vllm = _vllm_count(head_npu)
    worker_vllm = [_vllm_count(wn) for wn in worker_npus]

    # ---- vLLM die-aware selection ----
    # If each node exposes its vLLM devices (dies) and every node has enough for
    # tp, the physical board count is not a reliable proxy (one board may hold
    # all dies).  dp_local = vllm_devices // tp ranks per node; we leave
    # ASCEND_RT_VISIBLE_DEVICES unset so vLLM auto-enumerates its devices.
    use_die_aware = bool(worker_targets) and tp is not None and head_vllm > 0 and all(v > 0 for v in worker_vllm)

    if use_die_aware:
        for label, vc in [("head", head_vllm)] + [(wt.alias, vc) for wt, vc in zip(worker_targets, worker_vllm)]:
            if vc < tp:
                print_json({"status": "needs_input",
                            "error": f"{label} exposes {vc} vLLM device(s) but tp={tp} requires at least {tp}"})
                return 1
        head_local = max(1, head_vllm // tp)
        worker_locals = [max(1, v // tp) for v in worker_vllm]
        dp_local_min = min(head_local, *worker_locals)
        # Keep physical board ids for status reporting only.
        head_keep = (parse_device_csv(head_devices) or [])[:dp_local_min]
        head_devices = ",".join(str(d) for d in head_keep)
        trimmed_workers = []
        for wd in worker_devices_list:
            wd_list = (parse_device_csv(wd) or [])[:dp_local_min]
            trimmed_workers.append(",".join(str(d) for d in wd_list))
        worker_devices_list = trimmed_workers
        auto_visible_devices = True
        emit_progress("select-devices",
                      f"vLLM die-aware: dp_local={dp_local_min}, dp_size={dp_local_min * nnodes}, "
                      "leaving ASCEND_RT_VISIBLE_DEVICES unset for auto-enumeration")

    elif worker_targets and per_node_tp_dies is not None:
        # Legacy physical-board model: compute dp_local in terms of DIES.
        # Use largest CONTIGUOUS free block per node — vLLM DP mode requires
        # ASCEND_RT_VISIBLE_DEVICES to be contiguous (e.g. 0,1,2,3 not 0,1,5,6).
        head_devs = parse_device_csv(head_devices) or []

        # When the user pinned ASCEND_RT_VISIBLE_DEVICES on the head, honor it
        # verbatim instead of re-trimming with largest_contiguous_free (which
        # would silently discard non-contiguous user selections). Auto-selected
        # devices still go through the contiguous-block trim below.
        if explicit_devices:
            if len(largest_contiguous_free(head_devs)) != len(head_devs):
                print_json({"status": "needs_input",
                            "error": f"ASCEND_RT_VISIBLE_DEVICES={explicit_devices} must be a contiguous "
                                     f"device list for multi-node DP mode"})
                return 1
            head_dp_local = max(1, len(head_devs) // per_node_tp_devices)
        else:
            head_contig = largest_contiguous_free(head_devs)
            head_dies = len(head_contig) * DIES_PER_DEVICE
            head_dp_local = max(1, head_dies // per_node_tp_dies)

        worker_contigs: list[list[int]] = []
        worker_dp_locals: list[int] = []
        for wd in worker_devices_list:
            wd_list = parse_device_csv(wd) or []
            w_contig = largest_contiguous_free(wd_list)
            worker_contigs.append(w_contig)
            w_dies = len(w_contig) * DIES_PER_DEVICE
            worker_dp_locals.append(max(1, w_dies // per_node_tp_dies))

        dp_local_min = min(head_dp_local, *worker_dp_locals) if worker_dp_locals else head_dp_local
        # devices needed = dp_local_min TP groups x per_node_tp_devices each
        devices_needed = dp_local_min * per_node_tp_devices

        if not explicit_devices:
            # Auto-selected devices are trimmed to match dp_local_min; a
            # user-specified selection is kept as-is (already validated).
            head_devs = head_contig[:devices_needed]
            head_devices = ",".join(str(d) for d in head_devs)

        trimmed_workers = []
        for w_contig in worker_contigs:
            w_contig = w_contig[:devices_needed]
            trimmed_workers.append(",".join(str(d) for d in w_contig))
        worker_devices_list = trimmed_workers

        # Validate minimums
        if len(head_devs) < per_node_tp_devices:
            print_json({"status": "needs_input",
                        "error": f"head node has {len(head_devs)} contiguous device(s) ({len(head_devs)*DIES_PER_DEVICE} dies) but per_node needs {per_node_tp_devices} devices ({per_node_tp_dies} dies)"})
            return 1
        for i, wd in enumerate(worker_devices_list):
            wc = len(parse_device_csv(wd) or [])
            if wc < per_node_tp_devices:
                print_json({"status": "needs_input",
                            "error": f"worker {worker_targets[i].alias} has {wc} contiguous device(s) ({wc*DIES_PER_DEVICE} dies) but per_node needs {per_node_tp_devices} devices ({per_node_tp_dies} dies)"})
                return 1

    elif per_node_tp_devices is not None:
        # Single-node: prefer vLLM-visible device count when available.
        if head_vllm > 0:
            if head_vllm < tp:
                print_json({"status": "needs_input",
                            "error": f"head node exposes {head_vllm} vLLM device(s) but tp={tp} requires at least {tp}"})
                return 1
            head_devices = ""
            auto_visible_devices = True
        else:
            head_count = len(parse_device_csv(head_devices) or []) if head_devices else 0
            if head_count < per_node_tp_devices:
                print_json({"status": "needs_input",
                            "error": f"head node has {head_count} device(s) but tp={tp} requires at least {tp}"})
                return 1

    # ---- Phase 5: Allocate ports ----
    service_port = args.port if args.port else find_free_port(head_target)
    emit_progress("allocate-port", f"service port {service_port}", port=service_port)
    # DP-based multi-node uses --data-parallel-rpc-port (replaces torch.distributed master port).
    # Exclude the already-allocated service port so the two can never collide.
    dp_rpc_port = args.master_port if args.master_port else find_free_port(head_target, exclude={service_port})
    emit_progress("allocate-port", f"DP RPC port {dp_rpc_port}", port=dp_rpc_port)

    # ---- Phase 6: Compute multi-node config (DP-based) ----
    mn_config = resolve_multi_node_dp_config(
        head_addr=head_target.endpoint.host,
        dp_size_local=dp_local_min,
        nnodes=nnodes,
        dp_rpc_port=dp_rpc_port,
    )

    # ---- Phase 7: Launch head node ----
    instance_ts = now_utc().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    head_runtime_dir = f"{head_target.runtime_base}/{RUNTIME_DIR_BASE}/{instance_ts}_{_safe_machine_segment(head_alias)}"
    head_mn_args = build_multi_node_dp_args(mn_config, node_index=0)
    wrap_script = getattr(args, "wrap_script", "") or ""

    emit_progress("launch", f"[head {head_alias}] starting vLLM serve")
    head_launch_env = dict(launch_env)
    if head_devices and not auto_visible_devices:
        # Respect user-specified devices if provided; otherwise use computed.
        head_launch_env.setdefault("ASCEND_RT_VISIBLE_DEVICES", head_devices)
    # vLLM's --tensor-parallel-size counts LOGICAL DIES, not physical boards:
    # --tp 2 uses 2 dies (on A3 that is a single physical NPU).  head_devices
    # above are die ids; when auto_visible_devices is True vLLM auto-enumerates
    # its devices on each node instead.
    vllm_tp = per_node_tp_devices if (worker_targets and tp is not None) else tp
    # --headless is strictly forbidden on the head node in multi-node DP:
    # only worker nodes run engine-only (headless); head must host the API server.
    head_extra_args = [a for a in launch_extra_args if a != "--headless"]
    if launch_extra_args and len(head_extra_args) != len(launch_extra_args):
        print("WARN: stripped --headless from head extra_args", file=sys.stderr)
    head_script = build_launch_script(
        runtime_dir=head_runtime_dir, model=model, served_model_name=served_model_name,
        port=service_port, tp=vllm_tp, dp=dp,
        extra_env=head_launch_env, extra_args=head_extra_args,
        multi_node_args=head_mn_args, wrap_script=wrap_script,
        conda_env=head_target.conda_env, cann_dir=cann_dir,
        local_ip=head_alias,
        comm_iface=head_npu.get("comm_iface") or "",
    )
    result = ssh_exec_target(head_target, head_script, check=False)
    if result.returncode != 0:
        print_json({"status": "failed", "error": f"head node launch failed on {head_alias}",
                     "stderr_tail": result.stderr[-1000:], "stdout_tail": result.stdout[-500:]})
        return 1
    head_pid_line = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
    try:
        head_pid = int(head_pid_line)
    except ValueError:
        print_json({"status": "failed", "error": f"cannot parse head PID: {head_pid_line!r}"})
        return 1
    emit_progress("launch", f"[head {head_alias}] pid={head_pid}", pid=head_pid)

    # ---- Phase 7b: Wait for head DP RPC port before launching workers ----
    # The head must have its data-parallel RPC server listening before workers
    # connect, otherwise HCCL cross-node handshake may timeout.
    if worker_targets:
        emit_progress("probe", f"waiting for head DP RPC port {dp_rpc_port} (timeout={args.dp_rpc_timeout}s)")
        rpc_deadline = time.time() + args.dp_rpc_timeout
        rpc_ready = False
        while time.time() < rpc_deadline:
            if not check_alive(head_target, head_pid):
                stderr_tail = read_remote_tail(head_target, f"{head_runtime_dir}/stderr.log")
                print_json({
                    "status": "failed",
                    "error": "head process exited before DP RPC port was ready",
                    "stderr_tail": stderr_tail,
                })
                return 1
            r = ssh_exec_target(
                head_target,
                f"(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep -q ':{dp_rpc_port} ' && echo ready || echo waiting",
                check=False,
            )
            if r.stdout.strip() == "ready":
                rpc_ready = True
                break
            time.sleep(3)
        if not rpc_ready:
            print_json({
                "status": "failed",
                "error": f"head DP RPC port {dp_rpc_port} not ready within {args.dp_rpc_timeout}s",
            })
            ssh_exec_target(head_target, f"kill -15 {head_pid} 2>/dev/null || true", check=False)
            return 1
        emit_progress("probe", f"head DP RPC port {dp_rpc_port} ready, launching workers")

    # ---- Phase 8: Launch worker nodes ----
    worker_results = []
    for i, wt in enumerate(worker_targets):
        worker_runtime_dir = f"{wt.runtime_base}/{RUNTIME_DIR_BASE}/{instance_ts}_{_safe_machine_segment(wt.alias)}"
        worker_mn_args = build_multi_node_dp_args(mn_config, node_index=i + 1)
        emit_progress("launch", f"[worker {wt.alias}] starting vLLM serve")
        worker_launch_env = dict(launch_env)
        worker_launch_env.update(worker_extra_env)
        wdev = worker_devices_list[i] if i < len(worker_devices_list) else None
        if wdev and not auto_visible_devices:
            # Respect user-specified devices if provided; otherwise use computed.
            worker_launch_env.setdefault("ASCEND_RT_VISIBLE_DEVICES", wdev)
        # build_multi_node_dp_args(node_index>0) already adds --headless for
        # workers (engine-only).  Strip any user-supplied --headless from the
        # pass-through extra_args to avoid a duplicate flag on the worker CLI.
        # Also strip --api-server-count: headless workers start no API server,
        # so vLLM rejects --api-server-count together with --headless.  The
        # flag is head-only (unified API entry point on the head node).
        worker_extra_args = []
        _skip_next = False
        _stripped_api_count = False
        for _a in launch_extra_args:
            if _skip_next:
                _skip_next = False
                continue
            if _a == "--headless":
                continue
            if _a == "--api-server-count":
                _stripped_api_count = True
                _skip_next = True
                continue
            if _a.startswith("--api-server-count="):
                _stripped_api_count = True
                continue
            worker_extra_args.append(_a)
        if _stripped_api_count:
            print("WARN: stripped --api-server-count from worker extra_args "
                  "(headless workers run no API server)", file=sys.stderr)
        worker_script = build_launch_script(
            runtime_dir=worker_runtime_dir, model=model, served_model_name=served_model_name,
            port=service_port, tp=vllm_tp, dp=dp,
            extra_env=worker_launch_env, extra_args=worker_extra_args,
            multi_node_args=worker_mn_args,
            conda_env=wt.conda_env, cann_dir=cann_dir,
            local_ip=wt.alias,
            comm_iface=(worker_npus[i].get("comm_iface") or "") if i < len(worker_npus) else "",
        )
        wr = ssh_exec_target(wt, worker_script, check=False)
        if wr.returncode != 0:
            emit_progress("launch", f"[worker {wt.alias}] failed, stopping head and launched workers")
            # Clean up already-launched workers first
            _worker_map = {w.alias: w for w in worker_targets}
            for prev_wr in worker_results:
                prev_pid = prev_wr.get("pid", 0)
                prev_alias = prev_wr.get("host", "")
                if prev_pid and prev_alias in _worker_map:
                    emit_progress("launch", f"[worker {prev_alias}] cleaning up pid={prev_pid}")
                    ssh_exec_target(
                        _worker_map[prev_alias],
                        f"kill -15 {prev_pid} 2>/dev/null || true",
                        check=False,
                    )
            # Stop head
            ssh_exec_target(head_target, f"kill -15 {head_pid} 2>/dev/null || true", check=False)
            print_json({"status": "failed", "error": f"worker {wt.alias} launch failed",
                         "stderr_tail": wr.stderr[-1000:], "failed_worker": wt.alias})
            return 1
        wpid_line = wr.stdout.strip().splitlines()[-1].strip() if wr.stdout.strip() else ""
        try:
            wpid = int(wpid_line)
        except ValueError:
            wpid = 0
        emit_progress("launch", f"[worker {wt.alias}] pid={wpid}", pid=wpid)
        worker_results.append({"host": wt.alias, "pid": wpid,
                               "devices": worker_devices_list[i] if i < len(worker_devices_list) else "",
                               "runtime_dir": worker_runtime_dir})

    # ---- Phase 9: Wait for cluster ready ----
    emit_progress("probe", f"waiting for cluster ready (timeout={args.health_timeout}s)")
    readiness = wait_for_ready(head_target, head_pid, service_port, head_runtime_dir, timeout=args.health_timeout)

    # ---- Phase 10: Persist cluster state ----
    nodes_state = [{
        "role": "head", "host": head_alias, "pid": head_pid, "port": service_port,
        "devices": head_devices, "dp_size_local": mn_config.dp_size_local,
        "base_url": f"http://{head_ep.host}:{service_port}", "runtime_dir": head_runtime_dir,
    }]
    for wr in worker_results:
        w_alive = False
        for wt in worker_targets:
            if wt.alias == wr["host"]:
                w_alive = check_alive(wt, int(wr["pid"])) if wr["pid"] else False
                break
        nodes_state.append({"role": "worker", **wr, "alive": w_alive})

    all_alive = all(n.get("alive", True) for n in nodes_state if n["role"] == "worker")
    cluster_state = {
        "cluster_type": "multi-node", "head_addr": mn_config.head_addr,
        "dp_rpc_port": mn_config.dp_rpc_port, "dp_size": mn_config.dp_size,
        "dp_size_local": mn_config.dp_size_local, "nnodes": mn_config.nnodes,
        "connection_mode": head_target.connection_mode,
        "model": model, "served_model_name": served_model_name, "tp": tp, "dp": dp,
        "service_port": service_port, "extra_args": launch_extra_args,
        "status": "ready" if (readiness["ready"] and all_alive) else "started",
        "nodes": nodes_state, "started_at": now_utc(),
    }
    save_cluster_state(head_alias, cluster_state)

    # Also save per-machine state for backward compat
    hs = {"model": model, "served_model_name": served_model_name, "tp": tp, "dp": dp,
          "devices": head_devices, "env": launch_env, "extra_args": launch_extra_args,
          "machine": head_alias, "pid": head_pid, "port": service_port,
          "base_url": f"http://{head_ep.host}:{service_port}", "runtime_dir": head_runtime_dir,
          "started_at": now_utc(), "status": "ready" if readiness["ready"] else "started",
          "cluster_type": "multi-node", "cluster_head": head_alias,
          "connection_mode": head_target.connection_mode, "cann_dir": cann_dir}
    save_serving_state(head_alias, hs)
    for i, wt in enumerate(worker_targets):
        ws = {"model": model, "tp": tp, "dp": dp, "machine": wt.alias,
              "devices": worker_devices_list[i] if i < len(worker_devices_list) else "",
              "pid": worker_results[i]["pid"] if i < len(worker_results) else 0,
              "runtime_dir": worker_results[i]["runtime_dir"] if i < len(worker_results) else "",
              "started_at": now_utc(), "status": "started",
              "cluster_type": "multi-node", "cluster_head": head_alias,
              "connection_mode": wt.connection_mode, "cann_dir": cann_dir}
        save_serving_state(wt.alias, ws)

    # ---- Phase 11: Output ----
    output = {
        "status": "ready" if (readiness["ready"] and all_alive) else "failed",
        "cluster_type": "multi-node", "machine": head_alias, "nnodes": mn_config.nnodes,
        "base_url": f"http://{head_ep.host}:{service_port}", "service_port": service_port,
        "dp_rpc_port": mn_config.dp_rpc_port, "dp_size": mn_config.dp_size,
        "head_pid": head_pid, "served_model_name": served_model_name, "model": model,
        "tp": tp, "nodes": nodes_state,
        "readiness": readiness, "all_workers_alive": all_alive,
    }
    if not readiness["ready"]:
        output["stderr_tail"] = readiness.get("stderr_tail") or read_remote_tail(head_target, f"{head_runtime_dir}/stderr.log")
    write_serving_history("serve_start", sys.argv[1:], output, alias=head_alias)
    print_json(output)
    return 0 if readiness["ready"] else 1


def wait_for_ready(
    target: ExecutionTarget,
    pid: int,
    port: int,
    runtime_dir: str,
    timeout: int,
) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + timeout
    health_ok = False
    models_ok = False

    while time.monotonic() < deadline:
        if not check_alive(target, pid):
            stderr_tail = read_remote_tail(target, f"{runtime_dir}/stderr.log")
            return {
                "ready": False,
                "alive": False,
                "error": "process exited before becoming ready",
                "stderr_tail": stderr_tail,
                "elapsed_seconds": round(time.monotonic() - start, 1),
            }

        if not health_ok:
            health_ok = check_health(target, port)
            if health_ok:
                emit_progress("probe-health", "/health returned 200")

        if health_ok and not models_ok:
            if check_models(target, port) is not None:
                models_ok = True
                emit_progress("probe-models", "/v1/models returned model list")

        if health_ok and models_ok:
            return {
                "ready": True,
                "alive": True,
                "elapsed_seconds": round(time.monotonic() - start, 1),
            }

        time.sleep(HEALTH_POLL_INTERVAL)

    return {
        "ready": False,
        "alive": check_alive(target, pid),
        "health": health_ok,
        "models": models_ok,
        "error": f"timed out after {timeout}s waiting for service",
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }


# ---------------------------------------------------------------------------
# Relaunch merge
# ---------------------------------------------------------------------------

def merge_with_previous(
    previous: dict[str, Any],
    *,
    model: str | None,
    served_model_name: str | None,
    tp: int | None,
    dp: int | None,
    extra_env: dict[str, str],
    unset_env: list[str],
    extra_args: list[str],
    unset_args: list[str],
) -> dict[str, Any]:
    merged = dict(previous)
    if model is not None:
        merged["model"] = model
    if served_model_name is not None:
        merged["served_model_name"] = served_model_name
    if tp is not None:
        merged["tp"] = tp
    if dp is not None:
        merged["dp"] = dp

    prev_env = dict(merged.get("env", {}))
    for key in unset_env:
        prev_env.pop(key, None)
    prev_env.update(extra_env)
    merged["env"] = prev_env

    prev_args = list(merged.get("extra_args", []))
    if unset_args:
        cleaned: list[str] = []
        i = 0
        unset_set = set(unset_args)
        while i < len(prev_args):
            arg = prev_args[i]
            matched = any(arg.startswith(u) for u in unset_set)
            if not matched:
                cleaned.append(arg)
                i += 1
                continue
            # Skip the matched flag.  ``--flag=value`` is self-contained (1 token).
            # For ``--flag value ...``, skip the flag and every immediately
            # following token that does *not* start with ``-`` (multi-value flags
            # like ``--quantization-param-file fp8.json scale.json`` are handled).
            if "=" in arg or arg in unset_set:
                i += 1
            else:
                i += 1
                while i < len(prev_args) and not prev_args[i].startswith("-"):
                    i += 1
        prev_args = cleaned
    prev_args.extend(extra_args)
    merged["extra_args"] = prev_args

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    # ---- direct SSH connection ----
    p.add_argument("--host", required=True, help="direct SSH host IP")
    p.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    p.add_argument("--user", default="root", help="SSH user (default: root)")
    p.add_argument("--password", help="SSH password (if not using key)")
    p.add_argument("--key-file", help="SSH private key file path")
    p.add_argument("--container", help="Docker container name to exec into")
    p.add_argument("--conda-env", help="conda environment name on the remote host "
                   "(mutually exclusive with --container; when set, commands run "
                   "directly on the host inside the named conda env)")
    p.add_argument("--mode", "--connection-mode", dest="connection_mode",
                   choices=["docker", "conda", "direct"], default="docker",
                   help="连接模式: docker (SSH 到宿主机 + docker exec 进容器), "
                   "conda (SSH 到宿主机 + 激活 conda 环境), "
                   "direct (SSH 直连容器/虚拟机 IP) — 默认: docker")
    p.add_argument("--cann-dir", default="/usr/local/Ascend",
                   help="CANN 安装根目录 (默认: /usr/local/Ascend)。"
                   "脚本会依次 source <cann-dir>/cann/set_env.sh 和 "
                   "<cann-dir>/nnal/atb/set_env.sh，缺失时回退手动导出驱动库路径")
    # --------------------------------------------------------
    p.add_argument("--model", help="absolute model weight path on the remote container")
    p.add_argument(
        "--served-model-name", "--served-name",
        dest="served_model_name",
        help="model name exposed via /v1/models (default: directory basename of --model)",
    )
    p.add_argument("--tp", "--tensor-parallel-size", dest="tp", type=int)
    p.add_argument("--dp", "--data-parallel-size", dest="dp", type=int)
    p.add_argument(
        "--extra-env", action="append", default=[],
        help="KEY=VALUE (repeatable)",
    )
    p.add_argument(
        "--devices", dest="devices",
        help="(deprecated) comma-separated NPU device ids to pin; forwarded as "
        "ASCEND_RT_VISIBLE_DEVICES.  Prefer --extra-env ASCEND_RT_VISIBLE_DEVICES=...",
    )
    p.add_argument(
        "--unset-env", action="append", default=[],
        help="remove an env var from inherited config (repeatable)",
    )
    p.add_argument(
        "--unset-args", action="append", default=[],
        help="remove a vllm arg prefix from inherited config (repeatable)",
    )
    p.add_argument("--relaunch", action="store_true", help="reuse previous config as base")

    p.add_argument("--port", type=int, help="force a specific port")
    p.add_argument(
        "--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT,
        help=f"seconds to wait for /health + /v1/models (default: {DEFAULT_HEALTH_TIMEOUT})",
    )
    p.add_argument(
        "--dp-rpc-timeout", type=int, default=DEFAULT_DP_RPC_TIMEOUT,
        help=f"seconds to wait for head data-parallel RPC port before launching workers "
             f"(default: {DEFAULT_DP_RPC_TIMEOUT})",
    )
    p.add_argument(
        "--wrap-script", default="",
        help="remote path to a wrapper script that receives the serve script path "
        "and runtime dir as $1 and $2. The wrapper controls how the service is launched "
        "(e.g. msprof wrapping). The serving skill is agnostic to what the wrapper does.",
    )
    # ---- multi-node (optional) ----
    p.add_argument(
        "--worker-hosts",
        help="comma-separated worker node IPs for multi-node deployment. "
        "Uses vLLM Data Parallel (--data-parallel-size / --data-parallel-size-local "
        "/ --data-parallel-start-rank) for cross-node scaling. "
        "TP is per-node; DP instances = devices_per_node / tp.",
    )
    p.add_argument("--worker-key-file", help="SSH private key for worker nodes (default: same as head)")
    p.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    p.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    p.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    p.add_argument(
        "--worker-extra-env", action="append", default=[],
        help="KEY=VALUE env vars for worker nodes only (repeatable)",
    )
    p.add_argument("--master-port", type=int, help="DP RPC port for cross-node communication (default: auto-allocate)")
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Split on bare -- to separate our args from vllm passthrough args
    own_argv: list[str] = argv
    vllm_extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        own_argv = argv[:idx]
        vllm_extra = argv[idx + 1:]

    args = build_parser().parse_args(own_argv)

    try:
        # ---- resolve target ----
        emit_progress("resolve-target", f"looking up {args.host}")
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
        record = target.record
        alias = target.alias
        ep = target.endpoint
        password, key_file = _target_auth(target)
        container = target.container_name
        runtime_base = target.runtime_base


        # ---- parse env overrides ----
        extra_env: dict[str, str] = {}
        for item in args.extra_env:
            if "=" not in item:
                print_json({"status": "failed", "error": f"bad --extra-env {item!r}, expected KEY=VALUE"})
                return 1
            k, _, v = item.partition("=")
            try:
                extra_env[require_env_name(k.strip())] = v
            except ValueError as exc:
                print_json({"status": "needs_input", "error": str(exc)})
                return 1

        # ---- parse worker-specific env overrides ----
        worker_extra_env: dict[str, str] = {}
        for item in getattr(args, "worker_extra_env", []) or []:
            if "=" not in item:
                print_json({"status": "failed", "error": f"bad --worker-extra-env {item!r}, expected KEY=VALUE"})
                return 1
            k, _, v = item.partition("=")
            try:
                worker_extra_env[require_env_name(k.strip())] = v
            except ValueError as exc:
                print_json({"status": "needs_input", "error": str(exc)})
                return 1

        # ---- resolve launch params (fresh or relaunch) ----
        if args.relaunch:
            previous = load_serving_state(
                alias,
                state_repo_root=target.state_repo_root,
            )
            if previous is None:
                print_json({
                    "status": "failed",
                    "error": f"no previous launch state for {alias}; cannot --relaunch without a prior start",
                    "machine": alias,
                })
                return 1
            merged = merge_with_previous(
                previous,
                model=args.model,
                served_model_name=args.served_model_name,
                tp=args.tp, dp=args.dp,
                extra_env=extra_env, unset_env=args.unset_env,
                extra_args=vllm_extra, unset_args=args.unset_args,
            )
            model = merged["model"]
            served_model_name = merged["served_model_name"]
            tp = merged.get("tp")
            dp = merged.get("dp")
            launch_env = merged.get("env", {})
            launch_extra_args = merged.get("extra_args", [])
            emit_progress("resolve-params", "merged delta onto previous config", relaunch=True)
        else:
            if not args.model:
                print_json({
                    "status": "needs_input",
                    "error": "--model is required for a fresh start",
                    "machine": alias,
                })
                return 1
            model = args.model
            served_model_name = args.served_model_name or Path(model).name
            tp = args.tp
            dp = args.dp
            launch_env = extra_env
            launch_extra_args = vllm_extra

        # Deprecated --devices compat shim: device pinning now flows through
        # ASCEND_RT_VISIBLE_DEVICES (--extra-env) since vLLM expects die ids.
        # Keep accepting the old flag so existing callers don't break; an explicit
        # --extra-env ASCEND_RT_VISIBLE_DEVICES=... always wins over the shim.
        if getattr(args, "devices", None):
            launch_env.setdefault("ASCEND_RT_VISIBLE_DEVICES", args.devices)

        # ---- multi-node branch ----
        if getattr(args, "worker_hosts", None):
            return _run_multi_node(
                args=args,
                target=target,
                vllm_extra=vllm_extra,
                extra_env=extra_env,
                worker_extra_env=worker_extra_env,
                model=model,
                served_model_name=served_model_name,
                tp=tp,
                dp=dp,
                launch_env=launch_env,
                launch_extra_args=launch_extra_args,
            )

        # Validate the new launch target before touching an existing service.
        # A mistyped model path should be a needs_input response, not a reason
        # to stop a currently running service for this machine.

        # -- preflight: verify SSH connectivity --
        emit_progress("validate", "verifying SSH connectivity")
        r = ssh_exec_target(target, "echo ok", check=False)
        if r.returncode == 255:
            print_json({
                "status": "needs_input",
                "error": (
                    f"SSH connection to {alias} failed (rc=255). "
                    "Check network reachability, SSH key, and server MaxStartups limit."
                ),
                "machine": alias,
                "mode": target.mode,
                "detail": {
                    "ssh_stdout": (r.stdout or "").strip()[-500:],
                    "ssh_stderr": (r.stderr or "").strip()[-500:],
                },
            })
            return 1
        if r.returncode != 0:
            print_json({
                "status": "needs_input",
                "error": (
                    f"remote command failed on {alias} (rc={r.returncode}). "
                    "Verify the container is running and docker exec works."
                ),
                "machine": alias,
                "mode": target.mode,
                "detail": {
                    "stdout": (r.stdout or "").strip()[-500:],
                    "stderr": (r.stderr or "").strip()[-500:],
                },
            })
            return 1

        # -- preflight: verify model path --
        emit_progress("validate", f"checking model path: {model}")
        r = ssh_exec_target(target, f"test -d {shlex.quote(model)} || test -f {shlex.quote(model)}", check=False)
        if r.returncode != 0:
            print_json({
                "status": "needs_input",
                "error": f"model path not found on remote container: {model}",
                "machine": alias,
                "mode": target.mode,
            })
            return 1

        # ---- stop existing service on this machine ----
        prev_state = load_serving_state(
            alias,
            state_repo_root=target.state_repo_root,
        )
        if prev_state and prev_state.get("pid"):
            old_pid = prev_state["pid"]
            scope = f"machine {alias}"
            if not check_alive(target, int(old_pid)):
                emit_progress("stop-existing", f"previous service for {scope} is already stopped (pid={old_pid})")
                prev_state["status"] = "stopped"
                prev_state["stopped_at"] = now_utc()
                save_serving_state(
                    alias,
                    prev_state,
                    state_repo_root=target.state_repo_root,
                )
            else:
                emit_progress("stop-existing", f"stopping previous service for {scope} (pid={old_pid})")
                ssh_exec_target(
                    target,
                    f"kill -2 {old_pid} 2>/dev/null || true; sleep 2; kill -15 {old_pid} 2>/dev/null || true",
                    check=False,
                )
                deadline = time.time() + 20
                while check_alive(target, int(old_pid)) and time.time() < deadline:
                    time.sleep(1)
                if check_alive(target, int(old_pid)):
                    emit_progress("stop-existing", f"previous service still alive, sending SIGKILL to pid={old_pid}")
                    ssh_exec_target(target, f"kill -9 {old_pid} 2>/dev/null || true", check=False)
                    time.sleep(2)
                old_devices = _parse_devices_csv(str(prev_state.get("devices") or ""))
                if old_devices:
                    emit_progress("stop-existing", f"waiting for old service devices to free: {sorted(old_devices)}")
                    wait_for_devices_free(target, old_devices)
                if not check_alive(target, int(old_pid)):
                    prev_state["status"] = "stopped"
                    prev_state["stopped_at"] = now_utc()
                    save_serving_state(
                        alias,
                        prev_state,
                        state_repo_root=target.state_repo_root,
                    )
                else:
                    prev_state["status"] = "stopping"
                    prev_state["status_checked_at"] = now_utc()
                    save_serving_state(
                        alias,
                        prev_state,
                        state_repo_root=target.state_repo_root,
                    )

        # ---- port ----
        if args.port:
            port = args.port
        else:
            emit_progress("allocate-port", "finding free port")
            port = find_free_port(target)
        emit_progress("allocate-port", f"port {port}", port=port)

        # ---- launch ----
        instance_ts = now_utc().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        runtime_dir = f"{runtime_base}/{RUNTIME_DIR_BASE}/{instance_ts}_{_safe_machine_segment(alias)}"

        wrap_script = getattr(args, "wrap_script", "") or ""
        if wrap_script:
            emit_progress("launch", f"starting vllm serve (wrapped by {wrap_script})")
        else:
            emit_progress("launch", "starting vllm serve")
        cann_dir = getattr(args, "cann_dir", "/usr/local/Ascend")
        script = build_launch_script(
            runtime_dir=runtime_dir,
            model=model,
            served_model_name=served_model_name,
            port=port,
            tp=tp, dp=dp,
            extra_env=launch_env,
            extra_args=launch_extra_args,
            wrap_script=wrap_script,
            conda_env=target.conda_env,
            cann_dir=cann_dir,
        )
        result = ssh_exec_target(target, script, check=False)
        if result.returncode != 0:
            print_json({
                "status": "failed",
                "error": "launch script failed",
                "stderr_tail": result.stderr[-1000:],
                "stdout_tail": result.stdout[-500:],
                "machine": alias,
            })
            return 1

        pid_line = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        try:
            pid = int(pid_line)
        except ValueError:
            print_json({
                "status": "failed",
                "error": f"cannot parse PID from launch output: {pid_line!r}",
                "machine": alias,
            })
            return 1

        emit_progress("launch", f"process started pid={pid}", pid=pid)

        state = {
            "model": model,
            "served_model_name": served_model_name,
            "tp": tp,
            "dp": dp,
            "env": launch_env,
            "extra_args": launch_extra_args,
            "machine": alias,
            "mode": target.mode,
            "connection_mode": target.connection_mode,
            "pid": pid,
            "port": port,
            "base_url": f"http://{ep.host}:{port}",
            "runtime_dir": runtime_dir,
            "log_stdout": f"{runtime_dir}/stdout.log",
            "log_stderr": f"{runtime_dir}/stderr.log",
            "started_at": now_utc(),
            "status": "starting",
        }
        if target.conda_env:
            state["conda_env"] = target.conda_env
        state["cann_dir"] = cann_dir
        if wrap_script:
            state["wrap_script"] = wrap_script
        save_serving_state(
            alias,
            state,
            state_repo_root=target.state_repo_root,
        )

        # ---- probe readiness ----
        emit_progress("probe", f"waiting for ready (timeout={args.health_timeout}s)")
        readiness = wait_for_ready(target, pid, port, runtime_dir, timeout=args.health_timeout)

        # ---- persist state (always, even if not ready — so stop can clean up) ----
        state["status"] = "ready" if readiness["ready"] else "started"
        state["readiness_checked_at"] = now_utc()
        save_serving_state(
            alias,
            state,
            state_repo_root=target.state_repo_root,
        )

        # ---- build output ----
        output: dict[str, Any] = {
            "status": "ready" if readiness["ready"] else "failed",
            "machine": alias,
            "mode": target.mode,

            "base_url": f"http://{ep.host}:{port}",
            "container_ip": ep.host,
            "port": port,
            "pid": pid,
            "served_model_name": served_model_name,
            "model": model,
            "tp": tp,
            "dp": dp,
            "log_stdout": f"{runtime_dir}/stdout.log",
            "log_stderr": f"{runtime_dir}/stderr.log",
            "runtime_dir": runtime_dir,
            "readiness": readiness,
        }
        if launch_env:
            output["env"] = launch_env
        if launch_extra_args:
            output["extra_args"] = launch_extra_args
        if wrap_script:
            output["wrap_script"] = wrap_script
        if not readiness["ready"]:
            stderr_tail = readiness.get("stderr_tail") or read_remote_tail(target, f"{runtime_dir}/stderr.log")
            output["stderr_tail"] = stderr_tail
            diagnosis = diagnose_env_failure(stderr_tail, alias)
            if diagnosis:
                output["env_diagnosis"] = diagnosis
                emit_progress("diagnosis", diagnosis["recovery_command"],
                              error_tags=diagnosis["error_tags"])

        write_serving_history("serve_start", sys.argv[1:], output, alias=alias)
        print_json(output)
        return 0 if readiness["ready"] else 1

    except Exception as exc:
        error_msg = str(exc)
        host_id = getattr(args, "host", None) or ""
        result: dict[str, Any] = {
            "status": "failed",
            "error": error_msg,
            "host": host_id,
        }
        diagnosis = diagnose_env_failure(error_msg, host_id)
        if diagnosis:
            result["env_diagnosis"] = diagnosis
        write_serving_history("serve_start", sys.argv[1:], result, alias=host_id)
        print_json(result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
